#!/usr/bin/env python3
"""EstunDriver：四乳头 XY PBVS + 可选低带宽 Z + 250Hz CRI（集成版）。

基于 estun_driver_integrated.py 改造：
  - 目标从「单个 camera->target」升级为「base_link 系下 5 个点逐个跟随」
    （默认序列: udder_frame -> teat_front_left -> teat_front_right
      -> teat_rear_right -> teat_rear_left，可参数化）
  - 每点可配 base 系附加偏移（如乳头喷洒偏移 [0,0,-0.05]）
  - 逐个跟随：到达(误差<tolerance 且稳定) -> 停留 stay_duration -> 切下一个
  - 保留 250Hz 唯一 CRI 线程、CAKalman1D x3，使用 Ruckig 统一三轴整形
  - 新增 workspace 安全检查（目标出界则停止跟踪）
"""
import rclpy, time, threading, math, json
from collections import deque

import numpy as np
import tf2_ros
from tf2_ros import TransformBroadcaster, TransformListener
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import SetBool
from vision_msgs.msg import Detection2DArray
from codroid import CodroidClient, CriRealtimeDispatcher, TrajectorySpace, CriFilterType
from .motion_limiter import RuckigVelocityLimiter
from .tool import convert_tcp_pose
from .target_input import TopicTargetAdapter

# ---------- PBVS / 整形参数（默认值，可用 yaml 覆盖） ----------
DT = 0.004                  # 250Hz 固定控制周期
LAMBDA_GAIN = 1.5
FEEDFORWARD_GAIN = 0.75
VMAX = 0.15                 # 每轴速度上限 m/s
AMAX = 0.8                  # 每轴加速度上限 m/s^2
JMAX = 1.5                  # 每轴加加速度上限 m/s^3
TRACK_TIMEOUT = 0.20        # target_age <= 0.20 -> TRACK
LOSS_TIMEOUT = 0.50         # > 0.50 -> HOLD；中间 -> SMOOTH_HALT
MAX_PREDICTION = 0.08
TARGET_A_MAX = 2.0
GATE_SIGMA = 3.0
MEASUREMENT_NOISE = 0.005
PROCESS_NOISE = 1.0
BUSY_WAIT_MARGIN = 0.0002   # 忙等窗口：1ms→0.2ms，CPU空转从25%降到~5%，迟到靠points_due补发兜底


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


def _freshness_time(current_is_home, target_time, last_receive_time):
    if current_is_home or last_receive_time is None:
        return target_time
    return last_receive_time


def _freshness_fade(age, track_timeout, loss_timeout):
    """Map real-vision age to TRACK / fade / halt without refreshing it."""
    if age <= track_timeout:
        return 1.0
    if age >= loss_timeout:
        return 0.0
    return 1.0 - (age - track_timeout) / (loss_timeout - track_timeout)


def _quat_to_matrix(q):
    """四元数(x,y,z,w) -> 旋转矩阵。"""
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _vec3(v):
    return [float(v[0]), float(v[1]), float(v[2])]


def _compute_pbvs_velocity(robot_pos, target_pos, target_v, target_a,
                           prediction_horizon, lambda_gain, feedforward_gain,
                           vmax, fade, xy_only):
    """Compute PBVS velocity; teat mode deliberately excludes visual Z."""
    result = [0.0, 0.0, 0.0]
    axes = range(2) if xy_only else range(3)
    for axis in axes:
        predicted_p = (target_pos[axis]
                       + target_v[axis] * prediction_horizon
                       + 0.5 * target_a[axis] * prediction_horizon ** 2)
        predicted_v = _clamp(
            target_v[axis] + target_a[axis] * prediction_horizon,
            -vmax, vmax)
        result[axis] = fade * (
            lambda_gain * (predicted_p - robot_pos[axis])
            + feedforward_gain * predicted_v)
    return result


def _compute_camera_pbvs_velocity(target_pos, target_v, target_a,
                                  desired_target, rotation_base_camera,
                                  prediction_horizon, lambda_gain,
                                  feedforward_gain, vmax, fade):
    """Compute camera-frame PBVS and rotate its velocity into base_link."""
    velocity_camera = [0.0, 0.0, 0.0]
    for axis in range(3):
        predicted_p = (target_pos[axis]
                       + target_v[axis] * prediction_horizon
                       + 0.5 * target_a[axis] * prediction_horizon ** 2)
        predicted_v = _clamp(
            target_v[axis] + target_a[axis] * prediction_horizon,
            -vmax[axis], vmax[axis])
        velocity_camera[axis] = fade * (
            lambda_gain[axis] * (predicted_p - desired_target[axis])
            + feedforward_gain[axis] * predicted_v)
    velocity_base = (np.asarray(rotation_base_camera, dtype=float)
                     @ np.asarray(velocity_camera, dtype=float))
    return velocity_camera, velocity_base.tolist()


def _limit_vector_norm(values, maximum):
    norm = float(np.linalg.norm(values))
    if norm <= maximum or norm == 0.0:
        return list(values)
    return (np.asarray(values, dtype=float) * (maximum / norm)).tolist()


def _command_errors(command, robot):
    return (math.hypot(command[0] - robot[0], command[1] - robot[1]),
            abs(command[2] - robot[2]))


def _radial_deadband_scale(error_xy, deadband):
    """Continuous radial deadband: zero at the edge, no velocity step."""
    error_xy = float(error_xy)
    deadband = max(0.0, float(deadband))
    if error_xy <= deadband or error_xy <= 0.0:
        return 0.0
    return 1.0 - deadband / error_xy


def _command_lead_scale(error, warning, halt):
    """Continuously slow the reference while the physical arm catches up."""
    error = max(0.0, float(error))
    warning = float(warning)
    halt = float(halt)
    if error <= warning:
        return 1.0
    if error >= halt:
        return 0.0
    return (halt - error) / (halt - warning)


class DynamicTargetInitializer:
    """Short moving-window gate for four-ID lock; never assumes a static cow."""

    def __init__(self, window_frames, min_full_frames, max_duration,
                 structure_tolerance=0.03):
        self.window_frames = int(window_frames)
        self.min_full_frames = int(min_full_frames)
        self.max_duration = float(max_duration)
        self.structure_tolerance = float(structure_tolerance)
        self.frames = deque(maxlen=self.window_frames)
        self.started_at = None

    @property
    def full_count(self):
        return sum(frame is not None for _, frame in self.frames)

    def reset(self):
        self.frames.clear()
        self.started_at = None

    @staticmethod
    def _signature(local):
        names = sorted(local)
        points = [np.asarray(local[name], dtype=float) for name in names]
        return np.asarray([
            np.linalg.norm(points[i] - points[j])
            for i in range(len(points)) for j in range(i + 1, len(points))
        ])

    def observe(self, observation, stamp):
        stamp = float(stamp)
        if (self.started_at is None
                or stamp - self.started_at > self.max_duration):
            self.reset()
            self.started_at = stamp
        self.frames.append((stamp, observation))
        full = [(index, frame, self._signature(frame[0]))
                for index, (_, frame) in enumerate(self.frames)
                if frame is not None]
        if len(full) < self.min_full_frames:
            return None
        best = []
        for _, _, reference in full:
            group = [item for item in full
                     if np.all(np.abs(item[2] - reference)
                               <= self.structure_tolerance)]
            if len(group) > len(best):
                best = group
        if len(best) < self.min_full_frames:
            return None
        # Local geometry is robustly averaged; base position uses newest frame
        # so a moving cow is not delayed by the initialization window.
        names = sorted(best[-1][1][0])
        local = {name: np.median(
            np.asarray([item[1][0][name] for item in best]), axis=0).tolist()
                 for name in names}
        base = {name: list(best[-1][1][1][name]) for name in names}
        return local, base


class ZErrorStats:
    """Per-cycle Z evidence for deciding whether V2 needs slow Z servo."""

    def __init__(self):
        self.values = []

    def reset(self):
        self.values.clear()

    def add(self, value):
        value = float(value)
        if math.isfinite(value):
            self.values.append(value)

    def summary(self):
        if not self.values:
            return {
                'z_error_mean': 0.0, 'z_error_abs_mean': 0.0,
                'z_error_max': 0.0, 'z_error_std': 0.0,
                'z_error_over_5mm_ratio': 0.0,
                'z_error_over_10mm_ratio': 0.0,
                'z_error_over_20mm_ratio': 0.0,
                'z_sample_count': 0,
            }
        values = np.asarray(self.values, dtype=float)
        absolute = np.abs(values)
        return {
            'z_error_mean': float(np.mean(values)),
            'z_error_abs_mean': float(np.mean(absolute)),
            'z_error_max': float(np.max(absolute)),
            'z_error_std': float(np.std(values)),
            'z_error_over_5mm_ratio': float(np.mean(absolute > 0.005)),
            'z_error_over_10mm_ratio': float(np.mean(absolute > 0.010)),
            'z_error_over_20mm_ratio': float(np.mean(absolute > 0.020)),
            'z_sample_count': int(values.size),
        }


class LatencyStats:
    """Bounded diagnostic samples; never performs I/O in the CRI loop."""

    def __init__(self, max_samples=1000):
        self.values = deque(maxlen=int(max_samples))

    def add(self, value):
        value = float(value)
        if math.isfinite(value) and value >= 0.0:
            self.values.append(value)

    def summary(self):
        if not self.values:
            return {'mean': 0.0, 'p95': 0.0, 'max': 0.0, 'count': 0}
        values = np.asarray(self.values, dtype=float)
        return {
            'mean': float(np.mean(values)),
            'p95': float(np.percentile(values, 95)),
            'max': float(np.max(values)),
            'count': int(values.size),
        }


class CAKalman1D:
    """一维恒加速度目标估计器，状态 [p, v, a]；时间基准 time.monotonic()。"""

    def __init__(self, process_noise=PROCESS_NOISE, measurement_noise=MEASUREMENT_NOISE):
        self.q = float(process_noise)
        self.r = float(measurement_noise) ** 2
        self.x = None
        self.P = None
        self.last_t = None

    @staticmethod
    def _F(dt):
        return np.array([[1.0, dt, 0.5 * dt * dt],
                         [0.0, 1.0, dt],
                         [0.0, 0.0, 1.0]], dtype=float)

    def _Q(self, dt):
        return self.q * np.array(
            [[dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
             [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
             [dt**3 / 6.0, dt**2 / 2.0, dt]], dtype=float)

    def reset(self, p, t):
        self.x = np.array([float(p), 0.0, 0.0], dtype=float)
        self.P = np.diag([self.r, 0.04, 1.0])
        self.last_t = float(t)

    def update(self, p, t, gate_sigma):
        if self.x is None or self.last_t is None:
            self.reset(p, t)
            return True
        dt = float(t) - self.last_t
        if dt <= 0.0:
            return False
        F = self._F(dt)
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + self._Q(dt)
        innov = float(p - x_pred[0])
        S = float(P_pred[0, 0] + self.r)
        if abs(innov) > gate_sigma * math.sqrt(max(S, 1e-12)):
            return False
        K = P_pred[:, 0] / S
        self.x = x_pred + K * innov
        self.P = (np.eye(3) - np.outer(K, np.array([1.0, 0.0, 0.0]))) @ P_pred
        self.last_t = float(t)
        return True


class FixedCodroidClient(CodroidClient):
    def StopCriDataPush(self, ip=None, port=None):
        return super().StopCriDataPush(ip=ip or self.local_ip, port=port or self.udp_port)


class EstunDriver(Node):
    def __init__(self):
        super().__init__('estun_driver')
        self._declare_parameters()
        self._load_parameters()

        self.robot = FixedCodroidClient(host=self.robot_ip, local_ip=self.local_ip, udp_port=self.udp_port)
        self.robot.__enter__()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.topic_adapter = TopicTargetAdapter(
            self.tf_buffer,
            self.base,
            dict(zip(self.sequence, self.offsets)),
            self.min_target_confidence,
        )
        self.create_subscription(
            Detection2DArray, self.tracked_detections_topic,
            self._vision_stamp_cb, 1)
        self.stop_event = threading.Event()

        self.init_robot()
        self.tf2_estun_tcp_publisher()
        self.estun_tcp_publisher()

        self._init_sequence_state()
        # 安全使能：默认不跟随，调用 /estun_driver/enable 后才开始序列
        self.create_service(SetBool, '/estun_driver/enable', self._enable_cb)
        # 50Hz只处理Topic重试/状态推进；250Hz只留给CRI线程。
        self.create_timer(1.0 / 50.0, self.update_control_target)
        self.start_cri()

    def _declare_parameters(self):
        self.declare_parameter('robot_ip', '192.168.1.136')
        self.declare_parameter('local_ip', '192.168.1.10')
        self.declare_parameter('udp_port', 10086)
        self.declare_parameter('controller_udp_port', 9030)
        self.declare_parameter('cri_period_ms', 4)
        self.declare_parameter('start_buffer', 5)
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter(
            'tracked_detections_topic', '/udder/tracked_detections')
        self.declare_parameter('min_target_confidence', 0.50)
        self.declare_parameter('topic_tf_wait_timeout', 0.05)
        self.declare_parameter('udder_frame', 'udder_frame')
        self.declare_parameter('target_sequence', ['udder_frame', 'teat_front_left',
                                                   'teat_front_right', 'teat_rear_right', 'teat_rear_left'])
        self.declare_parameter('target_offsets',
                                [0.0, 0.0, 0.15, 0.0, 0.0, 0.15, 0.0, 0.0, 0.15,
                                 0.0, 0.0, 0.15, 0.0, 0.0, 0.15])
        self.declare_parameter('dry_run', False)
        self.declare_parameter('return_home', True)
        self.declare_parameter('loop', False)
        self.declare_parameter('init_window_frames', 5)
        self.declare_parameter('init_min_full_frames', 3)
        self.declare_parameter('init_max_duration', 0.35)
        # fixed保持原高度；low_bandwidth以独立死区和v/a/j慢速跟随视觉Z。
        self.declare_parameter('z_control_mode', 'fixed')
        self.declare_parameter('fixed_z_source', 'first_target')
        self.declare_parameter('z_fixed_offset', 0.0)
        self.declare_parameter('z_low_bandwidth_gain', 0.25)
        self.declare_parameter('z_low_bandwidth_vmax', 0.010)
        self.declare_parameter('z_low_bandwidth_amax', 0.030)
        self.declare_parameter('z_low_bandwidth_jmax', 0.080)
        self.declare_parameter('z_low_bandwidth_deadband', 0.005)
        self.declare_parameter('z_deviation_warn', 0.015)
        self.declare_parameter('z_deviation_halt', 0.030)
        self.declare_parameter('z_deviation_halt_duration', 0.10)
        self.declare_parameter('command_error_warn', 0.020)
        self.declare_parameter('command_error_halt', 0.040)
        self.declare_parameter('home_stay_duration', 1.0)
        self.declare_parameter('home_arrive_tolerance', 0.008)
        self.declare_parameter('z_approach_tolerance', 0.008)
        self.declare_parameter('arrive_tolerance_xy', 0.008)
        self.declare_parameter('flyby_enabled', True)
        self.declare_parameter('flyby_tolerance_xy', 0.010)
        self.declare_parameter('switch_speed_xy', 0.030)
        self.declare_parameter('arrive_stable_duration', 0.3)
        self.declare_parameter('stay_duration', 1.0)
        self.declare_parameter('workspace_min', [-1.0, -1.0, 0.0])
        self.declare_parameter('workspace_max', [1.0, 1.0, 1.0])
        # PBVS 参数可覆盖
        self.declare_parameter('lambda_gain', LAMBDA_GAIN)
        self.declare_parameter('feedforward_gain', FEEDFORWARD_GAIN)
        self.declare_parameter('vmax', VMAX)
        self.declare_parameter('amax', AMAX)
        self.declare_parameter('jmax', JMAX)
        # 滤波 / 超时 / 预测参数
        self.declare_parameter('process_noise', PROCESS_NOISE)
        self.declare_parameter('measurement_noise', MEASUREMENT_NOISE)
        self.declare_parameter('track_timeout', TRACK_TIMEOUT)
        self.declare_parameter('loss_timeout', LOSS_TIMEOUT)
        self.declare_parameter('max_prediction', MAX_PREDICTION)
        self.declare_parameter('target_a_max', TARGET_A_MAX)
        self.declare_parameter('gate_sigma', GATE_SIGMA)
        self.declare_parameter('busy_wait_margin', BUSY_WAIT_MARGIN)
        self.declare_parameter('enable_z_pbvs', True)
        self.declare_parameter('xy_deadband_m', 0.0)
        self.declare_parameter('pbvs_debug_rate_hz', 10.0)
        self.declare_parameter('pbvs_control_frame', 'base')
        self.declare_parameter('desired_target_cam', [0.0, 0.0, 0.5])
        self.declare_parameter('lambda_gain_xyz', [1.0, 1.0, 0.6])
        self.declare_parameter('feedforward_gain_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter('vmax_xyz', [0.22, 0.22, 0.06])
        self.declare_parameter('amax_xyz', [2.5, 2.5, 1.0])
        self.declare_parameter('jmax_xyz', [3.0, 3.0, 1.5])
        self.declare_parameter('vmax_total', 0.25)
        self.declare_parameter('process_noise_xyz', [0.06, 0.06, 0.05])
        self.declare_parameter('measurement_noise_xyz', [0.05, 0.05, 0.15])

    def _load_parameters(self):
        self.robot_ip = str(self.get_parameter('robot_ip').value)
        self.local_ip = str(self.get_parameter('local_ip').value)
        self.udp_port = int(self.get_parameter('udp_port').value)
        self.controller_udp_port = int(self.get_parameter('controller_udp_port').value)
        self.dt = float(self.get_parameter('cri_period_ms').value) / 1000.0
        self.start_buffer = int(self.get_parameter('start_buffer').value)
        self.base = str(self.get_parameter('base_frame').value)
        self.tracked_detections_topic = str(
            self.get_parameter('tracked_detections_topic').value)
        self.min_target_confidence = float(
            self.get_parameter('min_target_confidence').value)
        self.topic_tf_wait_timeout = float(
            self.get_parameter('topic_tf_wait_timeout').value)
        self.udder_frame = str(self.get_parameter('udder_frame').value)
        self.sequence = list(self.get_parameter('target_sequence').value)
        flat_offsets = list(self.get_parameter('target_offsets').value)
        assert len(flat_offsets) == 3 * len(self.sequence), \
            f"target_offsets 长度应为 3*N (N={len(self.sequence)})，实际 {len(flat_offsets)}"
        self.offsets = [list(flat_offsets[i*3:(i+1)*3]) for i in range(len(self.sequence))]
        self.dry_run = bool(self.get_parameter('dry_run').value)
        self.return_home = bool(self.get_parameter('return_home').value)
        self.loop = bool(self.get_parameter('loop').value)
        self.init_window_frames = int(
            self.get_parameter('init_window_frames').value)
        self.init_min_full_frames = int(
            self.get_parameter('init_min_full_frames').value)
        self.init_max_duration = float(
            self.get_parameter('init_max_duration').value)
        self.z_control_mode = str(self.get_parameter('z_control_mode').value)
        self.fixed_z_source = str(self.get_parameter('fixed_z_source').value)
        self.z_fixed_offset = float(self.get_parameter('z_fixed_offset').value)
        self.z_low_gain = float(
            self.get_parameter('z_low_bandwidth_gain').value)
        self.z_low_vmax = float(
            self.get_parameter('z_low_bandwidth_vmax').value)
        self.z_low_amax = float(
            self.get_parameter('z_low_bandwidth_amax').value)
        self.z_low_jmax = float(
            self.get_parameter('z_low_bandwidth_jmax').value)
        self.z_low_deadband = float(
            self.get_parameter('z_low_bandwidth_deadband').value)
        self.z_deviation_warn = float(
            self.get_parameter('z_deviation_warn').value)
        self.z_deviation_halt = float(
            self.get_parameter('z_deviation_halt').value)
        self.z_deviation_halt_duration = float(
            self.get_parameter('z_deviation_halt_duration').value)
        self.command_error_warn = float(
            self.get_parameter('command_error_warn').value)
        self.command_error_halt = float(
            self.get_parameter('command_error_halt').value)
        if not 0.0 < self.command_error_warn < self.command_error_halt:
            raise ValueError('command_error 必须满足 0 < warn < halt')
        if self.z_control_mode not in ('fixed', 'low_bandwidth'):
            raise ValueError('z_control_mode 必须是 fixed 或 low_bandwidth')
        if self.fixed_z_source not in ('robot_position', 'first_target'):
            raise ValueError('fixed_z_source 必须是 robot_position 或 first_target')
        self.home_stay_dur = float(self.get_parameter('home_stay_duration').value)
        self.home_arrive_tol = float(
            self.get_parameter('home_arrive_tolerance').value)
        self.z_approach_tol = float(
            self.get_parameter('z_approach_tolerance').value)
        self.arrive_tol_xy = float(
            self.get_parameter('arrive_tolerance_xy').value)
        if min(self.home_arrive_tol, self.z_approach_tol,
               self.arrive_tol_xy) <= 0.0:
            raise ValueError('到位容差必须大于0')
        self.flyby_enabled = bool(self.get_parameter('flyby_enabled').value)
        self.flyby_tol_xy = float(
            self.get_parameter('flyby_tolerance_xy').value)
        self.switch_speed_xy = float(
            self.get_parameter('switch_speed_xy').value)
        if self.switch_speed_xy < 0.0:
            raise ValueError('switch_speed_xy 必须大于等于0')
        self.arrive_stable = float(self.get_parameter('arrive_stable_duration').value)
        self.stay_dur = float(self.get_parameter('stay_duration').value)
        self.ws_min = np.asarray(self.get_parameter('workspace_min').value, float)
        self.ws_max = np.asarray(self.get_parameter('workspace_max').value, float)
        # PBVS 参数
        self.lambda_gain = float(self.get_parameter('lambda_gain').value)
        self.feedforward = float(self.get_parameter('feedforward_gain').value)
        self.vmax = float(self.get_parameter('vmax').value)
        self.amax = float(self.get_parameter('amax').value)
        self.jmax = float(self.get_parameter('jmax').value)
        self.process_noise = float(self.get_parameter('process_noise').value)
        self.measurement_noise = float(self.get_parameter('measurement_noise').value)
        self.track_timeout = float(self.get_parameter('track_timeout').value)
        self.loss_timeout = float(self.get_parameter('loss_timeout').value)
        self.max_prediction = float(self.get_parameter('max_prediction').value)
        self.target_a_max = float(self.get_parameter('target_a_max').value)
        self.gate_sigma = float(self.get_parameter('gate_sigma').value)
        self.busy_wait_margin = float(self.get_parameter('busy_wait_margin').value)
        self.enable_z_pbvs = bool(self.get_parameter('enable_z_pbvs').value)
        self.xy_deadband_m = float(self.get_parameter('xy_deadband_m').value)
        self.pbvs_debug_rate_hz = float(
            self.get_parameter('pbvs_debug_rate_hz').value)
        if self.xy_deadband_m < 0.0:
            raise ValueError('xy_deadband_m 必须大于等于0')
        if not 1.0 <= self.pbvs_debug_rate_hz <= 20.0:
            raise ValueError('pbvs_debug_rate_hz 必须在 1~20Hz')
        self.pbvs_control_frame = str(
            self.get_parameter('pbvs_control_frame').value)
        if self.pbvs_control_frame not in ('base', 'camera'):
            raise ValueError('pbvs_control_frame 必须是 base 或 camera')
        self.desired_target_cam = _vec3(
            self.get_parameter('desired_target_cam').value)
        self.lambda_gain_xyz = _vec3(
            self.get_parameter('lambda_gain_xyz').value)
        self.feedforward_xyz = _vec3(
            self.get_parameter('feedforward_gain_xyz').value)
        self.vmax_xyz = _vec3(self.get_parameter('vmax_xyz').value)
        self.amax_xyz = _vec3(self.get_parameter('amax_xyz').value)
        self.jmax_xyz = _vec3(self.get_parameter('jmax_xyz').value)
        self.vmax_total = float(self.get_parameter('vmax_total').value)
        self.process_noise_xyz = _vec3(
            self.get_parameter('process_noise_xyz').value)
        self.measurement_noise_xyz = _vec3(
            self.get_parameter('measurement_noise_xyz').value)
        if (any(value <= 0.0 for values in (
                self.vmax_xyz, self.amax_xyz, self.jmax_xyz,
                self.process_noise_xyz, self.measurement_noise_xyz)
                for value in values)
                or self.vmax_total <= 0.0):
            raise ValueError('camera PBVS限幅和噪声参数必须大于0')

    def init_robot(self):
        self.robot.EnterRemoteModeViaAuto()
        self.robot.ClearSystemError()
        r = self.robot.SwitchOn()
        if r.err is not None:
            raise RuntimeError(f'机器人上使能失败: {r.err}')
        self.get_logger().info('上电---成功')
        self.robot.ToActual()
        self.robot.StartListenUdp()
        self.robot.WaitForCriData(timeout=5.0)
        self.robot.StartCriDataPush(ip=self.robot.local_ip, port=self.robot.udp_port,
                                    duration=int(self.dt * 1000))
        self.get_logger().info(f'开启CRI---成功:{self.robot.local_ip}:{self.robot.udp_port}')

    # ---------- 反馈：base_link -> tool0 动态 TF ----------
    def tf2_estun_tcp_publisher(self):
        self.tf_broadcaster = TransformBroadcaster(self)
        # 100Hz is sufficient for timestamp interpolation and avoids competing
        # with the unique 250Hz CRI thread for the Python GIL.
        self.tf_timer = self.create_timer(0.01, self.tf2_estun_tcp)

    def tf2_estun_tcp(self):
        data = self.robot.CriData
        if data is None:
            return
        position, quaternion, _ = convert_tcp_pose(data.tcp_pose)
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.base
        transform.child_frame_id = "tool0"
        transform.transform.translation.x = position[0]
        transform.transform.translation.y = position[1]
        transform.transform.translation.z = position[2]
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]
        self.tf_broadcaster.sendTransform(transform)

    # ---------- 反馈：estun_pose_topic (30Hz) ----------
    def estun_tcp_publisher(self):
        self._publisher = self.create_publisher(Float64MultiArray, "estun_pose_topic", 10)
        self._timer = self.create_timer(1.0 / 30.0, self.estun_tcp_publish)

    def estun_tcp_publish(self):
        data = self.robot.CriData
        if data is None:
            return
        msg = Float64MultiArray()
        msg.data = list(data.tcp_pose)
        self._publisher.publish(msg)

    # =====================================================================
    # 5 点序列状态 + 目标更新
    # =====================================================================
    def _init_sequence_state(self):
        self.status_pub = self.create_publisher(String, '/estun_driver/status', 1)
        self.action_pub = self.create_publisher(String, '/estun_driver/action_log', 1)
        self.pbvs_debug_pub = self.create_publisher(String, '/pbvs/debug', 1)
        self.create_timer(0.2, self.publish_status)  # 5Hz 状态
        self.create_timer(1.0 / self.pbvs_debug_rate_hz, self.publish_pbvs_debug)
        self.enabled = False
        self.target_lock = threading.Lock()
        self.target_pos = [None, None, None]
        self.target_v = [0.0, 0.0, 0.0]
        self.target_a = [0.0, 0.0, 0.0]
        self.target_update_time = None
        self.filters = EstunDriver._new_target_filters(self)
        self.R_base_camera = None
        self.v_cmd = [0.0, 0.0, 0.0]
        self.a_cmd = [0.0, 0.0, 0.0]
        self.pos_cmd = [None, None, None]
        self.motion_limiter = RuckigVelocityLimiter(self.dt)
        self.fixed_motion_z = None
        self.z_stats = ZErrorStats()
        self.z_target = None
        self.z_error = None
        self.z_halt_since = None
        self.z_soft_halt = False
        self.spray_allowed = False
        self._z_summary_published = False
        self.command_error_xy = 0.0
        self.command_error_z = 0.0
        self.command_velocity_scale = 1.0
        self.command_soft_halt = False
        self.workspace_halt = False
        self.fixed_rx = self.fixed_ry = self.fixed_rz = 0.0
        # 序列状态
        self.home_pos = None
        self.current_is_home = False
        self.seq_index = 0
        self.last_receive_mono = None  # 最近一次收到目标测量的时刻（fade 新鲜度用）
        self._time_anchor = None  # (wall, mono) 时间锚点，用于 wall↔monotonic 统一换算
        self._last_wall = None
        self._last_mono = None
        self._cycle_local_targets: dict[str, list[float]] = {}
        self._cycle_base_targets: dict[str, list[float]] = {}
        self._topic_targets: dict[str, dict] = {}
        self._pending_topic_message = None
        self.topic_tf_drop_count = 0
        self.vision_processing_latency = LatencyStats()
        self.measurement_age_stats = LatencyStats()
        self.tf_wait_latency = LatencyStats()
        self.cri_lateness_stats = LatencyStats()
        self.cri_missed_cycles = 0
        self.latest_timing = {}
        self.initializer = DynamicTargetInitializer(
            self.init_window_frames, self.init_min_full_frames,
            self.init_max_duration)
        self.seq_phase = "waiting"    # waiting(待机) / track / staying / done
        self.phase_since = 0.0
        self.current_point = self.udder_frame
        self.pbvs_debug_snapshot = None

    def _new_target_filters(self):
        if getattr(self, 'pbvs_control_frame', 'base') == 'camera':
            return [CAKalman1D(
                self.process_noise_xyz[i], self.measurement_noise_xyz[i])
                for i in range(3)]
        return [CAKalman1D(self.process_noise, self.measurement_noise)
                for _ in range(3)]

    def publish_pbvs_debug(self):
        """低频发布最近一拍控制快照；250Hz CRI 线程不做 ROS 发布。"""
        snapshot = self.pbvs_debug_snapshot
        if snapshot is None:
            return
        (target, tool, error, raw_velocity, raw_velocity_camera,
         limited_velocity, velocity_command, acceleration, command,
         target_name, error_xy, inside_deadband, target_age,
         control_frame) = snapshot
        target_values = [round(v, 6) if v is not None else None for v in target]
        error_values = [round(v, 6) if v is not None else None for v in error]
        record = {
            'control_frame': control_frame,
            'target_base': target_values if control_frame == 'base' else None,
            'target_camera': target_values if control_frame == 'camera' else None,
            'tool_base': [round(v, 6) for v in tool],
            'error_base': error_values if control_frame == 'base' else None,
            'error_camera': error_values if control_frame == 'camera' else None,
            'raw_pbvs_velocity': [round(v, 6) for v in raw_velocity],
            'raw_pbvs_velocity_camera': [round(v, 6)
                                         for v in raw_velocity_camera],
            'limited_velocity': [round(v, 6) for v in limited_velocity],
            'v_cmd': [round(v, 6) for v in velocity_command],
            'limited_acceleration': [round(v, 6) for v in acceleration],
            'position_command': [round(v, 6) for v in command],
            'command_velocity_scale': round(self.command_velocity_scale, 4),
            'current_target': target_name,
            'e_xy': round(error_xy, 6) if error_xy is not None else None,
            'inside_xy_deadband': inside_deadband,
            'target_age': round(target_age, 6) if target_age is not None else None,
        }
        self.pbvs_debug_pub.publish(String(data=json.dumps(record)))

    def _log_action(self, event, point=None, error=None, seq_phase=None):
        """发布结构化动作日志，便于复盘。"""
        record = {
            "t": round(time.monotonic(), 3),
            "event": event,
            "point": point if point is not None else (
                "home" if self.current_is_home else self.current_point),
        }
        if error is not None:
            record["error_m"] = round(float(error), 4)
        if seq_phase is not None:
            record["phase"] = seq_phase
        self.action_pub.publish(String(data=json.dumps(record)))

    def publish_status(self):
        """发布序列状态，便于观察（不影响偏移量）。"""
        tcp = None
        data = self.robot.CriData
        if data is not None:
            tcp = [round(float(v) / 1000.0, 4) if i < 3 else round(float(v), 2)
                   for i, v in enumerate(data.tcp_pose[:3])]
        state = {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "seq_index": self.seq_index,
            "current_point": "home" if self.current_is_home else self.current_point,
            "seq_phase": self.seq_phase,
            "targets_locked": len(self._cycle_local_targets) == len(self.sequence),
            "locked_target_count": len(self._cycle_local_targets),
            "topic_tf_drop_count": self.topic_tf_drop_count,
            "timing": self.latest_timing,
            "vision_processing_latency": self.vision_processing_latency.summary(),
            "measurement_age": self.measurement_age_stats.summary(),
            "tf_wait_latency": self.tf_wait_latency.summary(),
            "cri_loop_lateness": self.cri_lateness_stats.summary(),
            "cri_missed_cycles": self.cri_missed_cycles,
            "target": [round(v, 4) if v is not None else None for v in self.target_pos],
            "home": [round(v, 4) for v in self.home_pos] if self.home_pos else None,
            "tcp": tcp,
            "fixed_motion_z": self.fixed_motion_z,
            "z_target": self.z_target,
            "z_error": self.z_error,
            "z_soft_halt": self.z_soft_halt,
            "spray_allowed": self.spray_allowed,
            "command_error_xy": self.command_error_xy,
            "command_error_z": self.command_error_z,
            "command_velocity_scale": self.command_velocity_scale,
            "command_soft_halt": self.command_soft_halt,
            "workspace_halt": self.workspace_halt,
            "z_stats": self.z_stats.summary(),
        }
        self.status_pub.publish(String(data=json.dumps(state)))

    def _enable_cb(self, request, response):
        if request.data and not self.enabled:
            self._clear_cycle_targets()
            data = self.robot.CriData
            if data is None:
                response.success = False
                response.message = 'no robot feedback'
                return response
            tcp = data.tcp_pose
            self.pos_cmd = [float(tcp[0]) / 1000.0,
                            float(tcp[1]) / 1000.0,
                            float(tcp[2]) / 1000.0]
            self.v_cmd = [0.0, 0.0, 0.0]
            self.a_cmd = [0.0, 0.0, 0.0]
            self.motion_limiter.reset(self.pos_cmd)
            self.command_velocity_scale = 1.0
            self.command_soft_halt = False
            self.enabled = True
            self._capture_home()
            self.seq_phase = "waiting"
            self.current_point = self.udder_frame
            self.get_logger().info('estun_driver 已使能，等待检测到目标后开始动作')
        elif not request.data:
            self.enabled = False
            self._clear_cycle_targets()
            self.get_logger().info('estun_driver 已禁用，停止跟随')
        response.success = True
        response.message = 'enabled' if self.enabled else 'disabled'
        return response

    def _capture_home(self):
        """记录 enable 时的机械臂 TCP 位置作为起始位置。"""
        data = self.robot.CriData
        if data is None:
            self.home_pos = None
            return
        tcp = data.tcp_pose
        self.home_pos = [float(tcp[0]) / 1000.0, float(tcp[1]) / 1000.0, float(tcp[2]) / 1000.0]
        self.get_logger().info(f'记录起始位置: {self.home_pos}')

    def _clear_cycle_targets(self):
        self._cycle_local_targets.clear()
        self._cycle_base_targets.clear()
        self._topic_targets.clear()
        self._pending_topic_message = None
        self.fixed_motion_z = None
        self.initializer.reset()
        self.z_stats.reset()
        self.z_target = None
        self.z_error = None
        self.z_halt_since = None
        self.z_soft_halt = False
        self.spray_allowed = False
        self._z_summary_published = False
        self.command_error_xy = 0.0
        self.command_error_z = 0.0
        self.command_velocity_scale = 1.0
        self.command_soft_halt = False
        self.workspace_halt = False

    def _record_z_measurement(self, z_target, now):
        if self.current_is_home:
            return
        self.z_target = float(z_target)
        if getattr(self, 'pbvs_control_frame', 'base') == 'camera':
            z_reference = self.desired_target_cam[2]
        elif self.z_control_mode == 'fixed':
            if self.fixed_motion_z is None:
                return
            z_reference = self.fixed_motion_z
        else:
            data = self.robot.CriData
            if data is None:
                return
            z_reference = float(data.tcp_pose[2]) / 1000.0
        self.z_error = self.z_target - z_reference
        self.z_stats.add(self.z_error)
        abs_error = abs(self.z_error)
        if abs_error > self.z_deviation_warn:
            self.get_logger().warning(
                f'Z工作距离偏差 {abs_error*1000.0:.1f}mm',
                throttle_duration_sec=1.0)
        if abs_error > self.z_deviation_halt:
            if self.z_halt_since is None:
                self.z_halt_since = now
            elif now - self.z_halt_since >= self.z_deviation_halt_duration:
                self.z_soft_halt = True
        else:
            self.z_halt_since = None
            self.z_soft_halt = False

    def _publish_z_summary(self):
        if self._z_summary_published:
            return
        summary = self.z_stats.summary()
        self.get_logger().info(
            '本轮Z偏差汇总: ' + json.dumps(summary, ensure_ascii=False))
        record = {'t': round(time.monotonic(), 3),
                  'event': 'z_summary', **summary}
        self.action_pub.publish(String(data=json.dumps(record)))
        self._z_summary_published = True

    def _vision_stamp_cb(self, message):
        receive_mono = time.monotonic()
        stamp_s = Time.from_msg(message.header.stamp).nanoseconds * 1e-9
        receive_wall = self.get_clock().now().nanoseconds * 1e-9
        self.vision_processing_latency.add(max(0.0, receive_wall - stamp_s))
        self.latest_timing = {
            'image_stamp': stamp_s,
            'detection_receive_time': receive_wall,
        }
        self._convert_topic_message(message, receive_mono, receive_mono)

    def _convert_topic_message(self, message, receive_mono, wait_started):
        try:
            frame = self.topic_adapter.convert(message)
            if (self.pbvs_control_frame == 'camera'
                    and self.R_base_camera is None):
                transform = self.tf_buffer.lookup_transform(
                    self.base, frame.frame_id,
                    Time(nanoseconds=int(round(frame.stamp * 1e9))))
                self.R_base_camera = _quat_to_matrix(
                    transform.transform.rotation)
        except (tf2_ros.TransformException, ValueError):
            with self.target_lock:
                self._pending_topic_message = (
                    message, receive_mono, wait_started)
            return False
        with self.target_lock:
            self._pending_topic_message = None
        transform_mono = time.monotonic()
        self.tf_wait_latency.add(transform_mono - receive_mono)
        self.latest_timing['tf_transform_time'] = (
            self.get_clock().now().nanoseconds * 1e-9)
        self._consume_topic_frame(frame, receive_mono)
        return True

    def _retry_pending_topic_message(self):
        with self.target_lock:
            pending = self._pending_topic_message
        if pending is None:
            return
        message, receive_mono, wait_started = pending
        now = time.monotonic()
        if now - wait_started > self.topic_tf_wait_timeout:
            with self.target_lock:
                if self._pending_topic_message is pending:
                    self._pending_topic_message = None
                    self.topic_tf_drop_count += 1
            return
        self._convert_topic_message(message, receive_mono, wait_started)

    def _consume_topic_frame(self, frame, receive_mono):
        measurement_mono, measurement_age = self._measurement_mono_time(frame.stamp)
        self.measurement_age_stats.add(measurement_age)
        control_points = (frame.camera_points
                          if self.pbvs_control_frame == 'camera'
                          else frame.base_points)
        with self.target_lock:
            for name, point in control_points.items():
                self._topic_targets[name] = {
                    'position': list(point),
                    'stamp': frame.stamp,
                    'measurement_mono': measurement_mono,
                    'receive_mono': receive_mono,
                    'confidence': frame.confidences[name],
                }

        if self.enabled and self.seq_phase == 'waiting':
            observation = None
            if set(self.sequence).issubset(frame.base_points):
                observation = (frame.camera_points, frame.base_points)
            result = self.initializer.observe(observation, frame.stamp)
            if result is not None and not self._cycle_base_targets:
                self._cycle_local_targets, self._cycle_base_targets = result
                self.get_logger().info(
                    f'Topic动态初始化完成：最近 {self.init_window_frames} 帧中 '
                    f'{self.initializer.full_count} 帧四点完整，锁定本轮ID')
                self._log_action('targets_locked')

        if (self.enabled and self.seq_phase == 'track'
                and not self.current_is_home
                and self.current_point in control_points):
            self._accept_target_measurement(
                control_points[self.current_point], frame.stamp,
                receive_mono=receive_mono)
            self.latest_timing['control_snapshot_time'] = (
                self.get_clock().now().nanoseconds * 1e-9)

    def _select_fixed_motion_z(self):
        """Select one static work height; target_offsets.z is already included."""
        data = self.robot.CriData
        if data is None:
            return False
        robot_z = float(data.tcp_pose[2]) / 1000.0
        if self.fixed_z_source == 'robot_position':
            fixed_z = robot_z + self.z_fixed_offset
        else:
            first = self._cycle_base_targets.get(self.sequence[0])
            if first is None:
                return False
            # first[2] already contains target_offsets.z. Add z_fixed_offset once.
            fixed_z = float(first[2]) + self.z_fixed_offset
        if not math.isfinite(fixed_z) or not self.ws_min[2] <= fixed_z <= self.ws_max[2]:
            self.get_logger().error(f'fixed_motion_z={fixed_z:.4f} 超出工作空间')
            return False
        self.fixed_motion_z = fixed_z
        return True

    def _switch_to_home(self):
        """5 点完成后切到回起始位置目标。"""
        if self.home_pos is None:
            self.seq_phase = "done"
            self.get_logger().info('无起始位置记录，直接完成')
            return
        self._publish_z_summary()
        self.current_is_home = True
        self.seq_phase = "track"
        self.phase_since = 0.0
        with self.target_lock:
            self.target_pos = list(self.home_pos)
            self.target_v = [0.0, 0.0, 0.0]
            self.target_a = [0.0, 0.0, 0.0]
            self.target_update_time = time.monotonic()
        self.get_logger().info(f'5 点完成，返回起始位置 {self.home_pos}')

    def _switch_target(self, index):
        """切换到第 index 个目标点，重置滤波器和到达状态。"""
        index = max(0, min(index, len(self.sequence) - 1))
        self.seq_index = index
        self.current_point = self.sequence[index]
        self.seq_phase = "track"
        # Starts only when the TCP first enters the final XY arrival tolerance.
        self.phase_since = 0.0
        self.target_pos = [None, None, None]
        self.target_v = [0.0, 0.0, 0.0]
        self.target_a = [0.0, 0.0, 0.0]
        self.target_update_time = None
        self.filters = EstunDriver._new_target_filters(self)
        with self.target_lock:
            topic_target = self._topic_targets.get(self.current_point)
        initial_target = (topic_target['position'] if topic_target is not None
                          else (None if getattr(self, 'pbvs_control_frame', 'base') == 'camera'
                                else self._cycle_base_targets.get(self.current_point)))
        if initial_target is not None:
            with self.target_lock:
                self.target_pos = list(initial_target)
                self.target_v = [0.0, 0.0, 0.0]
                self.target_a = [0.0, 0.0, 0.0]
                if topic_target is not None:
                    self.target_update_time = topic_target['measurement_mono']
                    self.last_receive_mono = topic_target['receive_mono']
                else:
                    # Cached fallback is not a new visual observation.
                    self.target_update_time = self.last_receive_mono
        self.get_logger().info(
            f'切换到目标[{index}] {self.current_point} '
            f'offset={self.offsets[index]}')
        self._log_action("switch")

    def _measurement_mono_time(self, stamp):
        """把 ROS 时间戳统一换算到 monotonic 轴。

        用 wall↔monotonic 锚点固定偏移换算，并检测 wall clock 跳变（NTP 校时等）
        自动刷新锚点，保证换算稳定。
        返回 (measurement_mono, age)，age 为测量年龄（管道延迟，0~1s）。
        """
        now_wall = self.get_clock().now().nanoseconds * 1e-9
        mono = time.monotonic()
        if self._time_anchor is None:
            self._time_anchor = (now_wall, mono)
            self._last_wall = now_wall
            self._last_mono = mono
        # 检测 wall clock 跳变（NTP 校时等）：用"时间增量差"，两次调用间 wall 与 mono 前进量不一致即跳变
        if self._last_wall is not None:
            wall_delta = now_wall - self._last_wall
            mono_delta = mono - self._last_mono
            if abs(wall_delta - mono_delta) > 0.5:
                self._time_anchor = (now_wall, mono)
                self.get_logger().warning('检测到系统时间跳变，刷新时间锚点',
                                          throttle_duration_sec=5.0)
        self._last_wall = now_wall
        self._last_mono = mono
        anchor_wall, anchor_mono = self._time_anchor
        offset = anchor_wall - anchor_mono
        measurement_mono = float(stamp) - offset
        age = mono - measurement_mono
        if age < 0.0:
            age = 0.0
            measurement_mono = mono
        elif age > 1.0:
            age = 1.0
            measurement_mono = mono - age
        return measurement_mono, age

    def update_control_target(self):
        """50Hz housekeeping; visual measurements arrive only by Topic."""
        if not self.enabled:
            return
        self._retry_pending_topic_message()
        if self.seq_phase == 'waiting':
            if len(self._cycle_base_targets) != len(self.sequence):
                self.get_logger().warning(
                    '等待Topic短时四点初始化完成',
                    throttle_duration_sec=1.0)
                return
            if not self.enable_z_pbvs or self.z_control_mode == 'low_bandwidth':
                self._switch_target(0)
                return
            if not self._select_fixed_motion_z():
                return
            data = self.robot.CriData
            robot_z = float(data.tcp_pose[2]) / 1000.0
            if (not self.dry_run
                    and abs(robot_z - self.fixed_motion_z) > self.z_approach_tol):
                self.seq_phase = 'z_approach'
                self.get_logger().info(
                    f'平滑移动到固定工作高度 Z={self.fixed_motion_z:.4f}m')
            else:
                self._switch_target(0)
            return
        if self.current_is_home and self.home_pos is not None:
            with self.target_lock:
                self.target_pos = list(self.home_pos)
                self.target_v = [0.0, 0.0, 0.0]
                self.target_a = [0.0, 0.0, 0.0]
                self.target_update_time = time.monotonic()

    def _accept_target_measurement(self, measured, stamp, receive_mono=None):
        if not all(math.isfinite(v) for v in measured):
            return False
        mono_now = time.monotonic() if receive_mono is None else float(receive_mono)
        measurement_time, _ = self._measurement_mono_time(stamp)
        updated_any = False
        with self.target_lock:
            last_t = self.target_update_time
            for i in range(3):
                if last_t is None or (mono_now - last_t) > self.loss_timeout:
                    self.filters[i].reset(measured[i], measurement_time)
                    updated_any = True
                elif self.filters[i].update(measured[i], measurement_time, self.gate_sigma):
                    updated_any = True
            if updated_any:
                for i in range(3):
                    self.target_pos[i] = float(self.filters[i].x[0])
                    self.target_v[i] = float(self.filters[i].x[1])
                    self.target_a[i] = _clamp(float(self.filters[i].x[2]), -self.target_a_max, self.target_a_max)
                self.target_update_time = measurement_time  # 测量时刻
                self.last_receive_mono = mono_now           # 收到时刻（fade 用）
                self._record_z_measurement(measured[2], mono_now)
        return updated_any

    # =====================================================================
    # 250Hz CRI 线程
    # =====================================================================
    def start_cri(self):
        self.dispatcher = CriRealtimeDispatcher(controller_ip=self.robot_ip,
                                                controller_udp_port=self.controller_udp_port,
                                                convert_to_si=False)
        r = self.robot.StartCriControl(filter_type=CriFilterType.AVERAGE,
                                       duration=int(self.dt * 1000),
                                       start_buffer=self.start_buffer)
        if r.err is not None:
            self.dispatcher.close()
            raise RuntimeError(f'开启实时控制失败: {r.err}')
        rt_ok = False
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            data = self.robot.CriData
            if data is not None and data.status.rt_control_mode:
                rt_ok = True
                break
            time.sleep(0.05)
        if not rt_ok:
            self.dispatcher.close()
            raise RuntimeError('控制器没有进入 rt_control_mode，禁止启动 250Hz 发送线程。')
        self.get_logger().info('rt_control_mode=True，进入实时控制')
        self.send_thread = threading.Thread(target=self.cri_send_loop, daemon=True)
        self.send_thread.start()
        self.get_logger().info(f'250Hz CRI 线程已启动: duration={int(self.dt*1000)}ms')

    def cri_send_loop(self):
        period = self.dt
        next_deadline = time.perf_counter() + period
        max_catch_up = self.start_buffer
        while not self.stop_event.is_set():
            remaining = next_deadline - time.perf_counter()
            if remaining > 0:
                if remaining > self.busy_wait_margin:
                    self.stop_event.wait(remaining - self.busy_wait_margin)
                while time.perf_counter() < next_deadline and not self.stop_event.is_set():
                    pass
            if self.stop_event.is_set():
                break
            now_perf = time.perf_counter()
            lateness = max(0.0, now_perf - next_deadline)
            self.cri_lateness_stats.add(lateness)
            missed = int(lateness / period)
            self.cri_missed_cycles += missed
            points_due = min(1 + missed, max_catch_up)
            if 1 + missed > max_catch_up:
                next_deadline = now_perf
            for _ in range(points_due):
                command = self.servo_step()
                if command is None:
                    break
                try:
                    self.dispatcher.SendCommand(command, TrajectorySpace.CARTESIAN)
                except OSError:
                    self.get_logger().error("CRI UDP send failed")
                    self.stop_event.set()
                    break
            next_deadline += points_due * period

    def servo_step(self):
        """单拍：读反馈 -> 序列推进/到达检测 -> 预测 PBVS -> 整形积分 -> 绝对命令。"""
        data = self.robot.CriData
        if data is None:
            return None
        tcp = data.tcp_pose
        robot_pos = [float(tcp[0]) / 1000.0, float(tcp[1]) / 1000.0, float(tcp[2]) / 1000.0]

        status = getattr(data, 'status', None)
        stop_reason = next((reason for attr, reason in (
            ('is_emergency_stop', '急停'),
            ('has_alarm', '机械臂报警'),
            ('collision_stop', '碰撞停止'),
            ('is_disabled', '机械臂未使能'),
        ) if status is not None and bool(getattr(status, attr, False))), None)
        if stop_reason is not None:
            self.enabled = False
            self.pos_cmd = robot_pos[:]
            self.v_cmd = [0.0, 0.0, 0.0]
            self.a_cmd = [0.0, 0.0, 0.0]
            self.motion_limiter.reset(robot_pos)
            self.get_logger().error(
                f'{stop_reason}，已停止CRI轨迹推进', throttle_duration_sec=1.0)
            return None

        if self.pos_cmd[0] is None:
            self.pos_cmd = robot_pos[:]
            self.motion_limiter.reset(robot_pos)
            self.fixed_rx = math.radians(float(tcp[3]))
            self.fixed_ry = math.radians(float(tcp[4]))
            self.fixed_rz = math.radians(float(tcp[5]))

        mono_now = time.monotonic()
        with self.target_lock:
            target_pos = list(self.target_pos)
            target_v = list(self.target_v)
            target_a = list(self.target_a)
            target_time = self.target_update_time

        # ---- 序列推进 + 到达检测 ----
        if target_pos[0] is not None and target_time is not None:
            self._advance_sequence(robot_pos, mono_now)
            # A fly-by switch must affect this same 250 Hz control cycle.
            with self.target_lock:
                target_pos = list(self.target_pos)
                target_v = list(self.target_v)
                target_a = list(self.target_a)
                target_time = self.target_update_time

        # ---- 计算期望速度 ----
        raw_pbvs_velocity = [0.0, 0.0, 0.0]
        raw_pbvs_velocity_camera = [0.0, 0.0, 0.0]
        error_xy = None
        inside_xy_deadband = False
        target_age = None
        if not self.enabled or self.dry_run or self.seq_phase == "waiting":
            desired_velocity = [0.0, 0.0, 0.0]
        elif self.seq_phase == 'z_approach':
            desired_velocity = [0.0, 0.0,
                                self.lambda_gain * (self.fixed_motion_z - robot_pos[2])]
        elif self.seq_phase in ("arrived", "staying", "done"):
            desired_velocity = [0.0, 0.0, 0.0]
        elif target_pos[0] is None or target_time is None:
            desired_velocity = [0.0, 0.0, 0.0]
        else:
            # fade 用"收到时刻"判断目标是否新鲜（数据持续更新则新鲜，不因管道延迟衰减）
            recv_time = _freshness_time(
                self.current_is_home, target_time, self.last_receive_mono
            )
            recv_age = mono_now - recv_time
            fade = _freshness_fade(
                recv_age, self.track_timeout, self.loss_timeout)
            # workspace 检查：目标出界 -> 不跟踪（安全）
            camera_tracking = (getattr(self, 'pbvs_control_frame', 'base') == 'camera'
                               and not self.current_is_home)
            target_inside = (True if camera_tracking else
                             (self._inside_workspace(target_pos)
                              if self.current_is_home
                              else self._inside_workspace_xy(target_pos)))
            if not target_inside:
                fade = 0.0
                self.get_logger().warning(f'目标 {self.current_point} 超出工作空间，停止跟踪',
                                          throttle_duration_sec=1.0)
            if self.workspace_halt:
                fade = 0.0
            # 预测时域：用测量年龄（从测量真实时刻到现在的延迟）+ 缓冲，精确外推到当前
            buffer_delay = self.start_buffer * self.dt
            measurement_age = max(0.0, mono_now - target_time)
            target_age = measurement_age
            prediction_horizon = min(measurement_age + buffer_delay, self.max_prediction)
            if camera_tracking:
                raw_pbvs_velocity_camera, desired_velocity = (
                    _compute_camera_pbvs_velocity(
                        target_pos, target_v, target_a,
                        self.desired_target_cam, self.R_base_camera,
                        prediction_horizon, self.lambda_gain_xyz,
                        self.feedforward_xyz, self.vmax_xyz, fade))
                if not self.enable_z_pbvs:
                    raw_pbvs_velocity_camera[2] = 0.0
                    desired_velocity = (self.R_base_camera
                                        @ np.asarray(raw_pbvs_velocity_camera)).tolist()
                error_xy = math.hypot(
                    target_pos[0] - self.desired_target_cam[0],
                    target_pos[1] - self.desired_target_cam[1])
                inside_xy_deadband = error_xy <= self.xy_deadband_m
                deadband_scale = _radial_deadband_scale(
                    error_xy, self.xy_deadband_m)
                raw_pbvs_velocity_camera[0] *= deadband_scale
                raw_pbvs_velocity_camera[1] *= deadband_scale
                desired_velocity = (self.R_base_camera
                                    @ np.asarray(raw_pbvs_velocity_camera)).tolist()
            else:
                desired_velocity = _compute_pbvs_velocity(
                    robot_pos, target_pos, target_v, target_a, prediction_horizon,
                    self.lambda_gain, self.feedforward, self.vmax, fade,
                    xy_only=not self.current_is_home)
            raw_pbvs_velocity = list(desired_velocity)
            if not self.current_is_home and not camera_tracking:
                error_xy = math.hypot(
                    target_pos[0] - robot_pos[0],
                    target_pos[1] - robot_pos[1])
                inside_xy_deadband = error_xy <= self.xy_deadband_m
                deadband_scale = _radial_deadband_scale(
                    error_xy, self.xy_deadband_m)
                desired_velocity[0] *= deadband_scale
                desired_velocity[1] *= deadband_scale
            if (not self.current_is_home
                    and not camera_tracking
                    and self.enable_z_pbvs
                    and self.z_control_mode == 'low_bandwidth'):
                predicted_z = (target_pos[2]
                               + target_v[2] * prediction_horizon
                               + 0.5 * target_a[2] * prediction_horizon ** 2)
                z_error = predicted_z - robot_pos[2]
                if abs(z_error) > self.z_low_deadband:
                    effective_error = z_error - math.copysign(
                        self.z_low_deadband, z_error)
                    desired_velocity[2] = fade * self.z_low_gain * effective_error
            if (not self.current_is_home and not camera_tracking
                    and not self.enable_z_pbvs):
                desired_velocity[2] = 0.0

        if (not self.current_is_home and target_pos[0] is not None
                and self.z_control_mode == 'low_bandwidth'):
            self.z_target = target_pos[2]
            self.z_error = (target_pos[2] - self.desired_target_cam[2]
                            if getattr(self, 'pbvs_control_frame', 'base') == 'camera'
                            else target_pos[2] - robot_pos[2])

        if (not self.current_is_home and target_pos[0] is not None
                and self.z_error is not None):
            if getattr(self, 'pbvs_control_frame', 'base') == 'camera':
                err_xy = math.hypot(
                    target_pos[0] - self.desired_target_cam[0],
                    target_pos[1] - self.desired_target_cam[1])
            else:
                err_xy = math.hypot(
                    target_pos[0] - robot_pos[0],
                    target_pos[1] - robot_pos[1])
            self.spray_allowed = (
                err_xy <= self.arrive_tol_xy
                and abs(self.z_error) <= self.z_deviation_warn
                and not self.z_soft_halt)
        else:
            self.spray_allowed = False

        if not self._inside_workspace(robot_pos):
            self.workspace_halt = True
            desired_velocity = [0.0, 0.0, 0.0]
            self.get_logger().error(
                '机器人TCP超出工作空间，停止生成运动参考',
                throttle_duration_sec=1.0)

        # ---- Ruckig 三轴统一 v/a/j 整形（仍在唯一 250Hz CRI 线程内） ----
        locked_z_command = self.pos_cmd[2]
        camera_tracking = (getattr(self, 'pbvs_control_frame', 'base') == 'camera'
                           and not self.current_is_home)
        max_velocity = []
        max_acceleration = []
        max_jerk = []
        for i in range(3):
            low_bandwidth_z = (i == 2 and not self.current_is_home
                               and not camera_tracking
                               and self.enable_z_pbvs
                               and self.z_control_mode == 'low_bandwidth')
            vmax = (self.vmax_xyz[i] if camera_tracking else
                    (self.z_low_vmax if low_bandwidth_z else self.vmax))
            amax = (self.amax_xyz[i] if camera_tracking else
                    (self.z_low_amax if low_bandwidth_z else self.amax))
            jmax = (self.jmax_xyz[i] if camera_tracking else
                    (self.z_low_jmax if low_bandwidth_z else self.jmax))
            desired_velocity[i] = _clamp(desired_velocity[i], -vmax, vmax)
            max_velocity.append(vmax)
            max_acceleration.append(amax)
            max_jerk.append(jmax)

        if camera_tracking:
            desired_velocity = _limit_vector_norm(
                desired_velocity, self.vmax_total)

        # Do not alternate between RUN and HALT.  Continuously reduce only the
        # velocity goal while the real arm catches the generated reference.
        previous_error = max(_command_errors(self.pos_cmd, robot_pos))
        self.command_velocity_scale = _command_lead_scale(
            previous_error, self.command_error_warn, self.command_error_halt)
        self.command_soft_halt = self.command_velocity_scale <= 0.0
        desired_velocity = [
            value * self.command_velocity_scale for value in desired_velocity]

        self.pos_cmd, self.v_cmd, self.a_cmd = self.motion_limiter.step(
            desired_velocity, max_velocity, max_acceleration, max_jerk)

        if (not self.current_is_home and self.seq_phase == 'track'
                and not camera_tracking and not self.enable_z_pbvs):
            self.v_cmd[2] = 0.0
            self.a_cmd[2] = 0.0
            self.pos_cmd[2] = locked_z_command
            self.motion_limiter.set_axis_state(2, locked_z_command)
        elif (not self.current_is_home and self.seq_phase == 'track'
                and not camera_tracking and self.z_control_mode == 'fixed'
                and self.fixed_motion_z is not None):
            # Remove all residual Z dynamics and prevent integration drift.
            self.v_cmd[2] = 0.0
            self.a_cmd[2] = 0.0
            self.pos_cmd[2] = self.fixed_motion_z
            self.motion_limiter.set_axis_state(2, self.fixed_motion_z)
        elif (self.seq_phase == 'z_approach'
              and abs(robot_pos[2] - self.fixed_motion_z) <= self.z_approach_tol
              and abs(self.v_cmd[2]) <= 0.01):
            self.v_cmd[2] = 0.0
            self.a_cmd[2] = 0.0
            self.pos_cmd[2] = self.fixed_motion_z
            self.motion_limiter.set_axis_state(2, self.fixed_motion_z)
            self._switch_target(0)

        if not self._inside_workspace(self.pos_cmd):
            self.workspace_halt = True
            self.pos_cmd = [
                _clamp(self.pos_cmd[i], self.ws_min[i], self.ws_max[i])
                for i in range(3)]
            self.v_cmd = [0.0, 0.0, 0.0]
            self.a_cmd = [0.0, 0.0, 0.0]
            self.motion_limiter.reset(self.pos_cmd)
            self.get_logger().error(
                'pos_cmd触及工作空间边界，命令已钳位',
                throttle_duration_sec=1.0)

        self.command_error_xy, self.command_error_z = _command_errors(
            self.pos_cmd, robot_pos)
        max_command_error = max(self.command_error_xy, self.command_error_z)
        if max_command_error >= self.command_error_warn:
            self.get_logger().warning(
                f'命令跟随误差: XY={self.command_error_xy*1000.0:.1f}mm '
                f'Z={self.command_error_z*1000.0:.1f}mm',
                throttle_duration_sec=1.0)

        debug_control_frame = ('base' if self.current_is_home else
                               getattr(self, 'pbvs_control_frame', 'base'))
        debug_reference = (self.desired_target_cam
                           if debug_control_frame == 'camera' else robot_pos)
        target_error = [
            (target_pos[i] - debug_reference[i])
            if target_pos[i] is not None else None for i in range(3)]
        self.pbvs_debug_snapshot = (
            tuple(target_pos), tuple(robot_pos), tuple(target_error),
            tuple(raw_pbvs_velocity), tuple(raw_pbvs_velocity_camera),
            tuple(desired_velocity), tuple(self.v_cmd), tuple(self.a_cmd),
            tuple(self.pos_cmd),
            'home' if self.current_is_home else self.current_point,
            error_xy, inside_xy_deadband, target_age, debug_control_frame)

        return [self.pos_cmd[0], self.pos_cmd[1], self.pos_cmd[2],
                self.fixed_rx, self.fixed_ry, self.fixed_rz]

    def _advance_sequence(self, robot_pos, mono_now):
        """到达检测 + 停留 + 序列推进。dry_run 模式下机械臂不动，用目标有效性代替到达判定。"""
        if self.dry_run:
            # dry_run：目标点有效且新鲜即视为"到达"，停留后切换，机械臂不动
            target_valid = (self.target_pos[0] is not None
                            and self.target_update_time is not None
                            and (mono_now - self.target_update_time) <= self.loss_timeout)
            if self.seq_phase == "track":
                if target_valid:
                    is_flyby = (self.flyby_enabled
                                and self.seq_index < len(self.sequence) - 1)
                    if is_flyby:
                        self._switch_target(self.seq_index + 1)
                    elif self.phase_since == 0.0:
                        self.phase_since = mono_now
                    elif mono_now - self.phase_since >= self.arrive_stable:
                        self.seq_phase = "staying"
                        self.phase_since = mono_now
                        self.get_logger().info(f'[dry_run] 到达 {self.current_point}，停留 {self.stay_dur}s')
                else:
                    self.phase_since = 0.0
            elif self.seq_phase == "staying":
                if mono_now - self.phase_since >= self.stay_dur:
                    if self.seq_index + 1 >= len(self.sequence):
                        self.seq_phase = "done"
                        self.get_logger().info('[dry_run] 全部目标点已完成')
                    else:
                        self._switch_target(self.seq_index + 1)
            return
        if self.current_is_home:
            # 回起始位置阶段
            err = math.sqrt(sum((self.home_pos[i] - robot_pos[i]) ** 2 for i in range(3)))
            if self.seq_phase == "track":
                if err <= self.home_arrive_tol:
                    if self.phase_since == 0.0:
                        self.phase_since = mono_now
                    elif mono_now - self.phase_since >= self.arrive_stable:
                        self.seq_phase = "staying"
                        self.phase_since = mono_now
                        self.get_logger().info(f'已回到起始位置 (err={err:.4f}m)，停留 {self.home_stay_dur}s')
                        self._log_action("home_arrive", error=err)
                else:
                    self.phase_since = 0.0
            elif self.seq_phase == "staying":
                if mono_now - self.phase_since >= self.home_stay_dur:
                    if self.loop:
                        # 回到起点后进入等待检测，检测到目标才开始下一轮（避免目标缺失时死循环）
                        self._clear_cycle_targets()
                        self.current_is_home = False
                        self.seq_phase = "waiting"
                        self.current_point = self.udder_frame
                        self.get_logger().info('循环：回到起点，等待检测到目标后开始下一轮')
                        self._log_action("loop")
                    else:
                        self.seq_phase = "done"
                        self.get_logger().info('动作完成，已回到起始位置')
                        self._log_action("done")
            return
        reference_xy = (self.desired_target_cam[:2]
                        if getattr(self, 'pbvs_control_frame', 'base') == 'camera'
                        else robot_pos[:2])
        err = math.hypot(
            self.target_pos[0] - reference_xy[0],
            self.target_pos[1] - reference_xy[1])
        speed_xy = math.hypot(self.v_cmd[0], self.v_cmd[1])
        if self.seq_phase == "track":
            is_flyby = (self.flyby_enabled
                        and self.seq_index < len(self.sequence) - 1)
            if (is_flyby and err <= self.flyby_tol_xy
                    and speed_xy <= self.switch_speed_xy):
                old_target = self.current_point
                new_target = self.sequence[self.seq_index + 1]
                self.get_logger().info(
                    f'waypoint switch: {old_target} -> {new_target}, '
                    f'error_xy={err*1000.0:.1f}mm, '
                    f'speed_xy={speed_xy*1000.0:.1f}mm/s')
                self._log_action("waypoint_switch", error=err)
                self._switch_target(self.seq_index + 1)
            elif not is_flyby and err <= self.arrive_tol_xy:
                if self.phase_since == 0.0:
                    self.phase_since = mono_now
                elif mono_now - self.phase_since >= self.arrive_stable:
                    self.seq_phase = "staying"
                    self.phase_since = mono_now
                    self.get_logger().info(
                        f'到达 {self.current_point} (err_xy={err:.4f}m)，停留 {self.stay_dur}s')
                    self._log_action("arrive", error=err)
            elif not is_flyby:
                self.phase_since = 0.0
        elif self.seq_phase == "staying":
            if mono_now - self.phase_since >= self.stay_dur:
                if self.seq_index + 1 >= len(self.sequence):
                    if self.return_home:
                        self._switch_to_home()
                    else:
                        self.seq_phase = "done"
                        self._publish_z_summary()
                        self.get_logger().info('全部目标点已完成')
                else:
                    self._switch_target(self.seq_index + 1)

    def _inside_workspace(self, pos):
        for i in range(3):
            if not (self.ws_min[i] <= pos[i] <= self.ws_max[i]):
                return False
        return True

    def _inside_workspace_xy(self, pos):
        return (self.ws_min[0] <= pos[0] <= self.ws_max[0]
                and self.ws_min[1] <= pos[1] <= self.ws_max[1])

    def destroy_node(self):
        self.stop_event.set()
        thread = getattr(self, 'send_thread', None)
        if thread is not None:
            thread.join(timeout=1.0)
        dispatcher = getattr(self, 'dispatcher', None)
        if dispatcher is not None:
            dispatcher.close()
        if hasattr(self, 'robot'):
            try:
                self.robot.StopCriControl()
                self.robot.__exit__(None, None, None)
            except Exception:
                pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = EstunDriver()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
