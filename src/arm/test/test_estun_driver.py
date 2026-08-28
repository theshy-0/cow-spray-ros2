import numpy as np
import pytest
from types import SimpleNamespace
import threading
import time

from arm.estun_driver import (
    EstunDriver,
    DynamicTargetInitializer,
    LatencyStats,
    ZErrorStats,
    _compute_camera_pbvs_velocity,
    _compute_pbvs_velocity,
    _command_errors,
    _command_lead_scale,
    _freshness_fade,
    _freshness_time,
    _limit_vector_norm,
    _radial_deadband_scale,
)
from arm.motion_limiter import RuckigVelocityLimiter


def test_camera_pbvs_uses_camera_error_and_maps_velocity_to_base():
    velocity_camera, velocity_base = _compute_camera_pbvs_velocity(
        target_pos=[0.10, -0.20, 0.60],
        target_v=[0.0, 0.0, 0.0],
        target_a=[0.0, 0.0, 0.0],
        desired_target=[0.0, 0.0, 0.50],
        rotation_base_camera=np.eye(3),
        prediction_horizon=0.0,
        lambda_gain=[1.0, 1.0, 0.6],
        feedforward_gain=[0.0, 0.0, 0.0],
        vmax=[0.22, 0.22, 0.06],
        fade=1.0,
    )
    assert velocity_camera == pytest.approx([0.10, -0.20, 0.06])
    assert velocity_base == pytest.approx(velocity_camera)


def test_camera_pbvs_rotates_camera_velocity_into_base():
    rotation = np.array([[0.0, -1.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0]])
    _, velocity_base = _compute_camera_pbvs_velocity(
        [0.10, 0.0, 0.50], [0.0] * 3, [0.0] * 3,
        [0.0, 0.0, 0.50], rotation, 0.0,
        [1.0, 1.0, 0.6], [0.0] * 3, [0.22, 0.22, 0.06], 1.0)
    assert velocity_base == pytest.approx([0.0, 0.10, 0.0])


def test_vector_norm_limit_preserves_direction():
    assert _limit_vector_norm([3.0, 4.0, 0.0], 2.5) == pytest.approx(
        [1.5, 2.0, 0.0])


def test_radial_deadband_is_continuous_instead_of_run_stop():
    assert _radial_deadband_scale(0.003, 0.004) == 0.0
    assert _radial_deadband_scale(0.004, 0.004) == 0.0
    assert _radial_deadband_scale(0.005, 0.004) == pytest.approx(0.2)


def test_command_lead_uses_continuous_slowdown():
    assert _command_lead_scale(0.010, 0.020, 0.040) == 1.0
    assert _command_lead_scale(0.030, 0.020, 0.040) == pytest.approx(0.5)
    assert _command_lead_scale(0.050, 0.020, 0.040) == 0.0


def test_ruckig_reference_respects_velocity_acceleration_and_jerk():
    dt = 0.004
    limiter = RuckigVelocityLimiter(dt)
    limiter.reset([0.0, 0.0, 0.0])
    velocities = []
    accelerations = []
    for _ in range(500):
        _, velocity, acceleration = limiter.step(
            [0.12, -0.08, 0.03],
            [0.12, 0.12, 0.04],
            [0.8, 0.8, 0.2],
            [1.5, 1.5, 0.5],
        )
        velocities.append(np.asarray(velocity))
        accelerations.append(np.asarray(acceleration))
    assert np.all(
        np.max(np.abs(velocities), axis=0)
        <= [0.120001, 0.120001, 0.040001])
    assert np.all(
        np.max(np.abs(accelerations), axis=0)
        <= [0.800001, 0.800001, 0.200001])
    jerk = np.diff(accelerations, axis=0) / dt
    assert np.all(
        np.max(np.abs(jerk), axis=0) <= [1.5001, 1.5001, 0.5001])


def test_home_target_does_not_expire_with_stale_visual_receive_time():
    assert _freshness_time(True, 10.0, 1.0) == 10.0


def test_visual_target_still_uses_receive_time():
    assert _freshness_time(False, 10.0, 1.0) == 1.0
    assert _freshness_time(False, 10.0, None) == 10.0


def test_disabled_driver_does_not_match_or_advance_targets():
    node = SimpleNamespace(enabled=False)
    EstunDriver.update_control_target(node)


def test_dynamic_initializer_accepts_three_consistent_single_target_frames():
    initializer = DynamicTargetInitializer(
        window_frames=5, min_full_frames=3, max_duration=0.35)
    result = None
    for stamp, x in [(1.00, 0.10), (1.05, 0.101), (1.10, 0.099)]:
        observation = (
            {'teat_front_left': [x, 0.20, 0.30]},
            {'teat_front_left': [x + 0.50, 0.20, 0.30]},
        )
        result = initializer.observe(observation, stamp)

    assert result is not None
    local, base = result
    assert set(local) == {'teat_front_left'}
    assert base['teat_front_left'] == [0.599, 0.20, 0.30]


def test_teat_pbvs_controls_xy_but_not_z():
    velocity = _compute_pbvs_velocity(
        robot_pos=[0.50, 0.10, 0.30],
        target_pos=[0.60, 0.00, 0.90],
        target_v=[0.02, -0.01, 1.00],
        target_a=[0.10, -0.10, 2.00],
        prediction_horizon=0.02,
        lambda_gain=1.8,
        feedforward_gain=0.35,
        vmax=0.35,
        fade=1.0,
        xy_only=True,
    )
    assert velocity[0] > 0.0
    assert velocity[1] < 0.0
    assert velocity[2] == 0.0


def test_home_pbvs_keeps_xyz_control():
    velocity = _compute_pbvs_velocity(
        robot_pos=[0.50, 0.10, 0.30],
        target_pos=[0.60, 0.00, 0.40],
        target_v=[0.0, 0.0, 0.0],
        target_a=[0.0, 0.0, 0.0],
        prediction_horizon=0.0,
        lambda_gain=1.8,
        feedforward_gain=0.35,
        vmax=0.35,
        fade=1.0,
        xy_only=False,
    )
    assert velocity[0] > 0.0
    assert velocity[1] < 0.0
    assert velocity[2] > 0.0


def _sequence_node(target, *, index=0, phase="track"):
    events = []
    return SimpleNamespace(
        dry_run=False,
        current_is_home=False,
        target_pos=list(target),
        robot=None,
        seq_phase=phase,
        phase_since=0.0,
        seq_index=index,
        sequence=["fl", "rl", "rr", "fr"],
        flyby_enabled=True,
        flyby_tol_xy=0.010,
        switch_speed_xy=0.030,
        v_cmd=[0.0, 0.0, 0.0],
        arrive_tol_xy=0.008,
        arrive_stable=0.03,
        stay_dur=0.05,
        return_home=True,
        z_control_mode="fixed",
        enable_z_pbvs=True,
        xy_deadband_m=0.0,
        z_error=None,
        z_deviation_warn=0.015,
        current_point=["fl", "rl", "rr", "fr"][index],
        get_logger=lambda: SimpleNamespace(info=lambda _message: None),
        _log_action=lambda *args, **kwargs: None,
        _switch_target=lambda next_index: events.append(next_index),
        events=events,
    )


def test_waypoint_does_not_switch_when_distance_is_too_large():
    node = _sequence_node([0.011, 0.0, 0.030])
    EstunDriver._advance_sequence(node, [0.0, 0.0, 0.0], 10.0)
    assert node.events == []


def test_waypoint_does_not_switch_when_xy_speed_is_too_high():
    node = _sequence_node([0.009, 0.0, 0.030])
    node.v_cmd = [0.031, 0.0, 0.0]
    EstunDriver._advance_sequence(node, [0.0, 0.0, 0.0], 10.0)
    assert node.events == []


def test_waypoint_switches_when_distance_and_speed_are_ready():
    node = _sequence_node([0.009, 0.0, 0.030])
    node.v_cmd = [0.029, 0.0, 0.0]
    node.z_control_mode = "low_bandwidth"
    node.z_error = 0.030
    EstunDriver._advance_sequence(node, [0.0, 0.0, 0.0], 10.0)
    assert node.events == [1]


def test_waypoint_switch_allows_exact_distance_and_speed_boundaries():
    node = _sequence_node([0.010, 0.0, 0.030])
    node.v_cmd = [0.030, 0.0, 0.0]
    EstunDriver._advance_sequence(node, [0.0, 0.0, 0.0], 10.0)
    assert node.events == [1]


def test_final_arrival_uses_xy_and_starts_stability_timer_on_entry():
    node = _sequence_node([0.005, 0.0, 0.020], index=3)
    node.v_cmd = [0.20, 0.0, 0.0]
    node.z_control_mode = "low_bandwidth"
    node.z_error = 0.050
    EstunDriver._advance_sequence(node, [0.0, 0.0, 0.0], 10.0)
    assert node.phase_since == 10.0
    assert node.seq_phase == "track"
    EstunDriver._advance_sequence(node, [0.0, 0.0, 0.0], 10.031)
    assert node.seq_phase == "staying"


def test_home_arrival_still_uses_xyz_distance():
    node = SimpleNamespace(
        dry_run=False,
        current_is_home=True,
        home_pos=[0.0, 0.0, 0.0],
        seq_phase="track",
        phase_since=0.0,
        home_arrive_tol=0.008,
        arrive_stable=0.03,
        home_stay_dur=0.20,
        loop=False,
        get_logger=lambda: SimpleNamespace(info=lambda _message: None),
        _log_action=lambda *_args, **_kwargs: None,
    )
    EstunDriver._advance_sequence(node, [0.0, 0.0, 0.020], 10.0)
    assert node.phase_since == 0.0
    EstunDriver._advance_sequence(node, [0.0, 0.0, 0.005], 10.1)
    assert node.phase_since == 10.1


def test_switch_target_preserves_xy_limiter_state_and_resets_phase_timer():
    lock = __import__("threading").Lock()
    node = SimpleNamespace(
        sequence=["fl", "rl"],
        offsets=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        process_noise=0.02,
        measurement_noise=0.006,
        _cycle_base_targets={"rl": [0.2, 0.3, 0.4]},
        _topic_targets={},
        target_lock=lock,
        v_cmd=[0.11, -0.07, 0.006],
        a_cmd=[0.20, -0.10, 0.015],
        pos_cmd=[0.5, 0.2, 0.3],
        last_receive_mono=1.0,
        get_logger=lambda: SimpleNamespace(info=lambda _message: None),
        _log_action=lambda *_args, **_kwargs: None,
    )
    EstunDriver._switch_target(node, 1)
    assert node.phase_since == 0.0
    assert node.v_cmd == [0.11, -0.07, 0.006]
    assert node.a_cmd == [0.20, -0.10, 0.015]
    assert node.pos_cmd == [0.5, 0.2, 0.3]
    assert node.last_receive_mono == 1.0


def test_freshness_track_fade_and_halt_ranges():
    assert _freshness_fade(0.10, 0.20, 0.80) == 1.0
    assert np.isclose(_freshness_fade(0.50, 0.20, 0.80), 0.5)
    assert _freshness_fade(0.81, 0.20, 0.80) == 0.0


def _init_frame(dx=0.0, broken=False):
    local = {
        "fl": [0.00, 0.00, 0.00],
        "rl": [0.00, 0.10, 0.00],
        "rr": [0.10, 0.10, 0.00],
        "fr": [0.10, 0.00, 0.00],
    }
    if broken:
        local["rr"] = [0.30, 0.10, 0.00]
    base = {name: [p[0] + dx, p[1], p[2] + 0.3]
            for name, p in local.items()}
    return local, base


def test_dynamic_initializer_accepts_three_consistent_full_frames_in_five():
    init = DynamicTargetInitializer(5, 3, 0.35)
    result = None
    observations = [
        _init_frame(0.00),
        None,
        _init_frame(0.01),
        _init_frame(0.02, broken=True),
        _init_frame(0.03),
    ]
    for index, observation in enumerate(observations):
        result = init.observe(observation, stamp=0.05 * index)
    assert result is not None
    local, base = result
    assert set(local) == {"fl", "rl", "rr", "fr"}
    assert np.allclose(base["fl"], [0.03, 0.0, 0.3])


def test_dynamic_initializer_restarts_after_duration_timeout():
    init = DynamicTargetInitializer(5, 3, 0.35)
    assert init.observe(_init_frame(), stamp=0.00) is None
    assert init.observe(_init_frame(0.01), stamp=0.10) is None
    assert init.observe(_init_frame(0.02), stamp=0.50) is None
    assert init.full_count == 1


def test_z_error_statistics_and_threshold_ratios():
    stats = ZErrorStats()
    for value in [0.0, 0.006, -0.012, 0.025]:
        stats.add(value)
    result = stats.summary()
    assert np.isclose(result["z_error_mean"], 0.00475)
    assert np.isclose(result["z_error_abs_mean"], 0.01075)
    assert np.isclose(result["z_error_max"], 0.025)
    assert np.isclose(result["z_error_std"], np.std([0.0, 0.006, -0.012, 0.025]))
    assert result["z_error_over_5mm_ratio"] == 0.75
    assert result["z_error_over_10mm_ratio"] == 0.50
    assert result["z_error_over_20mm_ratio"] == 0.25


def test_command_following_watchdog_uses_xy_and_z_separately():
    err_xy, err_z = _command_errors(
        command=[0.53, 0.14, 0.32], robot=[0.50, 0.10, 0.30])
    assert np.isclose(err_xy, 0.05)
    assert np.isclose(err_z, 0.02)


def test_latency_diagnostics_report_mean_p95_and_max():
    stats = LatencyStats()
    for value in [0.01, 0.02, 0.03, 0.04]:
        stats.add(value)
    result = stats.summary()
    assert result["mean"] == np.mean([0.01, 0.02, 0.03, 0.04])
    assert result["p95"] == np.percentile([0.01, 0.02, 0.03, 0.04], 95)
    assert result["max"] == 0.04


def test_topic_tf_unavailable_is_buffered_without_refreshing_freshness():
    class MissingAdapter:
        def convert(self, _message):
            raise ValueError("TF unavailable")

    node = SimpleNamespace(
        topic_adapter=MissingAdapter(),
        target_lock=threading.Lock(),
        _pending_topic_message=None,
        last_receive_mono=12.0,
    )
    message = object()
    assert not EstunDriver._convert_topic_message(
        node, message, receive_mono=20.0, wait_started=20.0)
    assert node._pending_topic_message == (message, 20.0, 20.0)
    assert node.last_receive_mono == 12.0


def _servo_node():
    limiter = RuckigVelocityLimiter(0.004)
    limiter.reset([0.50, 0.10, 0.30])
    node = SimpleNamespace(
        robot=SimpleNamespace(CriData=SimpleNamespace(
            tcp_pose=[500.0, 100.0, 300.0, 0.0, 0.0, 0.0],
            status=SimpleNamespace(has_alarm=False, is_emergency_stop=False,
                                   collision_stop=False, is_disabled=False))),
        pos_cmd=[0.50, 0.10, 0.30],
        fixed_rx=0.0, fixed_ry=0.0, fixed_rz=0.0,
        target_lock=threading.Lock(),
        target_pos=[0.55, 0.10, 0.90],
        target_v=[0.0, 0.0, 1.0],
        target_a=[0.0, 0.0, 2.0],
        target_update_time=time.monotonic(),
        last_receive_mono=time.monotonic(),
        enabled=True, dry_run=False, seq_phase="track",
        current_is_home=False, current_point="fl",
        track_timeout=0.20, loss_timeout=0.80,
        start_buffer=5, dt=0.004, max_prediction=0.02,
        lambda_gain=1.8, feedforward=0.35,
        vmax=0.35, amax=0.60, jmax=1.80,
        v_cmd=[0.0, 0.0, 0.0], a_cmd=[0.0, 0.0, 0.0],
        fixed_motion_z=0.30,
        z_approach_tol=0.008,
        z_control_mode="fixed",
        enable_z_pbvs=True,
        xy_deadband_m=0.0,
        pbvs_control_frame="base",
        desired_target_cam=[0.0, 0.0, 0.50],
        R_base_camera=np.eye(3),
        lambda_gain_xyz=[1.0, 1.0, 0.6],
        feedforward_xyz=[0.0, 0.0, 0.0],
        vmax_xyz=[0.22, 0.22, 0.06],
        amax_xyz=[2.5, 2.5, 1.0],
        jmax_xyz=[3.0, 3.0, 1.5],
        vmax_total=0.25,
        z_low_gain=0.25, z_low_vmax=0.010,
        z_low_amax=0.030, z_low_jmax=0.080,
        z_low_deadband=0.005,
        z_soft_halt=False, z_error=None,
        spray_allowed=False, arrive_tol_xy=0.008,
        z_deviation_warn=0.015,
        command_soft_halt=False, workspace_halt=False,
        command_velocity_scale=1.0,
        command_error_warn=0.020, command_error_halt=0.040,
        motion_limiter=limiter,
        ws_min=np.array([0.0, -1.0, 0.1]),
        ws_max=np.array([1.0, 1.0, 0.5]),
        get_logger=lambda: SimpleNamespace(
            warning=lambda *_args, **_kwargs: None,
            error=lambda *_args, **_kwargs: None),
    )
    node._advance_sequence = lambda _robot, _now: None
    node._inside_workspace = lambda pos: EstunDriver._inside_workspace(node, pos)
    node._inside_workspace_xy = lambda pos: EstunDriver._inside_workspace_xy(node, pos)
    node._switch_target = lambda _index: None
    return node


def test_xy_deadband_stops_reference_inside_four_millimetres():
    node = _servo_node()
    node.xy_deadband_m = 0.004
    node.target_pos = [0.503, 0.10, 0.30]

    command = EstunDriver.servo_step(node)

    assert command[:2] == [0.50, 0.10]
    assert node.v_cmd[:2] == [0.0, 0.0]


def test_xy_deadband_allows_reference_outside_four_millimetres():
    node = _servo_node()
    node.xy_deadband_m = 0.004
    node.target_pos = [0.505, 0.10, 0.30]

    EstunDriver.servo_step(node)

    assert node.v_cmd[0] > 0.0


def test_disabled_z_pbvs_locks_z_in_low_bandwidth_mode():
    node = _servo_node()
    node.enable_z_pbvs = False
    node.z_control_mode = "low_bandwidth"
    node.target_pos[2] = 0.35

    command = EstunDriver.servo_step(node)

    assert command[2] == 0.30
    assert node.v_cmd[2] == 0.0
    assert node.a_cmd[2] == 0.0


def test_camera_frame_servo_controls_xyz_from_camera_error():
    node = _servo_node()
    node.pbvs_control_frame = "camera"
    node.target_pos = [0.10, -0.10, 0.60]
    node.target_v = [0.0, 0.0, 0.0]
    node.target_a = [0.0, 0.0, 0.0]

    command = EstunDriver.servo_step(node)

    assert command[0] > 0.50
    assert command[1] < 0.10
    assert command[2] > 0.30
    assert np.linalg.norm(node.v_cmd) <= node.vmax_total
    assert np.all(np.abs(node.a_cmd) <= np.asarray(node.amax_xyz) + 1e-9)


def test_servo_step_keeps_fixed_z_when_visual_z_changes():
    node = _servo_node()
    for visual_z in [0.280, 0.292, 0.274, 0.288]:
        node.target_pos[2] = visual_z
        command = EstunDriver.servo_step(node)
        assert command[2] == node.fixed_motion_z
        assert node.v_cmd[2] == 0.0
        assert node.a_cmd[2] == 0.0


def test_low_bandwidth_z_moves_slowly_toward_visual_depth():
    node = _servo_node()
    node.z_control_mode = "low_bandwidth"
    node.target_pos[2] = 0.35
    node.target_v[2] = 0.0
    node.target_a[2] = 0.0
    command = EstunDriver.servo_step(node)
    assert 0.30 < command[2] < 0.301
    assert 0.0 < node.v_cmd[2] <= node.z_low_vmax


def test_low_bandwidth_z_deadband_does_not_move():
    node = _servo_node()
    node.z_control_mode = "low_bandwidth"
    node.target_pos[2] = 0.304
    node.target_v[2] = 0.0
    node.target_a[2] = 0.0
    command = EstunDriver.servo_step(node)
    assert command[2] == 0.30
    assert node.v_cmd[2] == 0.0


def test_servo_step_stops_reference_when_robot_reports_alarm():
    node = _servo_node()
    node.robot.CriData.status.has_alarm = True
    assert EstunDriver.servo_step(node) is None
    assert node.enabled is False
    assert node.pos_cmd == [0.5, 0.1, 0.3]
    assert node.v_cmd == [0.0, 0.0, 0.0]


def test_servo_step_uses_new_target_in_same_cycle_after_flyby_switch():
    node = _servo_node()
    node.target_pos = [0.40, 0.10, 0.30]  # old target asks for negative X

    def switch_during_advance(_robot, _now):
        node.target_pos = [0.60, 0.10, 0.30]  # new target asks for positive X
        node.target_v = [0.0, 0.0, 0.0]
        node.target_a = [0.0, 0.0, 0.0]
        node.target_update_time = time.monotonic()

    node._advance_sequence = switch_during_advance
    EstunDriver.servo_step(node)
    assert node.v_cmd[0] > 0.0


def test_xy_arrival_and_z_safety_produce_independent_spray_allowed():
    node = _servo_node()
    node.target_pos = [0.505, 0.10, 0.32]
    node.z_error = 0.020
    EstunDriver.servo_step(node)
    assert node.spray_allowed is False
    node.z_error = 0.010
    EstunDriver.servo_step(node)
    assert node.spray_allowed is True


def test_fixed_z_deviation_blocks_spray_but_not_xy_tracking():
    node = _servo_node()
    node.z_soft_halt = True
    node.z_error = 0.033
    EstunDriver.servo_step(node)
    assert node.v_cmd[0] > 0.0
    assert node.spray_allowed is False
