"""SICK Visionary 3D 相机 ROS2 节点。

通过封装的 SICK SDK（wrapper.LatestFrameCamera）读取帧数据，发布：
  ~/depth         sensor_msgs/Image        (16UC1, 单位 mm，0=无效)
  ~/intensity     sensor_msgs/Image        (16UC1, 原始强度)
  ~/confidence    sensor_msgs/Image        (16UC1, 置信度)
  ~/points        sensor_msgs/PointCloud2  (organized HxW, 字段 x/y/z float32, 单位 mm, 无效点为 NaN)
  ~/camera_info   sensor_msgs/CameraInfo   (相机内参)
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Float64, Header

_SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)
_CALIBRATION_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)

from .wrapper import (
    CameraConfig,
    CameraIntrinsics,
    LatestFrameCamera,
    SickCameraError,
    SickCameraFrameError,
)


class SickCameraNode(Node):
    def __init__(self) -> None:
        super().__init__('camera_node')

        # ---------- 参数 ----------
        self.declare_parameter('ip', '192.168.1.30')
        self.declare_parameter('device_type', 'Visionary-T Mini')
        self.declare_parameter('streaming_port', 2114)
        self.declare_parameter('control_port', 0)  # 0 表示使用默认端口
        self.declare_parameter('transport', 'TCP')
        self.declare_parameter('receiver_ip', '192.168.1.10')
        self.declare_parameter('timeout', 5.0)
        self.declare_parameter('frame_timeout', 0.3)
        self.declare_parameter('drop_warmup_frame', True)
        self.declare_parameter('reset_tcp_port', 2114)
        self.declare_parameter('camera_frame', 'sick_camera_optical_frame')
        self.declare_parameter('publish_rate', 0.0)  # 0 = 不限速，按相机帧率发布
        self.declare_parameter('points_publish_rate', 5.0)
        self.declare_parameter('publish_pointcloud', True)  # 生产链路不需要 points 时可关掉，省点云计算

        self.camera_frame = str(self.get_parameter('camera_frame').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)
        self.points_publish_rate = float(self.get_parameter('points_publish_rate').value)
        self.publish_pointcloud = bool(self.get_parameter('publish_pointcloud').value)

        # ---------- 构建相机配置 ----------
        config = CameraConfig(
            ip=str(self.get_parameter('ip').value),
            device_type=str(self.get_parameter('device_type').value),
            streaming_port=int(self.get_parameter('streaming_port').value),
            control_port=None if int(self.get_parameter('control_port').value) == 0
                            else int(self.get_parameter('control_port').value),
            transport=str(self.get_parameter('transport').value),
            receiver_ip=str(self.get_parameter('receiver_ip').value),
            timeout=float(self.get_parameter('timeout').value),
            frame_timeout=float(self.get_parameter('frame_timeout').value),
            drop_warmup_frame=bool(self.get_parameter('drop_warmup_frame').value),
            reset_tcp_port=int(self.get_parameter('reset_tcp_port').value),
        )

        # ---------- 启动相机 ----------
        self._camera: LatestFrameCamera | None = None
        self._started = self._start_camera(config)

        # ---------- 发布器 ----------
        self._cb = MutuallyExclusiveCallbackGroup()
        self.depth_pub = self.create_publisher(Image, '~/depth', _SENSOR_QOS)
        self.intensity_pub = self.create_publisher(Image, '~/intensity', _SENSOR_QOS)
        self.confidence_pub = self.create_publisher(Image, '~/confidence', _SENSOR_QOS)
        self.points_pub = self.create_publisher(PointCloud2, '~/points', _SENSOR_QOS)
        self.camera_info_pub = self.create_publisher(
            CameraInfo, '~/camera_info', _CALIBRATION_QOS
        )
        self.range_offset_pub = self.create_publisher(
            Float64, '~/range_offset_mm', _CALIBRATION_QOS
        )
        self._last_points_publish_at = 0.0
        self._camera_info_key = None

        # ---------- 定时回调 ----------
        period = 0.03  # 30Hz 默认
        if self.publish_rate > 0:
            period = 1.0 / self.publish_rate
        self.timer = self.create_timer(period, self._on_tick, callback_group=self._cb)

        self._frame_count = 0
        # monotonic → ROS wall-clock 偏移：每帧更新一次，供 header.stamp 使用
        # 让下游既能拿到稳定的 captured_at（monotonic 域），也能被 ROS 时间工具识别
        self._mono_to_ros_offset_ns: int = self._compute_mono_to_ros_offset_ns()

        self.get_logger().info(
            f'camera_node ready | ip={config.ip} device={config.device_type} '
            f'transport={config.transport} frame={self.camera_frame} '
            f'{"LIVE" if self._started else "SIM"}'
        )

    def _compute_mono_to_ros_offset_ns(self) -> int:
        """计算 monotonic 时钟到 ROS wall-clock 的偏移（纳秒）。

        ROS Time = monotonic + offset，每帧更新一次以适应系统时钟校时。
        """
        mono_ns = int(time.monotonic() * 1e9)
        ros_ns = self.get_clock().now().nanoseconds
        return ros_ns - mono_ns

    def _monotonic_to_ros_time(self, mono_sec: float) -> rclpy.time.Time:
        """monotonic 秒 → rclpy.time.Time（带最新偏移补偿）。"""
        # 实时刷新偏移，适应系统时钟漂移
        self._mono_to_ros_offset_ns = self._compute_mono_to_ros_offset_ns()
        mono_ns = int(mono_sec * 1e9)
        ros_ns = mono_ns + self._mono_to_ros_offset_ns
        return rclpy.time.Time(nanoseconds=ros_ns)

    # ============================================================
    # 相机启动
    # ============================================================
    def _start_camera(self, config: CameraConfig) -> bool:
        try:
            self._camera = LatestFrameCamera(config=config)
            self._camera.open()
            self.get_logger().info(
                f'相机已连接: {config.ip} | streaming_port={config.streaming_port}'
            )
            return True
        except Exception as exc:
            self.get_logger().warn(
                f'相机连接失败: {exc}，节点进入空跑模式（发布空帧）'
            )
            return False

    # ============================================================
    # 每帧回调
    # ============================================================
    def _on_tick(self, config: CameraConfig) -> None:
        if not self._started or self._camera is None:
            # 空跑模式：发布空帧占位
            self._publish_empty_frames()
            return

        try:
            frame = self._camera.read_frame(timeout=config.frame_timeout)
        except (SickCameraFrameError, SickCameraError) as exc:
            self.get_logger().warn(f'读取相机帧失败: {exc}')
            return

        self._frame_count += 1
        # 用 frame.host_received_at（monotonic，更接近采集时刻）作为 header.stamp
        # 优点：下游 detector/geometry/control 计算数据年龄时与 host_received_at 一致
        # 下游必须使用这个采集 stamp 查询历史 TF；眼在手上时禁止回退到最新 TF。
        stamp = self._monotonic_to_ros_time(frame.host_received_at).to_msg()
        header = Header(stamp=stamp, frame_id=self.camera_frame)

        height, width = frame.shape
        intrinsics = frame.intrinsics

        # 深度图（16UC1, mm）：把状态无效像素折叠为0，供生产检测单话题消费。
        depth_u16_mm = frame.distance_u16 // np.uint16(4)
        depth_u16_mm[frame.confidence_u16 != 0] = 0
        depth_msg = Image()
        depth_msg.header = header
        depth_msg.height = height
        depth_msg.width = width
        depth_msg.encoding = '16UC1'
        depth_msg.step = width * 2
        depth_msg.data = np.ascontiguousarray(depth_u16_mm).tobytes()
        self.depth_pub.publish(depth_msg)

        # 强度与原始状态图仅供标定/调试，有订阅者时才发布。
        if self.intensity_pub.get_subscription_count() > 0:
            intensity_msg = Image()
            intensity_msg.header = header
            intensity_msg.height = height
            intensity_msg.width = width
            intensity_msg.encoding = '16UC1'
            intensity_msg.step = width * 2
            intensity_msg.data = np.ascontiguousarray(frame.intensity_u16).tobytes()
            self.intensity_pub.publish(intensity_msg)

        if self.confidence_pub.get_subscription_count() > 0:
            confidence_msg = Image()
            confidence_msg.header = header
            confidence_msg.height = height
            confidence_msg.width = width
            confidence_msg.encoding = '16UC1'
            confidence_msg.step = width * 2
            confidence_msg.data = np.ascontiguousarray(frame.confidence_u16).tobytes()
            self.confidence_pub.publish(confidence_msg)

        # 完整点云仅供调试：publish_pointcloud=false 时完全不生成（省 CPU）；
        # 仍发布时，有订阅者才生成并单独限频。
        now = time.monotonic()
        points_due = (
            self.points_publish_rate <= 0.0
            or now - self._last_points_publish_at >= 1.0 / self.points_publish_rate
        )
        if (self.publish_pointcloud and self.points_pub.get_subscription_count() > 0
                and points_due):
            try:
                cloud_mm = frame.get_point_cloud()
                pts_msg = self._make_point_cloud2(cloud_mm, header)
                self.points_pub.publish(pts_msg)
                self._last_points_publish_at = now
            except Exception as exc:
                self.get_logger().warn(f'点云生成失败: {exc}')

        # 内参与径向深度原点偏移使用 transient-local，只在变化时发布。
        self._publish_camera_info(
            intrinsics, height, width, frame.projection.z_offset_mm
        )

    def _publish_empty_frames(self) -> None:
        """空跑模式：发布零值帧 + 默认内参，供下游调试。"""
        stamp = self.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id=self.camera_frame)
        h, w = 640, 640
        zeros_u16 = np.zeros((h, w), dtype=np.uint16)

        depth_msg = Image()
        depth_msg.header = header
        depth_msg.height, depth_msg.width = h, w
        depth_msg.encoding = '16UC1'
        depth_msg.step = w * 2
        depth_msg.data = zeros_u16.tobytes()
        self.depth_pub.publish(depth_msg)

        intensity_msg = Image()
        intensity_msg.header = header
        intensity_msg.height, intensity_msg.width = h, w
        intensity_msg.encoding = '16UC1'
        intensity_msg.step = w * 2
        intensity_msg.data = zeros_u16.tobytes()
        self.intensity_pub.publish(intensity_msg)

        confidence_msg = Image()
        confidence_msg.header = header
        confidence_msg.height, confidence_msg.width = h, w
        confidence_msg.encoding = '16UC1'
        confidence_msg.step = w * 2
        confidence_msg.data = zeros_u16.tobytes()
        self.confidence_pub.publish(confidence_msg)

        # SIM 模式也发布默认内参，保证 --once 等得到
        from .wrapper import CameraIntrinsics
        default_intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=320.0)
        self._publish_camera_info(default_intrinsics, h, w, 0.0)

    # ============================================================
    # 消息构造
    # ============================================================
    def _make_point_cloud2(self, cloud_mm: np.ndarray, header: Header) -> PointCloud2:
        """组织点云 HxWx3 (float32 mm) → PointCloud2。无效点 NaN。"""
        height, width, _ = cloud_mm.shape
        cloud_f32 = np.ascontiguousarray(cloud_mm, dtype=np.float32)
        msg = PointCloud2()
        msg.header = header
        msg.height = height
        msg.width = width
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * width
        msg.is_dense = False
        msg.data = cloud_f32.tobytes()
        return msg

    def _publish_camera_info(
        self,
        intrinsics: CameraIntrinsics,
        height: int,
        width: int,
        range_offset_mm: float,
    ) -> None:
        key = (
            height, width,
            intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            intrinsics.k1, intrinsics.k2, intrinsics.k3,
            intrinsics.p1, intrinsics.p2,
            float(range_offset_mm),
        )
        if key == self._camera_info_key:
            return
        msg = CameraInfo()
        msg.header = Header(
            stamp=self.get_clock().now().to_msg(), frame_id=self.camera_frame
        )
        msg.height = height
        msg.width = width
        msg.distortion_model = 'plumb_bob'
        msg.d = [intrinsics.k1, intrinsics.k2, intrinsics.p1,
                 intrinsics.p2, intrinsics.k3]
        msg.k = [
            intrinsics.fx, 0.0, intrinsics.cx,
            0.0, intrinsics.fy, intrinsics.cy,
            0.0, 0.0, 1.0,
        ]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [
            intrinsics.fx, 0.0, intrinsics.cx, 0.0,
            0.0, intrinsics.fy, intrinsics.cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
        self.camera_info_pub.publish(msg)
        self.range_offset_pub.publish(Float64(data=float(range_offset_mm)))
        self._camera_info_key = key
        self.get_logger().info(
            f'相机内参: fx={intrinsics.fx:.3f} fy={intrinsics.fy:.3f} '
            f'cx={intrinsics.cx:.3f} cy={intrinsics.cy:.3f} '
            f'range_offset={range_offset_mm:.3f}mm'
        )

    def destroy_node(self) -> bool:
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception:
                pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SickCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
