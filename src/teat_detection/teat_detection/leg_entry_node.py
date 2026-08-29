"""Publish stable ToF leg-entry geometry for the arm admission gate."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import rclpy
import tf2_ros
from camera.projection import RadialProjection
from cow_interfaces.msg import EntryStatus
from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, Float64

from .leg_entry import (
    LegEntryObservation,
    LegEntryTracker,
    detect_leg_entry,
    parse_depth_image,
    transform_observation,
)
from .transforms import quaternion_to_matrix


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
CALIBRATION_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class LegEntryNode(Node):
    def __init__(self) -> None:
        super().__init__("leg_entry_node")
        self._declare_parameters()
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.enabled = bool(self.get_parameter("enabled").value)
        self.camera_info: CameraInfo | None = None
        self.range_offset_mm: float | None = None
        self.projection: RadialProjection | None = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tracker = LegEntryTracker(
            int(self.get_parameter("stable_window_frames").value),
            int(self.get_parameter("stable_min_valid_frames").value),
            float(self.get_parameter("stable_duration").value),
            float(self.get_parameter("max_center_spread_m").value),
            float(self.get_parameter("max_gap_spread_m").value),
            np.asarray(self.get_parameter("production_axis").value, dtype=float),
        )
        self.publisher = self.create_publisher(
            EntryStatus, str(self.get_parameter("status_topic").value), 1
        )
        self.create_subscription(Image, self.depth_topic, self._on_depth, SENSOR_QOS)
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._on_camera_info,
            CALIBRATION_QOS,
        )
        self.create_subscription(
            Float64,
            str(self.get_parameter("range_offset_topic").value),
            self._on_range_offset,
            CALIBRATION_QOS,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("enable_topic").value),
            self._on_enable,
            1,
        )
        self.get_logger().info(
            f"leg_entry_node ready | depth={self.depth_topic} "
            f"status={self.get_parameter('status_topic').value} enabled={self.enabled}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("enabled", True)
        self.declare_parameter("depth_topic", "/camera_node/depth")
        self.declare_parameter("camera_info_topic", "/camera_node/camera_info")
        self.declare_parameter("range_offset_topic", "/camera_node/range_offset_mm")
        self.declare_parameter("status_topic", "/entry/status")
        self.declare_parameter("enable_topic", "/entry/detection_enabled")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "sick_camera_optical_frame")
        self.declare_parameter("tf_timeout", 0.05)
        self.declare_parameter("min_depth_mm", 400.0)
        self.declare_parameter("max_depth_mm", 1000.0)
        self.declare_parameter("leg_row_start_ratio", 0.60)
        self.declare_parameter("leg_row_end_ratio", 0.90)
        self.declare_parameter("min_leg_separation_px", 30)
        self.declare_parameter("min_leg_height_px", 20)
        self.declare_parameter("min_leg_aspect_ratio", 0.9)
        self.declare_parameter("min_leg_blob_area_px", 20)
        self.declare_parameter("leg_gap_min_m", 0.0)
        self.declare_parameter("leg_gap_max_m", 2.0)
        self.declare_parameter("stable_window_frames", 8)
        self.declare_parameter("stable_min_valid_frames", 6)
        self.declare_parameter("stable_duration", 0.20)
        self.declare_parameter("max_center_spread_m", 0.10)
        self.declare_parameter("max_gap_spread_m", 0.05)
        self.declare_parameter("production_axis", [1.0, 0.0, 0.0])

    @staticmethod
    def _stamp_seconds(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    @staticmethod
    def _point(values) -> Point:
        point = Point()
        if values is not None and np.isfinite(values).all():
            point.x, point.y, point.z = (float(v) for v in values)
        return point

    def _on_camera_info(self, message: CameraInfo) -> None:
        self.camera_info = message
        self._refresh_projection()

    def _on_range_offset(self, message: Float64) -> None:
        self.range_offset_mm = float(message.data)
        self._refresh_projection()

    def _refresh_projection(self) -> None:
        if self.camera_info is None or self.range_offset_mm is None:
            return
        distortion = list(self.camera_info.d) + [0.0, 0.0]
        self.projection = RadialProjection(
            width=int(self.camera_info.width),
            height=int(self.camera_info.height),
            fx=float(self.camera_info.k[0]),
            fy=float(self.camera_info.k[4]),
            cx=float(self.camera_info.k[2]),
            cy=float(self.camera_info.k[5]),
            k1=float(distortion[0]),
            k2=float(distortion[1]),
            z_offset_mm=float(self.range_offset_mm),
        )

    def _on_enable(self, message: Bool) -> None:
        requested = bool(message.data)
        if requested == self.enabled:
            return
        self.enabled = requested
        self.tracker.reset()
        self.get_logger().info(
            "牛腿入场检测已启用" if requested else "牛腿入场检测已锁定"
        )

    def _base_transform(self, stamp):
        transform = self.tf_buffer.lookup_transform(
            self.base_frame,
            self.camera_frame,
            Time.from_msg(stamp),
            timeout=Duration(seconds=float(self.get_parameter("tf_timeout").value)),
        )
        rotation = transform.transform.rotation
        translation = transform.transform.translation
        return (
            quaternion_to_matrix((rotation.x, rotation.y, rotation.z, rotation.w)),
            np.asarray((translation.x, translation.y, translation.z), dtype=float),
        )

    def _on_depth(self, message: Image) -> None:
        if not self.enabled:
            return
        depth = parse_depth_image(
            message.data, message.height, message.width, message.encoding
        )
        projection = self.projection
        if depth is None or projection is None:
            return
        stamp = self._stamp_seconds(message.header.stamp)

        def project(rows, cols):
            return projection.project(depth, rows, cols)

        observation = detect_leg_entry(
            depth,
            project,
            stamp=stamp,
            min_depth_mm=float(self.get_parameter("min_depth_mm").value),
            max_depth_mm=float(self.get_parameter("max_depth_mm").value),
            row_start_ratio=float(self.get_parameter("leg_row_start_ratio").value),
            row_end_ratio=float(self.get_parameter("leg_row_end_ratio").value),
            min_separation_px=int(self.get_parameter("min_leg_separation_px").value),
            min_height_px=int(self.get_parameter("min_leg_height_px").value),
            min_aspect_ratio=float(self.get_parameter("min_leg_aspect_ratio").value),
            min_blob_area_px=int(self.get_parameter("min_leg_blob_area_px").value),
        )
        if observation.valid:
            try:
                rotation, translation = self._base_transform(message.header.stamp)
                observation = transform_observation(
                    observation, rotation, translation
                )
            except tf2_ros.TransformException as exc:
                observation = replace(
                    observation, valid=False, reason="TF_UNAVAILABLE"
                )
                self.get_logger().warning(
                    f"牛腿历史TF不可用: {exc}", throttle_duration_sec=2.0
                )
        tracked = self.tracker.update(observation)
        self._publish(message, tracked)

    def _publish(self, source: Image, observation: LegEntryObservation) -> None:
        output = EntryStatus()
        output.header = source.header
        output.header.frame_id = self.base_frame
        output.valid = bool(observation.valid and observation.center_base is not None)
        output.stable = bool(observation.stable)
        output.cow_track_id = int(observation.cow_track_id)
        output.left_inner = self._point(observation.left_base)
        output.right_inner = self._point(observation.right_base)
        output.entry_center = self._point(observation.center_base)
        output.entry_center_camera = self._point(observation.center_camera)
        output.left_inner_u, output.left_inner_v = observation.left_pixel
        output.right_inner_u, output.right_inner_v = observation.right_pixel
        output.gap_m = float(observation.gap_m)
        output.line_speed_mps = float(observation.line_speed_mps)
        output.confidence = float(observation.confidence)
        gap_min = float(self.get_parameter("leg_gap_min_m").value)
        gap_max = float(self.get_parameter("leg_gap_max_m").value)
        calibrated = math.isfinite(gap_min) and gap_min > 0.0 and gap_max > gap_min
        output.corridor_clear = bool(
            calibrated and output.valid and gap_min <= output.gap_m <= gap_max
        )
        if not calibrated:
            output.reason = "GAP_NOT_CALIBRATED"
        elif output.valid and output.stable and not output.corridor_clear:
            output.reason = "LEG_GAP_OUT_OF_RANGE"
        else:
            output.reason = observation.reason
        self.publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LegEntryNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
