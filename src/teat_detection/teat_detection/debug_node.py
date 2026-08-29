"""Optional rqt image source for visualizing YOLO detections and teat IDs."""

from collections import OrderedDict
import json
import math

import cv2
import numpy as np
import rclpy
import tf2_ros
from cow_interfaces.msg import EntryStatus
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray

from .processing import intensity_to_bgr
from .transforms import quaternion_to_matrix
from .vision_stability import VisionStabilityRecorder


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class DetectionDebugNode(Node):
    def __init__(self) -> None:
        super().__init__("sick_yolo_debug")
        self.declare_parameter("intensity_topic", "/sick_camera/intensity")
        self.declare_parameter("depth_topic", "/sick_camera/depth")
        self.declare_parameter("detections_topic", "/detector_node/detections")
        self.declare_parameter("tracked_detections_topic", "/udder/tracked_detections")
        self.declare_parameter("udder_status_topic", "/udder/status")
        self.declare_parameter("entry_status_topic", "/entry/status")
        self.declare_parameter("image_topic", "/sick_yolo_debug/image")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("enable_vision_csv", False)
        self.declare_parameter("vision_csv_path", "/tmp/vision_stability.csv")
        self.declare_parameter(
            "vision_summary_path", "/tmp/vision_stability_summary.json"
        )
        self.declare_parameter("vision_csv_flush_rows", 32)
        self.declare_parameter("vision_tf_timeout", 0.05)
        self.declare_parameter("debug_pixel_jump_threshold", 8.0)
        self.declare_parameter("debug_depth_jump_threshold", 0.03)
        self.declare_parameter("debug_cam_jump_threshold", 0.03)
        self.declare_parameter("debug_base_jump_threshold", 0.03)
        self._images = OrderedDict()
        self._depths = OrderedDict()
        self._tf_cache = OrderedDict()
        self._prediction_label = ""
        self._entry_status = None
        self._tracked_topic = str(
            self.get_parameter("tracked_detections_topic").value
        )
        self.image_pub = self.create_publisher(
            Image, str(self.get_parameter("image_topic").value), SENSOR_QOS
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._recorder = None
        if bool(self.get_parameter("enable_vision_csv").value):
            self._recorder = VisionStabilityRecorder(
                str(self.get_parameter("vision_csv_path").value),
                str(self.get_parameter("vision_summary_path").value),
                flush_rows=int(self.get_parameter("vision_csv_flush_rows").value),
                jump_thresholds={
                    "pixel": float(self.get_parameter("debug_pixel_jump_threshold").value),
                    "depth": float(self.get_parameter("debug_depth_jump_threshold").value),
                    "cam": float(self.get_parameter("debug_cam_jump_threshold").value),
                    "base": float(self.get_parameter("debug_base_jump_threshold").value),
                },
                warning=self.get_logger().warning,
            )
            self.get_logger().info(
                "Vision stability CSV enabled: "
                f"{self.get_parameter('vision_csv_path').value}"
            )
        self.create_subscription(
            Image,
            str(self.get_parameter("intensity_topic").value),
            self._on_image,
            SENSOR_QOS,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._on_depth,
            SENSOR_QOS,
        )
        self.create_subscription(
            Detection2DArray,
            str(self.get_parameter("detections_topic").value),
            self._on_raw_detections,
            1,
        )
        self.create_subscription(
            Detection2DArray,
            self._tracked_topic,
            self._on_detections,
            1,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("udder_status_topic").value),
            self._on_udder_status,
            1,
        )
        self.create_subscription(
            EntryStatus,
            str(self.get_parameter("entry_status_topic").value),
            self._on_entry_status,
            1,
        )

    @staticmethod
    def _key(stamp) -> tuple[int, int]:
        return stamp.sec, stamp.nanosec

    def _on_image(self, msg: Image) -> None:
        if msg.encoding not in ("16UC1", "mono16"):
            self.get_logger().error(
                f"Expected 16UC1 image, got {msg.encoding}",
                throttle_duration_sec=5.0,
            )
            return
        intensity = np.frombuffer(msg.data, dtype=np.uint16).reshape(
            msg.height, msg.width
        )
        self._images[self._key(msg.header.stamp)] = (msg.header, intensity.copy())
        while len(self._images) > 32:
            self._images.popitem(last=False)

    def _on_depth(self, msg: Image) -> None:
        if msg.encoding not in ("16UC1", "mono16"):
            return
        depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(
            msg.height, msg.width
        )
        self._depths[self._key(msg.header.stamp)] = depth.copy()
        while len(self._depths) > 64:
            self._depths.popitem(last=False)

    @staticmethod
    def _stamp_seconds(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _base_transform(self, frame_id: str, stamp):
        key = (frame_id, stamp.sec, stamp.nanosec)
        if key in self._tf_cache:
            return self._tf_cache[key]
        base_frame = str(self.get_parameter("base_frame").value)
        if frame_id == base_frame:
            value = (np.eye(3), np.zeros(3), self._stamp_seconds(stamp))
        else:
            try:
                transform = self.tf_buffer.lookup_transform(
                    base_frame,
                    frame_id,
                    Time.from_msg(stamp),
                    timeout=Duration(
                        seconds=float(self.get_parameter("vision_tf_timeout").value)
                    ),
                )
            except tf2_ros.TransformException as exc:
                self.get_logger().warning(
                    f"Vision CSV base transform unavailable: {exc}",
                    throttle_duration_sec=2.0,
                )
                return None
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            value = (
                quaternion_to_matrix((rotation.x, rotation.y, rotation.z, rotation.w)),
                np.asarray((translation.x, translation.y, translation.z), dtype=float),
                self._stamp_seconds(transform.header.stamp),
            )
        self._tf_cache[key] = value
        while len(self._tf_cache) > 32:
            self._tf_cache.popitem(last=False)
        return value

    def _record_target(
        self,
        *,
        stamp,
        frame_id: str,
        target_id: str,
        source: str,
        confidence=math.nan,
        u=math.nan,
        v=math.nan,
        camera_point=None,
        depth_raw=math.nan,
        detection_index=math.nan,
        assignment_distance=math.nan,
    ) -> None:
        if self._recorder is None:
            return
        measurement_stamp = self._stamp_seconds(stamp)
        receive_time = self.get_clock().now().nanoseconds * 1e-9
        age = receive_time - measurement_stamp
        if measurement_stamp <= 0.0 or age < -0.001:
            age = math.nan
        camera = (
            np.asarray(camera_point, dtype=float)
            if camera_point is not None
            else np.full(3, math.nan)
        )
        base = np.full(3, math.nan)
        tf_time = math.nan
        if np.isfinite(camera).all():
            transform = self._base_transform(frame_id, stamp)
            if transform is not None:
                rotation, translation, tf_time = transform
                base = rotation @ camera + translation
        self._recorder.record(
            {
                "timestamp": measurement_stamp,
                "target_id": target_id,
                "source": source,
                "confidence": confidence,
                "u": u,
                "v": v,
                "depth_raw": depth_raw,
                # This is the projected camera optical Z actually consumed downstream.
                "depth_filtered": camera[2],
                "x_cam": camera[0],
                "y_cam": camera[1],
                "z_cam": camera[2],
                "x_base": base[0],
                "y_base": base[1],
                "z_base": base[2],
                "measurement_age": age,
                "measurement_stamp": measurement_stamp,
                "callback_receive_time": receive_time,
                "tf_transform_time": tf_time,
                "processing_latency": age,
                "detection_index": detection_index,
                "assignment_distance": assignment_distance,
            }
        )

    def _record_measured(self, msg: Detection2DArray) -> None:
        if self._recorder is None:
            return
        depth = self._depths.pop(self._key(msg.header.stamp), None)
        for index, detection in enumerate(msg.detections):
            target_id = detection.id.strip()
            if not target_id.startswith("teat_") or not detection.results:
                continue
            result = max(
                detection.results,
                key=lambda item: float(item.hypothesis.score),
            )
            u = float(detection.bbox.center.position.x)
            v = float(detection.bbox.center.position.y)
            depth_raw = math.nan
            if depth is not None:
                column, row = int(round(u)), int(round(v))
                if 0 <= row < depth.shape[0] and 0 <= column < depth.shape[1]:
                    value = int(depth[row, column])
                    if value > 0:
                        depth_raw = value * 0.001
            position = result.pose.pose.position
            self._record_target(
                stamp=msg.header.stamp,
                frame_id=msg.header.frame_id,
                target_id=target_id,
                source="MEASURED",
                confidence=float(result.hypothesis.score),
                u=u,
                v=v,
                camera_point=(position.x, position.y, position.z),
                depth_raw=depth_raw,
                detection_index=index,
            )

    def _on_raw_detections(self, msg: Detection2DArray) -> None:
        # When the identity tracker is running, keep the image for its message.
        if self.count_publishers(self._tracked_topic) == 0:
            self._on_detections(msg)

    def _on_udder_status(self, msg: String) -> None:
        try:
            status = json.loads(msg.data)
        except (TypeError, ValueError):
            self._prediction_label = ""
            return
        predicted = status.get("predicted", [])
        self._prediction_label = (
            "PREDICTED: " + ", ".join(predicted)
            if status.get("prediction") == "two_point" and predicted
            else ""
        )
        if self._recorder is None or "stamp" not in status:
            return
        stamp = Time(nanoseconds=int(float(status["stamp"]) * 1e9)).to_msg()
        frame_id = str(status.get("camera_frame", ""))
        for name, point in status.get("predicted_points", {}).items():
            target_id = name if str(name).startswith("teat_") else f"teat_{name}"
            self._record_target(
                stamp=stamp,
                frame_id=frame_id,
                target_id=target_id,
                source="PREDICTED",
                camera_point=point,
            )
        for name in status.get("lost", []):
            target_id = name if str(name).startswith("teat_") else f"teat_{name}"
            self._record_target(
                stamp=stamp,
                frame_id=frame_id,
                target_id=target_id,
                source="LOST",
            )

    def _on_entry_status(self, msg: EntryStatus) -> None:
        self._entry_status = msg

    def _draw_entry_overlay(self, image: np.ndarray) -> None:
        status = self._entry_status
        if status is None:
            return
        pixels = (
            (int(status.left_inner_u), int(status.left_inner_v)),
            (int(status.right_inner_u), int(status.right_inner_v)),
        )
        valid_pixels = all(x >= 0 and y >= 0 for x, y in pixels)
        color = (0, 210, 0) if status.corridor_clear and status.stable else (0, 0, 255)
        if valid_pixels:
            cv2.line(image, pixels[0], pixels[1], color, 2, cv2.LINE_AA)
            for point in pixels:
                cv2.circle(image, point, 5, color, -1, cv2.LINE_AA)
        label = (
            f"ENTRY id={status.cow_track_id} gap={status.gap_m:.3f}m "
            f"v={status.line_speed_mps:.3f}m/s {status.reason}"
        )
        cv2.putText(
            image, label, (6, 42), cv2.FONT_HERSHEY_SIMPLEX,
            0.38, color, 1, cv2.LINE_AA,
        )

    def _on_detections(self, msg: Detection2DArray) -> None:
        self._record_measured(msg)
        item = self._images.pop(self._key(msg.header.stamp), None)
        if item is None or self.image_pub.get_subscription_count() == 0:
            return
        header, intensity = item
        image = intensity_to_bgr(intensity, *intensity.shape)
        self._draw_entry_overlay(image)
        for detection in msg.detections:
            bbox = detection.bbox
            x1 = int(round(bbox.center.position.x - bbox.size_x * 0.5))
            y1 = int(round(bbox.center.position.y - bbox.size_y * 0.5))
            x2 = int(round(bbox.center.position.x + bbox.size_x * 0.5))
            y2 = int(round(bbox.center.position.y + bbox.size_y * 0.5))
            semantic_id = detection.id.strip()
            if semantic_id.startswith("teat_"):
                color = (0, 255, 0)
            elif semantic_id == "unassigned":
                color = (0, 180, 255)
            else:
                color = (0, 255, 0) if semantic_id == "target" else (0, 180, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            label = semantic_id or "target"
            if detection.results:
                result = detection.results[0]
                if not semantic_id:
                    label = f"class {result.hypothesis.class_id}"
                label += f" {result.hypothesis.score:.2f}"
            cv2.putText(
                image,
                label,
                (max(0, x1), max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
        if not msg.detections:
            cv2.putText(
                image,
                "NO TARGET",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        if self._prediction_label:
            cv2.putText(
                image,
                self._prediction_label,
                (6, image.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        output = Image()
        output.header = header
        output.height, output.width = image.shape[:2]
        output.encoding = "bgr8"
        output.is_bigendian = False
        output.step = output.width * 3
        output.data = np.ascontiguousarray(image).tobytes()
        self.image_pub.publish(output)

    def destroy_node(self):
        if self._recorder is not None:
            payload = self._recorder.close()
            self.get_logger().info(
                "Vision stability summary: "
                + json.dumps(payload, ensure_ascii=False, allow_nan=True)
            )
            self._recorder = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectionDebugNode()
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
