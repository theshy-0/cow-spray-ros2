"""YOLO detector: synchronized camera images -> detections and target TF."""

from __future__ import annotations

import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64
from tf2_ros import TransformBroadcaster
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/detector_node_ultralytics")
from ultralytics import YOLO

from .processing import intensity_to_bgr, select_target_xyz_from_depth_mm


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


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("detector_node")
        self.declare_parameter("intensity_topic", "/sick_camera/intensity")
        self.declare_parameter("depth_topic", "/sick_camera/depth")
        self.declare_parameter("camera_info_topic", "/sick_camera/camera_info")
        self.declare_parameter("range_offset_topic", "/sick_camera/range_offset_mm")
        self.declare_parameter("detections_topic", "/detector_node/detections")
        self.declare_parameter("model_path", "")
        self.declare_parameter("target_class_id", 0)
        self.declare_parameter("confidence", 0.30)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("inference_size", 256)
        self.declare_parameter("device", "")
        self.declare_parameter("parent_frame", "camera_optical_frame")
        self.declare_parameter("child_frame", "target")
        self.declare_parameter("roi_scale", 0.50)
        self.declare_parameter("min_valid_points", 8)
        self.declare_parameter("max_depth_m", 10.0)

        model_path = str(self._param("model_path")).strip()
        if not model_path:
            model_path = str(
                Path(get_package_share_directory("teat_detection"))
                / "models" / "best.pt"
            )
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.get_logger().info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        self.detection_pub = self.create_publisher(
            Detection2DArray, str(self._param("detections_topic")), 1
        )
        self.broadcaster = TransformBroadcaster(self)
        self._depth_frames: OrderedDict = OrderedDict()
        self._intensity_frames: OrderedDict = OrderedDict()
        self._camera_info: CameraInfo | None = None
        self._range_offset_mm = 0.0
        self._last_lost_log = 0.0

        self.create_subscription(Image, str(self._param("depth_topic")), self._on_depth, SENSOR_QOS)
        self.create_subscription(Image, str(self._param("intensity_topic")), self._on_intensity, SENSOR_QOS)
        self.create_subscription(CameraInfo, str(self._param("camera_info_topic")), self._on_camera_info, CALIBRATION_QOS)
        self.create_subscription(Float64, str(self._param("range_offset_topic")), self._on_range_offset, CALIBRATION_QOS)

    def _param(self, name: str):
        return self.get_parameter(name).value

    @staticmethod
    def _stamp_key(msg: Image) -> tuple[int, int]:
        return msg.header.stamp.sec, msg.header.stamp.nanosec

    @staticmethod
    def _decode_u16(msg: Image) -> np.ndarray:
        if msg.encoding not in ("16UC1", "mono16"):
            raise ValueError(f"Expected 16UC1 image, got {msg.encoding}")
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self._camera_info = msg

    def _on_range_offset(self, msg: Float64) -> None:
        self._range_offset_mm = float(msg.data)

    def _on_depth(self, msg: Image) -> None:
        self._store_and_try(self._depth_frames, msg)

    def _on_intensity(self, msg: Image) -> None:
        self._store_and_try(self._intensity_frames, msg)

    def _store_and_try(self, frames: OrderedDict, msg: Image) -> None:
        try:
            key = self._stamp_key(msg)
            frames[key] = (msg, self._decode_u16(msg))
            while len(frames) > 4:
                frames.popitem(last=False)
            self._try_process(key)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)

    def _try_process(self, key: tuple[int, int]) -> None:
        if key not in self._depth_frames or key not in self._intensity_frames:
            return
        _, depth = self._depth_frames.pop(key)
        intensity_msg, intensity = self._intensity_frames.pop(key)
        if self._camera_info is None:
            self.get_logger().warning("Waiting for camera calibration", throttle_duration_sec=2.0)
            return
        self._process_frame(intensity_msg, intensity, depth)

    def _process_frame(self, image_msg: Image, intensity: np.ndarray, depth: np.ndarray) -> None:
        image = intensity_to_bgr(intensity, image_msg.height, image_msg.width)
        args = {
            "source": image,
            "conf": float(self._param("confidence")),
            "iou": float(self._param("iou")),
            "imgsz": int(self._param("inference_size")),
            "verbose": False,
        }
        device = str(self._param("device")).strip()
        if device:
            args["device"] = device
        result = self.model.predict(**args)[0]

        output = Detection2DArray()
        output.header = image_msg.header
        candidates = []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            classes = boxes.cls.detach().cpu().numpy().astype(int)
            scores = boxes.conf.detach().cpu().numpy()
            coordinates = boxes.xyxy.detach().cpu().numpy()
            target_class = int(self._param("target_class_id"))
            for class_id, score, xyxy in zip(classes, scores, coordinates):
                if target_class >= 0 and class_id != target_class:
                    continue
                xyz_mm = self._project(depth, xyxy)
                detection = self._make_detection(image_msg, int(class_id), float(score), xyxy, xyz_mm)
                output.detections.append(detection)
                candidates.append((float(score), detection, xyz_mm))

        valid = [item for item in candidates if item[2] is not None]
        if valid:
            _, detection, xyz_mm = max(valid, key=lambda item: item[0])
            detection.id = "target"
            self._publish_tf(image_msg, xyz_mm)
        else:
            self._target_lost("no target with valid depth")
        self.detection_pub.publish(output)

    def _project(self, depth: np.ndarray, xyxy: np.ndarray) -> np.ndarray | None:
        info = self._camera_info
        if info is None or len(info.k) < 6:
            return None
        distortion = list(info.d) + [0.0, 0.0]
        return select_target_xyz_from_depth_mm(
            depth, xyxy, float(self._param("roi_scale")),
            int(self._param("min_valid_points")), float(self._param("max_depth_m")),
            float(info.k[0]), float(info.k[4]), float(info.k[2]), float(info.k[5]),
            float(distortion[0]), float(distortion[1]), self._range_offset_mm,
        )

    @staticmethod
    def _make_detection(image_msg, class_id, score, xyxy, xyz_mm) -> Detection2D:
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        msg = Detection2D()
        msg.header = image_msg.header
        msg.bbox.center.position.x = (x1 + x2) * 0.5
        msg.bbox.center.position.y = (y1 + y2) * 0.5
        msg.bbox.size_x = x2 - x1
        msg.bbox.size_y = y2 - y1
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = str(class_id)
        hypothesis.hypothesis.score = score
        if xyz_mm is not None:
            hypothesis.pose.pose.position.x = float(xyz_mm[0]) * 0.001
            hypothesis.pose.pose.position.y = float(xyz_mm[1]) * 0.001
            hypothesis.pose.pose.position.z = float(xyz_mm[2]) * 0.001
        msg.results.append(hypothesis)
        return msg

    def _publish_tf(self, image_msg: Image, xyz_mm: np.ndarray) -> None:
        transform = TransformStamped()
        transform.header = image_msg.header
        transform.header.frame_id = str(self._param("parent_frame"))
        transform.child_frame_id = str(self._param("child_frame"))
        transform.transform.translation.x = float(xyz_mm[0]) * 0.001
        transform.transform.translation.y = float(xyz_mm[1]) * 0.001
        transform.transform.translation.z = float(xyz_mm[2]) * 0.001
        transform.transform.rotation.w = 1.0
        self.broadcaster.sendTransform(transform)

    def _target_lost(self, reason: str) -> None:
        now = time.monotonic()
        if now - self._last_lost_log >= 2.0:
            self.get_logger().info(f"Target unavailable ({reason}); TF update stopped")
            self._last_lost_log = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
