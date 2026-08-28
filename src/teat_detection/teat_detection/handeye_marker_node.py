"""Detect one ArUco marker for hand-eye calibration and publish its pose."""

from __future__ import annotations

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from .processing import intensity_to_bgr


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


def marker_object_points(marker_size_m: float) -> np.ndarray:
    """Return IPPE square corners in ArUco corner order."""
    half = float(marker_size_m) * 0.5
    if half <= 0.0:
        raise ValueError("marker_size_m must be positive")
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )


def estimate_marker_pose(
    corners: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    marker_size_m: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate marker-to-camera pose from four detected image corners."""
    success, rotation, translation = cv2.solvePnP(
        marker_object_points(marker_size_m),
        np.asarray(corners, dtype=np.float32).reshape(4, 2),
        np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
        np.asarray(distortion, dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None
    return rotation.reshape(3, 1), translation.reshape(3)


def rotation_vector_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert an OpenCV Rodrigues vector to a normalized XYZW quaternion."""
    matrix, _ = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64).reshape(3, 1))
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = np.array(
                [0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                 (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[2, 1] - matrix[1, 2]) / scale]
            )
        elif index == 1:
            scale = 2.0 * np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = np.array(
                [(matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale,
                 (matrix[0, 2] - matrix[2, 0]) / scale]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = np.array(
                [(matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale,
                 (matrix[1, 0] - matrix[0, 1]) / scale]
            )
    return quaternion / np.linalg.norm(quaternion)


class HandeyeMarkerNode(Node):
    def __init__(self) -> None:
        super().__init__("handeye_marker_node")
        self.declare_parameter("intensity_topic", "/camera_node/intensity")
        self.declare_parameter("camera_info_topic", "/camera_node/camera_info")
        self.declare_parameter("debug_image_topic", "/handeye_marker/debug_image")
        self.declare_parameter("dictionary", "DICT_4X4_50")
        self.declare_parameter("marker_id", 0)
        self.declare_parameter("marker_size_m", 0.12)
        self.declare_parameter("parent_frame", "sick_camera_optical_frame")
        self.declare_parameter("child_frame", "calibration_marker")
        self.declare_parameter("axis_length_m", 0.06)

        dictionary_name = str(self.get_parameter("dictionary").value)
        dictionary_id = getattr(cv2.aruco, dictionary_name, None)
        if dictionary_id is None:
            raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(dictionary_id),
            cv2.aruco.DetectorParameters(),
        )
        self.marker_id = int(self.get_parameter("marker_id").value)
        self.marker_size_m = float(self.get_parameter("marker_size_m").value)
        self.axis_length_m = float(self.get_parameter("axis_length_m").value)
        marker_object_points(self.marker_size_m)
        if self.axis_length_m <= 0.0:
            raise ValueError("axis_length_m must be positive")

        self.camera_info: CameraInfo | None = None
        self.tf_broadcaster = TransformBroadcaster(self)
        self.debug_pub = self.create_publisher(
            Image, str(self.get_parameter("debug_image_topic").value), SENSOR_QOS
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._on_camera_info,
            CALIBRATION_QOS,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("intensity_topic").value),
            self._on_image,
            SENSOR_QOS,
        )
        self.get_logger().info(
            f"ArUco calibration marker: {dictionary_name} id={self.marker_id} "
            f"size={self.marker_size_m:.4f}m"
        )

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.camera_info = msg

    def _on_image(self, msg: Image) -> None:
        if msg.encoding not in ("16UC1", "mono16"):
            self.get_logger().error(
                f"Expected 16UC1 image, got {msg.encoding}",
                throttle_duration_sec=5.0,
            )
            return
        if self.camera_info is None:
            self.get_logger().warning(
                "Waiting for camera_info", throttle_duration_sec=2.0
            )
            return
        if (msg.width, msg.height) != (self.camera_info.width, self.camera_info.height):
            self.get_logger().error(
                "Image and camera_info resolutions differ",
                throttle_duration_sec=2.0,
            )
            return

        intensity = np.frombuffer(msg.data, dtype=np.uint16).reshape(
            msg.height, msg.width
        )
        debug = intensity_to_bgr(intensity, msg.height, msg.width)
        gray = cv2.cvtColor(debug, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(debug, corners, ids)
            matches = np.flatnonzero(ids.reshape(-1) == self.marker_id)
            if matches.size:
                self._publish_marker(msg, debug, corners[int(matches[0])])
            else:
                self._draw_status(debug, f"MARKER {self.marker_id} NOT FOUND", False)
        else:
            self._draw_status(debug, f"MARKER {self.marker_id} NOT FOUND", False)
        self.debug_pub.publish(self._image_message(msg, debug))

    def _publish_marker(self, msg: Image, debug: np.ndarray, corners: np.ndarray) -> None:
        info = self.camera_info
        camera_matrix = np.asarray(info.k, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(info.d, dtype=np.float64)
        pose = estimate_marker_pose(
            corners, camera_matrix, distortion, self.marker_size_m
        )
        if pose is None:
            self._draw_status(debug, "POSE FAILED", False)
            return
        rotation, translation = pose
        quaternion = rotation_vector_to_quaternion(rotation)
        cv2.drawFrameAxes(
            debug,
            camera_matrix,
            distortion,
            rotation,
            translation.reshape(3, 1),
            self.axis_length_m,
            2,
        )
        self._draw_status(
            debug,
            f"ID {self.marker_id} xyz=({translation[0]:.3f},"
            f"{translation[1]:.3f},{translation[2]:.3f})m",
            True,
        )

        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = str(self.get_parameter("parent_frame").value)
        transform.child_frame_id = str(self.get_parameter("child_frame").value)
        transform.transform.translation.x = float(translation[0])
        transform.transform.translation.y = float(translation[1])
        transform.transform.translation.z = float(translation[2])
        transform.transform.rotation.x = float(quaternion[0])
        transform.transform.rotation.y = float(quaternion[1])
        transform.transform.rotation.z = float(quaternion[2])
        transform.transform.rotation.w = float(quaternion[3])
        self.tf_broadcaster.sendTransform(transform)

    @staticmethod
    def _draw_status(image: np.ndarray, text: str, found: bool) -> None:
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(
            image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
            0.48, color, 1, cv2.LINE_AA,
        )

    @staticmethod
    def _image_message(source: Image, image: np.ndarray) -> Image:
        output = Image()
        output.header = source.header
        output.height, output.width = image.shape[:2]
        output.encoding = "bgr8"
        output.is_bigendian = False
        output.step = output.width * 3
        output.data = np.ascontiguousarray(image).tobytes()
        return output


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HandeyeMarkerNode()
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
