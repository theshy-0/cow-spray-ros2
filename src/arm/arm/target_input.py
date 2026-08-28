"""Convert timestamped tracked detections into base-frame target measurements."""

from dataclasses import dataclass
import math

from geometry_msgs.msg import PointStamped
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_point


@dataclass(frozen=True)
class TopicTargetFrame:
    stamp: float
    frame_id: str
    camera_points: dict[str, list[float]]
    base_points: dict[str, list[float]]
    confidences: dict[str, float]


class TopicTargetAdapter:
    """One-shot Topic measurement validation and historical TF2 conversion."""

    def __init__(self, tf_buffer, base_frame, offsets, min_confidence):
        self.tf_buffer = tf_buffer
        self.base_frame = str(base_frame)
        self.offsets = {name: list(value) for name, value in offsets.items()}
        self.min_confidence = float(min_confidence)

    def convert(self, message):
        frame_id = str(message.header.frame_id)
        if not frame_id:
            raise ValueError('tracked detection frame_id is empty')
        stamp_time = Time.from_msg(message.header.stamp)
        stamp = stamp_time.nanoseconds * 1e-9
        camera_points = {}
        confidences = {}
        for detection in message.detections:
            name = str(detection.id)
            if name not in self.offsets or not detection.results:
                continue
            result = max(
                detection.results,
                key=lambda item: float(item.hypothesis.score))
            confidence = float(result.hypothesis.score)
            position = result.pose.pose.position
            point = [float(position.x), float(position.y), float(position.z)]
            if (confidence < self.min_confidence
                    or not all(math.isfinite(value) for value in point)
                    or point[2] <= 0.0):
                continue
            camera_points[name] = point
            confidences[name] = confidence
        if not camera_points:
            return TopicTargetFrame(stamp, frame_id, {}, {}, {})

        transform = self.tf_buffer.lookup_transform(
            self.base_frame, frame_id, stamp_time)
        base_points = {}
        for name, coordinates in camera_points.items():
            source = PointStamped()
            source.header = message.header
            source.point.x, source.point.y, source.point.z = coordinates
            converted = do_transform_point(source, transform)
            offset = self.offsets[name]
            base_points[name] = [
                float(converted.point.x) + offset[0],
                float(converted.point.y) + offset[1],
                float(converted.point.z) + offset[2],
            ]
        return TopicTargetFrame(
            stamp, frame_id, camera_points, base_points, confidences)
