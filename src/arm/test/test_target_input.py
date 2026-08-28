import math

import pytest
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from arm.target_input import TopicTargetAdapter


def _message(stamp_sec=1, confidence=0.9, depth=1.0):
    message = Detection2DArray()
    message.header.frame_id = "sick_camera_optical_frame"
    message.header.stamp = Time(sec=stamp_sec)
    detection = Detection2D()
    detection.id = "teat_front_left"
    result = ObjectHypothesisWithPose()
    result.hypothesis.class_id = "0"
    result.hypothesis.score = confidence
    result.pose.pose.position.x = 1.0
    result.pose.pose.position.y = 0.0
    result.pose.pose.position.z = depth
    detection.results.append(result)
    message.detections.append(detection)
    return message


class FakeBuffer:
    def __init__(self, transform_for_stamp):
        self.transform_for_stamp = transform_for_stamp
        self.queries = []

    def lookup_transform(self, target, source, stamp):
        self.queries.append((target, source, stamp.nanoseconds))
        return self.transform_for_stamp(stamp.nanoseconds)


def _transform(tx=0.0, ty=0.0, tz=0.0, qz=0.0, qw=1.0):
    transform = TransformStamped()
    transform.header.frame_id = "base_link"
    transform.child_frame_id = "sick_camera_optical_frame"
    transform.transform.translation.x = tx
    transform.transform.translation.y = ty
    transform.transform.translation.z = tz
    transform.transform.rotation.z = qz
    transform.transform.rotation.w = qw
    return transform


def _adapter(buffer):
    return TopicTargetAdapter(
        tf_buffer=buffer,
        base_frame="base_link",
        offsets={"teat_front_left": [0.0, 0.0, 0.0]},
        min_confidence=0.5,
    )


def test_point_stamped_camera_to_base_translation():
    buffer = FakeBuffer(lambda _stamp: _transform(tx=0.5, ty=-0.2, tz=0.1))
    frame = _adapter(buffer).convert(_message())
    assert frame.base_points["teat_front_left"] == pytest.approx([1.5, -0.2, 1.1])


def test_camera_to_base_applies_rotation_not_only_translation():
    half = math.sqrt(0.5)
    buffer = FakeBuffer(lambda _stamp: _transform(qz=half, qw=half))
    frame = _adapter(buffer).convert(_message(depth=1.0))
    assert frame.base_points["teat_front_left"] == pytest.approx([0.0, 1.0, 1.0])


def test_measurement_timestamp_selects_historical_transform():
    def transform_for_stamp(stamp_ns):
        return _transform(tx=1.0 if stamp_ns == 1_000_000_000 else 2.0)

    buffer = FakeBuffer(transform_for_stamp)
    frame = _adapter(buffer).convert(_message(stamp_sec=1))
    assert frame.base_points["teat_front_left"][0] == pytest.approx(2.0)
    assert buffer.queries == [
        ("base_link", "sick_camera_optical_frame", 1_000_000_000)
    ]


def test_low_confidence_or_invalid_depth_is_not_a_measurement():
    buffer = FakeBuffer(lambda _stamp: _transform())
    assert _adapter(buffer).convert(_message(confidence=0.2)).base_points == {}
    assert _adapter(buffer).convert(_message(depth=0.0)).base_points == {}


def test_missing_timestamp_transform_is_reported_without_latest_fallback():
    def unavailable(_stamp):
        raise RuntimeError("transform unavailable")

    with pytest.raises(RuntimeError, match="transform unavailable"):
        _adapter(FakeBuffer(unavailable)).convert(_message())


def test_four_semantic_ids_survive_topic_conversion():
    message = _message()
    message.detections.clear()
    ids = [
        "teat_front_left", "teat_rear_left",
        "teat_rear_right", "teat_front_right",
    ]
    offsets = {}
    for index, name in enumerate(ids):
        detection = Detection2D()
        detection.id = name
        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id = "0"
        result.hypothesis.score = 0.9
        result.pose.pose.position.x = 0.1 * index
        result.pose.pose.position.z = 1.0
        detection.results.append(result)
        message.detections.append(detection)
        offsets[name] = [0.0, 0.0, 0.0]
    adapter = TopicTargetAdapter(
        FakeBuffer(lambda _stamp: _transform()),
        "base_link", offsets, 0.5)
    frame = adapter.convert(message)
    assert set(frame.base_points) == set(ids)
