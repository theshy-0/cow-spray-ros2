from types import SimpleNamespace

import numpy as np

from teat_detection.debug_node import DetectionDebugNode


def test_entry_overlay_uses_green_only_for_stable_clear_corridor():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    status = SimpleNamespace(
        left_inner_u=20,
        left_inner_v=80,
        right_inner_u=80,
        right_inner_v=80,
        corridor_clear=True,
        stable=True,
        cow_track_id=1,
        gap_m=0.45,
        line_speed_mps=0.15,
        reason="OK",
    )

    DetectionDebugNode._draw_entry_overlay(image, status)

    assert image[80, 50, 1] > 0
    assert image[80, 50, 2] == 0


def test_teat_overlay_does_not_draw_entry_overlay():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    hypothesis = SimpleNamespace(class_id="0", score=0.9)
    result = SimpleNamespace(hypothesis=hypothesis)
    position = SimpleNamespace(x=50.0, y=50.0)
    center = SimpleNamespace(position=position)
    bbox = SimpleNamespace(center=center, size_x=40.0, size_y=40.0)
    detection = SimpleNamespace(id="teat_front_left", bbox=bbox, results=[result])
    message = SimpleNamespace(detections=[detection])
    node = object.__new__(DetectionDebugNode)
    node._prediction_label = ""

    node._draw_teat_overlay(image, message)

    assert image[30, 30, 1] > 0
    assert image[80, 50].sum() == 0
