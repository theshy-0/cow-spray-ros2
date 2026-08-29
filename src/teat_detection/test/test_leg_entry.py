import numpy as np

from teat_detection.leg_entry import (
    LegEntryTracker,
    detect_leg_entry,
    parse_depth_image,
    transform_observation,
)


def _projector(depth):
    height, width = depth.shape
    x = np.broadcast_to(np.arange(width, dtype=np.float32), (height, width))
    y = np.broadcast_to(
        np.arange(height, dtype=np.float32)[:, None], (height, width)
    )
    cloud = np.dstack((x * 5.0, y * 5.0, depth)).astype(np.float32)
    return lambda rows, cols: cloud[np.asarray(rows), np.asarray(cols)]


def _detect(depth, stamp=1.0):
    return detect_leg_entry(
        depth,
        _projector(depth),
        stamp=stamp,
        min_depth_mm=400.0,
        max_depth_mm=1200.0,
        row_start_ratio=0.50,
        row_end_ratio=0.98,
        min_separation_px=20,
        min_height_px=20,
        min_aspect_ratio=0.8,
        min_blob_area_px=40,
    )


def test_depth_zero_is_invalid():
    data = np.array([[500, 0], [700, 800]], dtype=np.uint16)
    result = parse_depth_image(data.tobytes(), 2, 2, "16UC1")
    assert np.isnan(result[0, 1])
    assert result[1, 1] == 800.0


def test_two_independent_legs_produce_entry_geometry():
    depth = np.full((100, 140), np.nan, dtype=np.float32)
    depth[50:98, 15:40] = 600.0
    depth[50:98, 95:120] = 600.0
    observation = _detect(depth)
    assert observation.valid
    assert observation.reason == "OK"
    assert observation.left_pixel[0] < observation.right_pixel[0]


def test_single_blob_never_grants_entry():
    depth = np.full((100, 140), np.nan, dtype=np.float32)
    depth[50:98, 20:120] = 600.0
    observation = _detect(depth)
    assert not observation.valid
    assert observation.reason.startswith("EXPECTED_TWO_LEGS_GOT_")


def test_temporal_tracker_requires_stable_window_and_estimates_speed():
    depth = np.full((100, 140), np.nan, dtype=np.float32)
    depth[50:98, 15:40] = 600.0
    depth[50:98, 95:120] = 600.0
    tracker = LegEntryTracker(
        window_frames=4,
        min_valid_frames=3,
        stable_duration=0.2,
        max_center_spread_m=0.2,
        max_gap_spread_m=0.02,
        production_axis=np.array([1.0, 0.0, 0.0]),
    )
    result = None
    for index in range(3):
        observation = _detect(depth, stamp=1.0 + index * 0.1)
        observation = transform_observation(
            observation,
            np.eye(3),
            np.array([index * 0.01, 0.0, 0.0]),
        )
        result = tracker.update(observation)
    assert result.stable
    assert result.cow_track_id == 1
    np.testing.assert_allclose(result.line_speed_mps, 0.1, atol=1e-6)
