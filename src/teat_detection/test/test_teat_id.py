import numpy as np

from teat_detection.teat_id import TeatDetection, TeatTracker


def detections(offset=(0.0, 0.0, 0.0)):
    delta = np.asarray(offset, dtype=float)
    return [
        TeatDetection((100, 80), np.array([-0.04, 0.04, 0.60]) + delta, 0.9, 0),
        TeatDetection((200, 82), np.array([0.04, 0.04, 0.60]) + delta, 0.9, 1),
        TeatDetection((105, 170), np.array([-0.04, -0.04, 0.60]) + delta, 0.9, 2),
        TeatDetection((198, 168), np.array([0.04, -0.04, 0.60]) + delta, 0.9, 3),
    ]


def test_builds_semantic_frame_and_tracks_translation():
    tracker = TeatTracker(expected_z=(0.0, 0.0, -1.0))
    first = tracker.update(detections(), 1.0)
    second = tracker.update(list(reversed(detections((0.01, -0.02, 0.0)))), 1.1)
    assert np.allclose(first.origin, [0.0, 0.0, 0.60])
    assert np.allclose(second.origin, [0.01, -0.02, 0.60], atol=1e-6)
    assert second.rotation[2, 2] < 0.0
    assert second.residual < 1e-6


def test_rejects_large_identity_jump():
    tracker = TeatTracker(max_match_distance=0.03)
    tracker.update(detections(), 1.0)
    moved = detections((0.20, 0.0, 0.0))
    try:
        tracker.update(moved, 1.1)
    except ValueError as exc:
        assert "motion limit" in str(exc)
    else:
        raise AssertionError("large jump was accepted")


def test_recovers_when_fourth_teat_appears_after_three_point_initialization():
    tracker = TeatTracker()
    first = tracker.update(detections()[1:], 1.0)
    assert len(first.teats) == 3

    recovered = tracker.update(detections(), 1.1)
    assert set(recovered.teats) == {
        "front_left", "front_right", "rear_left", "rear_right"
    }
    assert set(recovered.source_indices) == set(recovered.teats)
    assert set(recovered.source_indices.values()) == {0, 1, 2, 3}


def test_matches_two_visible_teats_to_ids_from_previous_frame():
    tracker = TeatTracker()
    first = tracker.update(detections(), 1.0)
    names_by_source = {
        source_index: name for name, source_index in first.source_indices.items()
    }
    moved = detections((0.01, -0.01, 0.0))

    partial = tracker.match_partial([moved[3], moved[0]], 1.1)

    assert partial[names_by_source[0]] == 0
    assert partial[names_by_source[3]] == 3


def test_does_not_guess_ids_before_identity_has_been_established():
    tracker = TeatTracker()
    assert tracker.match_partial(detections()[:2], 1.0) == {}


def test_completes_four_positions_from_two_identified_teats():
    tracker = TeatTracker()
    first = tracker.update(detections(), 1.0)
    names_by_source = {
        source_index: name for name, source_index in first.source_indices.items()
    }
    angle = np.deg2rad(5.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([0.01, -0.01, 0.0])
    moved = [
        TeatDetection(
            detection.pixel_uv,
            rotation @ np.asarray(detection.position) + translation,
            detection.score,
            detection.source_index,
        )
        for detection in detections()
    ]

    predicted = tracker.predict_two([moved[3], moved[0]], 1.1)

    assert set(predicted.teats) == set(names_by_source.values())
    assert set(predicted.observed_names) == {names_by_source[0], names_by_source[3]}
    assert set(predicted.predicted_names) == {names_by_source[1], names_by_source[2]}
    for source_index, name in names_by_source.items():
        assert np.allclose(predicted.teats[name], moved[source_index].position, atol=1e-6)


def test_stops_two_point_completion_after_timeout():
    tracker = TeatTracker(two_point_timeout=0.2)
    tracker.update(detections(), 1.0)
    try:
        tracker.predict_two(detections()[:2], 1.3)
    except ValueError as exc:
        assert "timeout" in str(exc)
    else:
        raise AssertionError("expired two-point prediction was accepted")
