import cv2
import numpy as np

from teat_detection.handeye_marker_node import (
    estimate_marker_pose,
    marker_object_points,
    rotation_vector_to_quaternion,
)


def test_synthetic_square_pose_is_recovered():
    camera_matrix = np.array(
        [[300.0, 0.0, 128.0], [0.0, 300.0, 106.0], [0.0, 0.0, 1.0]]
    )
    expected_rotation = np.array([0.12, -0.08, 0.04])
    expected_translation = np.array([0.03, -0.02, 0.80])
    corners, _ = cv2.projectPoints(
        marker_object_points(0.12),
        expected_rotation,
        expected_translation,
        camera_matrix,
        np.zeros(5),
    )

    actual = estimate_marker_pose(corners, camera_matrix, np.zeros(5), 0.12)

    assert actual is not None
    rotation, translation = actual
    np.testing.assert_allclose(translation, expected_translation, atol=1e-5)
    assert abs(np.linalg.norm(rotation) - np.linalg.norm(expected_rotation)) < 1e-4
    assert abs(np.linalg.norm(rotation_vector_to_quaternion(rotation)) - 1.0) < 1e-12
