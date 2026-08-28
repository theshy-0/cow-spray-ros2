import numpy as np

from teat_detection.processing import (
    intensity_to_bgr,
    select_target_xyz_mm,
    select_target_xyz_from_depth_mm,
    xyz_to_optical_m,
)


def test_axis_and_unit_conversion():
    actual = xyz_to_optical_m([100.0, -200.0, 1000.0], "sick_sensor")
    np.testing.assert_allclose(actual, [-0.1, 0.2, 1.0])


def test_sdk_optical_only_changes_mm_to_m():
    actual = xyz_to_optical_m([100.0, 200.0, 1000.0], "ros_optical")
    np.testing.assert_allclose(actual, [0.1, 0.2, 1.0])


def test_xyz_median_rejects_background():
    cloud = np.zeros((20, 20, 3), dtype=np.float32)
    cloud[5:15, 5:15] = [100.0, 200.0, 1000.0]
    cloud[5:7, 5:7] = [900.0, 900.0, 4000.0]
    actual = select_target_xyz_mm(cloud, [4, 4, 16, 16], 1.0, 20, 10.0)
    np.testing.assert_allclose(actual, [100.0, 200.0, 1000.0])


def test_no_valid_xyz_returns_none():
    cloud = np.zeros((10, 10, 3), dtype=np.float32)
    assert select_target_xyz_mm(cloud, [0, 0, 10, 10], 1.0, 1, 10.0) is None


def test_intensity_image_shape_and_type():
    image = intensity_to_bgr(np.arange(12), 3, 4)
    assert image.shape == (3, 4, 3)
    assert image.dtype == np.uint8


def test_selected_depth_projection_at_principal_point():
    depth = np.full((5, 5), 1000, dtype=np.uint16)
    actual = select_target_xyz_from_depth_mm(
        depth, [1, 1, 4, 4], 1.0, 4, 10.0,
        fx=100.0, fy=100.0, cx=2.0, cy=2.0,
        k1=0.0, k2=0.0, z_offset_mm=0.0,
    )
    np.testing.assert_allclose(actual, [0.0, 0.0, 1000.0], atol=0.1)


def test_projection_origin_does_not_follow_unrelated_center_depth():
    baseline = np.full((7, 7), 1000, dtype=np.uint16)
    changed_background = baseline.copy()
    changed_background[3, 3] = 4000
    kwargs = dict(
        xyxy=[4, 4, 7, 7], roi_scale=1.0, min_valid_points=4,
        max_depth_m=10.0, fx=100.0, fy=100.0, cx=3.2, cy=2.4,
        k1=0.0, k2=0.0, z_offset_mm=0.0,
    )
    expected = select_target_xyz_from_depth_mm(baseline, **kwargs)
    actual = select_target_xyz_from_depth_mm(changed_background, **kwargs)
    np.testing.assert_allclose(actual, expected, atol=0.1)
