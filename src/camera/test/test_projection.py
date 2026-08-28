import numpy as np
import pytest

from camera.projection import RadialProjection


def test_radial_projection_projects_only_requested_pixels():
    depth = np.full((4, 6), 1000.0, dtype=np.float32)
    model = RadialProjection(
        width=6,
        height=4,
        fx=100.0,
        fy=100.0,
        cx=2.5,
        cy=1.5,
        z_offset_mm=12.0,
    )

    points = model.project(depth, np.array([1, 2]), np.array([2, 3]))

    assert points.shape == (2, 3)
    assert points.dtype == np.float32
    assert np.isfinite(points).all()
    assert points[:, 2] == pytest.approx([1011.975, 1011.975], abs=0.01)


def test_radial_projection_rejects_shape_mismatch_and_marks_bad_depth():
    model = RadialProjection(3, 2, 100.0, 100.0, 1.0, 0.5)
    depth = np.full((2, 3), 1000.0, dtype=np.float32)
    depth[0, 0] = np.nan

    assert np.isnan(model.project(depth, np.array([0]), np.array([0]))).all()
    with pytest.raises(ValueError):
        model.project(np.ones((3, 2), dtype=np.float32), np.array([0]), np.array([0]))



def load_tests(loader, standard_tests, pattern):
    """For `python3 -m unittest` (colcon fallback): collect TestCase classes
    and plain test_* functions alike, without double-collecting. pytest
    ignores this hook and keeps running everything directly."""
    import unittest
    suite = loader.suiteClass()
    for name, obj in list(globals().items()):
        if isinstance(obj, type) and name.startswith('Test') and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:
            suite.addTest(loader.loadTestsFromTestCase(obj))
        elif name.startswith('test_') and callable(obj) and not isinstance(obj, type):
            suite.addTest(unittest.FunctionTestCase(obj))
    return suite
