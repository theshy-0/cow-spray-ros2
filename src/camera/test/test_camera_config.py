import struct

import pytest

from camera.camera_config import BINNING, decode_binning, fps_to_period_us


def test_binning_values_and_resolution_are_fixed_by_camera_protocol():
    assert BINNING == {
        'none': (0, '512x424'),
        '2x2': (1, '256x212'),
        '4x4': (2, '128x106'),
    }
    assert decode_binning(struct.pack('>B', 1)) == 1


def test_unknown_binning_is_rejected():
    with pytest.raises(ValueError):
        decode_binning(b'\x09')


def test_fps_is_converted_to_supported_frame_period():
    assert fps_to_period_us(25) == 40_000
    assert fps_to_period_us(30) == 33_333
    with pytest.raises(ValueError):
        fps_to_period_us(31)



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
