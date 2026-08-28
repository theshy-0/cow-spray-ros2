"""Conversions between Codroid mm/degree feedback and ROS SI poses."""

from __future__ import annotations

import math
from typing import Sequence


def euler_to_quaternion(roll: float, pitch: float, yaw: float):
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def convert_tcp_pose(tcp_pose: Sequence[float]):
    if tcp_pose is None or len(tcp_pose) < 6:
        raise ValueError("tcp_pose must contain six values")
    x, y, z, rx, ry, rz = map(float, tcp_pose[:6])
    rpy = tuple(math.radians(value) for value in (rx, ry, rz))
    return (x / 1000.0, y / 1000.0, z / 1000.0), euler_to_quaternion(*rpy), rpy

