"""Small, ROS-independent image and XYZ processing helpers."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def intensity_to_bgr(intensity: Sequence[float], height: int, width: int) -> np.ndarray:
    """Convert a SICK intensity map into an 8-bit, three-channel YOLO image."""
    image = np.asarray(intensity, dtype=np.float32).reshape(height, width)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        gray = np.zeros((height, width), dtype=np.uint8)
    else:
        low, high = np.percentile(finite, (1.0, 99.0))
        if high <= low:
            low = float(finite.min())
            high = float(finite.max())
        if high <= low:
            gray = np.zeros((height, width), dtype=np.uint8)
        else:
            scaled = (image - low) * (255.0 / (high - low))
            gray = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def select_target_xyz_mm(
    cloud_mm: np.ndarray,
    xyxy: Sequence[float],
    roi_scale: float,
    min_valid_points: int,
    max_depth_m: float,
) -> Optional[np.ndarray]:
    """Return a robust median XYZ (mm) from the inner part of a detection box."""
    height, width, channels = cloud_mm.shape
    if channels != 3:
        raise ValueError("cloud_mm must have shape HxWx3")

    x1, y1, x2, y2 = (float(v) for v in xyxy)
    scale = min(max(float(roi_scale), 0.05), 1.0)
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half_w, half_h = (x2 - x1) * 0.5 * scale, (y2 - y1) * 0.5 * scale
    left = max(0, min(width, int(np.floor(cx - half_w))))
    right = max(0, min(width, int(np.ceil(cx + half_w))))
    top = max(0, min(height, int(np.floor(cy - half_h))))
    bottom = max(0, min(height, int(np.ceil(cy + half_h))))
    if right <= left or bottom <= top:
        return None

    points = np.asarray(cloud_mm[top:bottom, left:right], dtype=np.float64).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=1)
    valid &= points[:, 2] > 0.0
    if max_depth_m > 0.0:
        valid &= points[:, 2] <= max_depth_m * 1000.0
    points = points[valid]
    if points.shape[0] < int(min_valid_points):
        return None

    # Reject foreground/background mixing near a box edge without temporal filtering.
    z_median = float(np.median(points[:, 2]))
    z_mad = float(np.median(np.abs(points[:, 2] - z_median)))
    z_tolerance_mm = max(50.0, 3.0 * 1.4826 * z_mad)
    points = points[np.abs(points[:, 2] - z_median) <= z_tolerance_mm]
    if points.shape[0] < int(min_valid_points):
        return None
    return np.median(points, axis=0)


def select_target_xyz_from_depth_mm(
    depth_mm: np.ndarray,
    xyxy: Sequence[float],
    roi_scale: float,
    min_valid_points: int,
    max_depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    k1: float,
    k2: float,
    z_offset_mm: float,
) -> Optional[np.ndarray]:
    """Project only valid pixels inside a detection ROI, then take median XYZ."""
    depth = np.asarray(depth_mm, dtype=np.float32)
    height, width = depth.shape
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    scale = min(max(float(roi_scale), 0.05), 1.0)
    box_cx, box_cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    half_w, half_h = (x2 - x1) * 0.5 * scale, (y2 - y1) * 0.5 * scale
    left = max(0, min(width, int(np.floor(box_cx - half_w))))
    right = max(0, min(width, int(np.ceil(box_cx + half_w))))
    top = max(0, min(height, int(np.floor(box_cy - half_h))))
    bottom = max(0, min(height, int(np.ceil(box_cy + half_h))))
    if right <= left or bottom <= top:
        return None

    rows, cols = np.mgrid[top:bottom, left:right]
    selected = depth[rows, cols].ravel()
    rows, cols = rows.ravel(), cols.ravel()
    valid = np.isfinite(selected) & (selected > 0.0)
    if max_depth_m > 0.0:
        valid &= selected <= max_depth_m * 1000.0
    selected, rows, cols = selected[valid], rows[valid], cols[valid]
    if selected.size < int(min_valid_points):
        return None

    z_median = float(np.median(selected))
    z_mad = float(np.median(np.abs(selected - z_median)))
    keep = np.abs(selected - z_median) <= max(50.0, 3.0 * 1.4826 * z_mad)
    selected, rows, cols = selected[keep], rows[keep], cols[keep]
    if selected.size < int(min_valid_points):
        return None

    x = (cols.astype(np.float32) - np.float32(cx)) / np.float32(fx)
    y = (rows.astype(np.float32) - np.float32(cy)) / np.float32(fy)
    radius2 = x * x + y * y
    distortion = 1.0 + np.float32(k1) * radius2 + np.float32(k2) * radius2**2
    x *= distortion
    y *= distortion
    inverse_norm = 1.0 / np.sqrt(x * x + y * y + 1.0)

    points = np.column_stack(
        (
            selected * x * inverse_norm,
            selected * y * inverse_norm,
            selected * inverse_norm + np.float32(z_offset_mm),
        )
    )
    return np.median(points, axis=0)


def xyz_to_optical_m(xyz_mm: Sequence[float], convention: str) -> np.ndarray:
    """Convert SDK XYZ in millimetres to ROS optical coordinates in metres."""
    xyz = np.asarray(xyz_mm, dtype=np.float64)
    if xyz.shape != (3,):
        raise ValueError("xyz_mm must contain exactly three values")
    if convention == "sick_sensor":
        # SICK: +X left, +Y up, +Z forward. ROS optical: right, down, forward.
        xyz = np.array([-xyz[0], -xyz[1], xyz[2]], dtype=np.float64)
    elif convention != "ros_optical":
        raise ValueError("xyz_convention must be 'ros_optical' or 'sick_sensor'")
    return xyz * 0.001
