"""ROS-independent ToF leg-entry geometry and temporal validation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Callable

import cv2
import numpy as np


ProjectPixels = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LegEntryObservation:
    """One frame of measured entry geometry, expressed in metres."""

    valid: bool
    stamp: float
    left_camera: np.ndarray
    right_camera: np.ndarray
    center_camera: np.ndarray
    left_pixel: tuple[int, int]
    right_pixel: tuple[int, int]
    confidence: float
    reason: str
    left_base: np.ndarray | None = None
    right_base: np.ndarray | None = None
    center_base: np.ndarray | None = None
    gap_m: float = 0.0
    stable: bool = False
    line_speed_mps: float = 0.0
    cow_track_id: int = 0


def _invalid(stamp: float, reason: str) -> LegEntryObservation:
    nan = np.full(3, np.nan, dtype=float)
    return LegEntryObservation(
        False, float(stamp), nan, nan.copy(), nan.copy(),
        (-1, -1), (-1, -1), 0.0, reason,
    )


def parse_depth_image(data: bytes, height: int, width: int, encoding: str) -> np.ndarray | None:
    """Convert a ROS depth image to float32 millimetres with NaN invalids."""
    if encoding in ("16UC1", "mono16"):
        values = np.frombuffer(data, dtype=np.uint16).astype(np.float32)
    elif encoding == "32FC1":
        # sensor_msgs convention: floating-point depth is expressed in metres.
        values = np.frombuffer(data, dtype=np.float32).copy() * 1000.0
    else:
        return None
    if values.size != int(height) * int(width):
        return None
    depth = values.reshape(int(height), int(width))
    depth[~np.isfinite(depth) | (depth <= 0.0)] = np.nan
    return depth


def _edge_point(
    depth: np.ndarray,
    projector: ProjectPixels,
    x: int,
    row_start: int,
    row_end: int,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    column = depth[row_start:row_end, x]
    valid_rows = np.flatnonzero(np.isfinite(column))
    if valid_rows.size == 0:
        return None
    row = row_start + int(np.median(valid_rows))
    point = np.asarray(
        projector(np.array([row]), np.array([x]))[0], dtype=float
    )
    if point.shape != (3,) or not np.isfinite(point).all() or point[2] <= 0.0:
        return None
    return point * 0.001, (int(x), int(row))


def detect_leg_entry(
    depth_mm: np.ndarray,
    projector: ProjectPixels,
    *,
    stamp: float,
    min_depth_mm: float,
    max_depth_mm: float,
    row_start_ratio: float,
    row_end_ratio: float,
    min_separation_px: int,
    min_height_px: int,
    min_aspect_ratio: float,
    min_blob_area_px: int,
) -> LegEntryObservation:
    """Measure two independent leg blobs and their inner 3-D edges.

    A single merged blob is deliberately rejected: it may be one leg, tail,
    udder, tool, or background and must never grant motion admission.
    """
    depth = np.asarray(depth_mm, dtype=np.float32)
    if depth.ndim != 2 or depth.size == 0:
        return _invalid(stamp, "INVALID_DEPTH")
    height, width = depth.shape
    row_start = max(0, min(height - 1, int(height * row_start_ratio)))
    row_end = max(row_start + 1, min(height, int(height * row_end_ratio)))
    foreground = (
        np.isfinite(depth)
        & (depth >= float(min_depth_mm))
        & (depth <= float(max_depth_mm))
    )
    zone_values = depth[row_start:row_end][foreground[row_start:row_end]]
    if zone_values.size < max(20, int(min_blob_area_px) * 2):
        return _invalid(stamp, "NO_FOREGROUND")

    # Keep the nearer portion of the valid range. This suppresses floor and
    # background while leaving the physical thresholds configurable.
    near_limit = float(np.percentile(zone_values, 55.0))
    mask = foreground & (depth <= near_limit)
    mask[:row_start] = False
    mask[row_end:] = False
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    candidates: list[dict] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        x = int(stats[index, cv2.CC_STAT_LEFT])
        y = int(stats[index, cv2.CC_STAT_TOP])
        blob_width = int(stats[index, cv2.CC_STAT_WIDTH])
        blob_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        if area < int(min_blob_area_px) or blob_height < int(min_height_px):
            continue
        if blob_height / max(1, blob_width) < float(min_aspect_ratio):
            continue
        candidates.append(
            {"area": area, "x0": x, "x1": x + blob_width - 1,
             "y0": y, "y1": y + blob_height - 1}
        )

    # Admission requires exactly two independent leg-shaped components.
    if len(candidates) != 2:
        return _invalid(stamp, f"EXPECTED_TWO_LEGS_GOT_{len(candidates)}")
    candidates.sort(key=lambda item: item["x0"])
    left_blob, right_blob = candidates
    separation = right_blob["x0"] - left_blob["x1"]
    if separation < int(min_separation_px):
        return _invalid(stamp, "LEG_PIXEL_GAP_TOO_SMALL")

    left = _edge_point(
        depth, projector, left_blob["x1"], row_start, row_end
    )
    right = _edge_point(
        depth, projector, right_blob["x0"], row_start, row_end
    )
    if left is None or right is None:
        return _invalid(stamp, "INVALID_INNER_EDGE_DEPTH")
    left_camera, left_pixel = left
    right_camera, right_pixel = right
    center_camera = 0.5 * (left_camera + right_camera)
    confidence = min(1.0, (left_blob["area"] + right_blob["area"]) / max(1.0, binary.sum()))
    return LegEntryObservation(
        True,
        float(stamp),
        left_camera,
        right_camera,
        center_camera,
        left_pixel,
        right_pixel,
        float(confidence),
        "OK",
    )


def transform_observation(
    observation: LegEntryObservation,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> LegEntryObservation:
    """Attach base-frame coordinates using the transform at measurement time."""
    if not observation.valid:
        return observation
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("transform must contain a 3x3 rotation and 3-vector")
    left = rotation @ observation.left_camera + translation
    right = rotation @ observation.right_camera + translation
    center = 0.5 * (left + right)
    return replace(
        observation,
        left_base=left,
        right_base=right,
        center_base=center,
        gap_m=float(np.linalg.norm(right - left)),
    )


class LegEntryTracker:
    """Validate several measured frames and estimate signed line speed."""

    def __init__(
        self,
        window_frames: int,
        min_valid_frames: int,
        stable_duration: float,
        max_center_spread_m: float,
        max_gap_spread_m: float,
        production_axis: np.ndarray,
    ) -> None:
        if not 1 <= min_valid_frames <= window_frames:
            raise ValueError("valid frame count must be within the window")
        axis = np.asarray(production_axis, dtype=float)
        norm = float(np.linalg.norm(axis))
        if axis.shape != (3,) or norm <= 1e-9:
            raise ValueError("production_axis must be a non-zero 3-vector")
        self.frames = deque(maxlen=int(window_frames))
        self.min_valid_frames = int(min_valid_frames)
        self.stable_duration = float(stable_duration)
        self.max_center_spread_m = float(max_center_spread_m)
        self.max_gap_spread_m = float(max_gap_spread_m)
        self.axis = axis / norm
        self.cow_track_id = 0
        self._was_stable = False

    def reset(self) -> None:
        self.frames.clear()
        self._was_stable = False

    def update(self, observation: LegEntryObservation) -> LegEntryObservation:
        self.frames.append(observation)
        if not observation.valid or observation.center_base is None:
            self._was_stable = False
            return replace(observation, stable=False, cow_track_id=0)
        valid = [
            frame for frame in self.frames
            if frame.valid and frame.center_base is not None
            and np.isfinite(frame.center_base).all()
        ]
        if len(valid) < self.min_valid_frames:
            self._was_stable = False
            return replace(observation, stable=False, reason="UNSTABLE_FRAME_COUNT")
        recent = valid[-self.min_valid_frames:]
        duration = recent[-1].stamp - recent[0].stamp
        centers = np.asarray([frame.center_base for frame in recent])
        gaps = np.asarray([frame.gap_m for frame in recent])
        center_median = np.median(centers, axis=0)
        center_spread = float(np.max(np.linalg.norm(centers - center_median, axis=1)))
        gap_spread = float(np.ptp(gaps))
        stable = (
            duration + 1e-9 >= self.stable_duration
            and center_spread <= self.max_center_spread_m
            and gap_spread <= self.max_gap_spread_m
        )
        speed = 0.0
        if duration > 1e-6:
            times = np.asarray([frame.stamp for frame in recent], dtype=float)
            positions = centers @ self.axis
            speed = float(np.polyfit(times - times[0], positions, 1)[0])
        if stable and not self._was_stable:
            self.cow_track_id += 1
        self._was_stable = stable
        reason = "OK" if stable else "GEOMETRY_UNSTABLE"
        return replace(
            observation,
            stable=stable,
            line_speed_mps=speed,
            cow_track_id=self.cow_track_id if stable else 0,
            reason=reason,
        )
