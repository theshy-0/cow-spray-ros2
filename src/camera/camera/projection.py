"""SICK radial-depth pixels to camera-frame XYZ coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RadialProjection:
    """Project selected radial-depth pixels without building a full point cloud."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    z_offset_mm: float = 0.0

    def project(
        self,
        depth_mm: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> np.ndarray:
        depth = np.asarray(depth_mm, dtype=np.float32)
        rows = np.asarray(rows, dtype=np.intp).ravel()
        cols = np.asarray(cols, dtype=np.intp).ravel()
        if depth.shape != (self.height, self.width):
            raise ValueError(
                f'depth shape {depth.shape} does not match projection '
                f'{(self.height, self.width)}'
            )
        if rows.shape != cols.shape:
            raise ValueError('rows and cols must have the same shape')
        if rows.size and (
            rows.min() < 0
            or rows.max() >= self.height
            or cols.min() < 0
            or cols.max() >= self.width
        ):
            raise IndexError('pixel indices are outside the image')
        ray_x, ray_y, ray_z = self._rays(rows, cols)
        center_x, center_y = self._center_offsets(depth)
        selected_depth = depth[rows, cols]
        points = np.column_stack(
            (
                selected_depth * ray_x - center_x,
                selected_depth * ray_y - center_y,
                selected_depth * ray_z + np.float32(self.z_offset_mm),
            )
        ).astype(np.float32, copy=False)
        invalid = ~np.isfinite(selected_depth) | (selected_depth <= 0.0)
        points[invalid] = np.nan
        return points

    def _rays(self, rows, cols) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = cols.astype(np.float32) - np.float32(self.cx)
        x /= np.float32(self.fx)
        y = rows.astype(np.float32) - np.float32(self.cy)
        y /= np.float32(self.fy)
        radius2 = x * x + y * y
        distortion = (
            np.float32(1.0)
            + np.float32(self.k1) * radius2
            + np.float32(self.k2) * radius2 * radius2
        )
        x *= distortion
        y *= distortion
        inverse_norm = np.float32(1.0) / np.sqrt(
            x * x + y * y + np.float32(1.0)
        )
        return x * inverse_norm, y * inverse_norm, inverse_norm

    def _center_offsets(self, depth) -> tuple[np.float32, np.float32]:
        rows = np.array(
            [self.height // 2, (self.height - 1) // 2], dtype=np.intp
        )
        cols = np.array(
            [self.width // 2, (self.width - 1) // 2], dtype=np.intp
        )
        ray_x, ray_y, _ = self._rays(rows, cols)
        center_depth = np.nan_to_num(
            depth[rows, cols], nan=0.0, posinf=0.0, neginf=0.0
        )
        return (
            np.float32(0.5)
            * np.sum(center_depth * ray_x, dtype=np.float32),
            np.float32(0.5)
            * np.sum(center_depth * ray_y, dtype=np.float32),
        )