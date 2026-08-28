"""Data models used by the SICK camera wrapper."""

from dataclasses import dataclass, field
from functools import lru_cache
import time
from typing import Optional, Tuple

import numpy as np

from ..projection import RadialProjection

@dataclass(frozen=True)
class CameraConfig:
    """Connection and stream settings for a SICK Visionary camera."""

    ip: str = "192.168.101.30"
    device_type: str = "Visionary-T Mini"
    streaming_port: int = 2114
    control_port: Optional[int] = None
    transport: str = "TCP"
    receiver_ip: str = "192.168.101.10"
    timeout: float = 5.0
    drop_warmup_frame: bool = True
    reset_tcp_port: int = 2114


@dataclass(frozen=True)
class CameraIntrinsics:
    """Camera intrinsics parsed from the SICK frame XML.

    The distortion fields and ``f2rc`` are required for an accurate
    Visionary-T Mini range-to-point conversion.  Defaults keep the class
    backwards compatible with callers that only provide pinhole parameters.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    k1: float = 0.0
    k2: float = 0.0
    k3: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    f2rc: float = 0.0


@dataclass(frozen=True)
class CameraPose:
    """Camera pose in the user-defined world coordinate system."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0


@dataclass
class SickFrame:
    """Parsed frame data with consistent array shapes and units."""

    frame_no: int
    width: int
    height: int
    distance_u16: np.ndarray
    intensity_u16: np.ndarray
    confidence_u16: np.ndarray
    intrinsics: CameraIntrinsics
    depth_scale: float = 4.0
    timestamp: Optional[Tuple[int, int, int, int, int, int, int]] = None
    # 完整帧到达本机的 monotonic 时间。它比推理完成时间更接近采集时刻，
    # 且不会受系统时钟校时影响，供控制链路计算数据年龄。
    host_received_at: float = 0.0
    raw_sensor_data: object = None
    camera_params: object = None
    acquisition_ms: float = 0.0
    camera_timings_ms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.host_received_at:
            self.host_received_at = time.monotonic()

    @property
    def shape(self) -> Tuple[int, int]:
        return self.height, self.width

    @property
    def depth_mm(self) -> np.ndarray:
        return self.distance_u16.astype(np.float32) / float(self.depth_scale)

    def intensity_gray(self, vmax: int = 10000) -> np.ndarray:
        return _u16_to_gray(self.intensity_u16, vmax)

    def depth_gray(self, vmax_mm: int = 4000) -> np.ndarray:
        return _u16_to_gray(self.depth_mm, vmax_mm)

    def confidence_gray(self, vmax: int = 65535) -> np.ndarray:
        return _u16_to_gray(self.confidence_u16, vmax)

    def intensity_bgr(self, vmax: int = 10000) -> np.ndarray:
        import cv2

        return cv2.cvtColor(self.intensity_gray(vmax), cv2.COLOR_GRAY2BGR)

    def depth_at(self, u: int, v: int) -> float:
        u = int(np.clip(u, 0, self.width - 1))
        v = int(np.clip(v, 0, self.height - 1))
        return float(self.distance_u16[v, u]) / float(self.depth_scale)

    def get_point_cloud(self) -> np.ndarray:
        """Return the organized XYZ point cloud in millimetres."""
        rays, z_offset, center_x, center_y = self._projection_parameters()
        ray_x, ray_y, ray_z = rays
        depth = self.distance_u16.astype(np.float32) / np.float32(self.depth_scale)
        cloud = np.empty((self.height, self.width, 3), dtype=np.float32)
        cloud[..., 0] = depth * ray_x - center_x
        cloud[..., 1] = depth * ray_y - center_y
        cloud[..., 2] = depth * ray_z + z_offset
        cloud[self.confidence_u16 != 0] = 0.0
        return cloud

    def points_at(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Project selected image pixels instead of constructing a full cloud."""
        points = self.projection.project(self.depth_mm, rows, cols)
        rows = np.asarray(rows, dtype=np.intp).ravel()
        cols = np.asarray(cols, dtype=np.intp).ravel()
        points[self.confidence_u16[rows, cols] != 0] = 0.0
        return points

    @property
    def projection(self) -> RadialProjection:
        if self.camera_params is None:
            raise ValueError("camera_params is required for point projection")
        return RadialProjection(
            width=self.width,
            height=self.height,
            fx=self.intrinsics.fx,
            fy=self.intrinsics.fy,
            cx=self.intrinsics.cx,
            cy=self.intrinsics.cy,
            k1=self.intrinsics.k1,
            k2=self.intrinsics.k2,
            z_offset_mm=(
                float(self.camera_params.cam2worldMatrix[11])
                - self.intrinsics.f2rc
            ),
        )

    def _projection_parameters(self):
        if self.camera_params is None:
            raise ValueError("camera_params is required for point cloud computation")
        rays = _mono_rays(
            self.width, self.height,
            self.intrinsics.cx, self.intrinsics.cy,
            self.intrinsics.fx, self.intrinsics.fy,
            self.intrinsics.k1, self.intrinsics.k2,
        )
        row_a, col_a = self.height // 2, self.width // 2
        row_b, col_b = (self.height - 1) // 2, (self.width - 1) // 2
        depth_a = float(self.distance_u16[row_a, col_a]) / self.depth_scale
        depth_b = float(self.distance_u16[row_b, col_b]) / self.depth_scale
        center_x = 0.5 * (depth_a * rays[0][row_a, col_a] + depth_b * rays[0][row_b, col_b])
        center_y = 0.5 * (depth_a * rays[1][row_a, col_a] + depth_b * rays[1][row_b, col_b])
        z_offset = float(self.camera_params.cam2worldMatrix[11]) - self.intrinsics.f2rc
        return rays, z_offset, center_x, center_y


@lru_cache(maxsize=16)
def _mono_rays(width, height, cx, cy, fx, fy, k1, k2):
    """Cache the depth-to-XYZ ray grid for one camera calibration."""
    cols = (np.arange(width, dtype=np.float32) - np.float32(cx)) / np.float32(fx)
    rows = (np.arange(height, dtype=np.float32) - np.float32(cy)) / np.float32(fy)
    x, y = np.meshgrid(cols, rows)
    radius2 = x * x + y * y
    distortion = 1.0 + k1 * radius2 + k2 * radius2 * radius2
    x *= distortion
    y *= distortion
    inverse_norm = 1.0 / np.sqrt(x * x + y * y + 1.0)
    rays = x * inverse_norm, y * inverse_norm, inverse_norm
    for ray in rays:
        ray.setflags(write=False)
    return rays


def _u16_to_gray(values: np.ndarray, vmax: int) -> np.ndarray:
    vmax = max(int(vmax), 1)
    clipped = np.clip(values, 0, vmax)
    return (clipped.astype(np.float32) / float(vmax) * 255.0).astype(np.uint8)
