"""Protocol/API layer that turns raw SDK frames into SickFrame objects."""

import time
from typing import Optional

import numpy as np

from ..sdk.python_base.Streaming import Data

from .connection import SickCameraConnection
from .exceptions import SickCameraFrameError
from .models import CameraIntrinsics, SickFrame


class SickCameraAPI:
    """High-level frame API over a camera connection."""

    def __init__(self, connection: SickCameraConnection):
        self.connection = connection
        self._sensor_data = Data.Data()
        self._parameter_counter = None
        self._intrinsics = None
        self._camera_params = None

    def read_frame(self) -> SickFrame:
        total_started = time.perf_counter()
        started = time.perf_counter()
        raw_frame = self.connection.read_raw_frame()
        receive_ms = (time.perf_counter() - started) * 1000.0
        received_at = time.monotonic()
        started = time.perf_counter()
        try:
            self._sensor_data.read(raw_frame, convertToMM=False)
        except Exception as exc:
            raise SickCameraFrameError(f"Failed to parse camera frame: {exc}") from exc
        parse_ms = (time.perf_counter() - started) * 1000.0

        if not self._sensor_data.hasDepthMap:
            raise SickCameraFrameError("Frame does not contain a depth map")

        started = time.perf_counter()
        frame = self._build_frame(self._sensor_data, received_at)
        build_ms = (time.perf_counter() - started) * 1000.0
        frame.acquisition_ms = (time.perf_counter() - total_started) * 1000.0
        frame.camera_timings_ms = {
            "camera_receive": receive_ms,
            "camera_parse": parse_ms,
            "camera_build": build_ms,
        }
        return frame

    def _build_frame(self, sensor_data, received_at: float | None = None) -> SickFrame:
        height = int(sensor_data.cameraParams.height)
        width = int(sensor_data.cameraParams.width)
        shape = (height, width)

        distance = np.asarray(sensor_data.depthmap.distance, dtype=np.uint16).reshape(shape)
        intensity = np.asarray(sensor_data.depthmap.intensity, dtype=np.uint16).reshape(shape)
        confidence = np.asarray(sensor_data.depthmap.confidence, dtype=np.uint16).reshape(shape)

        if self._parameter_counter != sensor_data.changedCounter:
            self._parameter_counter = sensor_data.changedCounter
            self._camera_params = sensor_data.cameraParams
            self._intrinsics = CameraIntrinsics(
                fx=float(sensor_data.xmlParser.fx),
                fy=float(sensor_data.xmlParser.fy),
                cx=float(sensor_data.xmlParser.cx),
                cy=float(sensor_data.xmlParser.cy),
                k1=float(sensor_data.xmlParser.k1 or 0.0),
                k2=float(sensor_data.xmlParser.k2 or 0.0),
                k3=float(sensor_data.xmlParser.k3 or 0.0),
                p1=float(sensor_data.xmlParser.p1 or 0.0),
                p2=float(sensor_data.xmlParser.p2 or 0.0),
                f2rc=float(sensor_data.xmlParser.f2rc or 0.0),
            )

        timestamp = self._read_timestamp(sensor_data)
        return SickFrame(
            frame_no=int(sensor_data.depthmap.frameNumber),
            width=width,
            height=height,
            distance_u16=distance,
            intensity_u16=intensity,
            confidence_u16=confidence,
            intrinsics=self._intrinsics,
            timestamp=timestamp,
            host_received_at=time.monotonic() if received_at is None else received_at,
            raw_sensor_data=sensor_data,
            camera_params=self._camera_params,
        )

    @staticmethod
    def _read_timestamp(sensor_data) -> Optional[tuple]:
        try:
            return sensor_data.getDecodedTimestamp()
        except Exception:
            return None
