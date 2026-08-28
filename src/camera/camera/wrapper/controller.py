"""Controller layer intended for application code."""

import threading
import time
from typing import Optional

from .api import SickCameraAPI
from .connection import SickCameraConnection
from .exceptions import SickCameraFrameError
from .models import CameraConfig, CameraPose, SickFrame


class SickCameraClient:
    """Convenient application-facing camera client."""

    def __init__(
        self,
        ip: str = "192.168.101.30",
        config: Optional[CameraConfig] = None,
        pose: Optional[CameraPose] = None,
        **overrides,
    ):
        if config is None:
            config = CameraConfig(ip=ip, **overrides)
        self.config = config
        self.pose = pose or CameraPose()
        self.connection = SickCameraConnection(self.config)
        self.api = SickCameraAPI(self.connection)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def open(self) -> None:
        self.connection.open()

    def close(self) -> None:
        self.connection.close()

    def read_frame(self) -> SickFrame:
        return self.api.read_frame()


class LatestFrameCamera:
    """Read SICK frames on one worker and expose only the newest frame."""

    def __init__(self, config: Optional[CameraConfig] = None, reconnect_attempts: int = 2):
        self.config = config or CameraConfig()
        self.reconnect_attempts = max(0, int(reconnect_attempts))
        self.client = SickCameraClient(config=self.config)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread = None
        self._latest = None
        self._error = None
        self._sequence = 0
        self._delivered = 0

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def open(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.client.open()
        self._stop.clear()
        self._latest = self._error = None
        self._sequence = self._delivered = 0
        self._thread = threading.Thread(target=self._read_loop, name="sick-frame-reader", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread:
            thread.join(timeout=self.config.timeout + 1.0)
        self.client.close()
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        self._thread = None

    def read_frame(self, timeout: Optional[float] = None) -> SickFrame:
        deadline = time.monotonic() + (self.config.timeout if timeout is None else timeout)
        with self._condition:
            while self._sequence == self._delivered and self._error is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise SickCameraFrameError("Timed out waiting for a new camera frame")
                self._condition.wait(remaining)
            if self._sequence != self._delivered:
                self._delivered = self._sequence
                return self._latest
            raise SickCameraFrameError(f"Camera reader stopped: {self._error}") from self._error

    def _read_loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.client.read_frame()
                with self._condition:
                    self._latest = frame
                    self._sequence += 1
                    self._condition.notify_all()
            except Exception as exc:
                if self._stop.is_set():
                    return
                recovered = False
                for _ in range(self.reconnect_attempts):
                    try:
                        self.client.close()
                        if self._stop.wait(0.2):
                            return
                        self.client.open()
                        recovered = True
                        break
                    except Exception as reconnect_error:
                        exc = reconnect_error
                if recovered:
                    continue
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
                return
