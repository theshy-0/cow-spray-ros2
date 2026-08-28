"""Connection layer wrapping the official SICK SDK objects."""

from typing import Optional

from ..sdk.python_base.Control import Control
from ..sdk.python_base.Stream import Streaming
from ..sdk.python_base.Streaming.BlobServerConfiguration import BlobClientConfig
from ..sdk.python_base.Usertypes import FrontendMode
from ..sdk.shared.python.devices_config import get_device_config

from .exceptions import SickCameraConnectionError, SickCameraFrameError
from .models import CameraConfig


class SickCameraConnection:
    """Owns the control channel, stream channel, and cleanup sequence."""

    def __init__(self, config: Optional[CameraConfig] = None):
        self.config = config or CameraConfig()
        self.ctrl: Optional[Control] = None
        self.stream: Optional[Streaming] = None
        self.streaming_settings: Optional[BlobClientConfig] = None
        self.cola_protocol: Optional[str] = None
        self.control_port: Optional[int] = None
        self.sul_version: Optional[int] = None
        self.is_open = False

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def open(self) -> None:
        if self.is_open:
            return

        try:
            self.cola_protocol, default_port, self.sul_version = get_device_config(
                self.config.device_type
            )
            self.control_port = self.config.control_port or default_port
            self.ctrl = Control(
                self.config.ip,
                self.cola_protocol,
                self.control_port,
                timeout=self.config.timeout,
                sulVersion=self.sul_version,
            )
            self.ctrl.open()
            self.ctrl.login(Control.USERLEVEL_SERVICE, "CUST_SERV")

            self.streaming_settings = BlobClientConfig(self.ctrl)
            self.stream = self._open_stream()

            self.ctrl.setFrontendMode(FrontendMode.Continuous)
            self.ctrl.logout()
            self.is_open = True

            if self.config.drop_warmup_frame:
                self.read_raw_frame()
        except Exception as exc:
            self.close()
            raise SickCameraConnectionError(f"Failed to open camera: {exc}") from exc

    def close(self) -> None:
        # If open() failed before the connection became usable, do not issue
        # another protocol login on a disconnected socket.  Closing the raw
        # objects is sufficient and avoids masking the original error with a
        # second timeout.
        if not self.is_open:
            try:
                if self.stream is not None:
                    self.stream.closeStream()
            except Exception:
                pass
            try:
                if self.ctrl is not None:
                    self.ctrl.close()
            except Exception:
                pass
            finally:
                self.ctrl = None
                self.stream = None
                self.streaming_settings = None
                self.is_open = False
            return

        try:
            if self.ctrl is not None:
                self.ctrl.login(Control.USERLEVEL_AUTH_CLIENT, "CLIENT")
            if self.stream is not None:
                self.stream.closeStream()
            if self.streaming_settings is not None:
                if self.config.transport.upper() == "UDP":
                    self.streaming_settings.setTransportProtocol(
                        self.streaming_settings.PROTOCOL_TCP
                    )
                self.streaming_settings.setBlobTcpPort(self.config.reset_tcp_port)
            if self.ctrl is not None:
                self.ctrl.logout()
                self.ctrl.close()
        finally:
            self.ctrl = None
            self.stream = None
            self.streaming_settings = None
            self.is_open = False

    def read_raw_frame(self):
        if self.stream is None:
            raise SickCameraFrameError("Camera stream is not open")

        try:
            self.stream.getFrame()
        except Exception as exc:
            raise SickCameraFrameError(f"Failed to read camera frame: {exc}") from exc

        if self.stream.frame is None:
            raise SickCameraFrameError("Camera returned an empty frame")
        return self.stream.frame

    def _open_stream(self) -> Streaming:
        if self.streaming_settings is None:
            raise SickCameraConnectionError("Streaming settings are not initialized")

        transport = self.config.transport.upper()
        if transport == "TCP":
            self.streaming_settings.setTransportProtocol(
                self.streaming_settings.PROTOCOL_TCP
            )
            self.streaming_settings.setBlobTcpPort(self.config.streaming_port)
            stream = Streaming(self.config.ip, self.config.streaming_port)
            stream.openStream()
            return stream

        if transport == "UDP":
            self.streaming_settings.setTransportProtocol(
                self.streaming_settings.PROTOCOL_UDP
            )
            self.streaming_settings.setBlobUdpReceiverPort(self.config.streaming_port)
            self.streaming_settings.setBlobUdpReceiverIP(self.config.receiver_ip)
            self.streaming_settings.setBlobUdpControlPort(self.config.streaming_port)
            self.streaming_settings.setBlobUdpMaxPacketSize(1024)
            self.streaming_settings.setBlobUdpIdleTimeBetweenPackets(10)
            self.streaming_settings.setBlobUdpHeartbeatInterval(0)
            self.streaming_settings.setBlobUdpHeaderEnabled(True)
            self.streaming_settings.setBlobUdpFecEnabled(False)
            self.streaming_settings.setBlobUdpAutoTransmit(True)
            stream = Streaming(
                self.config.ip,
                self.config.streaming_port,
                protocol="UDP",
            )
            stream.openStream((self.config.receiver_ip, self.config.streaming_port))
            return stream

        raise SickCameraConnectionError(f"Unsupported transport: {self.config.transport}")