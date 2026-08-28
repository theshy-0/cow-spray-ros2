"""Exception types for the project-level SICK camera wrapper."""


class SickCameraError(Exception):
    """Base exception for camera wrapper failures."""


class SickCameraConnectionError(SickCameraError):
    """Raised when the camera connection or stream setup fails."""


class SickCameraFrameError(SickCameraError):
    """Raised when a frame cannot be read or parsed."""