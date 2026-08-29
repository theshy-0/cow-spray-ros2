"""Fault latching rules shared by the arm driver and tests."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class RecoveryState(Enum):
    IDLE = auto()
    FAULT_LATCHED = auto()
    RECOVERY_REQUESTED = auto()
    RECOVERING = auto()


@dataclass(frozen=True)
class FaultRecord:
    reason: str
    pose: tuple[float, float, float]
    emergency: bool


class RecoveryManager:
    def __init__(self) -> None:
        self.state = RecoveryState.IDLE
        self.fault: FaultRecord | None = None

    def latch(self, reason: str, pose, *, emergency: bool) -> bool:
        if self.state is not RecoveryState.IDLE:
            return False
        self.fault = FaultRecord(
            str(reason), tuple(float(value) for value in pose[:3]), bool(emergency)
        )
        self.state = RecoveryState.FAULT_LATCHED
        return True

    def request(self, *, emergency_active: bool, recovery_enabled: bool) -> tuple[bool, str]:
        if self.state is not RecoveryState.FAULT_LATCHED or self.fault is None:
            return False, "NO_LATCHED_FAULT"
        if self.fault.emergency or emergency_active:
            return False, "EMERGENCY_STOP_REQUIRES_MANUAL_RESET"
        if not recovery_enabled:
            return False, "RECOVERY_DISABLED"
        self.state = RecoveryState.RECOVERY_REQUESTED
        return True, "RECOVERY_REQUESTED"

    def clear_after_home(self) -> None:
        self.state = RecoveryState.IDLE
        self.fault = None
