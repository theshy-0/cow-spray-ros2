"""Pure admission and seven-second cycle checks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class AdmissionSnapshot:
    now: float
    measurement_stamp: float
    valid: bool
    stable: bool
    corridor_clear: bool
    confidence: float
    entry_center_base: tuple[float, float, float]
    line_speed_mps: float
    robot_ready: bool
    targets_locked: bool


@dataclass(frozen=True)
class AdmissionDecision:
    allowed: bool
    reason: str
    available_time_s: float


class EntryGate:
    """Decide whether pretracking or the fixed work cycle may start."""

    def __init__(
        self,
        *,
        data_timeout: float,
        min_confidence: float,
        production_axis,
        work_window_exit_m: float,
        estimated_cycle_time_s: float,
        cycle_reserve_s: float,
        minimum_line_speed_mps: float,
    ) -> None:
        axis = np.asarray(production_axis, dtype=float)
        norm = float(np.linalg.norm(axis))
        if axis.shape != (3,) or norm <= 1e-9:
            raise ValueError("production_axis must be a non-zero 3-vector")
        self.axis = axis / norm
        self.data_timeout = float(data_timeout)
        self.min_confidence = float(min_confidence)
        self.work_window_exit_m = float(work_window_exit_m)
        self.estimated_cycle_time_s = float(estimated_cycle_time_s)
        self.cycle_reserve_s = float(cycle_reserve_s)
        self.minimum_line_speed_mps = float(minimum_line_speed_mps)

    def _available_time(self, snapshot: AdmissionSnapshot) -> float:
        speed = float(snapshot.line_speed_mps)
        if abs(speed) < self.minimum_line_speed_mps:
            return math.inf
        center = np.asarray(snapshot.entry_center_base, dtype=float)
        remaining = self.work_window_exit_m - float(center @ self.axis)
        available = remaining / speed
        return available if available >= 0.0 else -math.inf

    def evaluate(
        self, snapshot: AdmissionSnapshot, *, require_targets: bool
    ) -> AdmissionDecision:
        available = self._available_time(snapshot)
        checks = (
            (snapshot.robot_ready, "ROBOT_NOT_READY"),
            (snapshot.valid, "ENTRY_INVALID"),
            (snapshot.stable, "ENTRY_UNSTABLE"),
            (snapshot.corridor_clear, "CORRIDOR_BLOCKED"),
            (snapshot.confidence >= self.min_confidence, "LOW_CONFIDENCE"),
            (
                snapshot.measurement_stamp > 0.0
                and snapshot.now - snapshot.measurement_stamp >= -0.05
                and snapshot.now - snapshot.measurement_stamp <= self.data_timeout,
                "ENTRY_STALE",
            ),
            (not require_targets or snapshot.targets_locked, "TARGETS_NOT_LOCKED"),
            (
                math.isinf(available)
                or available
                >= self.estimated_cycle_time_s + self.cycle_reserve_s,
                "INSUFFICIENT_CYCLE_TIME",
            ),
        )
        for passed, reason in checks:
            if not passed:
                return AdmissionDecision(False, reason, available)
        return AdmissionDecision(True, "OK", available)
