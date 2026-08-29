"""Fail-closed spray permission without hardware-specific I/O."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SprayState:
    requested: bool
    permitted: bool
    actual: bool
    reason: str


def evaluate_spray(
    *, requested: bool, permitted: bool, in_work_state: bool, faulted: bool
) -> SprayState:
    checks = (
        (not faulted, "FAULT"),
        (in_work_state, "NOT_IN_WORK_STATE"),
        (permitted, "TARGET_NOT_READY"),
        (requested, "NOT_REQUESTED"),
    )
    for passed, reason in checks:
        if not passed:
            return SprayState(bool(requested), bool(permitted), False, reason)
    return SprayState(True, True, True, "OK")
