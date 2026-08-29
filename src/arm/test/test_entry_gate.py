import math

from arm.entry_gate import AdmissionSnapshot, EntryGate


def _gate():
    return EntryGate(
        data_timeout=0.2,
        min_confidence=0.7,
        production_axis=[1.0, 0.0, 0.0],
        work_window_exit_m=1.5,
        estimated_cycle_time_s=6.0,
        cycle_reserve_s=0.7,
        minimum_line_speed_mps=0.01,
    )


def _snapshot(**changes):
    values = dict(
        now=10.1,
        measurement_stamp=10.0,
        valid=True,
        stable=True,
        corridor_clear=True,
        confidence=0.9,
        entry_center_base=(0.3, 0.0, 0.4),
        line_speed_mps=0.15,
        robot_ready=True,
        targets_locked=True,
    )
    values.update(changes)
    return AdmissionSnapshot(**values)


def test_gate_accepts_stable_geometry_with_time_margin():
    decision = _gate().evaluate(_snapshot(), require_targets=True)
    assert decision.allowed
    assert decision.available_time_s == 8.0


def test_gate_rejects_stale_measurement():
    decision = _gate().evaluate(
        _snapshot(measurement_stamp=9.0), require_targets=False
    )
    assert not decision.allowed
    assert decision.reason == "ENTRY_STALE"


def test_gate_rejects_cycle_that_cannot_finish():
    decision = _gate().evaluate(
        _snapshot(entry_center_base=(0.8, 0.0, 0.4)), require_targets=True
    )
    assert not decision.allowed
    assert decision.reason == "INSUFFICIENT_CYCLE_TIME"


def test_stationary_line_has_no_time_expiry():
    decision = _gate().evaluate(
        _snapshot(line_speed_mps=0.0), require_targets=True
    )
    assert decision.allowed
    assert math.isinf(decision.available_time_s)
