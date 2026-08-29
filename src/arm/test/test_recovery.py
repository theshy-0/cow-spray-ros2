from arm.recovery import RecoveryManager, RecoveryState
from arm.sprayer import evaluate_spray


def test_collision_latches_and_requires_enabled_recovery():
    manager = RecoveryManager()
    assert manager.latch("collision", [0.5, 0.1, 0.3], emergency=False)
    assert manager.state is RecoveryState.FAULT_LATCHED
    ok, reason = manager.request(emergency_active=False, recovery_enabled=False)
    assert not ok
    assert reason == "RECOVERY_DISABLED"


def test_emergency_stop_never_requests_automatic_recovery():
    manager = RecoveryManager()
    manager.latch("emergency", [0.5, 0.1, 0.3], emergency=True)
    ok, reason = manager.request(emergency_active=False, recovery_enabled=True)
    assert not ok
    assert reason == "EMERGENCY_STOP_REQUIRES_MANUAL_RESET"


def test_sprayer_is_fail_closed():
    assert evaluate_spray(
        requested=True, permitted=True, in_work_state=True, faulted=False
    ).actual
    assert not evaluate_spray(
        requested=True, permitted=True, in_work_state=True, faulted=True
    ).actual
