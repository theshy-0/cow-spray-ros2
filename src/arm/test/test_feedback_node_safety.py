"""Keep the hand-eye feedback node read-only."""

import ast
from pathlib import Path


def test_feedback_node_never_calls_motion_interfaces():
    source = Path(__file__).parents[1] / "arm" / "estun_feedback_node.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert {"StartListenUdp", "StartCriDataPush", "WaitForCriData"} <= calls
    assert {
        "EnterRemoteModeViaAuto",
        "SwitchOn",
        "StartCriControl",
        "SendCommand",
    }.isdisjoint(calls)
