import csv
import json
import math

from teat_detection.vision_stability import VisionStabilityRecorder


def _row(target, u, x):
    return {
        "timestamp": 1.0,
        "target_id": target,
        "source": "MEASURED",
        "confidence": 0.9,
        "u": u,
        "v": 20.0,
        "depth_raw": 1.0,
        "depth_filtered": 1.0,
        "x_cam": x,
        "y_cam": 0.0,
        "z_cam": 1.0,
        "x_base": x + 0.5,
        "y_base": 0.0,
        "z_base": 1.0,
    }


def test_recorder_keeps_target_histories_separate(tmp_path):
    csv_path = tmp_path / "vision.csv"
    summary_path = tmp_path / "summary.json"
    recorder = VisionStabilityRecorder(str(csv_path), str(summary_path), flush_rows=2)

    assert math.isnan(recorder.record(_row("teat_front_left", 10.0, 0.0))["delta_cam"])
    assert math.isnan(recorder.record(_row("teat_front_right", 100.0, 1.0))["delta_cam"])
    result = recorder.record(_row("teat_front_left", 13.0, 0.004))
    payload = recorder.close()

    assert result["delta_pixel"] == 3.0
    assert result["delta_cam"] == 0.004
    assert payload["targets"]["teat_front_left"]["sample_count"] == 2
    assert payload["targets"]["teat_front_right"]["sample_count"] == 1
    assert json.loads(summary_path.read_text(encoding="utf-8"))["targets"]
    with csv_path.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 3


def test_lost_row_does_not_replace_last_valid_position(tmp_path):
    recorder = VisionStabilityRecorder(
        str(tmp_path / "vision.csv"), str(tmp_path / "summary.json")
    )
    recorder.record(_row("teat_rear_left", 10.0, 0.0))
    recorder.record({"timestamp": 2.0, "target_id": "teat_rear_left", "source": "LOST"})
    result = recorder.record(_row("teat_rear_left", 12.0, 0.002))
    recorder.close()

    assert result["delta_pixel"] == 2.0
    assert result["delta_cam"] == 0.002
