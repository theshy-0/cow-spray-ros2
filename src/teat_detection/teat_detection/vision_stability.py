"""CSV recording and compact statistics for passive vision diagnostics."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


CSV_FIELDS = (
    "timestamp", "target_id", "source", "confidence", "u", "v",
    "du", "dv", "delta_pixel", "depth_raw", "depth_filtered",
    "delta_depth", "x_cam", "y_cam", "z_cam", "delta_cam_x",
    "delta_cam_y", "delta_cam_z", "delta_cam", "x_base", "y_base",
    "z_base", "delta_base", "measurement_age", "measurement_stamp",
    "callback_receive_time", "tf_transform_time", "processing_latency",
    "detection_index", "assignment_distance",
)


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _delta(current, previous):
    if not all(_finite(value) for value in (*current, *previous)):
        return [math.nan] * len(current), math.nan
    difference = np.asarray(current, dtype=float) - np.asarray(previous, dtype=float)
    return difference.tolist(), float(np.linalg.norm(difference))


class VisionStabilityRecorder:
    """Record one row per semantic target without touching measurement values."""

    def __init__(
        self,
        csv_path: str,
        summary_path: str,
        *,
        flush_rows: int = 32,
        jump_thresholds=None,
        warning=None,
    ) -> None:
        path = Path(csv_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=CSV_FIELDS)
        self._writer.writeheader()
        self._summary_path = Path(summary_path).expanduser()
        self._summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._flush_rows = max(1, int(flush_rows))
        self._buffer = []
        self._rows = defaultdict(list)
        self._previous = {}
        self._thresholds = dict(jump_thresholds or {})
        self._warning = warning
        self._last_warning = {}
        self._closed = False

    def record(self, values: dict) -> dict:
        row = {name: values.get(name, math.nan) for name in CSV_FIELDS}
        target = str(row["target_id"])
        previous = self._previous.get(target)
        if previous is not None:
            pixel_delta, row["delta_pixel"] = _delta(
                (row["u"], row["v"]), (previous["u"], previous["v"])
            )
            row["du"], row["dv"] = pixel_delta
            camera_delta, row["delta_cam"] = _delta(
                (row["x_cam"], row["y_cam"], row["z_cam"]),
                (previous["x_cam"], previous["y_cam"], previous["z_cam"]),
            )
            row["delta_cam_x"], row["delta_cam_y"], row["delta_cam_z"] = camera_delta
            _, row["delta_base"] = _delta(
                (row["x_base"], row["y_base"], row["z_base"]),
                (previous["x_base"], previous["y_base"], previous["z_base"]),
            )
            if _finite(row["depth_filtered"]) and _finite(previous["depth_filtered"]):
                row["delta_depth"] = float(row["depth_filtered"]) - float(
                    previous["depth_filtered"]
                )
        if any(
            _finite(row[name])
            for name in ("u", "v", "x_cam", "y_cam", "z_cam")
        ):
            self._previous[target] = row.copy()
        self._rows[target].append(row.copy())
        self._buffer.append(row)
        if len(self._buffer) >= self._flush_rows:
            self.flush()
        self._warn_jump(row)
        return row

    def _warn_jump(self, row) -> None:
        exceeded = (
            (_finite(row["delta_pixel"]) and row["delta_pixel"] > self._thresholds.get("pixel", math.inf))
            or (_finite(row["delta_depth"]) and abs(row["delta_depth"]) > self._thresholds.get("depth", math.inf))
            or (_finite(row["delta_cam"]) and row["delta_cam"] > self._thresholds.get("cam", math.inf))
            or (_finite(row["delta_base"]) and row["delta_base"] > self._thresholds.get("base", math.inf))
        )
        now = time.monotonic()
        target = str(row["target_id"])
        if not exceeded or now - self._last_warning.get(target, -math.inf) < 1.0:
            return
        self._last_warning[target] = now
        if self._warning is not None:
            self._warning(
                "[VISION_JUMP] "
                f"target={target} source={row['source']} "
                f"delta_cam={row['delta_cam']:.4f}m "
                f"delta_base={row['delta_base']:.4f}m "
                f"pixel_jump={row['delta_pixel']:.2f}px "
                f"depth_jump={row['delta_depth']:.4f}m"
            )

    def flush(self) -> None:
        if not self._buffer:
            return
        self._writer.writerows(self._buffer)
        self._stream.flush()
        self._buffer.clear()

    @staticmethod
    def _stat(rows, field, operation="std"):
        values = np.asarray(
            [float(row[field]) for row in rows if _finite(row[field])], dtype=float
        )
        if values.size == 0:
            return math.nan
        if operation == "mean":
            return float(np.mean(values))
        if operation == "max":
            return float(np.max(values))
        if operation == "p95":
            return float(np.percentile(values, 95.0))
        return float(np.std(values))

    def summary(self) -> dict:
        result = {}
        for target, rows in sorted(self._rows.items()):
            item = {
                "sample_count": len(rows),
                "measured_count": sum(row["source"] == "MEASURED" for row in rows),
                "predicted_count": sum(row["source"] == "PREDICTED" for row in rows),
                "lost_count": sum(row["source"] == "LOST" for row in rows),
            }
            for field in ("u", "v", "depth_filtered", "x_cam", "y_cam", "z_cam", "x_base", "y_base", "z_base"):
                item[("mean_" if field in ("u", "v", "depth_filtered") else "std_") + field] = self._stat(
                    rows, field, "mean" if field in ("u", "v", "depth_filtered") else "std"
                )
                if field in ("u", "v", "depth_filtered"):
                    item["std_" + field] = self._stat(rows, field)
            item["mean_depth"] = item["mean_depth_filtered"]
            item["std_depth"] = item["std_depth_filtered"]
            for field in ("delta_pixel", "delta_cam", "delta_base"):
                item["max_" + field] = self._stat(rows, field, "max")
            item["p95_delta_cam"] = self._stat(rows, "delta_cam", "p95")
            item["p95_delta_base"] = self._stat(rows, "delta_base", "p95")
            result[target] = item
        return result

    def conclusions(self, summary: dict) -> list[str]:
        findings = []
        pixel_limit = self._thresholds.get("pixel", 8.0)
        depth_limit = self._thresholds.get("depth", 0.03)
        cam_limit = self._thresholds.get("cam", 0.03)
        base_limit = self._thresholds.get("base", 0.03)
        for target, item in summary.items():
            pixel_std = math.hypot(item.get("std_u", math.nan), item.get("std_v", math.nan))
            depth_std = item.get("std_depth", math.nan)
            cam_std = math.sqrt(sum(item.get(f"std_{axis}_cam", 0.0) ** 2 for axis in "xyz"))
            base_std = math.sqrt(sum(item.get(f"std_{axis}_base", 0.0) ** 2 for axis in "xyz"))
            if _finite(pixel_std) and _finite(depth_std) and pixel_std > pixel_limit / 4.0 and depth_std < depth_limit / 3.0:
                findings.append(f"{target}: bbox波动突出，更可能是YOLO 2D定位抖动")
            if _finite(pixel_std) and _finite(depth_std) and pixel_std < pixel_limit / 4.0 and depth_std > depth_limit / 3.0:
                findings.append(f"{target}: bbox稳定但深度波动突出，更可能是Depth/ROI问题")
            if _finite(cam_std) and _finite(base_std) and cam_std < cam_limit / 6.0 and base_std > base_limit / 3.0:
                findings.append(f"{target}: Camera XYZ稳定但Base XYZ波动突出，更可能是TF/timestamp问题")
            if item.get("max_delta_cam", 0.0) > cam_limit and item.get("max_delta_pixel", 0.0) > pixel_limit:
                findings.append(f"{target}: 像素和3D同时发生大跳变，需重点检查ID交换")
        if not findings and summary:
            findings.append("未发现单一级别占主导的大跳变；视觉总体稳定或需结合更多指标判断")
        return findings

    def close(self) -> dict:
        if self._closed:
            return self.summary()
        self.flush()
        summary = self.summary()
        payload = {"targets": summary, "conclusions": self.conclusions(summary)}
        self._summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        self._stream.close()
        self._closed = True
        return payload
