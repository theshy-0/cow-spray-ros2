"""ROS-independent four-teat identity tracking and udder-frame estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Mapping, Sequence

import numpy as np


TEAT_NAMES = ("front_left", "front_right", "rear_left", "rear_right")


def _vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result


def _normalize(value: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-9:
        raise ValueError(f"{name} is degenerate")
    return value / norm


@dataclass(frozen=True)
class TeatDetection:
    """One camera detection with a pixel centre and a 3-D camera-frame point."""

    pixel_uv: Sequence[float]
    position: Sequence[float]
    score: float
    source_index: int = -1

    def validated(self) -> "TeatDetection":
        pixel = _vector(self.pixel_uv, 2, "pixel_uv")
        position = _vector(self.position, 3, "position")
        score = float(self.score)
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be between zero and one")
        if position[2] <= 0.0:
            raise ValueError("teat depth must be positive")
        return TeatDetection(pixel, position, score, int(self.source_index))


@dataclass(frozen=True)
class UdderObservation:
    """A named four-teat observation and its fitted moving task frame."""

    stamp: float
    origin: np.ndarray
    rotation: np.ndarray
    teats: Mapping[str, np.ndarray]
    residual: float
    source_indices: Mapping[str, int] = field(default_factory=dict)
    observed_names: tuple[str, ...] = ()
    predicted_names: tuple[str, ...] = ()
    stale: bool = False


def build_semantic_frame(
    teats: Mapping[str, Sequence[float]], expected_z: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Build X=left-to-right, Y=rear-to-front, and constrained Z axes.

    支持 3 点降级：允许缺失一个乳头，用可用点估计坐标轴；
    若某方向（左右/前后）缺一侧则用 SVD 主方向补全。
    """
    names = [name for name in TEAT_NAMES if name in teats]
    if len(names) < 3:
        raise ValueError(f"at least three teats are required, got {sorted(names)}")
    p = {name: _vector(teats[name], 3, name) for name in names}
    origin = np.mean([p[name] for name in names], axis=0)

    left = [p[n] for n in names if n.endswith("left")]
    right = [p[n] for n in names if n.endswith("right")]
    if left and right:
        x_axis = _normalize(np.mean(right, axis=0) - np.mean(left, axis=0),
                            "udder lateral axis")
    else:
        centered = np.vstack([p[n] - origin for n in names])
        _, _, vh = np.linalg.svd(centered)
        x_axis = _normalize(vh[0], "udder lateral axis")

    front = [p[n] for n in names if n.startswith("front")]
    rear = [p[n] for n in names if n.startswith("rear")]
    if front and rear:
        y_raw = np.mean(front, axis=0) - np.mean(rear, axis=0)
    else:
        centered = np.vstack([p[n] - origin for n in names])
        _, _, vh = np.linalg.svd(centered)
        candidate = (vh[0] if abs(np.dot(vh[0], x_axis)) <= abs(np.dot(vh[1], x_axis))
                     else vh[1])
        y_raw = candidate
    y_axis = _normalize(y_raw - np.dot(y_raw, x_axis) * x_axis, "udder long axis")
    z_axis = _normalize(np.cross(x_axis, y_axis), "udder normal")
    expected = _normalize(_vector(expected_z, 3, "expected_z"), "expected_z")
    if float(np.dot(z_axis, expected)) < 0.0:
        z_axis = -z_axis
        y_axis = -y_axis
    return origin, np.column_stack((x_axis, y_axis, z_axis))


def _rigid_fit(
    reference: np.ndarray, measured: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted Kabsch fit: measured ~= rotation @ reference + translation."""
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("detection weights must be positive")
    ref_center = np.sum(reference * weights[:, None], axis=0) / total
    measured_center = np.sum(measured * weights[:, None], axis=0) / total
    ref_zero = reference - ref_center
    measured_zero = measured - measured_center
    covariance = (ref_zero * weights[:, None]).T @ measured_zero
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = measured_center - rotation @ ref_center
    fitted = (rotation @ reference.T).T + translation
    residual = float(np.sqrt(np.sum(weights * np.sum((fitted - measured) ** 2, axis=1)) / total))
    return translation, rotation, residual


def _minimal_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the shortest 3-D rotation that maps source onto target."""
    a = _normalize(source, "previous teat pair")
    b = _normalize(target, "current teat pair")
    cross = np.cross(a, b)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if sine < 1e-9:
        if cosine > 0.0:
            return np.eye(3)
        basis = np.eye(3)[int(np.argmin(np.abs(a)))]
        axis = _normalize(np.cross(a, basis), "opposite-pair rotation axis")
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    kx, ky, kz = cross
    skew = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / (sine**2))


class TeatTracker:
    """Assign stable teat IDs once, then track the four-point constellation."""

    def __init__(
        self,
        *,
        expected_z: Sequence[float] = (0.0, 0.0, 1.0),
        front_is_smaller_v: bool = True,
        left_is_smaller_u: bool = True,
        min_score: float = 0.35,
        max_match_distance: float = 0.08,
        max_fit_residual: float = 0.02,
        two_point_timeout: float = 0.30,
        max_prediction_translation: float = 0.05,
        max_prediction_rotation_deg: float = 15.0,
        partial_prediction_timeout: float = 0.50,
        zero_point_timeout: float = 0.20,
        reacquire_after_failures: int = 5,
        reacquire_stable_frames: int = 5,
    ) -> None:
        self.expected_z = _vector(expected_z, 3, "expected_z")
        self.front_is_smaller_v = bool(front_is_smaller_v)
        self.left_is_smaller_u = bool(left_is_smaller_u)
        self.min_score = float(min_score)
        self.max_match_distance = float(max_match_distance)
        self.max_fit_residual = float(max_fit_residual)
        self.two_point_timeout = float(two_point_timeout)
        self.max_prediction_translation = float(max_prediction_translation)
        self.max_prediction_rotation = np.deg2rad(float(max_prediction_rotation_deg))
        self.partial_prediction_timeout = float(partial_prediction_timeout)
        self.zero_point_timeout = float(zero_point_timeout)
        self.reacquire_after_failures = int(reacquire_after_failures)
        self.reacquire_stable_frames = int(reacquire_stable_frames)
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be between zero and one")
        if self.max_match_distance <= 0.0 or self.max_fit_residual <= 0.0:
            raise ValueError("tracking limits must be positive")
        if (
            self.two_point_timeout <= 0.0
            or self.max_prediction_translation <= 0.0
            or self.max_prediction_rotation <= 0.0
        ):
            raise ValueError("two-point prediction limits must be positive")
        if self.reacquire_after_failures < 1 or self.reacquire_stable_frames < 1:
            raise ValueError("reacquisition limits must be positive")
        self.reset()

    def reset(self) -> None:
        self._last_stamp: float | None = None
        self._last_positions: dict[str, np.ndarray] | None = None
        self._reference_local_dict: dict[str, np.ndarray] | None = None
        self._tracked_names: list[str] | None = None
        self._last_origin: np.ndarray | None = None
        self._last_rotation: np.ndarray | None = None
        self._last_full_stamp: float | None = None
        self._last_real_stamp: float | None = None
        self._last_four_teats: dict[str, np.ndarray] | None = None
        self.initialized = False
        self._consecutive_failures = 0
        self._stable_frames = 0
        self._reacquiring = False

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def stable_frames(self) -> int:
        return self._stable_frames

    @property
    def reacquiring(self) -> bool:
        return self._reacquiring

    def update(
        self, detections: Sequence[TeatDetection], stamp: float
    ) -> UdderObservation:
        stamp = float(stamp)
        if not np.isfinite(stamp):
            raise ValueError("stamp must be finite")
        if self._last_stamp is not None and stamp <= self._last_stamp:
            raise ValueError("detection timestamp must increase")
        try:
            observation = self._update_once(detections, stamp)
        except ValueError:
            self._consecutive_failures += 1
            self._stable_frames = 0
            if self._consecutive_failures >= self.reacquire_after_failures:
                self.reset()
                self._reacquiring = True
            raise

        self._consecutive_failures = 0
        if self._reacquiring:
            self._stable_frames += 1
            if self._stable_frames < self.reacquire_stable_frames:
                raise ValueError(
                    f"reacquiring: {self._stable_frames}/"
                    f"{self.reacquire_stable_frames} stable frames"
                )
            self._reacquiring = False
        return observation

    def update_partial(
        self, detections: Sequence[TeatDetection], stamp: float
    ) -> UdderObservation | None:
        """统一入口：0~4 个观测点 → 固定四点 observation。

        核心语义：
          - 初始化前（模板未建立）：>=3 点初始化，否则返回 None（无输出依据）
          - 初始化后：每一帧必须输出四点；任何失败（匹配失败/超时/reacquire）
            都走历史/几何传播，绝不把 tracked 清空。

        模型统一为 p_i = R @ template_i + t。
        返回 None 仅表示"尚未建立四点模板"。
        """
        stamp = float(stamp)
        if not np.isfinite(stamp):
            raise ValueError("stamp must be finite")
        if self._last_stamp is not None and stamp <= self._last_stamp:
            raise ValueError("detection timestamp must increase")
        valid = [
            detection.validated()
            for detection in detections
            if float(detection.score) >= self.min_score
        ]
        n = len(valid)

        # ---------- 初始化前：必须 >=3 点建立模板，否则不输出 ----------
        if not self.initialized:
            if n < 3:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.reacquire_after_failures:
                    self._reacquiring = True
                self._last_stamp = stamp
                return None
            try:
                observation = self._update_once(valid, stamp)
            except ValueError:
                self._consecutive_failures += 1
                self._last_stamp = stamp
                return None
            self.initialized = True
            self._reacquiring = False
            self._consecutive_failures = 0
            self._stable_frames = 0
            self._last_four_teats = {
                name: value.copy() for name, value in observation.teats.items()
            }
            return observation

        # ---------- 初始化后：每一帧必须输出四点 ----------
        try:
            if n == 0:
                observation = self._predict_zero(stamp)
            elif n == 1:
                observation = self._predict_one(valid, stamp)
            elif n == 2:
                observation = self.predict_two(valid, stamp)
            else:
                observation = self._update_once(valid, stamp)
        except ValueError:
            # 软失败：拒绝异常检测 / 超时，不破坏输出连续性 → 传播历史四点
            self._consecutive_failures += 1
            self._stable_frames = 0
            if self._consecutive_failures >= self.reacquire_after_failures:
                self._reacquiring = True
            self._last_stamp = stamp
            return self._propagate_last(stamp)

        # 成功：更新状态
        self._consecutive_failures = 0
        if self._reacquiring:
            self._stable_frames += 1
            if self._stable_frames >= self.reacquire_stable_frames:
                self._reacquiring = False
                self._stable_frames = 0
        self._last_four_teats = {
            name: value.copy() for name, value in observation.teats.items()
        }
        return observation

    def _propagate_last(self, stamp: float) -> UdderObservation:
        """失败时传播最后已知四点（历史/几何传播），保持输出连续性。

        stale=True 表示不是本帧真实观测，status.valid 应为 False。
        不刷新 _last_real_stamp（预测不能追溯为真实测量）。
        """
        if self._last_four_teats is None:
            raise ValueError("no four-teat history to propagate")
        origin = (
            self._last_origin.copy()
            if self._last_origin is not None
            else np.zeros(3)
        )
        rotation = (
            self._last_rotation.copy()
            if self._last_rotation is not None
            else np.eye(3)
        )
        positions = {
            name: value.copy() for name, value in self._last_four_teats.items()
        }
        return UdderObservation(
            stamp, origin, rotation, positions, -1.0, {},
            (), tuple(TEAT_NAMES), stale=True,
        )

    def _predict_zero(self, stamp: float) -> UdderObservation:
        """0 点：无新视觉约束。zero_point_timeout 内传播上一帧姿态，超时失败。

        不刷新 _last_real_stamp（预测不能追溯为真实测量）。
        """
        if (
            self._reference_local_dict is None
            or set(self._reference_local_dict) != set(TEAT_NAMES)
            or self._last_origin is None
            or self._last_rotation is None
            or self._last_real_stamp is None
        ):
            raise ValueError("four-teat template has not been established")
        if stamp - self._last_real_stamp > self.zero_point_timeout:
            raise ValueError("zero-point prediction timeout")
        origin = self._last_origin.copy()
        rotation = self._last_rotation.copy()
        positions = {
            name: origin + rotation @ self._reference_local_dict[name]
            for name in TEAT_NAMES
        }
        self._last_stamp = stamp
        return UdderObservation(
            stamp, origin, rotation, positions, 0.0, {},
            (), tuple(TEAT_NAMES),
        )

    def _predict_one(
        self, detections: Sequence[TeatDetection], stamp: float
    ) -> UdderObservation:
        """1 点：保持上一帧旋转，用 1 个实测锚点求平移，恢复全部 4 点。

        实测点输出真实测量，其余 3 点 predicted。不更新模板（_reference_local_dict）。
        """
        if (
            self._reference_local_dict is None
            or set(self._reference_local_dict) != set(TEAT_NAMES)
            or self._last_rotation is None
            or self._last_real_stamp is None
        ):
            raise ValueError("four-teat template has not been established")
        if stamp - self._last_real_stamp > self.partial_prediction_timeout:
            raise ValueError("one-point prediction timeout")
        named = self._match_previous(detections)
        name = next(iter(named))
        detection = named[name]
        rotation = self._last_rotation.copy()
        translation = (
            np.asarray(detection.position, dtype=float)
            - rotation @ self._reference_local_dict[name]
        )
        positions = {
            n: translation + rotation @ self._reference_local_dict[n]
            for n in TEAT_NAMES
        }
        positions[name] = np.asarray(detection.position, dtype=float)
        self._last_positions = {n: v.copy() for n, v in positions.items()}
        self._last_stamp = stamp
        self._last_real_stamp = stamp
        source = (
            {name: int(detection.source_index)}
            if detection.source_index >= 0
            else {}
        )
        predicted = tuple(n for n in TEAT_NAMES if n != name)
        return UdderObservation(
            stamp, translation, rotation, positions, 0.0,
            source, (name,), predicted,
        )

    def match_partial(
        self, detections: Sequence[TeatDetection], stamp: float
    ) -> Mapping[str, int]:
        """Match one or two visible teats to established identities only."""
        stamp = float(stamp)
        if not np.isfinite(stamp):
            raise ValueError("stamp must be finite")
        if self._last_stamp is not None and stamp <= self._last_stamp:
            raise ValueError("detection timestamp must increase")
        valid = [
            detection.validated()
            for detection in detections
            if float(detection.score) >= self.min_score
        ]
        if len(valid) >= 3:
            raise ValueError("partial identity matching accepts at most two teats")
        if not valid or self._last_positions is None:
            return {}
        named = self._match_previous(valid)
        for name, detection in named.items():
            self._last_positions[name] = np.asarray(
                detection.position, dtype=float
            ).copy()
        self._last_stamp = stamp
        return {
            name: int(detection.source_index)
            for name, detection in named.items()
            if detection.source_index >= 0
        }

    def predict_two(
        self, detections: Sequence[TeatDetection], stamp: float
    ) -> UdderObservation:
        """Complete a four-teat template from two observed, identified teats."""
        stamp = float(stamp)
        if not np.isfinite(stamp):
            raise ValueError("stamp must be finite")
        if self._last_stamp is not None and stamp <= self._last_stamp:
            raise ValueError("detection timestamp must increase")
        if (
            self._last_positions is None
            or self._last_origin is None
            or self._last_rotation is None
            or self._last_full_stamp is None
            or self._reference_local_dict is None
            or set(self._reference_local_dict) != set(TEAT_NAMES)
        ):
            raise ValueError("four-teat template has not been established")
        if stamp - self._last_full_stamp > self.two_point_timeout:
            raise ValueError("two-point prediction timeout")
        valid = [
            detection.validated()
            for detection in detections
            if float(detection.score) >= self.min_score
        ]
        if len(valid) != 2:
            raise ValueError("two-point prediction requires exactly two teats")
        named = self._match_previous(valid)
        names = [name for name in TEAT_NAMES if name in named]
        first, second = names
        previous_pair = self._last_rotation @ (
            self._reference_local_dict[second] - self._reference_local_dict[first]
        )
        measured_pair = np.asarray(named[second].position) - np.asarray(
            named[first].position
        )
        delta_rotation = _minimal_rotation(previous_pair, measured_pair)
        rotation = delta_rotation @ self._last_rotation
        weights = np.asarray([named[name].score for name in names], dtype=float)
        candidates = np.vstack(
            [
                np.asarray(named[name].position)
                - rotation @ self._reference_local_dict[name]
                for name in names
            ]
        )
        origin = np.average(candidates, axis=0, weights=weights)
        fitted = np.vstack(
            [origin + rotation @ self._reference_local_dict[name] for name in names]
        )
        measured = np.vstack([named[name].position for name in names])
        residual = float(
            np.sqrt(np.average(np.sum((fitted - measured) ** 2, axis=1), weights=weights))
        )
        angle = float(
            np.arccos(
                np.clip(
                    np.dot(_normalize(previous_pair, "previous teat pair"),
                           _normalize(measured_pair, "current teat pair")),
                    -1.0,
                    1.0,
                )
            )
        )
        if residual > self.max_fit_residual:
            raise ValueError("two-point geometry residual exceeds limit")
        if float(np.linalg.norm(origin - self._last_origin)) > self.max_prediction_translation:
            raise ValueError("two-point prediction translation exceeds limit")
        if angle > self.max_prediction_rotation:
            raise ValueError("two-point prediction rotation exceeds limit")

        positions = {
            name: origin + rotation @ self._reference_local_dict[name]
            for name in TEAT_NAMES
        }
        for name in names:
            positions[name] = np.asarray(named[name].position, dtype=float)
        self._last_positions = {name: value.copy() for name, value in positions.items()}
        self._last_origin = origin.copy()
        self._last_rotation = rotation.copy()
        self._last_stamp = stamp
        self._last_real_stamp = stamp
        source_indices = {
            name: int(named[name].source_index)
            for name in names
            if named[name].source_index >= 0
        }
        predicted = tuple(name for name in TEAT_NAMES if name not in named)
        return UdderObservation(
            stamp,
            origin,
            rotation,
            positions,
            residual,
            source_indices,
            tuple(names),
            predicted,
        )

    def _update_once(
        self, detections: Sequence[TeatDetection], stamp: float
    ) -> UdderObservation:
        valid: list[TeatDetection] = []
        for detection in detections:
            candidate = detection.validated()
            if candidate.score >= self.min_score:
                valid.append(candidate)
        # 3 点降级：至少 3 个有效乳头即可拟合（缺一个用可用点估计坐标轴）
        if len(valid) < 3:
            raise ValueError("at least three valid teat detections are required")
        selected = sorted(valid, key=lambda item: item.score, reverse=True)[:4]

        if self._last_positions is None:
            named = self._initial_assignment(selected)
            names = [n for n in TEAT_NAMES if n in named]
            positions = {n: np.asarray(named[n].position, dtype=float) for n in names}
            origin, rotation = build_semantic_frame(positions, self.expected_z)
            # 3 点初始化：模板不完整，用平行四边形推断缺失点补全模板。
            # 仅作为初始化 fallback（没有完整四点模板时），之后真实 4 点会修正。
            if len(names) == 3:
                missing = [n for n in TEAT_NAMES if n not in names][0]
                diag_map = {
                    "front_left": "rear_right",
                    "front_right": "rear_left",
                    "rear_left": "front_right",
                    "rear_right": "front_left",
                }
                diag_point = diag_map[missing]
                others = [n for n in names if n != diag_point]
                # 平行四边形：FL+RR = FR+RL（对角线交于中心）
                positions[missing] = (
                    positions[others[0]] + positions[others[1]]
                    - positions[diag_point]
                )
            self._reference_local_dict = {
                n: rotation.T @ (positions[n] - origin) for n in TEAT_NAMES
            }
            self._tracked_names = list(positions.keys())
            residual = 0.0
        else:
            named = self._match_previous(selected)
            names = [n for n in TEAT_NAMES if n in named]
            positions = {n: np.asarray(named[n].position, dtype=float) for n in names}
            # 用当前可用名字与参考坐标的交集做 rigid fit（4 点或 3 点）
            use_names = [n for n in names if n in self._reference_local_dict]
            if len(use_names) < 3:
                raise ValueError("insufficient tracked teats for rigid fit")
            reference = np.vstack([self._reference_local_dict[n] for n in use_names])
            measured = np.vstack([positions[n] for n in use_names])
            weights = np.asarray([named[n].score for n in use_names], dtype=float)
            origin, rotation, residual = _rigid_fit(reference, measured, weights)
            if residual > self.max_fit_residual:
                raise ValueError(
                    f"udder rigid-fit residual {residual:.4f} exceeds limit"
                )
            # 补全缺失点：只有模板含该点时才用几何恢复（3 点补第 4 点）
            for n in TEAT_NAMES:
                if n not in positions and n in self._reference_local_dict:
                    positions[n] = origin + rotation @ self._reference_local_dict[n]
            # 用新拟合更新参考局部坐标（覆盖当前可用点，缺的点保留旧值以便恢复）
            updated = dict(self._reference_local_dict)
            for n in names:
                updated[n] = rotation.T @ (positions[n] - origin)
            self._reference_local_dict = updated
            self._tracked_names = names

        self._last_stamp = stamp
        self._last_origin = origin.copy()
        self._last_rotation = rotation.copy()
        self._last_full_stamp = stamp
        # 保留所有见过名字的最后位置（缺失的点保留旧值，便于点恢复时正确匹配）
        if self._last_positions is None:
            self._last_positions = {name: value.copy() for name, value in positions.items()}
        else:
            new_last = dict(self._last_positions)
            for name, value in positions.items():
                new_last[name] = value.copy()
            self._last_positions = new_last
        source_indices = {
            name: int(detection.source_index)
            for name, detection in named.items()
            if detection.source_index >= 0
        }
        predicted = tuple(n for n in TEAT_NAMES if n not in names)
        self._last_real_stamp = stamp
        return UdderObservation(
            stamp,
            origin,
            rotation,
            positions,
            residual,
            source_indices,
            tuple(names),
            predicted,
        )

    def _initial_assignment(
        self, detections: Sequence[TeatDetection]
    ) -> dict[str, TeatDetection]:
        # 用 3D 相机坐标绑定 ID（比像素 u/v 更稳）：
        #   前后(front/rear)按深度 position[2]（z 小=近=front）
        #   左右(left/right)按 position[0]（x 小=left）
        ordered_z = sorted(detections, key=lambda item: float(item.position[2]))
        n = len(ordered_z)
        if n == 4:
            first, second = ordered_z[:2], ordered_z[2:]
            front_row, rear_row = (
                (first, second) if self.front_is_smaller_v else (second, first)
            )
            def split_row(row: Sequence[TeatDetection]):
                ordered_x = sorted(row, key=lambda item: float(item.position[0]))
                return ordered_x if self.left_is_smaller_u else list(reversed(ordered_x))
            front_left, front_right = split_row(front_row)
            rear_left, rear_right = split_row(rear_row)
            return {
                "front_left": front_left,
                "front_right": front_right,
                "rear_left": rear_left,
                "rear_right": rear_right,
            }
        # 3 点降级：按深度 z 间隙分行（前/后），每行 1~2 个；单点行用 x 相对另一行判定左右
        zs = [float(item.position[2]) for item in ordered_z]
        gaps = [zs[i + 1] - zs[i] for i in range(2)]
        split = 1 + int(gaps.index(max(gaps)))
        if self.front_is_smaller_v:
            front_row, rear_row = ordered_z[:split], ordered_z[split:]
        else:
            front_row, rear_row = ordered_z[split:], ordered_z[:split]

        def split_row(row: Sequence[TeatDetection]):
            ordered_x = sorted(row, key=lambda item: float(item.position[0]))
            return ordered_x if self.left_is_smaller_u else list(reversed(ordered_x))

        result: dict[str, TeatDetection] = {}
        # 每行可能 1 或 2 个；单点行用 x 与另一行的 x 中位数比较定左右
        def assign_row(row, prefix: str, other_row):
            if len(row) == 2:
                left_item, right_item = split_row(row)
                result[f"{prefix}_left"] = left_item
                result[f"{prefix}_right"] = right_item
            else:
                item = row[0]
                other_x = [float(d.position[0]) for d in other_row]
                median_x = float(np.median(other_x)) if other_x else float(item.position[0])
                if self.left_is_smaller_u:
                    side = "left" if float(item.position[0]) < median_x else "right"
                else:
                    side = "right" if float(item.position[0]) < median_x else "left"
                result[f"{prefix}_{side}"] = item

        assign_row(front_row, "front", rear_row)
        assign_row(rear_row, "rear", front_row)
        return result

    def _match_previous(
        self, detections: Sequence[TeatDetection]
    ) -> dict[str, TeatDetection]:
        n = len(detections)
        # 从已知名字里选 n 个（4 点或 3 点），与检测点做最近匹配；
        # 已知名字 = 上次观测保留的全部名字（含缺失点的历史位置）
        available = [name for name in TEAT_NAMES if name in self._last_positions]
        if len(available) < n:
            # Three-point initialization followed by four detections: match the
            # three known identities first, then assign the remaining point to
            # the only missing identity.  Do not read history that does not exist.
            if len(available) != 3 or n != 4:
                raise ValueError("unsupported teat identity expansion")
            best_expansion = None
            for candidate_order in permutations(detections, len(available)):
                previous = np.vstack(
                    [self._last_positions[name] for name in available]
                )
                current = np.vstack([item.position for item in candidate_order])
                distances = np.linalg.norm(current - previous, axis=1)
                cost = float(np.sum(distances**2))
                if best_expansion is None or cost < best_expansion[0]:
                    best_expansion = cost, candidate_order, distances
            assert best_expansion is not None
            _, candidate_order, distances = best_expansion
            if float(np.max(distances)) > self.max_match_distance:
                raise ValueError("teat identity match exceeds motion limit")
            matched = dict(zip(available, candidate_order))
            used = {id(item) for item in candidate_order}
            remaining = [item for item in detections if id(item) not in used]
            missing = [name for name in TEAT_NAMES if name not in matched]
            assert len(remaining) == len(missing) == 1
            matched[missing[0]] = remaining[0]
            return matched
        best: tuple[float, tuple[str, ...], tuple[TeatDetection, ...]] | None = None
        for names_subset in permutations(available, n):
            previous = np.vstack([self._last_positions[name] for name in names_subset])
            for candidate_order in permutations(detections, n):
                current = np.vstack([item.position for item in candidate_order])
                distances = np.linalg.norm(current - previous, axis=1)
                cost = float(np.sum(distances**2))
                if best is None or cost < best[0]:
                    best = cost, names_subset, candidate_order
        assert best is not None
        _, names_subset, candidate_order = best
        matched = dict(zip(names_subset, candidate_order))
        distances = [
            float(np.linalg.norm(np.asarray(matched[name].position)
                                 - self._last_positions[name]))
            for name in names_subset
        ]
        if max(distances) > self.max_match_distance:
            raise ValueError("teat identity match exceeds motion limit")
        return matched
