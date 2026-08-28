"""Jerk-limited Cartesian reference generation with Ruckig."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from ruckig import ControlInterface, InputParameter, OutputParameter, Result, Ruckig


def _vec3(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain three finite values")
    return result


class RuckigVelocityLimiter:
    """Turn a Cartesian velocity goal into a continuous position reference."""

    def __init__(self, period: float) -> None:
        self.period = float(period)
        if self.period <= 0.0:
            raise ValueError("period must be positive")
        self.otg = Ruckig(3, self.period)
        self.input = InputParameter(3)
        self.output = OutputParameter(3)
        self.input.control_interface = ControlInterface.Velocity
        self.initialized = False

    def reset(
        self,
        position: Sequence[float],
        velocity: Sequence[float] = (0.0, 0.0, 0.0),
        acceleration: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.input.current_position = _vec3(position, "position").tolist()
        self.input.current_velocity = _vec3(velocity, "velocity").tolist()
        self.input.current_acceleration = _vec3(
            acceleration, "acceleration").tolist()
        self.input.target_velocity = [0.0, 0.0, 0.0]
        self.input.target_acceleration = [0.0, 0.0, 0.0]
        self.initialized = True

    def set_axis_state(
        self, axis: int, position: float, velocity: float = 0.0,
        acceleration: float = 0.0,
    ) -> None:
        if not self.initialized:
            raise RuntimeError("limiter is not initialized")
        self.input.current_position[axis] = float(position)
        self.input.current_velocity[axis] = float(velocity)
        self.input.current_acceleration[axis] = float(acceleration)

    def step(
        self,
        target_velocity: Sequence[float],
        max_velocity: Sequence[float],
        max_acceleration: Sequence[float],
        max_jerk: Sequence[float],
    ) -> tuple[list[float], list[float], list[float]]:
        if not self.initialized:
            raise RuntimeError("limiter is not initialized")
        self.input.target_velocity = _vec3(
            target_velocity, "target_velocity").tolist()
        self.input.target_acceleration = [0.0, 0.0, 0.0]
        self.input.max_velocity = _vec3(max_velocity, "max_velocity").tolist()
        self.input.max_acceleration = _vec3(
            max_acceleration, "max_acceleration").tolist()
        self.input.max_jerk = _vec3(max_jerk, "max_jerk").tolist()
        result = self.otg.update(self.input, self.output)
        if result not in (Result.Working, Result.Finished):
            raise RuntimeError(f"Ruckig update failed: {result}")
        position = list(self.output.new_position)
        velocity = list(self.output.new_velocity)
        acceleration = list(self.output.new_acceleration)
        self.output.pass_to_input(self.input)
        return position, velocity, acceleration
