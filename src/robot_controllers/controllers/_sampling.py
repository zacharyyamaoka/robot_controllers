"""What the trajectory controllers share: a fast time-driven sampler, the clock, joint selection by name,
and the per-joint parameter helpers their configs use.

WHY a sampler of our own instead of calling `JointTrajectory.joint_state_at` every tick: that call builds a
new trajectory and a JointState through robot_msgs' validation, about 170 µs per call on this machine
(measured 2026-11-12), more than the rest of a tick together, and compute time is a metric every controller
is ranked on. `TrajectorySampler.sample` is the same law (linear between samples, the ends held,
`(1 - w)·a + w·b` so both endpoints come back bit-exact) in plain numpy; tests/test_controllers.py pins it
bit-for-bit against `joint_state_at(time, extrapolate="clamp")`.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import field
from typing import Any

import numpy as np
from robot_msgs.sensor import JointState
from robot_msgs.trajectory import JointTrajectory

PerJoint = float | tuple[float, ...]
"""A per-joint parameter: one value for every joint, or one per joint in the input's `joint_names` order."""

END_TOLERANCE_S = 1e-9
"""A clock this close to the trajectory's end counts as at the end (summed dts carry float round-off)."""


def param(default: Any, unit: str, doc: str) -> Any:
    """A config field with its unit and one-line meaning attached, so the site can print the parameters
    table from `dataclasses.fields(config)[i].metadata` without parsing source comments."""
    return field(default=default, metadata={"unit": unit, "doc": doc})


def check_dt(dt: float) -> float:
    """The measured period since the last tick (s): finite and not negative."""
    dt = float(dt)
    if not (dt >= 0.0 and dt != math.inf):
        raise ValueError(f"dt must be finite and >= 0 s, got {dt!r}")
    return dt


def normalize_per_joint(value: Any, field_name: str, *, allow_none: bool = False) -> PerJoint | None:
    """A config field's value as a float or a tuple of floats, each finite and >= 0 (configs stay
    hashable)."""
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must be set")
    if np.ndim(value) == 0:
        number = float(value)
        if not (math.isfinite(number) and number >= 0.0):
            raise ValueError(f"{field_name} must be finite and >= 0, got {value!r}")
        return number
    numbers = tuple(float(v) for v in value)
    if not numbers or not all(math.isfinite(v) and v >= 0.0 for v in numbers):
        raise ValueError(f"{field_name} values must be finite and >= 0, got {value!r}")
    return numbers


def per_joint(value: PerJoint | None, joint_names: Sequence[str], field_name: str) -> np.ndarray | None:
    """`value` as one float per joint: a scalar is broadcast, a tuple must have one entry per joint."""
    if value is None:
        return None
    count = len(joint_names)
    if np.ndim(value) == 0:
        return np.full(count, float(value))
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (count,):
        raise ValueError(
            f"{field_name} has {values.size} values but the input has {count} joints {tuple(joint_names)}; "
            "give one value, or one per joint in the input's order"
        )
    return values


class TrajectorySampler:
    """One JointTrajectory, sampled at a time on its own `t` axis, holding the first and last samples
    outside it.

    `sample(time) -> (positions, velocities, effort)`, each `(J,)`: positions and velocities are linear
    between samples exactly as `JointTrajectory.joint_state_at(time, extrapolate="clamp")`; effort is that
    too, or zeros when the trajectory has none. A trajectory without velocities gets the slope of the segment
    `time` falls in (zero outside the trajectory). WHY the slope and not zero: a zero velocity target makes
    the board's damping brake the very motion it is being asked to make.
    """

    __slots__ = (
        "trajectory",
        "joint_names",
        "start_time",
        "end_time",
        "_times",
        "_positions",
        "_velocities",
        "_effort",
        "_zeros",
    )

    def __init__(self, trajectory: JointTrajectory) -> None:
        if not isinstance(trajectory, JointTrajectory):
            raise TypeError(f"expected a robot_msgs JointTrajectory, got {type(trajectory).__name__}")
        if len(trajectory) == 0:
            raise ValueError("an empty JointTrajectory has nothing to follow")
        self.trajectory = trajectory
        self.joint_names: tuple[str, ...] = trajectory.joint_names
        self._times: list[float] = trajectory.t.tolist()
        self.start_time: float = self._times[0]
        self.end_time: float = self._times[-1]
        self._positions = trajectory.positions
        self._velocities = trajectory.velocities
        self._effort = trajectory.effort
        zeros = np.zeros(len(self.joint_names))
        zeros.setflags(write=False)
        self._zeros = zeros

    def sample(self, time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = self._times
        positions, velocities, effort = self._positions, self._velocities, self._effort
        if time >= self.end_time or len(times) == 1:
            last = len(times) - 1
            return (
                positions[last],
                self._zeros if velocities is None else velocities[last],
                self._zeros if effort is None else effort[last],
            )
        if time < self.start_time:
            return (
                positions[0],
                self._zeros if velocities is None else velocities[0],
                self._zeros if effort is None else effort[0],
            )
        # start <= time < end, so lower is in [0, N - 2]: the bracket of robot_msgs' Trajectory._bracket.
        lower = bisect_right(times, time) - 1
        upper = lower + 1
        span = times[upper] - times[lower]
        weight = (time - times[lower]) / span
        keep = 1.0 - weight
        position = keep * positions[lower] + weight * positions[upper]
        if velocities is None:
            velocity = (positions[upper] - positions[lower]) / span
        else:
            velocity = keep * velocities[lower] + weight * velocities[upper]
        if effort is None:
            return position, velocity, self._zeros
        return position, velocity, keep * effort[lower] + weight * effort[upper]


class TrajectoryClock:
    """Time since the active input began (s): the time-driven clock of the JTC and the impedance
    controllers.

    Each tick adds its measured dt, except the first tick after an input arrives (construction or
    `replace_input`), which samples t = 0. WHY: that tick's dt is the time since the previous wake, which
    elapsed before this input began; the loop delivers a new input right before the tick of the same wake,
    so "now" is t = 0 of the new input (ros2_control JTC: a trajectory stamped 0 starts at the update that
    adopts it). Adding it would run the whole input one period early.
    """

    __slots__ = ("time", "_fresh")

    def __init__(self) -> None:
        self.time = 0.0
        self._fresh = True

    def restart(self) -> None:
        self.time = 0.0
        self._fresh = True

    def advance(self, dt: float) -> float:
        dt = check_dt(dt)
        if self._fresh:
            self._fresh = False
        else:
            self.time += dt
        return self.time


class JointSelector:
    """The controller's joints out of a measured JointState, by name and in the input's order.

    The state may carry more joints, in any order (CONTRACT §2). The index map is cached per state name
    tuple, so a steady loop pays one tuple comparison per tick; a missing joint raises KeyError naming it.
    """

    __slots__ = ("joint_names", "_state_names", "_indices")

    def __init__(self, joint_names: tuple[str, ...]) -> None:
        self.joint_names = joint_names
        self._state_names: tuple[str, ...] | None = None
        self._indices: np.ndarray | None = None

    def check(self, state: JointState) -> np.ndarray | None:
        """The indices of the controller's joints in `state` (None when the state has exactly that
        order)."""
        names = state.name
        if names is not self._state_names and names != self._state_names:
            if names == self.joint_names:
                indices = None
            else:
                indices = np.array([state.index_of(name) for name in self.joint_names], dtype=np.intp)
            self._state_names, self._indices = names, indices
        return self._indices

    def positions(self, state: JointState) -> np.ndarray:
        indices = self.check(state)
        return state.position if indices is None else state.position[indices]

    def velocities(self, state: JointState, who: str) -> np.ndarray:
        indices = self.check(state)
        if state.velocity is None:
            raise ValueError(f"{who} needs measured joint velocities; the JointState has none")
        return state.velocity if indices is None else state.velocity[indices]
