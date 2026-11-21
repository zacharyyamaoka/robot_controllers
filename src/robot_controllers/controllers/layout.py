"""The layout controller: a feed-forward model and a feedback model, each a pure function, placed by a Layout
(CONTRACT amendment 3, §3.3).

Zach, 2026-11-20: "Surely thats just one that like just fowrwards the joint trajectory its given, and feedback
runs on the driver. that is just alike a first step" - that first step is `LayoutConfig()`, tick for tick the
joint trajectory controller. Every other built controller of the JointTrajectory box is a point in the same
space:

    LayoutConfig()                                                   # = joint_trajectory_controller
    LayoutConfig(feedback=PD(K, D), layout=Layout("trajectory", "board"))  # = joint_impedance_board
    LayoutConfig(feedback=PD(K, D), layout=Layout("trajectory", "host"))   # = joint_impedance_host
    LayoutConfig(feedback=PD(K, D), layout=Layout("trajectory", "split"))  # = joint_impedance_host_spring
    LayoutConfig(feedforward=ff, feedback=fb, layout=Layout("plan", "board"))
    controller = LayoutController(trajectory, config)

Time-driven exactly as the JTC (`TrajectoryClock`, `TrajectorySampler`): the clock adds each tick's dt (the
first tick after an input samples t = 0), the reference is sampled at clock + lookahead (one lookahead, or
each joint at its own, §4.1), held at rest after the end (`status == "done"`), the optional path tolerance
aborts and holds where the arm was measured, and `replace_input` runs a new trajectory from the next tick.
The reference acceleration is the trajectory's own `accelerations`, else the finite difference of its
velocities on its own grid (`np.gradient`), sampled linearly and held at the ends.

What a tick evaluates is the layout's choice (see `models.Layout`): `plan` evaluates `ff` once per trajectory
sample at load (build and replace_input) and the tick only interpolates that column; `host_*` evaluate `ff`
every tick; `host` and `split` evaluate the feedback every tick on the measured state and send it inside the
feed-forward field. `last_split()` says how much of that field was feed-forward and how much host feedback;
`diagnostics()["plan_compute_s"]` is the thread CPU time spent evaluating `ff` at plan time (the cost the
tick's own timing never sees).

Interpolation (amendment 4, §4.1): between two commands the board walks its setpoint at the sent velocity and
holds the feed-forward. `tangent` (the default) sends the reference velocity at the sample time s; `secant`
sends `(p(s + T) - p(s)) / T`, T = `control_period_s`, so the walk lands on the next sample; `secant_mid_ff`
also reads everything the feed-forward takes from the reference at `s + T/2`, the middle of the hold. Host and
split feedback still act on the reference at s; split's board damper tracks the secant velocity.
"""

from __future__ import annotations

import math
import time as _time
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any

import numpy as np
from robot_msgs.control import ServoCommand
from robot_msgs.sensor import JointState
from robot_msgs.trajectory import JointTrajectory

from robot_controllers.controllers._sampling import (
    END_TOLERANCE_S,
    JointSelector,
    PerJoint,
    TrajectoryClock,
    TrajectorySampler,
    normalize_per_joint,
    param,
    per_joint,
)
from robot_controllers.core import Status
from robot_controllers.models import FeedbackModel, FeedforwardModel, Layout


@dataclass(frozen=True)
class LayoutConfig:
    """Parameters of the layout controller: two pure functions and where each runs. Per-joint values follow
    the trajectory's `joint_names` order; a single value applies to every joint."""

    feedforward: FeedforwardModel | None = param(
        None,
        "",
        "Feed-forward model ff(q, qd, qdd) -> tau, a pure function (rad, rad/s, rad/s² -> N·m); None = no "
        "model (only the trajectory and none placements).",
    )
    feedback: FeedbackModel | None = param(
        None,
        "",
        "Feedback model fb(q_ref, qd_ref, q, qd) -> tau, a pure function; one with gains(q_ref) -> (kp, kd) "
        "can run on the board. None = the board's configured gains.",
    )
    layout: Layout = param(
        Layout("trajectory", "board"),
        "",
        "Where and how often each model is evaluated: feedforward_at (trajectory, none, plan, "
        "host_reference, host_measured), feedback_at (board, board_scheduled, host, split), lookahead_s.",
    )
    max_effort_nm: PerJoint | None = param(
        None, "N·m", "Torque cap sent with every command; None = the board's own cap."
    )
    path_tolerance_rad: PerJoint | None = param(
        None,
        "rad",
        "Largest |reference - measured position| before the controller aborts and holds where the arm is; "
        "None = no guard (ros2_control JTC path tolerance).",
    )
    control_period_s: float = param(
        0.01,
        "s",
        "The host period T a secant interpolation spans (the bench sets it from the rung's rate, as a "
        "deployment would); the tangent default never reads it. Default 0.01 s = the rack host's 100 Hz.",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.layout, Layout):
            raise TypeError(
                f"layout must be a robot_controllers.models.Layout, got {type(self.layout).__name__}"
            )
        period = float(self.control_period_s)
        if not (math.isfinite(period) and period > 0.0):
            raise ValueError(f"control_period_s must be finite and > 0, got {self.control_period_s!r}")
        object.__setattr__(self, "control_period_s", period)
        for name in ("max_effort_nm", "path_tolerance_rad"):
            object.__setattr__(self, name, normalize_per_joint(getattr(self, name), name, allow_none=True))


class _Column:
    """An (N, J) column on a trajectory's own `t` axis, sampled as TrajectorySampler samples positions: linear
    between samples with `(1 - w)·a + w·b` (so a sample time gives the sample back exactly), the ends held."""

    __slots__ = ("_times", "_values", "_start", "_end")

    def __init__(self, times: list[float], values: np.ndarray) -> None:
        values = np.array(values, dtype=np.float64)
        values.setflags(write=False)
        self._times, self._values = times, values
        self._start, self._end = times[0], times[-1]

    def at(self, time: float) -> np.ndarray:
        times, values = self._times, self._values
        if time >= self._end or len(times) == 1:
            return values[-1]
        if time < self._start:
            return values[0]
        lower = bisect_right(times, time) - 1
        weight = (time - times[lower]) / (times[lower + 1] - times[lower])
        return (1.0 - weight) * values[lower] + weight * values[lower + 1]


def _grid_velocities(trajectory: JointTrajectory) -> np.ndarray:
    """The velocity at every sample, as TrajectorySampler gives it there: the trajectory's own, else the slope
    of the segment each sample starts (zero at the last, as the sampler holds the end at rest)."""
    if trajectory.velocities is not None:
        return np.asarray(trajectory.velocities, dtype=np.float64)
    positions = np.asarray(trajectory.positions, dtype=np.float64)
    velocities = np.zeros_like(positions)
    if len(positions) > 1:
        velocities[:-1] = (
            np.diff(positions, axis=0) / np.diff(np.asarray(trajectory.t, dtype=np.float64))[:, None]
        )
    return velocities


def _grid_accelerations(trajectory: JointTrajectory, velocities: np.ndarray) -> np.ndarray:
    """The trajectory's own accelerations, else np.gradient of its velocities on its own grid (§3.3)."""
    if trajectory.accelerations is not None:
        return np.asarray(trajectory.accelerations, dtype=np.float64)
    if len(velocities) < 2:
        return np.zeros_like(velocities)
    return np.gradient(velocities, np.asarray(trajectory.t, dtype=np.float64), axis=0)


def _as_joint_vector(values: Any, count: int, what: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count,):
        raise ValueError(f"{what} returned shape {array.shape}; expected ({count},), one value per joint")
    return array


class LayoutController:
    """A JointTrajectory followed with a feed-forward model and a feedback model placed by a Layout.

    Built as `LayoutController(trajectory, LayoutConfig(...))`; an invalid layout for the given models raises
    ValueError here (CONTRACT §3.2). Replannable, self-reporting, and `last_split() -> (tau_ff, tau_fb_host)`:
    the feed-forward and the host feedback inside the last command's feed-forward field (zeros before the
    first tick and
    while holding after an abort; tau_fb_host is zero whenever the feedback runs on the board).
    """

    status: Status

    def __init__(self, trajectory: JointTrajectory, config: LayoutConfig | None = None) -> None:
        self.config = config if config is not None else LayoutConfig()
        layout = self.config.layout
        layout.check(self.config.feedforward, self.config.feedback)
        self.layout = layout
        self._ff = self.config.feedforward
        self._fb = self.config.feedback
        self._ff_at = layout.feedforward_at
        self._fb_at = layout.feedback_at
        self._period = self.config.control_period_s
        self._secant = layout.interpolation != "tangent"
        self._mid_ff = layout.interpolation == "secant_mid_ff"
        # what a tick has to read from the measured state
        self._needs_position = self._ff_at == "host_measured" or self._fb_at in ("host", "split")
        self._needs_velocity = self._ff_at == "host_measured" or self._fb_at == "host"
        self.status = "running"
        self.message = ""
        self.active_input = 0
        self.plan_compute_s = 0.0
        self.plan_loads = 0
        self._clock = TrajectoryClock()
        self._load(trajectory)

    # ── loading an input (build and replace_input) ─────────────────────────────────────────────────────────

    def _load(self, trajectory: JointTrajectory) -> None:
        # Everything that can fail is computed before anything is assigned: a refused input changes nothing.
        sampler = TrajectorySampler(trajectory)
        names = sampler.joint_names
        count = len(names)
        lookaheads = self.layout.lookaheads(count)  # raises on a per-joint lookahead of another length
        max_effort = per_joint(self.config.max_effort_nm, names, "max_effort_nm")
        tolerance = per_joint(self.config.path_tolerance_rad, names, "path_tolerance_rad")
        times: list[float] = trajectory.t.tolist()
        acceleration_column = plan_column = None
        plan_cpu_s = 0.0
        if self._ff_at in ("plan", "host_reference", "host_measured"):
            velocities = _grid_velocities(trajectory)
            accelerations = _grid_accelerations(trajectory, velocities)
            if self._ff_at == "plan":
                positions = np.asarray(trajectory.positions, dtype=np.float64)
                # WHY thread CPU time: the same clock the loop scores a tick on (amendment 1), so plan-time
                # and tick-time costs add up on one scale; this call is the whole plan-time cost of the
                # layout.
                start = _time.thread_time_ns()
                column = np.asarray(self._ff(positions, velocities, accelerations), dtype=np.float64)
                plan_cpu_s = (_time.thread_time_ns() - start) * 1e-9
                if column.shape != positions.shape:
                    raise ValueError(
                        f"feedforward on the {positions.shape} trajectory samples returned "
                        f"shape {column.shape}"
                    )
                plan_column = _Column(times, column)
            else:
                acceleration_column = _Column(times, accelerations)
        stiffness = damping = None
        if self._fb is not None and self._fb_at in ("board", "split"):
            # WHY from the first sample, once per input: constant gains are what a board renders between
            # commands (CONTRACT §3.2); scheduling them every tick is the board_scheduled layout.
            kp, kd = self._fb.gains(np.asarray(trajectory.positions[0], dtype=np.float64))
            stiffness = _as_joint_vector(kp, count, "feedback.gains (kp)")
            damping = _as_joint_vector(kd, count, "feedback.gains (kd)")
        # The constant fields of every command, frozen once by robot_msgs so a tick does not copy them.
        template = ServoCommand(
            name=names,
            position=np.full(count, np.nan),
            velocity=np.zeros(count),
            stiffness=stiffness,
            damping=damping,
            max_effort=max_effort,
        )
        self.trajectory = trajectory
        self._sampler = sampler
        self._count = count
        # WHY one float when every joint looks as far ahead: that path is today's tick, operation for
        # operation, so the default (and a tuple of equal values) commands bit for bit what it did before.
        if np.all(lookaheads == lookaheads[0]):
            self._lookahead: float | np.ndarray = float(lookaheads[0])
            self._groups: tuple[tuple[int, np.ndarray], ...] = ()
        else:
            lookaheads.setflags(write=False)
            self._lookahead = lookaheads
            self._groups = tuple(
                (int(np.flatnonzero(lookaheads == value)[0]), np.flatnonzero(lookaheads == value))
                for value in dict.fromkeys(lookaheads.tolist())
            )
        self._names = template.name
        self._selector = JointSelector(names)
        self._tolerance = tolerance
        self._max_effort = template.max_effort
        self._stiffness = template.stiffness
        self._damping = template.damping
        self._no_position = template.position  # NaN: no position term on the board
        self._zeros = template.velocity  # the at-rest velocity, zero gains, zero torque
        self._plan_column = plan_column
        self._acceleration_column = acceleration_column
        self._hold: ServoCommand | None = None
        self._last_ff = self._zeros
        self._last_fb = self._zeros
        if plan_column is not None:
            self.plan_compute_s += plan_cpu_s
            self.plan_loads += 1
        self.status = "running"
        self.message = ""

    # ── one tick ───────────────────────────────────────────────────────────────────────────────────────────

    def tick(self, state: JointState, dt: float) -> tuple[ServoCommand, dict[str, float]]:
        time = self._clock.advance(dt)
        if self._hold is not None:
            self._selector.check(state)
            self._last_ff = self._last_fb = self._zeros
            return self._hold, {}
        sample_time = time + self._lookahead  # a float, or one time per joint
        reference_position, reference_velocity, reference_effort = self._sample(sample_time)
        end = self._sampler.end_time
        at_rest = sample_time + END_TOLERANCE_S >= end
        # WHY at rest: as the JTC, the hold after the end must not carry the last sample's velocity
        reference_velocity = self._rest(reference_velocity, at_rest)
        command_velocity = reference_velocity
        if self._secant:
            # WHY the secant (§4.1): the board walks its setpoint at the sent velocity for one host period;
            # sent as (p(s + T) - p(s)) / T the walk lands on the next sample instead of leaving the curve
            # along its tangent, an error that grows with T². The trajectory is known ahead: one more sample.
            ahead = self._sample(sample_time + self._period)[0]
            command_velocity = self._rest((ahead - reference_position) / self._period, at_rest)
        if time + END_TOLERANCE_S >= end and self.status == "running":
            self.status = "done"
            self.message = f"trajectory ended at {end:.3f} s; holding its last sample"

        measured_position = measured_velocity = None
        if self._tolerance is not None or self._needs_position:
            measured_position = self._selector.positions(state)
        else:
            self._selector.check(state)
        if self._needs_velocity:
            measured_velocity = self._selector.velocities(state, "layout_controller")
        if self._tolerance is not None:
            error = np.abs(reference_position - measured_position)
            off_path = ~(error <= self._tolerance)  # a NaN measurement counts as off the path
            if off_path.any():
                return self._abort(measured_position, error, off_path), {}

        feedforward = self._feedforward(
            sample_time + 0.5 * self._period if self._mid_ff else sample_time,
            at_rest,
            reference_position,
            reference_velocity,
            reference_effort,
            measured_position,
            measured_velocity,
        )
        fb_at = self._fb_at
        if fb_at == "board" or fb_at == "board_scheduled":
            self._last_ff, self._last_fb = feedforward, self._zeros
            if fb_at == "board":
                stiffness, damping = self._stiffness, self._damping
            else:
                stiffness, damping = self._fb.gains(reference_position)
            return ServoCommand(
                name=self._names,
                position=reference_position,
                velocity=command_velocity,
                # WHY None for "none": no feed-forward is one register fewer on the bus, and the board adds 0
                feedforward_effort=None if self._ff_at == "none" else feedforward,
                stiffness=stiffness,
                damping=damping,
                max_effort=self._max_effort,
            ), {}
        if fb_at == "host":
            host_feedback = np.asarray(
                self._fb(reference_position, reference_velocity, measured_position, measured_velocity),
                dtype=np.float64,
            )
            self._last_ff, self._last_fb = feedforward, host_feedback
            # WHY velocity None and gains 0 (as joint_impedance_host): the board's PD is off, so a velocity
            # target does nothing; 0, not None, because None keeps the board's configured gains.
            return ServoCommand(
                name=self._names,
                position=self._no_position,
                velocity=None,
                feedforward_effort=feedforward + host_feedback,
                stiffness=self._zeros,
                damping=self._zeros,
                max_effort=self._max_effort,
            ), {}
        # split: the spring kp (q_ref - q) on the host, the damper kd (v - qd) on the board around v = qd_ref
        host_feedback = self._stiffness * (reference_position - measured_position)
        self._last_ff, self._last_fb = feedforward, host_feedback
        return ServoCommand(
            name=self._names,
            position=self._no_position,
            velocity=command_velocity,
            feedforward_effort=feedforward + host_feedback,
            stiffness=self._zeros,
            damping=self._damping,
            max_effort=self._max_effort,
        ), {}

    def _feedforward(
        self,
        ff_time: float | np.ndarray,
        at_rest: bool | np.ndarray,
        reference_position: np.ndarray,
        reference_velocity: np.ndarray,
        reference_effort: np.ndarray,
        measured_position: np.ndarray | None,
        measured_velocity: np.ndarray | None,
    ) -> np.ndarray:
        """The feed-forward the layout places, from the reference at `ff_time`: the sample time itself, or
        the middle of the hold for `secant_mid_ff` (then everything it reads from the reference moves)."""
        ff_at = self._ff_at
        if ff_at == "none":
            return self._zeros
        mid = self._mid_ff
        if ff_at == "trajectory":
            return self._sample(ff_time)[2] if mid else reference_effort
        if ff_at == "plan":
            return self._column(self._plan_column, ff_time)
        if mid:
            at_rest = ff_time + END_TOLERANCE_S >= self._sampler.end_time
            reference_position, reference_velocity, _ = self._sample(ff_time)
            reference_velocity = self._rest(reference_velocity, at_rest)
        # WHY zero acceleration at rest: the hold after the end is at rest (velocity zeroed above), so its
        # acceleration is zero too; inside the trajectory it is the sampled reference acceleration.
        acceleration = self._rest(self._column(self._acceleration_column, ff_time), at_rest)
        if ff_at == "host_reference":
            values = self._ff(reference_position, reference_velocity, acceleration)
        else:  # host_measured: position/velocity terms at the measured state, acceleration from the reference
            values = self._ff(measured_position, measured_velocity, acceleration)
        return np.asarray(values, dtype=np.float64)

    # ── sampling at one time, or at each joint's own (§4.1) ──────────────────────────────────────────────

    def _sample(self, time: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The sampler's (position, velocity, effort) at `time`, or with one time per joint each joint's at
        its own: one sampler call per distinct lookahead."""
        if not self._groups:
            return self._sampler.sample(time)
        position, velocity, effort = (np.empty(self._count) for _ in range(3))
        for first, joints in self._groups:
            p, v, e = self._sampler.sample(float(time[first]))
            position[joints], velocity[joints], effort[joints] = p[joints], v[joints], e[joints]
        return position, velocity, effort

    def _column(self, column: _Column, time: float | np.ndarray) -> np.ndarray:
        if not self._groups:
            return column.at(time)
        out = np.empty(self._count)
        for first, joints in self._groups:
            out[joints] = column.at(float(time[first]))[joints]
        return out

    def _rest(self, values: np.ndarray, at_rest: bool | np.ndarray) -> np.ndarray:
        """`values` with the joints whose sample is at rest (past the end) set to zero."""
        if not self._groups:
            return self._zeros if at_rest else values
        return np.where(at_rest, 0.0, values) if at_rest.any() else values

    def _abort(self, measured: np.ndarray, error: np.ndarray, off_path: np.ndarray) -> ServoCommand:
        worst = int(np.argmax(np.where(off_path, np.nan_to_num(error, nan=np.inf), -1.0)))
        self.status = "aborted"
        self.message = (
            f"{self._names[worst]} was {math.degrees(error[worst]):.2f} deg off the path "
            f"(tolerance {math.degrees(self._tolerance[worst]):.2f} deg); holding where it was measured"
        )
        # WHY the JTC's hold whatever the layout: latched once at the measured pose, feed-forward 0, the
        # board's own configured gains (stiffness/damping None) - the one law that holds without the host
        # (bam_homing H3).
        self._hold = ServoCommand(
            name=self._names,
            position=np.array(measured, dtype=np.float64),
            velocity=self._zeros,
            feedforward_effort=self._zeros,
            max_effort=self._max_effort,
        )
        self._last_ff = self._last_fb = self._zeros
        return self._hold

    # ── the rest of the controller contract ───────────────────────────────────────────────────────────────

    def replace_input(self, message: JointTrajectory) -> None:
        """Run `message` from now: its t = 0 is the next tick. Accepted in any status (as ros2's JTC). A
        `plan` layout evaluates the feed-forward on the new trajectory here; its CPU time adds to
        plan_compute_s."""
        self._load(message)
        self._clock.restart()
        self.active_input += 1

    def last_split(self) -> tuple[np.ndarray, np.ndarray]:
        """(tau_ff, tau_fb_host) of the last tick, in the trajectory's joint order (N·m): the feed-forward and
        the host feedback the command's feed-forward field carries."""
        return self._last_ff, self._last_fb

    @property
    def clock_s(self) -> float:
        return self._clock.time

    def diagnostics(self) -> dict[str, float | str]:
        report: dict[str, float | str] = {
            "mode": "time_driven",
            "layout": self.layout.name,
            "feedforward_at": self._ff_at,
            "feedback_at": self._fb_at,
            # a per-joint lookahead reports its largest here and each joint's below
            "lookahead_s": float(np.max(self._lookahead)),
            "interpolation": self.layout.interpolation,
            "control_period_s": self._period,
            "plan_compute_s": self.plan_compute_s,
            "plan_loads": self.plan_loads,
            "status": self.status,
            "clock_s": self._clock.time,
            "active_trajectory": self.active_input,
            "trajectory_end_s": self._sampler.end_time,
            "message": self.message,
        }
        if self.layout.per_joint:
            for name, lookahead in zip(self._names, self.layout.lookaheads(self._count), strict=True):
                report[f"lookahead_s.{name}"] = float(lookahead)
        if self._stiffness is not None:
            for name, stiffness, damping in zip(self._names, self._stiffness, self._damping, strict=True):
                report[f"stiffness_nm_rad.{name}"] = float(stiffness)
                report[f"damping_nm_s_rad.{name}"] = float(damping)
        return report
