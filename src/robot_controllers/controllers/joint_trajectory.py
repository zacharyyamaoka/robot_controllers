"""Joint trajectory controllers: stream a JointTrajectory to the servo boards' own PD (CONTRACT §2).

Two clocks, one law:

- `JointTrajectoryController` is time-driven (ros2_control's joint_trajectory_controller): its clock adds
  each tick's measured dt and it samples the trajectory there, so a late tick skips ahead and the arm stays
  on schedule. `replace_input` restarts the clock at 0 on the new trajectory ("now" is its t = 0).
- `TapeFeeder` is index-driven (bam_controllers v1, 2026-07-28): tick k sends the sample at
  k x control_period whatever dt says, so a late tick slips the rest of the trajectory, and it refuses a new
  trajectory while running (`NotReplannable`).

Both command the sampled position and velocity with feed-forward = the trajectory's effort (or 0), and leave
stiffness and damping None so the board's configured gains close the loop. After the trajectory ends they
keep commanding its last sample, at rest, with `status == "done"`. The only way to `aborted` is the optional
path tolerance (off by default): the controller then holds where the arm was measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    check_dt,
    normalize_per_joint,
    param,
    per_joint,
)
from robot_controllers.core import Status


class NotReplannable(RuntimeError):
    """A controller refused a new input while it runs (the tape feeder); the bench records `refused`."""


def _normalize_guard_and_cap(config: object) -> None:
    for name in ("path_tolerance_rad", "max_effort_nm"):
        object.__setattr__(config, name, normalize_per_joint(getattr(config, name), name, allow_none=True))


@dataclass(frozen=True)
class JointTrajectoryConfig:
    """Parameters of the time-driven joint trajectory controller. Per-joint values follow the trajectory's
    `joint_names` order; a single value applies to every joint."""

    path_tolerance_rad: PerJoint | None = param(
        None,
        "rad",
        "Largest |reference - measured position| before the controller aborts and holds where the arm "
        "is; None = no guard (ros2_control JTC path tolerance).",
    )
    max_effort_nm: PerJoint | None = param(
        None, "N·m", "Torque cap sent with every command; None = the board's own cap."
    )

    def __post_init__(self) -> None:
        _normalize_guard_and_cap(self)


@dataclass(frozen=True)
class TapeFeederConfig:
    """Parameters of the index-driven tape feeder (bam_controllers v1)."""

    control_period_s: float = param(
        0.01,
        "s",
        "The tick period the tape assumes: tick k commands the trajectory at k x control_period_s, "
        "whatever the measured dt. Default 0.01 s = the rack host's 100 Hz (HostTiming's default rate).",
    )
    path_tolerance_rad: PerJoint | None = param(
        None,
        "rad",
        "Largest |reference - measured position| before it aborts and holds where the arm is; None = no "
        "guard. WHY off (v1 defaulted to 0.05 rad): as the JTC, so the comparison isolates the clock.",
    )
    max_effort_nm: PerJoint | None = param(
        None, "N·m", "Torque cap sent with every command; None = the board's own cap."
    )

    def __post_init__(self) -> None:
        period = float(self.control_period_s)
        if not (math.isfinite(period) and period > 0.0):
            raise ValueError(f"control_period_s must be finite and > 0, got {self.control_period_s!r}")
        object.__setattr__(self, "control_period_s", period)
        _normalize_guard_and_cap(self)


class _TrajectoryStreamer:
    """What both clocks share: the trajectory, the hold at its end and the optional path-tolerance guard."""

    status: Status
    _mode = ""

    def __init__(
        self,
        trajectory: JointTrajectory,
        *,
        max_effort_nm: PerJoint | None,
        path_tolerance_rad: PerJoint | None,
    ) -> None:
        self.status = "running"
        self.message = ""
        self.active_input = 0  # the running input: 0 = the one it was built with, +1 per accepted replace
        self._max_effort_setting = max_effort_nm
        self._path_tolerance_setting = path_tolerance_rad
        self._load(trajectory)

    def _load(self, trajectory: JointTrajectory) -> None:
        # Everything that can fail is computed before anything is assigned: a refused input changes nothing.
        sampler = TrajectorySampler(trajectory)
        names = sampler.joint_names
        max_effort = per_joint(self._max_effort_setting, names, "max_effort_nm")
        tolerance = per_joint(self._path_tolerance_setting, names, "path_tolerance_rad")
        # The constant fields of every command, frozen once by robot_msgs so a tick does not copy them.
        template = ServoCommand(
            name=names, position=np.zeros(len(names)), velocity=np.zeros(len(names)), max_effort=max_effort
        )
        self.trajectory = trajectory
        self._sampler = sampler
        self._names = template.name
        self._selector = JointSelector(names)
        self._tolerance = tolerance
        self._max_effort = template.max_effort
        self._rest = template.velocity
        self._hold: ServoCommand | None = None
        self.status = "running"
        self.message = ""

    def _command_at(self, time: float, state: JointState) -> ServoCommand:
        if self._hold is not None:
            self._selector.check(state)
            return self._hold
        position, velocity, effort = self._sampler.sample(time)
        if time + END_TOLERANCE_S >= self._sampler.end_time:
            # WHY at rest: the hold after the end must not carry the last sample's velocity into the
            # board's PD (bam_controllers v1 and ros2's JTC both hold the last point at zero velocity).
            velocity = self._rest
            if self.status == "running":
                self.status = "done"
                self.message = f"trajectory ended at {self._sampler.end_time:.3f} s; holding its last sample"
        if self._tolerance is not None:
            measured = self._selector.positions(state)
            error = np.abs(position - measured)
            off_path = ~(error <= self._tolerance)  # a NaN measurement counts as off the path
            if off_path.any():
                return self._abort(measured, error, off_path)
        else:
            self._selector.check(state)
        return ServoCommand(
            name=self._names,
            position=position,
            velocity=velocity,
            feedforward_effort=effort,
            max_effort=self._max_effort,
        )

    def _abort(self, measured: np.ndarray, error: np.ndarray, off_path: np.ndarray) -> ServoCommand:
        worst = int(np.argmax(np.where(off_path, np.nan_to_num(error, nan=np.inf), -1.0)))
        self.status = "aborted"
        self.message = (
            f"{self._names[worst]} was {math.degrees(error[worst]):.2f} deg off the path "
            f"(tolerance {math.degrees(self._tolerance[worst]):.2f} deg); holding where it was measured"
        )
        # WHY latched once, not re-read each tick: re-latching at the measured (sagging) pose walks the arm
        # down one sag per tick (bam_homing H3). Feed-forward 0 as bam_controllers v1's fault hold.
        self._hold = ServoCommand(
            name=self._names,
            position=np.array(measured, dtype=np.float64),
            velocity=self._rest,
            feedforward_effort=self._rest,
            max_effort=self._max_effort,
        )
        return self._hold

    def _diagnostics(self, clock_s: float) -> dict[str, float | str]:
        return {
            "mode": self._mode,
            "status": self.status,
            "clock_s": clock_s,
            "active_trajectory": self.active_input,
            "trajectory_end_s": self._sampler.end_time,
            "message": self.message,
        }


class JointTrajectoryController(_TrajectoryStreamer):
    """Time-driven joint trajectory controller (ros2_control's JTC law, without its splines).

    Built as `JointTrajectoryController(trajectory, config)`. Its clock is the time since the active
    trajectory began: each tick adds the measured dt (the first tick after an input samples t = 0, see
    `TrajectoryClock`), and the trajectory is sampled there on its own `t` axis, so a tick that comes
    3 periods late commands the sample 3 periods on. Times before the first sample hold the first sample.
    """

    _mode = "time_driven"

    def __init__(self, trajectory: JointTrajectory, config: JointTrajectoryConfig | None = None) -> None:
        self.config = config if config is not None else JointTrajectoryConfig()
        self._clock = TrajectoryClock()
        super().__init__(
            trajectory,
            max_effort_nm=self.config.max_effort_nm,
            path_tolerance_rad=self.config.path_tolerance_rad,
        )

    def tick(self, state: JointState, dt: float) -> tuple[ServoCommand, dict[str, float]]:
        return self._command_at(self._clock.advance(dt), state), {}

    def replace_input(self, message: JointTrajectory) -> None:
        """Run `message` from now: its t = 0 is the next tick. Accepted in any status (an abort included,
        as ros2's JTC takes a new goal after aborting one)."""
        self._load(message)
        self._clock.restart()
        self.active_input += 1

    @property
    def clock_s(self) -> float:
        return self._clock.time

    def diagnostics(self) -> dict[str, float | str]:
        return self._diagnostics(self._clock.time)


class TapeFeeder(_TrajectoryStreamer):
    """Index-driven trajectory streaming, as bam_controllers v1: one sample per tick, regardless of dt.

    Tick k (0-based) commands the trajectory at k x `control_period_s`; on a trajectory sampled at that
    period that is exactly v1's "emit sample k" (a trajectory on another grid is interpolated there instead
    of rejected). Its own clock never looks at dt, so a late tick slips the rest of the trajectory:
    `diagnostics()["schedule_slip_s"]` = the sum of the measured dts (from the first tick on) - its clock.
    """

    _mode = "index_driven"

    def __init__(self, trajectory: JointTrajectory, config: TapeFeederConfig | None = None) -> None:
        self.config = config if config is not None else TapeFeederConfig()
        self._period = self.config.control_period_s
        self._next_index = 0
        self._elapsed_s = 0.0
        super().__init__(
            trajectory,
            max_effort_nm=self.config.max_effort_nm,
            path_tolerance_rad=self.config.path_tolerance_rad,
        )

    def tick(self, state: JointState, dt: float) -> tuple[ServoCommand, dict[str, float]]:
        dt = check_dt(dt)
        if self._next_index > 0:  # the first tick's dt elapsed before the tape started (TrajectoryClock)
            self._elapsed_s += dt
        time = self._next_index * self._period
        self._next_index += 1
        return self._command_at(time, state), {}

    def replace_input(self, message: JointTrajectory) -> None:
        """Refuses while running or aborted (raises NotReplannable); once done it loads `message` from its
        start, as v1's `set_trajectory` accepts a new trajectory only while holding."""
        if self.status != "done":
            raise NotReplannable(
                f"tape_feeder refuses a new trajectory while {self.status}: it replays one trajectory "
                "sample by sample and takes the next only once holding (bam_controllers v1)"
            )
        self._load(message)
        self._next_index = 0
        self._elapsed_s = 0.0
        self.active_input += 1

    @property
    def clock_s(self) -> float:
        """The trajectory time of the sample it last commanded."""
        return max(self._next_index - 1, 0) * self._period

    @property
    def schedule_slip_s(self) -> float:
        return self._elapsed_s - self.clock_s

    def diagnostics(self) -> dict[str, float | str]:
        report = self._diagnostics(self.clock_s)
        report["elapsed_s"] = self._elapsed_s
        report["schedule_slip_s"] = self.schedule_slip_s
        report["ticks"] = self._next_index
        return report
