"""Joint impedance: a spring K and damper D per joint around a JointTrajectory, closed in one of three places.

Zach, 2026-11-12: "compare like doing impedance control ON moteus vs on the computer". One law,

    tau = K (q_ref - q) + D (qd_ref - qd) + effort_ref

rendered three ways (CONTRACT §2):

- `board` (runs_on board): the board gets position q_ref, velocity qd_ref, stiffness K, damping D and
  feed-forward effort_ref; its fast PD closes both K and D.
- `host` (runs_on host): position NaN, stiffness 0, damping 0, feed-forward = the whole law from the
  measured q and qd; the host closes K and D at its own rate, behind the bus delay.
- `host_spring` (runs_on split): position NaN, velocity qd_ref, stiffness 0, damping D, feed-forward =
  K (q_ref - q) + effort_ref; the host closes K, the board closes D on its own velocity estimate.

D = 2 zeta sqrt(K I) on the per-joint design inertia I unless an explicit damping is given. The reference is
sampled exactly as the time-driven JTC samples it (same clock, same hold at the end), and a new trajectory
replaces the old one with its t = 0 at the next tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

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

Location = Literal["board", "host", "host_spring"]
LOCATIONS: tuple[str, ...] = ("board", "host", "host_spring")

# Where each location renders the spring and the damper (diagnostics, and the site's host <-> board picture).
_SPRING_ON = {"board": "board", "host": "host", "host_spring": "host"}
_DAMPER_ON = {"board": "board", "host": "host", "host_spring": "board"}


@dataclass(frozen=True)
class JointImpedanceConfig:
    """Parameters of the joint impedance controller. Per-joint values follow the trajectory's `joint_names`
    order; a single value applies to every joint. Defaults are the planar arm's (shoulder, elbow)."""

    location: Location = param(
        "board",
        "",
        "Where the law closes: board (K and D sent to the board's PD), host (the whole law as a torque, "
        "board gains 0) or host_spring (K as a host torque, D on the board).",
    )
    stiffness_nm_rad: PerJoint = param((40.0, 25.0), "N·m/rad", "Joint spring stiffness K.")
    damping_ratio: float = param(
        0.8, "", "Damping ratio zeta the damping is designed for: D = 2 zeta sqrt(K I)."
    )
    inertia_kg_m2: PerJoint = param(
        (0.10, 0.03), "kg·m²", "Per-joint inertia I the damping is designed on (link + reflected rotor)."
    )
    damping_nm_s_rad: PerJoint | None = param(
        None, "N·m·s/rad", "Explicit joint damping D; overrides damping_ratio when set."
    )
    max_effort_nm: PerJoint | None = param(
        None, "N·m", "Torque cap sent with every command; None = the board's own cap."
    )

    def __post_init__(self) -> None:
        if self.location not in LOCATIONS:
            raise ValueError(f"location must be one of {LOCATIONS}, got {self.location!r}")
        ratio = float(self.damping_ratio)
        if not (math.isfinite(ratio) and ratio >= 0.0):
            raise ValueError(f"damping_ratio must be finite and >= 0, got {self.damping_ratio!r}")
        object.__setattr__(self, "damping_ratio", ratio)
        for name in ("stiffness_nm_rad", "inertia_kg_m2"):
            object.__setattr__(self, name, normalize_per_joint(getattr(self, name), name))
        for name in ("damping_nm_s_rad", "max_effort_nm"):
            object.__setattr__(self, name, normalize_per_joint(getattr(self, name), name, allow_none=True))

    def stiffness(self, joint_names: tuple[str, ...]) -> np.ndarray:
        """K per joint (N·m/rad)."""
        return per_joint(self.stiffness_nm_rad, joint_names, "stiffness_nm_rad")

    def damping(self, joint_names: tuple[str, ...]) -> np.ndarray:
        """D per joint (N·m·s/rad): the explicit damping, else 2 zeta sqrt(K I)."""
        if self.damping_nm_s_rad is not None:
            return per_joint(self.damping_nm_s_rad, joint_names, "damping_nm_s_rad")
        inertia = per_joint(self.inertia_kg_m2, joint_names, "inertia_kg_m2")
        return 2.0 * self.damping_ratio * np.sqrt(self.stiffness(joint_names) * inertia)


class JointImpedanceController:
    """A joint spring and damper around a JointTrajectory, closed on the board, the host, or split.

    Built as `JointImpedanceController(trajectory, config)`. Time-driven like the JTC: its clock adds each
    tick's measured dt (the first tick after an input samples t = 0), the reference is sampled there and held
    at rest after the end (`status == "done"`). It never aborts. The host locations read the measured
    position (and, for `host`, velocity) of the trajectory's joints from the state by name.
    """

    status: Status

    def __init__(self, trajectory: JointTrajectory, config: JointImpedanceConfig | None = None) -> None:
        self.config = config if config is not None else JointImpedanceConfig()
        self.location: str = self.config.location
        self.status = "running"
        self.message = ""
        self.active_input = 0
        self._clock = TrajectoryClock()
        renderers = {
            "board": self._render_board,
            "host": self._render_host,
            "host_spring": self._render_host_spring,
        }
        self._render = renderers[self.location]
        self._load(trajectory)

    def _load(self, trajectory: JointTrajectory) -> None:
        # Everything that can fail is computed before anything is assigned: a bad input changes nothing.
        sampler = TrajectorySampler(trajectory)
        names = sampler.joint_names
        count = len(names)
        stiffness = self.config.stiffness(names)
        damping = self.config.damping(names)
        max_effort = per_joint(self.config.max_effort_nm, names, "max_effort_nm")
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
        self._names = template.name
        self._selector = JointSelector(names)
        self._stiffness = template.stiffness
        self._damping = template.damping
        self._max_effort = template.max_effort
        self._no_position = template.position  # NaN: no position target, no position term on the board
        self._zeros = template.velocity  # zero gains (host) and the at-rest velocity of the hold
        self.status = "running"
        self.message = ""

    def tick(self, state: JointState, dt: float) -> tuple[ServoCommand, dict[str, float]]:
        time = self._clock.advance(dt)
        reference_position, reference_velocity, reference_effort = self._sampler.sample(time)
        if time + END_TOLERANCE_S >= self._sampler.end_time:
            reference_velocity = self._zeros  # WHY at rest: as the JTC, the hold after the end must not move
            if self.status == "running":
                self.status = "done"
                self.message = f"trajectory ended at {self._sampler.end_time:.3f} s; holding its last sample"
        return self._render(state, reference_position, reference_velocity, reference_effort), {}

    def _render_board(
        self,
        state: JointState,
        reference_position: np.ndarray,
        reference_velocity: np.ndarray,
        reference_effort: np.ndarray,
    ) -> ServoCommand:
        self._selector.check(state)
        return ServoCommand(
            name=self._names,
            position=reference_position,
            velocity=reference_velocity,
            feedforward_effort=reference_effort,
            stiffness=self._stiffness,
            damping=self._damping,
            max_effort=self._max_effort,
        )

    def _render_host(
        self,
        state: JointState,
        reference_position: np.ndarray,
        reference_velocity: np.ndarray,
        reference_effort: np.ndarray,
    ) -> ServoCommand:
        measured_position = self._selector.positions(state)
        measured_velocity = self._selector.velocities(state, "joint_impedance_host")
        feedforward = (
            self._stiffness * (reference_position - measured_position)
            + self._damping * (reference_velocity - measured_velocity)
            + reference_effort
        )
        # WHY velocity None: with damping 0 the board's velocity target has no effect, so it is not sent
        # (one register fewer on the bus). WHY stiffness/damping 0, not None: None keeps the board's gains.
        return ServoCommand(
            name=self._names,
            position=self._no_position,
            velocity=None,
            feedforward_effort=feedforward,
            stiffness=self._zeros,
            damping=self._zeros,
            max_effort=self._max_effort,
        )

    def _render_host_spring(
        self,
        state: JointState,
        reference_position: np.ndarray,
        reference_velocity: np.ndarray,
        reference_effort: np.ndarray,
    ) -> ServoCommand:
        measured_position = self._selector.positions(state)
        feedforward = self._stiffness * (reference_position - measured_position) + reference_effort
        # The board closes D (v - qd) at its own rate on its own velocity estimate, around v = qd_ref.
        return ServoCommand(
            name=self._names,
            position=self._no_position,
            velocity=reference_velocity,
            feedforward_effort=feedforward,
            stiffness=self._zeros,
            damping=self._damping,
            max_effort=self._max_effort,
        )

    def replace_input(self, message: JointTrajectory) -> None:
        """Run `message` from now: its t = 0 is the next tick (per-joint parameters are re-matched to its
        joints)."""
        self._load(message)
        self._clock.restart()
        self.active_input += 1

    @property
    def clock_s(self) -> float:
        return self._clock.time

    def diagnostics(self) -> dict[str, float | str]:
        report: dict[str, float | str] = {
            "mode": "time_driven",
            "location": self.location,
            "spring_on": _SPRING_ON[self.location],
            "damper_on": _DAMPER_ON[self.location],
            "status": self.status,
            "clock_s": self._clock.time,
            "active_trajectory": self.active_input,
            "trajectory_end_s": self._sampler.end_time,
            "message": self.message,
        }
        for name, stiffness, damping in zip(self._names, self._stiffness, self._damping, strict=True):
            report[f"stiffness_nm_rad.{name}"] = float(stiffness)
            report[f"damping_nm_s_rad.{name}"] = float(damping)
        return report
