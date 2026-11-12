"""The servo boards: one moteus board per joint in position mode, ticking at 4 kHz (CONTRACT §3). numpy-free.

Per joint, every tick (fw bldc_servo_position.h, as `bam_pid_tunning.plants.servo_sim.controller_tick`):

    control_position += control_velocity * dt          (the latched setpoint walks at its velocity)
    tau = clip(kp (control_position - q_meas) + kd (control_velocity - qd_meas) + feedforward, +-limit)

- `kp`, `kd` are the configured gains unless the command sends `stiffness` / `damping` (absolute N*m/rad,
  N*m*s/rad, as robot_msgs' ServoCommand: the driver turns them into kp_scale/kd_scale); `position` NaN drops
  the position term; `velocity` / `feedforward_effort` missing or NaN count as 0; limit = min(cap,
  max_effort).
- The encoder: q_meas is the joint `encoder_delay_s` ago, quantised to 2 pi / counts; qd_meas is the
  derivative of that quantised position through a first-order low-pass at `velocity_filter_hz`. WHY the
  derivative of the quantised count, not a filtered true velocity: moteus estimates velocity from its counts,
  so kd multiplies the quantisation noise (the "kd buzz" that bounds a real joint's damping; bam_pid_tunning,
  2026-10-25).
- The watchdog: no command latched for `watchdog_s` -> moteus mode 11, tau = -kd qd_meas (a brake on the
  configured kd) until the next command latches. Each entry is reported once in `watchdog_entries`.

A command reaches a board through `schedule(command, latch_t)` and takes effect on the first tick at or after
latch_t (the bus delay is the caller's: the plant schedules at sent time + command delay). A joint the command
does not name gets nothing, so its watchdog keeps counting, as on a real bus. `defines` (joint -> rad, a
homing controller's "your position is now this", moteus set_output_exact) shift that board's position frame on
the same tick: the reported position jumps to the value, the joint does not move.

WHY pure-Python scalars per joint, not numpy arrays: a board tick is a dozen multiply-adds on two joints;
numpy's per-call overhead on 2-element arrays is ~10x the arithmetic, and the board runs 4000 times per
simulated second.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from robot_msgs.control import ServoCommand

OK_MODE = 0  # position mode, running
FAULT_MODE = 1  # a board fault (this model never raises one; the column exists for the metrics)
WATCHDOG_MODE = 11  # moteus mode 11: command timeout, braking
LATCH_TOLERANCE_S = 1e-9  # a command due at a tick's time latches on that tick despite float rounding


@dataclass(frozen=True)
class BoardConfig:
    """The moteus boards' configuration, per joint (shoulder, elbow), SI units. Defaults are CONTRACT §3."""

    kp_nm_rad: tuple[float, ...] = (300.0, 200.0)  # configured position gain (N*m/rad)
    kd_nm_s_rad: tuple[float, ...] = (
        7.5,
        4.0,
    )  # configured velocity gain, also the mode-11 brake (N*m*s/rad)
    torque_cap_nm: tuple[float, ...] = (12.0, 8.0)  # the board's torque limit (N*m)
    rate_hz: float = 4000.0  # control loop rate, = the physics rate (Hz)
    encoder_delay_s: float = 0.0005  # measurement delay of the position the loop sees (s)
    velocity_filter_hz: float = 200.0  # first-order low-pass on the velocity estimate (Hz)
    encoder_counts_per_rev: int = 2**14  # position quantisation: 2 pi / counts rad
    watchdog_s: float = 0.25  # no command latched for this long -> mode 11 (s)

    @property
    def dt_s(self) -> float:
        return 1.0 / self.rate_hz

    @property
    def quantum_rad(self) -> float:
        # WHY written as (2 pi) / counts: bit-identical to bam_pid_tunning's TAU / counts (the cross-check
        # oracle)
        return (2.0 * math.pi) / self.encoder_counts_per_rev if self.encoder_counts_per_rev > 0 else 0.0

    @property
    def delay_ticks(self) -> int:
        return int(round(self.encoder_delay_s * self.rate_hz))

    @property
    def filter_alpha(self) -> float:
        return 1.0 - math.exp(-2.0 * math.pi * self.velocity_filter_hz * self.dt_s)

    def check(self, joint_count: int) -> None:
        for name in ("kp_nm_rad", "kd_nm_s_rad", "torque_cap_nm"):
            if len(getattr(self, name)) != joint_count:
                raise ValueError(
                    f"BoardConfig.{name} has {len(getattr(self, name))} values for {joint_count} joints"
                )


@dataclass(frozen=True)
class _Latched:
    """One joint's share of a command, resolved to numbers the board loop uses."""

    position: float  # NaN = no position term
    velocity: float
    feedforward: float
    kp: float
    kd: float
    limit: float


def _value(values, index: int, default: float) -> float:
    if values is None:
        return default
    value = float(values[index])
    return default if math.isnan(value) else value


class ServoBoards:
    """Every joint's board. `reset`, then `schedule` commands and call `step` once per physics tick.

    After `step`: `tau` (applied torque per joint), `at_cap`, `mode`, `measured_position`, `measured_velocity`
    and `watchdog_entries` (joint indices that entered mode 11 on this tick) describe that tick.
    """

    def __init__(self, joint_names: Sequence[str], config: BoardConfig | None = None) -> None:
        self.joint_names = tuple(joint_names)
        self.config = config or BoardConfig()
        self.config.check(len(self.joint_names))
        self._index = {name: index for index, name in enumerate(self.joint_names)}
        self._dt = self.config.dt_s
        self._alpha = self.config.filter_alpha
        self._quantum = self.config.quantum_rad
        self._kp = [float(v) for v in self.config.kp_nm_rad]
        self._kd = [float(v) for v in self.config.kd_nm_s_rad]
        self._cap = [float(v) for v in self.config.torque_cap_nm]
        self.reset(0.0, [0.0] * len(self.joint_names))

    # ── setup ──────────────────────────────────────────────────────────────────────────────────────────────

    def reset(self, t: float, q: Sequence[float], *, feedforward: Sequence[float] | None = None) -> None:
        """Start every board at rest at `q` (rad) holding it with the configured gains and `feedforward`
        (N*m), as if that hold had latched at time `t` (s): the arm is already held when the controller takes
        over."""
        joints = len(self.joint_names)
        q = [float(v) for v in q]
        feedforward = [0.0] * joints if feedforward is None else [float(v) for v in feedforward]
        self._pending: deque[tuple[float, dict[int, _Latched]]] = deque()
        self._latched = [
            _Latched(q[j], 0.0, feedforward[j], self._kp[j], self._kd[j], self._cap[j]) for j in range(joints)
        ]
        self._control_position = list(q)
        self._offset = [0.0] * joints  # position frame shift from defines (rad)
        self._define_due: list[float | None] = [None] * joints
        self._last_latch_t = [float(t)] * joints
        self._history = [
            deque([q[j]] * (self.config.delay_ticks + 1), maxlen=self.config.delay_ticks + 1)
            for j in range(joints)
        ]
        self._previous_measured = [self._quantise(q[j]) for j in range(joints)]
        self.measured_position = list(self._previous_measured)
        self.measured_velocity = [0.0] * joints
        self.tau = [0.0] * joints
        self.at_cap = [False] * joints
        self.mode = [OK_MODE] * joints
        self.watchdog_entries: list[int] = []

    def _quantise(self, q: float) -> float:
        quantum = self._quantum
        return math.floor(q / quantum + 0.5) * quantum if quantum else q

    # ── commands ───────────────────────────────────────────────────────────────────────────────────────────

    def schedule(
        self, command: ServoCommand, latch_t: float, defines: Mapping[str, float] | None = None
    ) -> None:
        """Queue `command` (and any position `defines`) to latch at `latch_t` (s). Joints it names that these
        boards do not drive are ignored."""
        shares: dict[int, _Latched] = {}
        for i, name in enumerate(command.name):
            j = self._index.get(name)
            if j is None:
                continue
            cap = self._cap[j]
            max_effort = _value(command.max_effort, i, cap)
            shares[j] = _Latched(
                position=float(command.position[i]),
                velocity=_value(command.velocity, i, 0.0),
                feedforward=_value(command.feedforward_effort, i, 0.0),
                kp=_value(command.stiffness, i, self._kp[j]),
                kd=_value(command.damping, i, self._kd[j]),
                limit=min(cap, max_effort),
            )
        frame_shifts = {
            self._index[name]: float(value) for name, value in (defines or {}).items() if name in self._index
        }
        if self._pending and latch_t < self._pending[-1][0]:
            raise ValueError(f"commands must latch in order: {latch_t} is before {self._pending[-1][0]}")
        self._pending.append((float(latch_t), shares, frame_shifts))

    # ── one tick ───────────────────────────────────────────────────────────────────────────────────────────

    def step(self, t: float, q_true: Sequence[float]) -> list[float]:
        """One board tick at time `t` (s) with the joints truly at `q_true` (rad) at that instant; returns
        tau."""
        dt, alpha = self._dt, self._alpha
        pending = self._pending
        fresh: set[int] = set()
        while pending and pending[0][0] <= t + LATCH_TOLERANCE_S:
            latch_t, shares, frame_shifts = pending.popleft()
            for j, share in shares.items():
                self._latched[j] = share
                self._control_position[j] = share.position
                self._last_latch_t[j] = latch_t
                fresh.add(j)
                if self.mode[j] == WATCHDOG_MODE:
                    self.mode[j] = OK_MODE
            for j, value in frame_shifts.items():
                self._define_due[j] = value
        self.watchdog_entries = []
        watchdog_s = self.config.watchdog_s
        for j in range(len(self.joint_names)):
            history = self._history[j]
            history.append(float(q_true[j]))
            raw = self._quantise(history[0])
            if self._define_due[j] is not None:
                # WHY shift the filter's memory too: the joint did not move, so the velocity must not spike
                shift = (self._define_due[j] - raw) - self._offset[j]
                self._offset[j] += shift
                self._previous_measured[j] += shift
                if j not in fresh:  # a setpoint latched earlier is still in the old frame
                    self._control_position[j] += shift
                self._define_due[j] = None
            measured = raw + self._offset[j] if self._offset[j] else raw
            velocity = self.measured_velocity[j]
            velocity += alpha * ((measured - self._previous_measured[j]) / dt - velocity)
            self._previous_measured[j] = measured
            self.measured_position[j] = measured
            self.measured_velocity[j] = velocity

            if self.mode[j] != WATCHDOG_MODE and t - self._last_latch_t[j] >= watchdog_s:
                self.mode[j] = WATCHDOG_MODE
                self.watchdog_entries.append(j)
            latched = self._latched[j]
            if self.mode[j] == WATCHDOG_MODE:
                raw = -self._kd[j] * velocity
                limit = self._cap[j]
            else:
                if latched.position == latched.position:  # not NaN
                    control = self._control_position[j] + latched.velocity * dt
                    self._control_position[j] = control
                    position_term = latched.kp * (control - measured)
                else:
                    position_term = 0.0
                raw = position_term + latched.kd * (latched.velocity - velocity) + latched.feedforward
                limit = latched.limit
            if raw > limit:
                self.tau[j], self.at_cap[j] = limit, True
            elif raw < -limit:
                self.tau[j], self.at_cap[j] = -limit, True
            else:
                self.tau[j], self.at_cap[j] = raw, False
        return self.tau
