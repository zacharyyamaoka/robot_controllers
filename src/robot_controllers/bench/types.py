"""The bench contract: what a rung is, what an episode records, what a scored run returns.

Frozen before the build lanes start (CONTRACT.md). The plant and loop (bench lane) produce `EpisodeLog`s, the
metrics lane turns one into numbers named in `metrics.names`, the gates here judge them, and the W&B and site
lanes read `RunResult`s from results JSON. Nothing here imports MuJoCo.

Time is simulated seconds since the episode started. The host loop and the board are separate clocks on one time
axis: the board samples, its reply arrives at the host `reply_delay_s` later, the host computes and sends, the
command latches on the board `command_delay_s` after it was sent.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from robot_controllers.core import Box
from robot_controllers.timing import BusTiming

# ── timing and scene: what a rung sets up ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HostTiming:
    """The host loop and the bus. Defaults are the rack: 100 Hz, ~7 ms round trip (bam_fivebar RACK_BOARD).

    Amendment 4 (§4.2): `command_delay_s` may be one delay per plant joint (one adapter serves its boards in
    turn, so each joint's part of a command latches at sent + its own delay), and `command_delay_jitter_s`
    adds one draw u ~ U(-jitter, +jitter) to every joint's delay of each command, from the plant's own
    generator seeded from the episode seed (the host's wake-jitter draws never shift; the plant rounds u to
    whole board ticks). Latches stay first in, first out per joint, as on a CAN bus. The defaults are the
    episode as before.

    Amendment 5 (§5.2): `command_delay_steps = ((t_s, delay_s | per-joint tuple), ...)` changes the bus mid-run:
    a command sent at or after t_s uses that delay (the last step at or before its send time wins), so the twin
    can play a bus that slows down. `from_bus(bus, rate_hz)` is the timing a `BusTiming` implies.
    """

    rate_hz: float = 100.0
    command_delay_s: float | tuple[float, ...] = 0.0055  # host sends -> the board latches it (or per joint)
    reply_delay_s: float = 0.0015  # the board samples -> the host has it
    jitter_s: float = 0.0  # std of the host's wake jitter (gaussian, clipped to +-period/2), seeded
    stalls: tuple[tuple[float, float], ...] = ()  # (start_s, duration_s): the host does not wake in that window
    include_compute: bool = True  # send time = wake time + the measured tick compute time
    command_delay_jitter_s: float = 0.0  # +- bound of the uniform draw added to each command's delay (s)
    # (t_s, delay): from each t_s on, commands sent use that delay (one value or one per joint); () = never
    command_delay_steps: tuple[tuple[float, float | tuple[float, ...]], ...] = ()

    def __post_init__(self) -> None:
        delay = self.command_delay_s
        if np.ndim(delay) == 0:
            values: tuple[float, ...] = (float(delay),)
        else:
            values = tuple(float(v) for v in delay)
            object.__setattr__(self, "command_delay_s", values)  # a list becomes a (hashable) tuple
        if not values or not all(math.isfinite(v) and v >= 0.0 for v in values):
            raise ValueError(f"command_delay_s must be finite and >= 0 s, got {delay!r}")
        jitter = float(self.command_delay_jitter_s)
        if not (math.isfinite(jitter) and jitter >= 0.0):
            raise ValueError(f"command_delay_jitter_s must be finite and >= 0 s, got {jitter!r}")
        steps = []
        for step in self.command_delay_steps:
            at, step_delay = step
            at = float(at)
            step_values = (float(step_delay),) if np.ndim(step_delay) == 0 else tuple(float(v) for v in step_delay)
            if not math.isfinite(at) or not step_values or not all(
                math.isfinite(v) and v >= 0.0 for v in step_values
            ):
                raise ValueError(f"command_delay_steps entries are (t_s, delay_s >= 0), got {step!r}")
            steps.append((at, step_values[0] if np.ndim(step_delay) == 0 else step_values))
        if any(later[0] < earlier[0] for earlier, later in zip(steps, steps[1:], strict=False)):
            raise ValueError(f"command_delay_steps must be in time order, got {self.command_delay_steps!r}")
        object.__setattr__(self, "command_delay_steps", tuple(steps))

    @classmethod
    def from_bus(cls, bus: BusTiming, rate_hz: float, **fields: Any) -> HostTiming:
        """The host timing a bus implies at `rate_hz`: its command delay (per joint when it has one per
        joint), reply delay and jitter (as `command_delay_jitter_s`); every other field at its default or
        `fields`."""
        return cls(
            rate_hz=float(rate_hz),
            command_delay_s=bus.command_delay_s,
            reply_delay_s=bus.reply_delay_s,
            command_delay_jitter_s=bus.jitter_s,
            **fields,
        )

    @property
    def period_s(self) -> float:
        return 1.0 / self.rate_hz

    def command_delays(self, joint_count: int) -> tuple[float, ...]:
        """Each joint's command delay (s); a per-joint tuple of another length raises ValueError."""
        if np.ndim(self.command_delay_s) == 0:
            return (float(self.command_delay_s),) * joint_count
        if len(self.command_delay_s) != joint_count:
            raise ValueError(
                f"command_delay_s has {len(self.command_delay_s)} values for {joint_count} plant joints"
            )
        return tuple(self.command_delay_s)

    def delay_steps(self, joint_count: int) -> tuple[tuple[float, tuple[float, ...]], ...]:
        """`command_delay_steps` with each delay as one value per joint (a tuple of another length raises)."""
        out = []
        for at, delay in self.command_delay_steps:
            if np.ndim(delay) == 0:
                out.append((at, (float(delay),) * joint_count))
            elif len(delay) != joint_count:
                raise ValueError(f"command_delay_steps at {at:g} s has {len(delay)} values for {joint_count} joints")
            else:
                out.append((at, tuple(delay)))
        return tuple(out)


@dataclass(frozen=True)
class Push:
    """An external force on the tool, world frame (N), for a window."""

    start_s: float
    duration_s: float
    force_n: tuple[float, float, float]


STOP_SIDES: tuple[str, ...] = ("upper", "lower")


@dataclass(frozen=True)
class Scene:
    """What the plant contains for this rung. Plant-specific extras go in `extras`.

    Amendment 5 (§5.4, §5.10): `stops = ((joint_index, angle_rad, side), ...)` puts a stiff hard stop (a joint-range
    limit) on one side of a plant joint: side `upper` blocks the joint above `angle_rad`, `lower` below it. Every
    plant honours it (`ServoPlant._stopped_model`), records the stop's constraint torque per joint (`tau_stop`)
    and a `stop_touch` event, and refuses a start pose already past a stop. `joint_index` is in the plant's joint
    order.
    """

    belt: bool = False  # a moving conveyor belt under the tool
    belt_speed_m_s: float = 0.25
    contact_stiffness_n_m: float = 1.0e5  # an assumed belt contact, not a measured one
    contact_damping_n_s_m: float = 200.0
    pushes: tuple[Push, ...] = ()
    payload_kg: float = 0.0
    start_q: tuple[float, ...] | None = None  # rad; None = the plant's home
    extras: Mapping[str, Any] = field(default_factory=dict)
    stops: tuple[tuple[int, float, str], ...] = ()  # (plant joint index, angle rad, "upper" | "lower")

    def __post_init__(self) -> None:
        stops = []
        for stop in self.stops:
            joint, angle, side = stop
            if side not in STOP_SIDES or not math.isfinite(float(angle)) or int(joint) < 0:
                raise ValueError(f"a stop is (joint_index >= 0, finite angle_rad, 'upper' | 'lower'), got {stop!r}")
            stops.append((int(joint), float(angle), str(side)))
        object.__setattr__(self, "stops", tuple(stops))


# ── gates and rungs ─────────────────────────────────────────────────────────────────────────────────────────

Op = Literal["<=", ">=", "=="]


@dataclass(frozen=True)
class Gate:
    """A pass/fail line on one metric. A metric that was not measured fails the gate ("not measured")."""

    metric: str  # a name in metrics.names
    op: Op
    value: float
    why: str  # one line: what failing it would mean
    # WHY "if_measured": some metrics do not exist for some controllers by design (a host-torque impedance law never
    # commands a position, so it has no command jump). Such a gate passes as "n/a" instead of failing as "not
    # measured"; every other gate keeps "coverage counts" (2026-11-12, after the build lanes found the case).
    applies: Literal["always", "if_measured"] = "always"

    @property
    def label(self) -> str:
        return f"{self.metric} {self.op} {self.value:g}"


@dataclass(frozen=True)
class GateResult:
    gate: Gate
    value: float | None
    passed: bool  # a not-measured value is a fail (coverage counts)
    note: str = ""  # "not measured", "unsupported: refuses replanning", ...


def judge(gate: Gate, value: float | None, note: str = "") -> GateResult:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        if gate.applies == "if_measured":
            return GateResult(gate, None, True, note or "n/a for this controller")
        return GateResult(gate, None, False, note or "not measured")
    v = float(value)
    ok = {"<=": v <= gate.value, ">=": v >= gate.value, "==": v == gate.value}[gate.op]
    return GateResult(gate, v, bool(ok), note)


InputScript = Callable[[np.ndarray], Sequence[tuple[float, Any]]]
"""start joint positions (rad) -> [(t_s, message), ...]: the first at t = 0 builds the controller; later ones are
replans (`Replannable.replace_input`) or, for a non-replannable controller, refusals."""


@dataclass(frozen=True)
class Rung:
    """One benchmark environment of one box: an input script, a scene, timing, gates and one objective.

    `id` is readable and stable ("R3.BeltImpact@0.10m/s"); its identity is the id plus its gates, never the
    run (house rule: an environment's identity is fixed).
    """

    id: str
    title: str  # "Belt impact at 0.10 m/s"
    box: Box
    family: str  # free_space | contact | disturbance | replanning | timing | sequence | sync
    description: str  # two sentences: what happens and what it tests
    duration_s: float
    inputs: InputScript
    scene: Scene = field(default_factory=Scene)
    timing: HostTiming = field(default_factory=HostTiming)
    gates: tuple[Gate, ...] = ()
    objective: str = "tracking_p95_deg"  # the one number this rung ranks controllers on
    minimize: bool = True
    params: Mapping[str, float | str] = field(default_factory=dict)  # shown on the site and in W&B config
    order: int = 0  # ladder position, simplest first
    plant: str = "planar_arm"  # the plant run_episode builds for it: planar_arm | ur5e (amendment 4, §4.5)


# ── the episode log: what the bench records, what the metrics read ─────────────────────────────────────────


@dataclass
class EpisodeLog:
    """One controller on one rung for one seed. Arrays, SI units, NaN where a command field was None.

    Host ticks (K rows): one per command actually sent.
    Plant samples (N rows): every board/physics step (or a fixed decimation of it, at least 1 kHz).
    """

    controller: str
    rung: str
    seed: int
    joint_names: tuple[str, ...]
    nominal_period_s: float
    # host ticks
    tick_t: np.ndarray  # (K,) host wake time
    sent_t: np.ndarray  # (K,) time the command went on the bus
    state_stamp: np.ndarray  # (K,) board sample time of the state the tick used
    state_seq: np.ndarray  # (K,) int seq of that state
    compute_s: np.ndarray  # (K,) measured wall time of controller.tick
    cmd_position: np.ndarray  # (K, J)
    cmd_velocity: np.ndarray  # (K, J)
    cmd_feedforward: np.ndarray  # (K, J)
    cmd_stiffness: np.ndarray  # (K, J)
    cmd_damping: np.ndarray  # (K, J)
    cmd_max_effort: np.ndarray  # (K, J)
    ref_position: np.ndarray | None  # (K, J) the active input on the intended schedule, at sent_t; None = no ref
    ref_velocity: np.ndarray | None  # (K, J)
    ref_tool_pos: np.ndarray | None  # (K, 3) forward kinematics of ref_position
    status: np.ndarray  # (K,) str: running | done | aborted
    input_index: np.ndarray  # (K,) int: which input of the script is active (0 = first)
    # plant samples
    t: np.ndarray  # (N,)
    q: np.ndarray  # (N, J)
    qd: np.ndarray  # (N, J)
    tau: np.ndarray  # (N, J) motor torque applied
    at_cap: np.ndarray  # (N, J) bool: torque clipped at max_effort / the board cap
    board_mode: np.ndarray  # (N, J) int: 0 ok, 1 fault, 11 watchdog timeout (moteus codes)
    tool_pos: np.ndarray  # (N, 3) world
    contact_force: np.ndarray  # (N, 3) world, on the tool from the belt; zeros when no belt
    in_contact: np.ndarray  # (N,) bool
    external_force: np.ndarray  # (N, 3) push applied to the tool
    # what happened
    events: list[dict[str, Any]] = field(default_factory=list)  # {"t", "kind", ...}: replan, refused,
    # stall_start, stall_end, push_start, push_end, touch, watchdog, abort
    meta: dict[str, Any] = field(default_factory=dict)  # plant name, timing, board config, rung params, targets
    # the torque split (CONTRACT amendment 3, §3.4); None = this log has none (a plant or log from before it)
    tau_ff: np.ndarray | None = None  # (N, J) the feed-forward field the board added (0 in watchdog mode 11)
    tau_fb_board: np.ndarray | None = None  # (N, J) the board's PD term kp (p - q) + kd (v - qd), pre-clip
    latched_tick: np.ndarray | None = None  # (N, J) int: the host-tick row whose command ran (-1 = none)
    split_ff: np.ndarray | None = None  # (K, J) the controller's last_split() feed-forward; NaN = it has none
    split_fb_host: np.ndarray | None = None  # (K, J) ... and the host feedback inside the feed-forward field
    # amendment 5 (§5.4): the hard stops' constraint torque on each joint (N·m, signed, 0 off the stop); None = the
    # scene had no stops
    tau_stop: np.ndarray | None = None  # (N, J)

    def event_times(self, kind: str) -> list[float]:
        return [float(e["t"]) for e in self.events if e.get("kind") == kind]


# ── a scored run: what results JSON holds per (controller, rung, seed) ──────────────────────────────────────


@dataclass
class RunResult:
    controller: str
    rung: str
    seed: int
    metrics: dict[str, float | int | None]
    gates: list[dict[str, Any]]  # [{"metric", "op", "value", "why", "measured", "passed", "note"}]
    passed: bool  # every gate passed
    objective: float | None
    trace: dict[str, list[float]] = field(default_factory=dict)  # downsampled for plots (results-1 schema)
    events: list[dict[str, Any]] = field(default_factory=list)
    wandb_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def gate_rows(results: Sequence[GateResult]) -> list[dict[str, Any]]:
    return [
        {
            "metric": r.gate.metric,
            "op": r.gate.op,
            "value": r.gate.value,
            "why": r.gate.why,
            "applies": r.gate.applies,
            "measured": r.value,
            "passed": r.passed,
            "note": r.note,
        }
        for r in results
    ]
