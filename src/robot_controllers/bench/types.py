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

# ── timing and scene: what a rung sets up ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HostTiming:
    """The host loop and the bus. Defaults are the rack: 100 Hz, ~7 ms round trip (bam_fivebar RACK_BOARD)."""

    rate_hz: float = 100.0
    command_delay_s: float = 0.0055  # host sends -> the board latches it
    reply_delay_s: float = 0.0015  # the board samples -> the host has it
    jitter_s: float = 0.0  # std of the host's wake jitter (gaussian, clipped to +-period/2), seeded
    stalls: tuple[tuple[float, float], ...] = ()  # (start_s, duration_s): the host does not wake in that window
    include_compute: bool = True  # send time = wake time + the measured tick compute time

    @property
    def period_s(self) -> float:
        return 1.0 / self.rate_hz


@dataclass(frozen=True)
class Push:
    """An external force on the tool, world frame (N), for a window."""

    start_s: float
    duration_s: float
    force_n: tuple[float, float, float]


@dataclass(frozen=True)
class Scene:
    """What the plant contains for this rung. Plant-specific extras go in `extras`."""

    belt: bool = False  # a moving conveyor belt under the tool
    belt_speed_m_s: float = 0.25
    contact_stiffness_n_m: float = 1.0e5  # an assumed belt contact, not a measured one
    contact_damping_n_s_m: float = 200.0
    pushes: tuple[Push, ...] = ()
    payload_kg: float = 0.0
    start_q: tuple[float, ...] | None = None  # rad; None = the plant's home
    extras: Mapping[str, Any] = field(default_factory=dict)


# ── gates and rungs ─────────────────────────────────────────────────────────────────────────────────────────

Op = Literal["<=", ">=", "=="]


@dataclass(frozen=True)
class Gate:
    """A pass/fail line on one metric. A metric that was not measured fails the gate ("not measured")."""

    metric: str  # a name in metrics.names
    op: Op
    value: float
    why: str  # one line: what failing it would mean

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
    family: str  # free_space | contact | disturbance | replanning | timing | sequence
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
            "measured": r.value,
            "passed": r.passed,
            "note": r.note,
        }
        for r in results
    ]
