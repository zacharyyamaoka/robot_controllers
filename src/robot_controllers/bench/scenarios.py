"""Scenarios: one fixed target at one host rate, on one bus, with a feed-forward this wrong and board gains this stiff
(CONTRACT amendment 5, §5.4).

    from robot_controllers.bench.scenarios import base_scenario, grid
    base = base_scenario("bam_fivebar")       # path@nominal, 50 Hz, the datasheet bus, ff±10 %, gains x1
    base.id                                   # 'bam_fivebar/path@nominal·50Hz·bus=datasheet·ff±10%·gains×1'
    rung = base.rung()                        # the target's Rung at that rate and bus, its effort column as given
    log = run_episode(spec, rung, plant_factory=base.plant_factory())   # the plant's boards at the scenario's gains

Zach, 2026-11-20: "How good or bad is the feedforward/feedback we are given, Host Rate, Speed of trajectories,
Whether its running on host/driver, whether its on plan or live ... fixed target trajectories".

A scenario is the twin's truth plus what the controller is given:

- `bus` is the bus the twin runs (the truth). The base runs the datasheet bus (§5.4: the fdcanusb's per-device
  exchange time, the boards served in joint order); the [bus] axis runs others while the controller's "nominal"
  lookahead still reads the datasheet (`population.config_spec`), which is the comparison Zach asked for ("compare
  that against reality").
- `model_error` is how wrong every model the controller is given is. WHY the trajectory's effort column too: on the
  rig the planner fills that column from the same identified model the controller has, so a 10 % wrong model is 10 %
  wrong in the column (P2's feed-forward) as well; `rung()` refills it with the arm's `rigid_body` model under the
  scenario's error. The plant never changes.
- `gain_scale` multiplies the plant's board kp and kd (`plant_factory()`, `scaled_board`) and the gains a host-side
  PD / CTC is built from (`layout_spec(gain_scale=)`).

The grid (`grid(arm)`, the base first; one axis at a time around the base, plus one 2-D grid):

- [speed] path@slow, path@fast; [rate] 25, 100, 200 Hz; [model_error] 0, 30 %; [gains] x0.5, x2;
- [bus] ideal (5.5 ms every joint, the twin rack), the rig as measured (2.2 ms per board in turn: seven boards in
  ~15.5 ms, `rig_bus`), the rig + ±1 ms jitter;
- [event] the collision target; planar only, the belt contact R3 at 0.10 m/s;
- [rate_error] host rate {25, 50, 100, 200} x model error {0, 10, 30 %} at nominal speed, less the points another
  axis already has (an id appears once).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from robot_msgs.trajectory import JointTrajectory

from robot_controllers.bench.types import Gate, Rung
from robot_controllers.models import ModelError
from robot_controllers.timing import BusTiming

BASE_RATE_HZ = 50.0  # the rig's host rate (§5.4)
BASE_MODEL_ERROR = ModelError(0.1)
SPEEDS_AXIS: tuple[str, ...] = ("slow", "fast")
RATES_AXIS_HZ: tuple[float, ...] = (25.0, 100.0, 200.0)
MODEL_ERRORS_AXIS: tuple[float, ...] = (0.0, 0.3)
GAINS_AXIS: tuple[float, ...] = (0.5, 2.0)
RATE_ERROR_RATES_HZ: tuple[float, ...] = (25.0, 50.0, 100.0, 200.0)
RATE_ERROR_ERRORS: tuple[float, ...] = (0.0, 0.1, 0.3)
AXES: tuple[str, ...] = ("base", "speed", "rate", "model_error", "gains", "bus", "event", "rate_error")
# The bus axis (§5.4). WHY 15.5 ms / 7: the rig's fdcanusb serves seven boards in ~15.5 ms (2026-11-20), so each
# board's command lands ~2.2 ms after the previous one; an arm's joints take the first slots in joint order.
IDEAL_DELAY_S = 0.0055
RIG_EXCHANGE_S = 0.0155
RIG_DEVICES = 7
RIG_JITTER_S = 0.001
IDEAL_SOURCE = "twin:ideal (5.5 ms every joint, the twin rack)"
RIG_SOURCE = "measured:rig (seven boards in ~15.5 ms, 2026-11-20)"
BELT_RUNG = "R3.BeltImpact@0.10m/s"  # [event], planar only (§5.4)
# WHY a gate on detection (§5.6: "None = never: a fail"): a config that never notices the stop cannot be recommended
# for that scenario, whatever its other numbers.
DETECT_GATE = Gate(
    "collision_detect_ms", ">=", 0.0, "the hard stop is detected at all (the residual crosses its threshold)"
)


def joint_count(arm: str) -> int:
    from robot_controllers.bench.arms import arm as registered

    return len(registered(arm).joint_names)


def datasheet_bus(arm: str) -> BusTiming:
    """The arm's datasheet bus (§5.4): the fdcanusb's per-device exchange time x the joint count, joint order."""
    from robot_controllers.calibration import datasheet_bus as datasheet

    return datasheet(joint_count(arm))


def ideal_bus() -> BusTiming:
    return BusTiming(IDEAL_DELAY_S, source=IDEAL_SOURCE)


def rig_bus(arm: str, jitter_s: float = 0.0) -> BusTiming:
    """The rig as measured: seven boards in ~15.5 ms, the arm's joints in the first slots; `jitter_s` ± per command."""
    count = joint_count(arm)
    bus = BusTiming.from_datasheet(RIG_EXCHANGE_S, RIG_DEVICES, slot_order=tuple(range(count)), source=RIG_SOURCE)
    return dataclasses.replace(bus, jitter_s=float(jitter_s)) if jitter_s else bus


def bus_label(bus: BusTiming) -> str:
    """How a scenario id names its bus: `datasheet`, `stagger` (the rig), a uniform `5.5ms`, else the per-joint ms;
    `±<ms>ms` for jitter."""
    if bus.source.startswith("datasheet"):
        text = "datasheet"
    elif bus.source.startswith("measured:rig"):
        text = "stagger"
    elif isinstance(bus.command_delay_s, tuple) and len(set(bus.command_delay_s)) > 1:
        text = "/".join(f"{1e3 * v:.3g}" for v in bus.command_delay_s) + "ms"
    else:
        value = bus.command_delay_s[0] if isinstance(bus.command_delay_s, tuple) else bus.command_delay_s
        text = f"{1e3 * float(value):.3g}ms"
    return text + (f"±{1e3 * bus.jitter_s:g}ms" if bus.jitter_s else "")


def _given_effort(message: JointTrajectory, arm: str, model_error: ModelError) -> JointTrajectory:
    from robot_controllers.bench.arms import arm as registered

    t = np.asarray(message.t, dtype=float)
    q = np.asarray(message.positions, dtype=float)
    qd = np.asarray(message.velocities, dtype=float) if message.velocities is not None else np.gradient(q, t, axis=0)
    qdd = (
        np.asarray(message.accelerations, dtype=float)
        if message.accelerations is not None
        else np.gradient(qd, t, axis=0)
    )
    effort = np.asarray(registered(arm).feedforward("rigid_body", model_error)(q, qd, qdd), dtype=float)
    return dataclasses.replace(message, effort=effort)


@functools.lru_cache(maxsize=256)
def _given_script(base: Any, arm: str, model_error: ModelError, start_q: tuple[float, ...]) -> tuple:
    out = []
    for t, message in base(np.asarray(start_q, dtype=float)):
        if isinstance(message, JointTrajectory) and message.effort is not None:
            message = _given_effort(message, arm, model_error)
        out.append((t, message))
    return tuple(out)


@dataclass(frozen=True)
class GivenEffortScript:
    """`base`'s input script with every JointTrajectory's effort column refilled by `arm`'s `rigid_body` model under
    `model_error`: the column a planner with that model would have filled (see the module docstring)."""

    base: Any
    arm: str
    model_error: ModelError

    def __call__(self, start_q: np.ndarray) -> list[tuple[float, Any]]:
        start = tuple(float(v) for v in np.asarray(start_q, dtype=float))
        return list(_given_script(self.base, self.arm, self.model_error, start))

    @property
    def duration_s(self) -> float:
        return float(self.base.duration_s)


def _scaled_plant(arm: str, gain_scale: float) -> Any:
    from robot_controllers.bench.arms import arm as registered
    from robot_controllers.bench.board import scaled_board

    spec = registered(arm)
    return spec.plant(scaled_board(spec.board_config(), gain_scale))


@dataclass(frozen=True)
class Scenario:
    """One graded situation (§5.4). `target` is a target id of `arm` (`<arm>/path@nominal`, `<arm>/collision`, ...;
    the planar belt is `planar_arm/R3.BeltImpact@0.10m/s`), `bus` the twin's truth, `axis` the grid axis it belongs
    to ("base" for the base)."""

    arm: str
    target: str
    host_rate_hz: float
    bus: BusTiming
    model_error: ModelError
    gain_scale: float = 1.0
    axis: str = "base"

    def __post_init__(self) -> None:
        object.__setattr__(self, "host_rate_hz", float(self.host_rate_hz))
        object.__setattr__(self, "gain_scale", float(self.gain_scale))
        if not self.target.startswith(f"{self.arm}/"):
            raise ValueError(f"target {self.target!r} is not one of {self.arm!r}'s")
        if not (math.isfinite(self.host_rate_hz) and self.host_rate_hz > 0.0):
            raise ValueError(f"host_rate_hz must be finite and > 0, got {self.host_rate_hz!r}")
        if not (math.isfinite(self.gain_scale) and self.gain_scale > 0.0):
            raise ValueError(f"gain_scale must be finite and > 0, got {self.gain_scale!r}")

    @property
    def id(self) -> str:
        """`<target>·<rate>Hz·bus=<label>·ff±<error>%·gains×<scale>` (readable and unique within a grid)."""
        return (
            f"{self.target}·{self.host_rate_hz:g}Hz·bus={bus_label(self.bus)}·ff±{100 * self.model_error.scale:g}%"
            + (
                f"·fric{100 * self.model_error.friction_scale:+g}%"
                if self.model_error.friction_scale is not None
                and self.model_error.friction_scale != self.model_error.scale
                else ""
            )
            + f"·gains×{self.gain_scale:g}"
        )

    @property
    def target_name(self) -> str:
        """The target id without the arm: `path@nominal`, `collision`, `calibration`, `R3.BeltImpact@0.10m/s`."""
        return self.target.split("/", 1)[1]

    @property
    def kind(self) -> str:
        """`path`, `collision`, `calibration` or `contact` (the planar belt)."""
        name = self.target_name
        if name.startswith("path@"):
            return "path"
        if name in ("collision", "calibration"):
            return name
        return "contact"

    @property
    def speed(self) -> str:
        name = self.target_name
        return name.split("@", 1)[1] if name.startswith("path@") else "nominal"

    @property
    def joints(self) -> int:
        return joint_count(self.arm)

    def free(self) -> Scenario:
        """The same scenario without the hard stop (the collision's threshold run): its target's free path."""
        if self.kind != "collision":
            return self
        free_target = self.base_rung().params.get("free_target", f"{self.arm}/path@nominal")
        return dataclasses.replace(self, target=str(free_target))

    def base_rung(self) -> Rung:
        """The target's own rung (100 Hz, its own bus), before this scenario's rate, bus and effort."""
        return _target_rung(self.arm, self.target)

    def rung(self) -> Rung:
        """The target's rung at this scenario's host rate (its timing gates with it), on its bus (command delay per
        joint, reply delay, jitter), with the effort column as the given model fills it; id = the scenario id."""
        return _scenario_rung(self)

    def plant_factory(self) -> Callable[[], Any] | None:
        """None at gains x1 (the rung's own plant), else a plant on `scaled_board(board, gain_scale)`."""
        if self.gain_scale == 1.0:
            return None
        return functools.partial(_scaled_plant, self.arm, self.gain_scale)

    def fields(self) -> dict[str, Any]:
        """The scenario's coordinates for a ledger row."""
        count = self.joints
        return {
            "arm": self.arm,
            "scenario": self.id,
            "axis": self.axis,
            "target": self.target,
            "kind": self.kind,
            "speed": self.speed,
            "host_rate_hz": self.host_rate_hz,
            "bus": {
                "label": bus_label(self.bus),
                "command_delay_ms": [round(1e3 * v, 6) for v in self.bus.per_joint(count)],
                "reply_delay_ms": round(1e3 * self.bus.reply_delay_s, 6),
                "jitter_ms": round(1e3 * self.bus.jitter_s, 6),
                "source": self.bus.source,
            },
            "model_error": self.model_error.scale,
            "model_error_seed": self.model_error.seed,
            "gain_scale": self.gain_scale,
        }


@functools.lru_cache(maxsize=64)
def _target_rung(arm: str, target: str) -> Rung:
    if target == f"planar_arm/{BELT_RUNG}":
        from robot_controllers.bench.rungs import rung

        return rung(BELT_RUNG)
    from robot_controllers.bench.arms import arm as registered

    targets = {t.id: t for t in registered(arm).targets()}
    try:
        return targets[target].rung()
    except KeyError:
        raise ValueError(f"no target {target!r}; {arm} has {sorted(targets)}") from None


@functools.lru_cache(maxsize=512)
def _scenario_rung(scenario: Scenario) -> Rung:
    from robot_controllers.bench.rungs import rate_variant

    base = scenario.base_rung()
    rung = base if float(base.timing.rate_hz) == scenario.host_rate_hz else rate_variant(base, scenario.host_rate_hz)
    bus = scenario.bus
    timing = dataclasses.replace(
        rung.timing,
        command_delay_s=bus.command_delay_s,
        reply_delay_s=bus.reply_delay_s,
        command_delay_jitter_s=bus.jitter_s,
    )
    inputs = rung.inputs
    if not scenario.model_error.is_nominal:
        inputs = GivenEffortScript(inputs, scenario.arm, scenario.model_error)
    gates = rung.gates
    if scenario.kind == "collision" and all(g.metric != DETECT_GATE.metric for g in gates):
        gates = (*gates, DETECT_GATE)
    return dataclasses.replace(
        rung,
        id=scenario.id,
        title=f"{base.title}, {scenario.host_rate_hz:g} Hz, bus {bus_label(bus)}, "
        f"ff±{100 * scenario.model_error.scale:g} %, gains ×{scenario.gain_scale:g}",
        timing=timing,
        inputs=inputs,
        gates=gates,
        params={
            **rung.params,
            "scenario": scenario.id,
            "axis": scenario.axis,
            "host_rate_hz": scenario.host_rate_hz,
            "bus": bus_label(bus),
            "model_error": scenario.model_error.scale,
            "gain_scale": scenario.gain_scale,
        },
    )


def base_scenario(arm: str) -> Scenario:
    """`path@nominal` at 50 Hz on the datasheet bus with a 10 % wrong model and the boards' own gains (§5.4)."""
    return Scenario(
        arm=arm,
        target=f"{arm}/path@nominal",
        host_rate_hz=BASE_RATE_HZ,
        bus=datasheet_bus(arm),
        model_error=BASE_MODEL_ERROR,
        gain_scale=1.0,
        axis="base",
    )


@functools.lru_cache(maxsize=16)
def grid(arm: str) -> tuple[Scenario, ...]:
    """Every scenario of `arm`, the base first (the module docstring has the axes)."""
    base = base_scenario(arm)
    out = [base]
    replace = dataclasses.replace
    out += [replace(base, target=f"{arm}/path@{speed}", axis="speed") for speed in SPEEDS_AXIS]
    out += [replace(base, host_rate_hz=rate, axis="rate") for rate in RATES_AXIS_HZ]
    out += [replace(base, model_error=ModelError(scale), axis="model_error") for scale in MODEL_ERRORS_AXIS]
    out += [replace(base, gain_scale=scale, axis="gains") for scale in GAINS_AXIS]
    out += [
        replace(base, bus=bus, axis="bus") for bus in (ideal_bus(), rig_bus(arm), rig_bus(arm, RIG_JITTER_S))
    ]
    out.append(replace(base, target=f"{arm}/collision", axis="event"))
    if arm == "planar_arm":
        out.append(replace(base, target=f"{arm}/{BELT_RUNG}", axis="event"))
    seen = {s.id for s in out}
    for rate in RATE_ERROR_RATES_HZ:
        for scale in RATE_ERROR_ERRORS:
            point = replace(base, host_rate_hz=rate, model_error=ModelError(scale), axis="rate_error")
            if point.id not in seen:
                seen.add(point.id)
                out.append(point)
    return tuple(out)
