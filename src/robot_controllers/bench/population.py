"""The population: sixteen controller/driver configurations, each a `layout_spec` with one more knob, where its
lookahead comes from (CONTRACT amendment 5, §5.5).

    from robot_controllers.bench.population import POPULATION, config_spec
    from robot_controllers.bench.scenarios import base_scenario
    spec = config_spec("P4s", base_scenario("bam_fivebar"))     # model FF at plan time, board PD, calibrated lookahead

Zach, 2026-11-20: "we start with like feedback given and feedforward given. And we're interested in basically like
the layout ... a bunch of different controller/driver configurations (that's the population)".

| # | feed-forward | feedback | lookahead |
|---|---|---|---|
| P1 | none | board PD | calibrated |
| P2 | the trajectory's column (as built) | board PD | none |
| P3 | model at plan time | board PD | nominal |
| P4 | model at plan time | board PD | calibrated |
| P5 | model + Coulomb at plan time | board PD | calibrated |
| P6 | model at the host, measured state | board PD | calibrated |
| P7 | model at plan time | computed-torque gains re-sent each tick | calibrated |
| P8 | model at plan time | host spring + board damper | none |
| P9 | model at the host, measured state | computed torque on the host | none |
| P10 | model at plan time | PD on the host | none |

P2-P7 also come as `P<n>s` with `secant_mid_ff` interpolation (16 configs). "model" is the arm's `rigid_body`
(`rigid_body_coulomb` for P5), built with the scenario's `ModelError`; "board PD" is the board's own configured
gains, which the scenario's `gain_scale` scales on the plant.

Lookahead modes:

- `none`: 0;
- `nominal`: the datasheet bus (`calibration.datasheet_bus`, §5.4), whatever bus the twin runs: what a controller
  that trusts the datasheet samples ahead by (`nominal=` overrides it, e.g. `recommend` passes the user's bus);
- `calibrated`: the friction-aware bus delay (§5.2, `calibrate_bus(..., friction=True)`) from `calibrate_bus` on the
  arm's calibration target at that scenario's bus and host rate, run once per (arm, bus, rate)
  (`scenario_calibration`); the plain fit (the "best lookahead") is recorded beside it. WHY the bus delay (contract
  §5.2 as amended after the first full grade, 2026-11-20): at 50 Hz with the secant velocity the plain fit
  over-advances; P4s at base, tool p95, plain fit vs bus delay: planar 1.58 vs 1.36 mm, UR5e 4.65 vs 1.80 mm,
  pendulum 15.98 vs 11.61 mm, five-bar 12.05 vs 12.08 mm. Friction lag is not a pure delay, so sampling ahead by it
  adds error elsewhere. WHY one calibration per bus and rate, not per config: it is a routine the rig runs once, with
  one recipe (the `robot-controllers calibrate` one: the trajectory's effort, board PD, secant_mid_ff, on the nominal
  model and gains), and every config then reads the same number.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from robot_controllers.core import ControllerSpec
from robot_controllers.timing import BusTiming, lookahead_from

POPULATION: tuple[str, ...] = (
    "P1", "P2", "P2s", "P3", "P3s", "P4", "P4s", "P5", "P5s", "P6", "P6s", "P7", "P7s", "P8", "P9", "P10",
)
# WHY these on a real arm (§5.5 as amended after the grade, 2026-11-20): every config whose feedback the board closes
# and that has a feed-forward. Never P1 (no feed-forward: the arm sags) nor P8-P10 (feedback on the host, which
# saturated the boards at 50 Hz in every sim). The first list (P2, P4, P4s, P5s) was a guess that excluded the grade's
# best accuracy on three arms (P7 / P7s: gains re-sent each tick, still closed by the board) and on the pendulum (P3s).
REAL_SAFE: tuple[str, ...] = ("P2", "P2s", "P3", "P3s", "P4", "P4s", "P5", "P5s", "P6", "P6s", "P7", "P7s")
LOOKAHEAD_MODES: tuple[str, ...] = ("none", "nominal", "calibrated")
# The calibration recipe (the CLI's `robot-controllers calibrate`): the trajectory's effort, board PD, secant_mid_ff
CALIBRATION_LAYOUT: tuple[str, str, str, str, str] = (
    "trajectory", "trajectory", "board_gains", "board", "secant_mid_ff",
)


@dataclass(frozen=True)
class Config:
    """One population member: `layout_spec`'s models and placements, its interpolation and its lookahead mode."""

    id: str
    ff_model: str
    ff_at: str
    fb_model: str
    fb_at: str
    lookahead_mode: str
    interpolation: str = "tangent"
    title: str = ""

    @property
    def needs_calibration(self) -> bool:
        return self.lookahead_mode == "calibrated"

    @property
    def feedforward_needed(self) -> str:
        """The feed-forward a setup must have to run it: `none`, `trajectory` (an effort column from any model),
        `rigid_body` or `rigid_body_coulomb` (`recommend`)."""
        if self.ff_model == "none":
            return "none"
        if self.ff_model == "trajectory":
            return "trajectory"
        return self.ff_model


_BASE: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    ("P1", "none", "none", "board_gains", "board", "calibrated", "no feed-forward, board PD, calibrated lookahead"),
    ("P2", "trajectory", "trajectory", "board_gains", "board", "none", "the trajectory's column (as built), board PD"),
    ("P3", "rigid_body", "plan", "board_gains", "board", "nominal",
     "model at plan time, board PD, the datasheet's lookahead"),
    ("P4", "rigid_body", "plan", "board_gains", "board", "calibrated",
     "model at plan time, board PD, calibrated lookahead"),
    ("P5", "rigid_body_coulomb", "plan", "board_gains", "board", "calibrated",
     "model + Coulomb at plan time, board PD, calibrated lookahead"),
    ("P6", "rigid_body", "host_measured", "board_gains", "board", "calibrated",
     "model at the host on the measured state, board PD, calibrated lookahead"),
    ("P7", "rigid_body", "plan", "ctc", "board_scheduled", "calibrated",
     "model at plan time, computed-torque gains re-sent each tick, calibrated lookahead"),
    ("P8", "rigid_body", "plan", "pd", "split", "none", "model at plan time, host spring + board damper"),
    ("P9", "rigid_body", "host_measured", "ctc", "host", "none",
     "model at the host on the measured state, computed torque on the host"),
    ("P10", "rigid_body", "plan", "pd", "host", "none", "model at plan time, PD on the host"),
)
SECANT_PAIRED = ("P2", "P3", "P4", "P5", "P6", "P7")


def _configs() -> dict[str, Config]:
    out: dict[str, Config] = {}
    for cid, ff_model, ff_at, fb_model, fb_at, mode, title in _BASE:
        out[cid] = Config(cid, ff_model, ff_at, fb_model, fb_at, mode, "tangent", title)
        if cid in SECANT_PAIRED:
            out[f"{cid}s"] = Config(
                f"{cid}s", ff_model, ff_at, fb_model, fb_at, mode, "secant_mid_ff", f"{title}, secant_mid_ff"
            )
    return {cid: out[cid] for cid in POPULATION}


CONFIGS: Mapping[str, Config] = _configs()


def config(config_id: str) -> Config:
    try:
        return CONFIGS[config_id]
    except KeyError:
        raise KeyError(f"no config {config_id!r}; the population is {POPULATION}") from None


# ── the calibration a "calibrated" config reads ─────────────────────────────────────────────────────────────


def _ms(values: Any) -> list[float]:
    return [round(1e3 * float(v), 6) for v in values]


@functools.lru_cache(maxsize=128)
def scenario_calibration(arm: str, bus: BusTiming, host_rate_hz: float, seed: int = 0) -> dict[str, Any]:
    """The calibration of `arm` on `bus` at `host_rate_hz` (§5.2): two xcorr fits of one zero-lookahead run of the
    calibration recipe on the arm's calibration target, the friction-aware one (the bus delay: what is compared with
    the datasheet, and what "calibrated" configs sample ahead by, `lookahead_s`) and the plain one (the "best
    lookahead", recorded beside it). JSON-safe; seconds as tuples, ms as lists."""
    from robot_controllers.bench.layouts import layout_spec
    from robot_controllers.bench.scenarios import Scenario
    from robot_controllers.calibration import calibrate_bus, datasheet_bus
    from robot_controllers.models import ModelError

    scenario = Scenario(arm, f"{arm}/calibration", host_rate_hz, bus, ModelError(), 1.0, axis="calibration")
    rung = scenario.rung()
    count = scenario.joints
    ff_model, ff_at, fb_model, fb_at, interpolation = CALIBRATION_LAYOUT

    def spec_at(lookahead: float | tuple[float, ...]) -> ControllerSpec:
        return layout_spec(ff_model, ff_at, fb_model, fb_at, lookahead, interpolation, arm=arm)

    nominal = datasheet_bus(count)
    best = calibrate_bus(spec_at, rung, nominal=nominal, method="xcorr", seed=seed)
    delay = calibrate_bus(spec_at, rung, nominal=nominal, method="xcorr", seed=seed, friction=True)
    best_s = tuple(float(v) for v in best.measured.per_joint(count))
    delay_s = tuple(float(v) for v in delay.measured.per_joint(count))
    return {
        "rung": rung.id,
        "method": "xcorr",
        "lookahead": "bus_delay",
        "lookahead_s": delay_s,
        "lookahead_ms": _ms(delay_s),
        "recipe": "ff=trajectory@trajectory+fb=board_gains@board+interp=secant_mid_ff, zero lookahead",
        "best_lookahead_s": best_s,
        "bus_delay_s": delay_s,
        "best_lookahead_ms": _ms(best_s),
        "bus_delay_ms": _ms(delay_s),
        "datasheet_ms": _ms(nominal.per_joint(count)),
        "twin_bus_ms": _ms(bus.per_joint(count)),
        "bus_delay_minus_datasheet_ms": _ms(delay.delta_s),
        "bus_within_datasheet": bool(delay.within),
        "datasheet_source": nominal.source,
        "why": best.why,
        "bus_why": delay.why,
    }


def calibration_for(scenario: Any) -> dict[str, Any]:
    """`scenario_calibration` at the scenario's arm, bus and host rate."""
    return scenario_calibration(scenario.arm, scenario.bus, scenario.host_rate_hz)


def _calibrated_lookahead(calibration: Any, count: int) -> tuple[float, ...]:
    """The lookahead a "calibrated" config reads from a calibration record (this module's dict: `lookahead_s`, else
    `lookahead_ms`), a `calibration.Calibration`, a `BusTiming` or plain seconds."""
    if isinstance(calibration, Mapping):
        values = calibration.get("lookahead_s")
        if values is None:
            values = [v / 1e3 for v in calibration["lookahead_ms"]]
        return tuple(float(v) for v in values)
    measured = getattr(calibration, "measured", calibration)
    if isinstance(measured, BusTiming):
        return tuple(float(v) for v in lookahead_from(measured, count))
    if isinstance(measured, (int, float)):
        return (float(measured),) * count
    return tuple(float(v) for v in measured)


def lookahead_for(
    config_id: str,
    scenario: Any,
    calibration: Any = None,
    *,
    nominal: BusTiming | None = None,
) -> float | tuple[float, ...]:
    """The lookahead `config_id` samples ahead by on `scenario` (the module docstring has the three modes)."""
    from robot_controllers.calibration import datasheet_bus

    member = config(config_id)
    count = scenario.joints
    if member.lookahead_mode == "none":
        return 0.0
    if member.lookahead_mode == "nominal":
        return tuple(float(v) for v in lookahead_from(nominal or datasheet_bus(count), count))
    values = _calibrated_lookahead(calibration if calibration is not None else calibration_for(scenario), count)
    if len(values) != count:
        raise ValueError(f"a calibration of {len(values)} joints for {scenario.arm}'s {count}")
    return values


def config_spec(
    config_id: str,
    scenario: Any,
    calibration: Any = None,
    *,
    nominal: BusTiming | None = None,
) -> ControllerSpec:
    """The `layout_spec` of `config_id` on `scenario`'s arm with its model error and gains (§5.5). `calibration`
    (default: `calibration_for(scenario)`, run once per arm, bus and rate) gives a "calibrated" config's lookahead;
    `nominal` (default: the arm's datasheet bus) a "nominal" one's."""
    from robot_controllers.bench.layouts import layout_spec

    member = config(config_id)
    lookahead = lookahead_for(config_id, scenario, calibration, nominal=nominal)
    return layout_spec(
        member.ff_model,
        member.ff_at,
        member.fb_model,
        member.fb_at,
        lookahead,
        member.interpolation,
        arm=scenario.arm,
        model_error=scenario.model_error,
        gain_scale=scenario.gain_scale,
    )
