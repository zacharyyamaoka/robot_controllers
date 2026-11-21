"""`recommend()`: tell it the driver, the state of your feed-forward and feedback, the host rate and the bus, and it
looks the best controller up in the graded ledger (CONTRACT amendment 5, §5.7).

    from robot_controllers.recommend import Setup, recommend
    from robot_controllers.calibration import datasheet_bus
    answer = recommend(Setup(arm="bam_fivebar", host_rate_hz=50, bus=datasheet_bus(6), feedforward="rigid_body",
                             model_error=0.1), objective="accuracy")
    answer.config, answer.why                    # "P4s", "P4s on bam_fivebar/path@nominal·50Hz·...: ..."
    answer.controller                            # a built ControllerSpec for that setup
    answer.expected["tool_p95_mm"]               # what it did in the twin there

    robot-controllers recommend --arm bam_fivebar --rate 50 --ff-error 0.1 --objective accuracy

Zach, 2026-11-20: "once you tell me what driver your using, and the state of your FF/FB and the host rate etc ... I
can tell you exactly what the best controller is you should be using. And perhaps if you want like to really optimize
for less impact, or better collision detection".

How it answers:

1. **The arm.** The setup's arm if the ledger graded it; else the graded arm with the nearest joint count (ties: the
   registry's order), and `why` says "nearest".
2. **The scenario.** Among that arm's scenarios where the objective is graded (accuracy and cost on the paths, impact
   on the collision and the belt, collision detection on the collision), the nearest to the setup on host rate (log
   scale: one octave = 1), model error (10 % = 1), speed (one step = 1), board gains (one octave = 1) and bus (the
   mean per-joint command-delay difference, 2 ms = 1, plus the jitter difference, 1 ms = 1). Ties: the base first.
3. **The config.** The best passing config there (`grade.rank`) that the driver can run (`DriverCaps.can_run`, its
   `max_rate_hz`) and that the setup's feed-forward can feed (`none` < `trajectory` < `gravity` < `rigid_body` <
   `rigid_body_coulomb`: P1 needs nothing, P2 an effort column, P5 the Coulomb term, the rest `rigid_body`); the next
   one is the runner-up. None of them passing gives `controller=None` and the reason.
4. **The controller** is built for the setup (`population.config_spec` on its arm, rate, model error and gains): a
   "nominal" lookahead reads the setup's bus; a "calibrated" one the twin's calibrated lookahead (the friction-aware
   bus delay, §5.2) on the matched scenario shifted by (your bus - that scenario's bus) per joint. WHY shifted, not re-run: recommend is a lookup; the routine
   that measures it on the real arm is `robot-controllers calibrate` (or the sweep on a real log, §5.9).
"""

from __future__ import annotations

import functools
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robot_controllers.core import ControllerSpec
from robot_controllers.models import MOTEUS, DriverCaps, Layout, ModelError
from robot_controllers.timing import BusTiming

FEEDFORWARD_ORDER: tuple[str, ...] = ("none", "trajectory", "gravity", "rigid_body", "rigid_body_coulomb")
SPEED_ORDER: tuple[str, ...] = ("slow", "nominal", "fast")
BUS_SCALE_MS = 2.0  # 2 ms of mean command-delay difference counts as much as one octave of host rate
JITTER_SCALE_MS = 1.0
ERROR_SCALE = 0.1  # 10 % of model error counts as one octave of host rate


@dataclass(frozen=True)
class Setup:
    """What you have (§5.7): the arm, the host rate, the bus (its datasheet or a calibration), the feed-forward you
    are given and how wrong it is (0.1 = 10 %), the driver, the board gains relative to the arm's nominal, the speed
    you run at; `joint_count` lets an arm the ledger has never seen be matched."""

    arm: str
    host_rate_hz: float
    bus: BusTiming
    feedforward: str = "rigid_body"
    model_error: float = 0.1
    driver_caps: DriverCaps = MOTEUS
    gain_scale: float = 1.0
    speed: str = "nominal"
    joint_count: int | None = None

    def __post_init__(self) -> None:
        if self.feedforward not in FEEDFORWARD_ORDER:
            raise ValueError(f"feedforward must be one of {FEEDFORWARD_ORDER}, got {self.feedforward!r}")
        if self.speed not in SPEED_ORDER:
            raise ValueError(f"speed must be one of {SPEED_ORDER}, got {self.speed!r}")
        if not (math.isfinite(float(self.host_rate_hz)) and float(self.host_rate_hz) > 0.0):
            raise ValueError(f"host_rate_hz must be finite and > 0, got {self.host_rate_hz!r}")


@dataclass(frozen=True)
class Recommendation:
    """The answer: the built controller and its config id (None when nothing fits), its metrics on the matched
    scenario, the runner-up config, one plain reason, and the evidence (the matched scenario and every candidate's
    numbers there, best first)."""

    controller: ControllerSpec | None
    config: str | None
    expected: dict[str, Any]
    runner_up: str | None
    why: str
    evidence: list[dict[str, Any]]
    scenario: str | None = None
    arm: str | None = None


# ── the ledger ─────────────────────────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=4)
def _ledger(path: str) -> dict[str, Any]:
    from robot_controllers.grade import load_ledger

    return load_ledger(Path(path))


def _default_path() -> str:
    from robot_controllers.grade import default_out_dir

    return str(default_out_dir() / "results.json")


# ── matching ───────────────────────────────────────────────────────────────────────────────────────────────


def _joint_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts.setdefault(row["arm"], int(row.get("joints") or len(row["bus"]["command_delay_ms"])))
    return counts


def _arm_order(name: str) -> int:
    from robot_controllers.bench.arms import ARM_MODULES

    names = [n for n, _ in ARM_MODULES]
    return names.index(name) if name in names else len(names)


def _bus_ms(bus: BusTiming, count: int) -> list[float]:
    delay = bus.command_delay_s
    if isinstance(delay, tuple) and len(delay) != count:
        mean = statistics.fmean(delay)
        return [1e3 * mean] * count
    return [1e3 * v for v in bus.per_joint(count)]


def distance(setup: Setup, row: Mapping[str, Any], count: int) -> float:
    """How far a ledger scenario is from the setup (the module docstring's weights)."""
    rate = abs(math.log2(float(setup.host_rate_hz) / float(row["host_rate_hz"])))
    error = abs(float(setup.model_error) - float(row["model_error"])) / ERROR_SCALE
    speed = abs(SPEED_ORDER.index(setup.speed) - SPEED_ORDER.index(row["speed"])) if row["kind"] == "path" else 0.0
    gains = abs(math.log2(float(setup.gain_scale) / float(row["gain_scale"])))
    mine = _bus_ms(setup.bus, count)
    theirs = [float(v) for v in row["bus"]["command_delay_ms"]]
    if len(theirs) != len(mine):
        theirs = [statistics.fmean(theirs)] * len(mine)
    bus = statistics.fmean(abs(a - b) for a, b in zip(mine, theirs, strict=True)) / BUS_SCALE_MS
    jitter = abs(1e3 * setup.bus.jitter_s - float(row["bus"]["jitter_ms"])) / JITTER_SCALE_MS
    return rate + error + speed + gains + bus + jitter


def _feeds(setup_ff: str, needed: str) -> bool:
    """Whether a setup with feed-forward `setup_ff` can run a config that needs `needed` (population.Config)."""
    have = FEEDFORWARD_ORDER.index(setup_ff)
    if needed == "none":
        return True
    if needed == "trajectory":
        return have >= FEEDFORWARD_ORDER.index("trajectory")
    return have >= FEEDFORWARD_ORDER.index(needed)


def _layout_of(config_id: str) -> Layout:
    from robot_controllers.bench.population import config

    member = config(config_id)
    return Layout(
        member.ff_at, member.fb_at, lookahead_s=0.001 if member.lookahead_mode != "none" else 0.0,
        interpolation=member.interpolation,
    )


def allowed(config_id: str, setup: Setup) -> tuple[bool, str]:
    """(can this setup run it, why not)."""
    from robot_controllers.bench.population import config

    member = config(config_id)
    caps = setup.driver_caps
    if not caps.can_run(_layout_of(config_id)):
        return False, f"the driver cannot run feedback at {member.fb_at} / feed-forward at {member.ff_at}"
    if float(setup.host_rate_hz) > float(caps.max_rate_hz):
        return False, f"the driver takes at most {caps.max_rate_hz:g} Hz"
    if not _feeds(setup.feedforward, member.feedforward_needed):
        return False, f"needs a {member.feedforward_needed} feed-forward (you have {setup.feedforward})"
    return True, ""


def _metric_text(row: Mapping[str, Any], names: Sequence[str]) -> str:
    from robot_controllers.metrics.names import info

    parts = []
    for name in names:
        value = row["metrics"].get(name)
        unit = info(name).unit
        parts.append(f"{name} {'n/a' if value is None else format(float(value), '.3g')}{' ' + unit if unit else ''}")
    return ", ".join(parts)


def _shifted_calibration(row: Mapping[str, Any], setup: Setup, count: int) -> tuple[float, ...] | None:
    calibration = row.get("calibration")
    if not calibration:
        return None
    best = [float(v) for v in calibration["lookahead_ms"]]  # what the config sampled ahead by (the bus delay, §5.2)
    twin = [float(v) for v in calibration["twin_bus_ms"]]
    mine = _bus_ms(setup.bus, count)
    if len(best) != count:
        mean_best, mean_twin = statistics.fmean(best), statistics.fmean(twin)
        best, twin = [mean_best] * count, [mean_twin] * count
    return tuple(max(0.0, (b + m - t) / 1e3) for b, m, t in zip(best, mine, twin, strict=True))


def recommend(
    setup: Setup,
    objective: str | Mapping[str, float] = "accuracy",
    *,
    ledger: Mapping[str, Any] | str | Path | None = None,
) -> Recommendation:
    """The best controller for `setup` on `objective` (a name of `grade.OBJECTIVES` or {name: weight}), looked up in
    `ledger` (default: the committed `results/grade/results.json`). The module docstring has the rules."""
    from robot_controllers.bench.population import config_spec
    from robot_controllers.bench.scenarios import Scenario
    from robot_controllers.grade import OBJECTIVE_KINDS, OBJECTIVES, objective_metrics, rank

    if isinstance(objective, str) and objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {tuple(OBJECTIVES)} or a dict of weights, got {objective!r}")
    document = (
        _ledger(_default_path() if ledger is None else str(ledger))
        if not isinstance(ledger, Mapping)
        else ledger
    )
    rows = [r for r in document["rows"] if r.get("source", "sim") == "sim"]
    counts = _joint_counts(rows)
    if not counts:
        return Recommendation(None, None, {}, None, "the ledger has no graded rows", [])
    notes: list[str] = []
    if setup.arm in counts:
        arm_name = setup.arm
        count = counts[arm_name]
    else:
        wanted = setup.joint_count
        if wanted is None:
            try:
                from robot_controllers.bench.arms import arm as registered

                wanted = len(registered(setup.arm).joint_names)
            except KeyError:
                wanted = None
        if wanted is None:
            return Recommendation(
                None, None, {}, None,
                f"{setup.arm!r} is not in the ledger and its joint count is unknown: pass Setup(joint_count=...)", [],
            )
        arm_name = min(counts, key=lambda a: (abs(counts[a] - int(wanted)), _arm_order(a)))
        count = counts[arm_name]
        notes.append(
            f"{setup.arm} is not in the ledger: the nearest arm by joint count is {arm_name} "
            f"({count} joints vs your {wanted})"
        )
    kinds = (
        OBJECTIVE_KINDS[objective]
        if isinstance(objective, str)
        else tuple(sorted({k for name in objective for k in OBJECTIVE_KINDS[name]}))
    )
    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["arm"] == arm_name and row["kind"] in kinds:
            by_scenario.setdefault(row["scenario"], []).append(row)
    if isinstance(objective, str):
        by_scenario = {sid: rs for sid, rs in by_scenario.items() if objective_metrics(objective, rs)}
    if not by_scenario:
        return Recommendation(
            None, None, {}, None,
            f"the ledger grades no {arm_name} scenario on {objective}" + ("; " + "; ".join(notes) if notes else ""),
            [], arm=arm_name,
        )
    def primary(sid: str) -> int:
        """0 when the scenario measures the objective's first metric (impact: the collision before the belt)."""
        if not isinstance(objective, str):
            return 0
        measured = objective_metrics(objective, by_scenario[sid])
        return OBJECTIVES[objective].index(measured[0]) if measured else len(OBJECTIVES[objective])

    scored = sorted(
        by_scenario,
        key=lambda sid: (
            round(distance(setup, by_scenario[sid][0], count), 9),
            by_scenario[sid][0]["axis"] != "base",
            primary(sid),
            sid,
        ),
    )
    scenario_id = scored[0]
    candidates = by_scenario[scenario_id]
    first = candidates[0]
    gap = distance(setup, first, count)
    order = rank(candidates, objective)
    names = (
        objective_metrics(objective, candidates)
        if isinstance(objective, str)
        else tuple(n for name in objective for n in OBJECTIVES[name])
    )
    refused: dict[str, str] = {}
    fitting = []
    for row in order:
        ok, reason = allowed(row["config"], setup)
        if ok:
            fitting.append(row)
        else:
            refused[row["config"]] = reason
    evidence: list[dict[str, Any]] = [
        {
            "scenario": scenario_id,
            "distance": round(gap, 4),
            "nearest_others": [{"scenario": sid, "distance": round(distance(setup, by_scenario[sid][0], count), 4)}
                               for sid in scored[1:4]],
        }
    ]
    ranked_ids = [r["config"] for r in order]
    for row in sorted(candidates, key=lambda r: (ranked_ids.index(r["config"]) if r["config"] in ranked_ids
                                                 else len(ranked_ids), r["config"])):
        ok, reason = allowed(row["config"], setup)
        evidence.append({
            "config": row["config"],
            "controller": row["controller"],
            "passed": row["passed"],
            "failed_gates": row["failed_gates"],
            "can_run": ok,
            "why_not": reason or None,
            **{name: row["metrics"].get(name) for name in names},
        })
    where = f"{scenario_id} (the nearest graded scenario, distance {gap:.2g})"
    prefix = "; ".join(notes) + ". " if notes else ""
    if not fitting:
        passing = [r["config"] for r in candidates if r["passed"]]
        why = (
            f"{prefix}no config that this driver and feed-forward can run passes on {where}: "
            + (f"the passing ones ({', '.join(passing)}) need what you lack ("
               + "; ".join(f"{c}: {refused[c]}" for c in passing if c in refused) + ")"
               if passing else "none passed")
        )
        failing = [r for r in candidates if not r["passed"] and allowed(r["config"], setup)[0]]
        if failing:
            why += "; the runnable ones fail " + "; ".join(
                f"{r['config']} ({', '.join(r['failed_gates'])})" for r in failing[:3]
            )
        return Recommendation(None, None, {}, None, why, evidence, scenario=scenario_id, arm=arm_name)
    pick, runner = fitting[0], (fitting[1] if len(fitting) > 1 else None)
    from robot_controllers.bench.arms import arm as registered

    built_arm = arm_name
    spec_count = len(registered(built_arm).joint_names)
    target = first["target"]
    scenario = Scenario(
        built_arm, target, float(setup.host_rate_hz), setup.bus if _bus_fits(setup.bus, spec_count)
        else BusTiming(tuple(v / 1e3 for v in _bus_ms(setup.bus, spec_count)), setup.bus.reply_delay_s,
                       setup.bus.jitter_s, setup.bus.source),
        ModelError(float(setup.model_error)), float(setup.gain_scale), axis="setup",
    )
    calibration = _shifted_calibration(pick, setup, spec_count)
    nominal = scenario.bus
    controller = config_spec(pick["config"], scenario, calibration, nominal=nominal)
    objective_text = objective if isinstance(objective, str) else ", ".join(f"{k}×{v:g}" for k, v in objective.items())
    why = (
        f"{prefix}{pick['config']} ({pick['feedforward']} + {pick['feedback']}, {pick['interpolation']}, lookahead "
        f"{pick['lookahead_mode']}) is the best of the {len(fitting)} passing configs this driver and feed-forward can "
        f"run on {where} for {objective_text}: {_metric_text(pick, names)}"
    )
    if runner is not None:
        why += f"; runner-up {runner['config']}: {_metric_text(runner, names)}"
    skipped = [c for c in ranked_ids[: ranked_ids.index(pick["config"])] if c in refused]
    if skipped:
        why += "; better but not runnable here: " + ", ".join(f"{c} ({refused[c]})" for c in skipped)
    if pick["lookahead_mode"] == "calibrated":
        why += (
            "; its lookahead is the twin's calibrated bus delay there shifted by your bus - measure yours "
            "with `robot-controllers calibrate`"
        )
    # WHY said, not filtered (§5.5, §5.9): a real arm runs only REAL_SAFE (the board closes the feedback and there is a
    # feed-forward; host-side feedback saturated the boards at 50 Hz in every sim, and no feed-forward sags the arm);
    # the twin's answer stays the answer, and the real-safe one is named beside it
    from robot_controllers.bench.population import REAL_SAFE

    if pick["config"] not in REAL_SAFE:
        safe = next((r for r in fitting if r["config"] in REAL_SAFE), None)
        why += (
            f"; {pick['config']} is not real-safe (§5.5: feedback on the host, or no feed-forward): on a real arm the "
            "best real-safe one here is "
            + (f"{safe['config']} ({_metric_text(safe, names)})" if safe else "none (no REAL_SAFE config passes)")
        )
        evidence[0]["best_real_safe"] = None if safe is None else safe["config"]
    return Recommendation(
        controller=controller,
        config=pick["config"],
        expected=dict(pick["metrics"]),
        runner_up=None if runner is None else runner["config"],
        why=why,
        evidence=evidence,
        scenario=scenario_id,
        arm=arm_name,
    )


def _bus_fits(bus: BusTiming, count: int) -> bool:
    return not isinstance(bus.command_delay_s, tuple) or len(bus.command_delay_s) == count


def setup_from_args(
    arm: str,
    rate_hz: float,
    *,
    bus: BusTiming | None = None,
    feedforward: str = "rigid_body",
    model_error: float = 0.1,
    driver_caps: DriverCaps = MOTEUS,
    gain_scale: float = 1.0,
    speed: str = "nominal",
    joint_count: int | None = None,
) -> Setup:
    """A Setup with the datasheet bus (§5.4) of the arm (or of `joint_count` joints) when `bus` is None."""
    from robot_controllers.calibration import datasheet_bus

    if bus is None:
        count = joint_count
        if count is None:
            from robot_controllers.bench.arms import arm as registered

            count = len(registered(arm).joint_names)
        bus = datasheet_bus(int(count))
    return Setup(arm, float(rate_hz), bus, feedforward, float(model_error), driver_caps, float(gain_scale), speed,
                 joint_count)


__all__ = ["Recommendation", "Setup", "recommend", "setup_from_args"]
