"""The layout sweep: given a feed-forward model and a feedback model, which layout? (CONTRACT §3.6)

    robot-controllers layouts      # env -u VIRTUAL_ENV uv run --frozen --directory <repo> --extra bench ...

Zach, 2026-11-20: "Given a feed forward model, Given a feedback model. what is the best way you basically lay
these things out...? computation cost, time, etc."

Every valid (feed-forward model @ placement, feedback model @ placement) is one `LayoutController` config,
named `ff=<model>@<at>+fb=<model>@<at>`, run on the swept rungs (R1, R2, R4, R6, R8 and both R9: free space,
fast at 100 Hz, push, host stall, model error, fast at 50 and 500 Hz). The ledger is
`results/layouts/results.json` (results-1, box JointTrajectory); its `summary.layouts` answers the question
per (FF model, FB model):

    the recommended layout = the cheapest (lowest host_duty_pct, then plan_compute_ms_per_s, then
    wire_bytes_per_s) among those that pass every gate on every swept rung and whose R2 tracking_p95_deg is
    within 10 % of the best passing layout's

The models (`feedforward_model`, `feedback_model`):

- feed-forward: `trajectory` (the rung's own effort column; only `@trajectory`), `none` (only `@none`),
  `gravity`, `rigid_body` (gravity + inertia + coriolis + viscous = the nominal model the rungs fill effort
  with), `rigid_body_coulomb` (+ the gearbox's dry friction). Every placement but `trajectory` ignores the
  effort column, so the feed-forward is only ever the named model.
- feedback: `board_gains` (no model: the board's configured kp/kd, the simple JTC), `pd` (the same kp/kd as a
  pure function, so it can move to the host or split), `ctc` (computed torque on the planar arm's mass matrix,
  ζ 0.9, its bandwidth chosen so its shoulder kp at home equals the board's shoulder kp: CTC and PD compared
  at equal stiffness where the arm starts; `ctc_bandwidth_hz()`).

WHY sequential by default (as `bench/run.py`): host_duty_pct and plan_compute_ms_per_s are measured CPU time,
and episodes side by side on a shared machine inflate it. `workers > 1` is for quick looks; its cost numbers
are flagged.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import socket
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from robot_controllers.bench.arm import ARM, HOME_Q, ArmConfig
from robot_controllers.bench.arm_models import planar_arm_mass_matrix
from robot_controllers.bench.board import BoardConfig, scaled_board
from robot_controllers.bench.types import HostTiming
from robot_controllers.core import ControllerSpec
from robot_controllers.models import (
    FEEDBACK_AT,
    MODEL_FEEDFORWARD_AT,
    PD,
    InertiaScaledPD,
    Layout,
    ModelError,
)
from robot_controllers.registry import JOINT_TRAJECTORY

FEEDFORWARD_MODELS: tuple[str, ...] = ("trajectory", "none", "gravity", "rigid_body", "rigid_body_coulomb")
FEEDBACK_MODELS: tuple[str, ...] = ("board_gains", "pd", "ctc")
ARMS: tuple[str, ...] = ("planar_arm", "ur5e")  # the arms of the layout sweep and the study (§4.5)
# WHY a named joint per arm (§3.6, §4.5): CTC is matched to the board where the arm carries the most inertia, the
# planar shoulder and the UR5e's shoulder_lift; an arm not listed uses the joint with the largest M_jj(home), which
# picks those same two joints.
CTC_REFERENCE_JOINT: dict[str, str] = {"planar_arm": "shoulder", "ur5e": "shoulder_lift_joint"}
SWEPT_RUNGS: tuple[str, ...] = (
    "R1.FreeSpace",
    "R2.FastFreeSpace",
    "R4.SidewaysPush",
    "R6.HostStall",
    "R8.FastFreeSpace·payload",
    "R9.FastFreeSpace·host50Hz",
    "R9.FastFreeSpace·host500Hz",
)
REFERENCE_RUNG = "R2.FastFreeSpace"  # the tracking (and plan / wire numbers) a recommendation compares
# WHY 0 and the bus delay (CONTRACT §3.8.2): a board-side law tracks a reference that reaches the board
# command_delay_s late; sampling it that far ahead is the fix the first sweep left out.
BUS_DELAY_LOOKAHEAD_S = HostTiming().command_delay_s  # 5.5 ms on the rack
LOOKAHEADS_S: tuple[float, ...] = (0.0, BUS_DELAY_LOOKAHEAD_S)
DUTY_RATE_HZ = 100.0  # the median compute_us_p50 a recommendation ranks on is expressed at 100 Hz
TRACKING_WITHIN = 0.10  # a layout within 10 % of the best passing R2 tracking counts as "as good"
CTC_DAMPING_RATIO = 0.9
COST_ORDER: tuple[str, ...] = ("host_duty_pct", "plan_compute_ms_per_s", "wire_bytes_per_s")
LAYOUTS_SLUG = "layouts"
_RUNS_ON = {"board": "board", "board_scheduled": "board", "host": "host", "split": "split"}
_LC = "robot_controllers.controllers.layout"


# ── the models ─────────────────────────────────────────────────────────────────────────────────────────────


def ctc_bandwidth_hz(board: BoardConfig | None = None, arm: ArmConfig = ARM) -> float:
    """The CTC bandwidth (Hz) whose shoulder kp at home, M11(home)·ωn², equals the board's configured shoulder
    kp.

    WHY this rule (CONTRACT §3.6): CTC and the board's PD are then compared at equal stiffness where the arm
    starts, so what the sweep measures is the inertia scaling and the coupling, not a stiffer or softer loop.
    """
    board = board or BoardConfig()
    shoulder_inertia = float(planar_arm_mass_matrix(arm)(HOME_Q)[0, 0])
    return math.sqrt(float(board.kp_nm_rad[0]) / shoulder_inertia) / (2.0 * math.pi)


def ctc_record(board: BoardConfig | None = None, arm: ArmConfig = ARM) -> dict[str, Any]:
    """The CTC feedback's numbers, for the ledger: its bandwidth and the gains it renders at home."""
    board = board or BoardConfig()
    feedback = feedback_model("ctc")
    kp, kd = feedback.gains(HOME_Q)
    return {
        "bandwidth_hz": feedback.bandwidth_hz,
        "natural_frequency_rad_s": feedback.natural_frequency_rad_s,
        "damping_ratio": feedback.damping_ratio,
        "rule": "shoulder kp at home = M11(home) * wn^2 = the board's configured shoulder kp",
        "board_kp_nm_rad": list(board.kp_nm_rad),
        "board_kd_nm_s_rad": list(board.kd_nm_s_rad),
        "home_q_rad": list(HOME_Q),
        "home_kp_nm_rad": [float(v) for v in kp],
        "home_kd_nm_s_rad": [float(v) for v in kd],
        "home_mass_matrix": planar_arm_mass_matrix(arm)(HOME_Q).tolist(),
    }


def arm_spec(arm: str) -> Any:
    """The registry's `ArmSpec` for `arm` (§5.1); an unknown arm is a ValueError, as before the registry."""
    from robot_controllers.bench.arms import arm as registered

    try:
        return registered(arm)
    except KeyError as error:
        raise ValueError(str(error.args[0])) from None


def feedforward_model(name: str, arm: str = "planar_arm", model_error: ModelError | None = None) -> Any:
    """The feed-forward function a sweep name stands for on `arm`, built with `model_error` (§5.3; None = the
    nominal model). None for `trajectory` and `none`: no model."""
    if name in ("trajectory", "none"):
        return None
    if name not in FEEDFORWARD_MODELS:
        raise ValueError(f"unknown feed-forward model {name!r}; known: {FEEDFORWARD_MODELS}")
    return arm_spec(arm).feedforward(name, model_error)


def ctc_reference_joint(spec: Any, mass_matrix: Any) -> int:
    """The joint CTC's bandwidth is matched on: `CTC_REFERENCE_JOINT`, else the largest M_jj at home."""
    name = CTC_REFERENCE_JOINT.get(spec.name)
    if name is not None:
        return spec.joint_names.index(name)
    return int(np.argmax(np.diag(np.asarray(mass_matrix(np.asarray(spec.home_q, dtype=float))))))


def feedback_model(
    name: str, arm: str = "planar_arm", model_error: ModelError | None = None, gain_scale: float = 1.0
) -> Any:
    """The feedback function a sweep name stands for on `arm` (None for `board_gains`: the board's own
    gains). `pd` is the arm's board gains × `gain_scale` as a pure function; `ctc` is computed torque on the arm's
    mass matrix with `model_error`, its bandwidth set so its kp at home on the reference joint equals that board kp
    (§3.6, §4.5): with the defaults these are the models the layout sweep and the study ran."""
    if name == "board_gains":
        return None
    if name not in FEEDBACK_MODELS:
        raise ValueError(f"unknown feedback model {name!r}; known: {FEEDBACK_MODELS}")
    spec = arm_spec(arm)
    board = scaled_board(spec.board_config(), gain_scale)
    if name == "pd":
        return PD(board.kp_nm_rad, board.kd_nm_s_rad)
    mass = spec.mass_matrix(model_error)
    joint = ctc_reference_joint(spec, mass)
    inertia = float(mass(spec.home_q)[joint, joint])
    bandwidth = math.sqrt(float(board.kp_nm_rad[joint]) / inertia) / (2.0 * math.pi)
    return InertiaScaledPD(bandwidth, CTC_DAMPING_RATIO, mass)


def feedforward_placements(model: str) -> tuple[str, ...]:
    if model == "trajectory":
        return ("trajectory",)
    if model == "none":
        return ("none",)
    feedforward_model(model)  # raises on an unknown name
    return MODEL_FEEDFORWARD_AT


def feedback_placements(model: str) -> tuple[str, ...]:
    if model == "board_gains":
        return ("board",)
    feedback_model(model)
    return FEEDBACK_AT


def lookahead_text(lookahead_s: float | Sequence[float]) -> str:
    """A lookahead in ms as names spell it: `5.5`, or one per joint `1.5/3.7`; "" for none (all zero)."""
    if np.ndim(lookahead_s) == 0:
        return "" if float(lookahead_s) == 0.0 else f"{float(lookahead_s) * 1e3:g}"
    values = [float(v) for v in lookahead_s]
    return "" if not any(values) else "/".join(f"{v * 1e3:g}" for v in values)


def layout_name(
    ff_model: str,
    ff_at: str,
    fb_model: str,
    fb_at: str,
    lookahead_s: float | Sequence[float] = 0.0,
    interpolation: str = "tangent",
) -> str:
    """`ff=<model>@<at>+fb=<model>@<at>`, plus `+la=<ms>ms` (`+la=<ms>/<ms>/...ms` per joint) for a non-zero
    lookahead and `+interp=<interpolation>` for a non-default interpolation (CONTRACT §4.1)."""
    name = f"ff={ff_model}@{ff_at}+fb={fb_model}@{fb_at}"
    ahead = lookahead_text(lookahead_s)
    if ahead:
        name += f"+la={ahead}ms"
    return name if interpolation == "tangent" else f"{name}+interp={interpolation}"


def sweep_points(
    ff_models: Sequence[str] = FEEDFORWARD_MODELS,
    fb_models: Sequence[str] = FEEDBACK_MODELS,
    lookaheads_s: Sequence[float] = LOOKAHEADS_S,
) -> list[tuple[str, str, str, str, float]]:
    """Every valid (ff model, ff at, fb model, fb at) at every lookahead, in model order."""
    return [
        (ff_model, ff_at, fb_model, fb_at, float(lookahead))
        for ff_model in ff_models
        for ff_at in feedforward_placements(ff_model)
        for fb_model in fb_models
        for fb_at in feedback_placements(fb_model)
        for lookahead in lookaheads_s
    ]


def layout_spec(
    ff_model: str,
    ff_at: str,
    fb_model: str,
    fb_at: str,
    lookahead_s: float | Sequence[float] = 0.0,
    interpolation: str = "tangent",
    arm: str = "planar_arm",
    model_error: ModelError | None = None,
    gain_scale: float = 1.0,
) -> ControllerSpec:
    """The LayoutController for one sweep point, as a built ControllerSpec named
    `ff=<model>@<at>+fb=<model>@<at>` (`+la=<ms>ms` with a lookahead, `+interp=<value>` with a non-default
    interpolation).

    `arm` is any registered arm (§5.1). `model_error` is how wrong the feed-forward and the CTC mass matrix it is
    given are, and `gain_scale` multiplies the board gains `pd` and `ctc` are built from (§5.3; the plant's own
    boards are the run's: `scaled_board` on the plant for a scenario with other gains). Neither changes the name:
    a scenario names them.

    Raises ValueError for a pair the layout cannot place (`trajectory` only @trajectory, `none` only @none, a
    model only @plan/@host_*, `board_gains` only @board), and for an arm the registry does not have.
    """
    arm_spec(arm)
    if ff_at not in feedforward_placements(ff_model):
        raise ValueError(
            f"feed-forward {ff_model!r} cannot be placed at {ff_at!r}; "
            f"it goes at {feedforward_placements(ff_model)}"
        )
    if fb_at not in feedback_placements(fb_model):
        raise ValueError(
            f"feedback {fb_model!r} cannot be placed at {fb_at!r}; it goes at {feedback_placements(fb_model)}"
        )
    feedforward = feedforward_model(ff_model, arm, model_error)
    feedback = feedback_model(fb_model, arm, model_error, gain_scale)
    layout = Layout(ff_at, fb_at, lookahead_s=lookahead_s, interpolation=interpolation)
    layout.check(feedforward, feedback)
    ff_text = {"trajectory": "the trajectory's own effort column", "none": "no feed-forward"}.get(
        ff_model, f"the {ff_model} model evaluated at {ff_at.replace('_', ' ')}"
    )
    fb_text = (
        "the board's configured gains"
        if fb_model == "board_gains"
        else f"{fb_model} feedback on the {fb_at.replace('_', ' ')}"
    )
    ahead_ms = lookahead_text(layout.lookahead_s)
    ahead = "" if not ahead_ms else f", the reference sampled {ahead_ms} ms ahead"
    if layout.interpolation != "tangent":
        ahead += f", {layout.interpolation.replace('_', ' ')} interpolation"
    return ControllerSpec(
        name=layout_name(ff_model, ff_at, fb_model, fb_at, layout.lookahead_s, layout.interpolation),
        title=f"ff {ff_model} @ {ff_at} + fb {fb_model} @ {fb_at}"
        + ("" if not ahead_ms else f" + lookahead {ahead_ms} ms")
        + ("" if layout.interpolation == "tangent" else f" + {layout.interpolation}"),
        box=JOINT_TRAJECTORY,
        runs_on=_RUNS_ON[fb_at],
        summary=f"The layout controller with {ff_text} and {fb_text}{ahead}"
        + _given_text(model_error, gain_scale)
        + ("." if arm == "planar_arm" else f", on the {arm}."),
        target=f"{_LC}:LayoutController",
        config_target=f"{_LC}:LayoutConfig",
        prior_art=("robot_controllers layout_controller",),
        variant_of="layout_controller",
        default_config={"feedforward": feedforward, "feedback": feedback, "layout": layout},
    )


def _given_text(model_error: ModelError | None, gain_scale: float) -> str:
    """", models ff±10%, gains ×0.5" when the controller was given a wrong model or other gains; "" otherwise."""
    parts = []
    if model_error is not None and not model_error.is_nominal:
        parts.append(f"models {model_error.label}")
    if float(gain_scale) != 1.0:
        parts.append(f"gains ×{float(gain_scale):g}")
    return "" if not parts else ", " + ", ".join(parts)


# ── the recommendation ─────────────────────────────────────────────────────────────────────────────────────


def _number(value: Any) -> float:
    """A cost for sorting: a missing one sorts last."""
    return math.inf if value is None or not math.isfinite(float(value)) else float(value)


def recommend(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The cheapest layout among those that pass and track within 10 % of the best passing one (CONTRACT
    §3.6).

    Each candidate: `name`, `passed` (every gate on every swept rung), `tracking_p95_deg` (R2),
    `host_duty_pct`,
    `plan_compute_ms_per_s`, `wire_bytes_per_s`. Returns `{"name": <str | None>, "why": <str>}`; None when
    nothing passed, and the why then says so and names the closest one.
    """
    rows = list(candidates)
    passing = [c for c in rows if c.get("passed") and c.get("tracking_p95_deg") is not None]
    if not passing:
        if not rows:
            return {"name": None, "why": "no layouts were run"}
        closest = min(
            rows,
            key=lambda c: (-int(c.get("rungs_passed") or 0), _number(c.get("tracking_p95_deg")), c["name"]),
        )
        detail = f"; closest: {closest['name']}"
        if closest.get("rungs_passed") is not None and closest.get("rungs_total") is not None:
            detail += f" ({closest['rungs_passed']}/{closest['rungs_total']} rungs passed"
            failed = closest.get("failed_rungs") or []
            detail += f", failed {', '.join(failed)})" if failed else ")"
        return {
            "name": None,
            "why": f"none of the {len(rows)} layouts passed every gate on every swept rung{detail}",
        }
    best = min(passing, key=lambda c: (float(c["tracking_p95_deg"]), c["name"]))
    limit = float(best["tracking_p95_deg"]) * (1.0 + TRACKING_WITHIN)
    close = [c for c in passing if float(c["tracking_p95_deg"]) <= limit]
    pick = min(close, key=lambda c: (*(_number(c.get(name)) for name in COST_ORDER), c["name"]))

    def cost(c: Mapping[str, Any]) -> str:
        return (
            f"host duty {_fmt(c.get('host_duty_pct'))} %, plan {_fmt(c.get('plan_compute_ms_per_s'))} ms/s, "
            f"wire {_fmt(c.get('wire_bytes_per_s'), 0)} B/s"
        )

    if pick["name"] == best["name"]:
        tracking = (
            f"the best R2 tracking p95 of the {len(passing)} passing layouts "
            f"({_fmt(pick['tracking_p95_deg'])} deg)"
        )
    else:
        tracking = (
            f"R2 tracking p95 {_fmt(pick['tracking_p95_deg'])} deg, within 10 % of the best passing "
            f"{best['name']} ({_fmt(best['tracking_p95_deg'])} deg)"
        )
    why = (
        f"{pick['name']}: passes every gate on every swept rung with {tracking}, and is the cheapest of the "
        f"{len(close)} layout{'s' if len(close) != 1 else ''} that close: {cost(pick)}"
    )
    return {"name": pick["name"], "why": why}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if digits == 0:
        return f"{number:.0f}"
    return f"{number:.{digits}g}"


def median_duty_pct(runs: Mapping[str, Mapping[str, Any]]) -> float | None:
    """The median over a layout's rungs of compute_us_p50, as % of one core at 100 Hz (CONTRACT §3.8.3).

    WHY the median and one rate: per-tick CPU time on the shared host varied 2-4x between identical laws, so
    one noisy rung flipped picks; the rate is fixed so a 500 Hz rung's tick costs the same as a 100 Hz one's.
    """
    ticks = [row.get("metrics", {}).get("compute_us_p50") for row in runs.values()]
    ticks = [float(v) for v in ticks if v is not None and math.isfinite(float(v))]
    return statistics.median(ticks) * DUTY_RATE_HZ / 1.0e4 if ticks else None


def layout_candidates(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One candidate per swept controller of a layouts ledger: passed (every gate on every rung it ran), the
    reference rung's tracking, plan and wire numbers, the median duty over the swept rungs, and which rungs
    and gates failed."""
    rung_ids = [r["id"] for r in document["rungs"]]
    by_controller: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in document["results"]:
        by_controller.setdefault(row["controller"], {})[row["rung"]] = row
    out = []
    for record in document["controllers"]:
        runs = by_controller.get(record["name"], {})
        failed = {
            rung: [
                f"{g['metric']} {g['op']} {g['value']:g} ({_fmt(g['measured'])})"
                for g in runs[rung]["gates"]
                if not g["passed"]
            ]
            for rung in rung_ids
            if rung in runs and not runs[rung]["passed"]
        }
        missing = [rung for rung in rung_ids if rung not in runs]
        reference = runs.get(REFERENCE_RUNG, {}).get("metrics", {})
        out.append(
            {
                "name": record["name"],
                "feedforward_model": record.get("feedforward_model"),
                "feedforward_at": record.get("feedforward_at"),
                "feedback_model": record.get("feedback_model"),
                "feedback_at": record.get("feedback_at"),
                "lookahead_s": record.get("lookahead_s", 0.0),
                "passed": not failed and not missing,
                "rungs_passed": sum(1 for rung in rung_ids if rung in runs and runs[rung]["passed"]),
                "rungs_total": len(rung_ids),
                "failed_rungs": [*failed, *missing],
                "failed_gates": failed,
                "tracking_p95_deg": reference.get("tracking_p95_deg"),
                "fb_share_pct": reference.get("fb_share_pct"),
                "fb_share_lf_pct": reference.get("fb_share_lf_pct"),
                "host_duty_pct": median_duty_pct(runs),
                "host_duty_pct_reference_rung": reference.get("host_duty_pct"),
                "plan_compute_ms_per_s": reference.get("plan_compute_ms_per_s"),
                "wire_bytes_per_s": reference.get("wire_bytes_per_s"),
            }
        )
    return out


def layout_summary(document: Mapping[str, Any]) -> dict[str, Any]:
    """`summary.layouts`: per (FF model, FB model) the candidates and the recommended layout; one overall."""
    candidates = layout_candidates(document)
    pairs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for candidate in candidates:
        pairs.setdefault((candidate["feedforward_model"], candidate["feedback_model"]), []).append(candidate)
    per_pair = [
        {
            "feedforward_model": ff_model,
            "feedback_model": fb_model,
            "recommendation": recommend(rows),
            "candidates": [row["name"] for row in rows],
        }
        for (ff_model, fb_model), rows in pairs.items()
    ]
    return {
        "rule": "per (feed-forward model, feedback model), over every placement and lookahead: the cheapest "
        "layout (lowest host_duty_pct, then plan_compute_ms_per_s, then wire_bytes_per_s) among those that "
        f"pass every gate on every swept rung and whose {REFERENCE_RUNG} tracking_p95_deg is within 10 % of "
        "the best passing layout's; host_duty_pct is the median over the swept rungs of compute_us_p50 "
        f"expressed at {DUTY_RATE_HZ:g} Hz (median us x {DUTY_RATE_HZ:g} / 10^4), plan and wire are "
        f"{REFERENCE_RUNG}'s",
        "reference_rung": REFERENCE_RUNG,
        "lookaheads_s": sorted({float(c.get("lookahead_s", 0.0)) for c in document["controllers"]}),
        "swept_rungs": [r["id"] for r in document["rungs"]],
        "ctc": ctc_record(),
        "per_pair": per_pair,
        "overall": recommend(candidates),
        "candidates": candidates,
    }


# ── the sweep ──────────────────────────────────────────────────────────────────────────────────────────────


def _layout_episode(job: tuple[str, str, str, str, float, str, int]) -> dict[str, Any]:
    """One (layout, rung, seed) -> RunResult dict. Top level so a process pool can run it."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ff_model, ff_at, fb_model, fb_at, lookahead_s, rung_id, seed = job
    from robot_controllers.bench.loop import run_episode
    from robot_controllers.bench.rungs import rung
    from robot_controllers.metrics.episode import run_result

    started = time.perf_counter()
    log = run_episode(layout_spec(ff_model, ff_at, fb_model, fb_at, lookahead_s), rung(rung_id), seed=seed)
    # WHY no trace: ~1400 episodes of 200 Hz series would make the ledger ~70 MB; the numbers are the answer
    # here
    result = run_result(log, rung(rung_id), trace=False).to_dict()
    result["wall_s"] = round(time.perf_counter() - started, 3)
    return result


def run_layouts(
    *,
    ff_models: Sequence[str] = FEEDFORWARD_MODELS,
    fb_models: Sequence[str] = FEEDBACK_MODELS,
    rungs: Sequence[str] = SWEPT_RUNGS,
    lookaheads_s: Sequence[float] = LOOKAHEADS_S,
    seeds: Sequence[int] = (0,),
    workers: int = 1,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """The results-1 document of the layout sweep (box JointTrajectory, `summary.layouts` added)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from robot_controllers.bench.run import code_version, controller_record, robot_msgs_version, rung_record
    from robot_controllers.bench.rungs import rung as rung_by_id
    from robot_controllers.metrics.summary import summarize

    points = sweep_points(ff_models, fb_models, lookaheads_s)
    swept = [rung_by_id(rung_id) for rung_id in rungs]
    jobs = [(*point, r.id, seed) for point in points for r in swept for seed in seeds]
    if progress:
        progress(
            f"layouts: {len(points)} layouts x {len(swept)} rungs x {len(seeds)} seeds = {len(jobs)} episodes"
        )
    results: list[dict[str, Any]] = []

    def report(job: tuple, result: dict[str, Any]) -> None:
        results.append(result)
        if progress:
            print_name = layout_name(*job[:5])
            progress(
                f"  [{len(results)}/{len(jobs)}] {print_name} @ {job[5]}: "
                f"{'pass' if result['passed'] else 'FAIL'} ({result['wall_s']} s)"
            )

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for job, result in zip(jobs, pool.map(_layout_episode, jobs), strict=True):
                report(job, result)
    else:
        for job in jobs:
            report(job, _layout_episode(job))

    controllers = []
    for point in points:
        record = controller_record(layout_spec(*point))
        record.update(
            dict(
                zip(
                    ("feedforward_model", "feedforward_at", "feedback_model", "feedback_at", "lookahead_s"),
                    point,
                    strict=True,
                )
            )
        )
        controllers.append(record)
    sha, dirty = code_version()
    import mujoco

    box = JOINT_TRAJECTORY
    document: dict[str, Any] = {
        "schema": "robot-controllers/results-1",
        "box": {
            "slug": box.slug,
            "name": box.name,
            "input": box.input_name,
            "output": box.output_name,
            "wandb_project": box.wandb_project,
        },
        "sweep": LAYOUTS_SLUG,
        "generated": {
            "commit": sha,
            "dirty": dirty,
            "robot_msgs": robot_msgs_version(),
            "plant": "planar_arm",
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "date": _dt.date.today().isoformat(),
            "workers": workers,
            "cost_numbers_contended": workers > 1,
            "seeds": list(seeds),
            "load_average_1m": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
        },
        "rungs": [rung_record(r) for r in swept],
        "controllers": controllers,
        "results": results,
        "summary": {},
        "wandb": {"project_url": None, "reports": {}},
    }
    document["summary"] = summarize(box, swept, results)
    document["summary"]["layouts"] = layout_summary(document)
    return document


def write_layouts(document: dict[str, Any], out_dir: Path | None = None) -> Path:
    """results/layouts/results.json (the ledger of the sweep)."""
    from robot_controllers.bench.run import RESULTS_DIR

    path = (out_dir or RESULTS_DIR) / LAYOUTS_SLUG / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=1, allow_nan=False) + "\n")
    return path
