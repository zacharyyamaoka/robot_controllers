"""The study: how host rate, speed, lookahead, bus jitter, a staggered bus and a moving target move seven
layouts, on the planar arm and the UR5e (CONTRACT amendment 4, §4.6).

    robot-controllers study --workers 2     # -> results/study/results.json (schema robot-controllers/study-1)

Zach, 2026-11-20: "why is 50hz a meaning control hz? I would be curious to see more clearly how the results
change with different control frequency ... try trajectories at different speeds. I kinda want to make this a
nice stable conclusion." And: "how that can be calibrated on how far the look ahead should be? how that that
impact when you are doing like syncing to catch a moving item where timing is very important? ... how
reliabiliy does look ahead work?"

One row per (arm, dimension, value, layout, seed); `value` is a number for a one-axis dimension and a list for
a grid (`rate_speed` [rate_hz, move_s], `lookahead` [rate_hz, lookahead_ms], `stagger` [rate_hz, way], the
sync lookahead sweep [belt_speed_m_s, lookahead_ms]); every row also spells its coordinates out (`rate_hz`,
`move_s`, `lookahead_ms`, ...). The dimensions:

- `rate`: `rate_variant(base, r)` for r in RATES_HZ, every layout.
- `speed`: `speed_variant(base, m)` at 100 Hz, every layout (the planar and UR5e move times differ).
- `rate_speed`: the grid of both for layouts 2, 3, 4, 6 and 7.
- `lookahead`: layouts 3 and 4 at every lookahead in LOOKAHEADS_MS, on the base at 100 and at 50 Hz.
- `jitter`: layout 4 on `bus_variant(base, 5.5 ms, j)`, j in JITTER_MS, seeds 0-4.
- `stagger`: `bus_variant(base, d)` with d_j = 1.5 + 2.2·j ms (one adapter, ~2.2 ms per board) and layout 4
  four ways: lookahead 5.5 ms (miscalibrated), the scalar mean of d, per-joint d (the truth) and per-joint
  `calibrate_lookahead` (the one-run procedure; the row carries the calibrated values), at 50 and 100 Hz.
- `sync` (planar only): `sync_variant(v)` at every belt speed for every layout, and layouts 3 and 4 over the
  lookahead list at 0.25 m/s.

The summary answers per arm: the rate above which layout 4 is within 10 % of its 1 kHz tracking, the lookahead
that minimises tracking at each rate and what ±1 ms of lookahead error costs there, the calibrated-vs-true
per-joint lookahead, and (planar) the sync timing and error per layout per belt speed.

WHY no traces: ~800 episodes of 200 Hz series would be ~40 MB; the ledger keeps the numbers (a few MB) and a
figure re-runs one episode. WHY sequential by default (as `bench/run.py`): compute is measured CPU time and
episodes side by side contend; `workers > 1` flags the cost numbers as contended.
"""

from __future__ import annotations

import dataclasses
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

from robot_controllers.bench.layouts import layout_name, layout_spec
from robot_controllers.bench.types import Rung

SCHEMA = "robot-controllers/study-1"
STUDY_SLUG = "study"
# The seven layouts (§4.6), as layout_spec's (ff model, ff at, fb model, fb at, lookahead_s, interpolation)
STUDY_POINTS: tuple[tuple[str, str, str, str, float, str], ...] = (
    ("none", "none", "board_gains", "board", 0.0055, "tangent"),
    ("trajectory", "trajectory", "board_gains", "board", 0.0, "tangent"),
    ("trajectory", "trajectory", "board_gains", "board", 0.0055, "tangent"),
    ("trajectory", "trajectory", "board_gains", "board", 0.0055, "secant_mid_ff"),
    ("rigid_body", "plan", "pd", "split", 0.0, "tangent"),
    ("rigid_body", "host_measured", "ctc", "host", 0.0, "tangent"),
    ("rigid_body", "plan", "pd", "host", 0.0, "tangent"),
)
STUDY_LAYOUTS: tuple[str, ...] = tuple(layout_name(*point) for point in STUDY_POINTS)
ARMS: tuple[str, ...] = ("planar_arm", "ur5e")
DIMENSIONS: tuple[str, ...] = ("rate", "speed", "rate_speed", "lookahead", "jitter", "stagger", "sync")
BASE_RUNG = {"planar_arm": "R2.FastFreeSpace", "ur5e": "U2.FastFreeSpace"}
RATES_HZ: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
MOVES_S = {"planar_arm": (1.2, 0.8, 0.5, 0.3, 0.2), "ur5e": (2.0, 1.2, 0.8, 0.6, 0.45, 0.35)}
RATE_SPEED_LAYOUTS = (2, 3, 4, 6, 7)  # 1-based, as the contract numbers them
LOOKAHEAD_LAYOUTS = (3, 4)
LOOKAHEADS_MS: tuple[float, ...] = (0, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 10, 12)
LOOKAHEAD_RATES_HZ: tuple[float, ...] = (100.0, 50.0)
JITTER_LAYOUT = 4
JITTER_MS: tuple[float, ...] = (0, 0.5, 1, 2, 3)
JITTER_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
BUS_DELAY_S = 0.0055  # the rack's command delay, the jitter rows' mean
STAGGER_FIRST_MS, STAGGER_STEP_MS = 1.5, 2.2  # d_j = 1.5 + 2.2·j ms
STAGGER_RATES_HZ: tuple[float, ...] = (50.0, 100.0)
STAGGER_WAYS: tuple[str, ...] = ("miscalibrated", "mean", "true", "calibrated")
SYNC_SPEEDS_M_S: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)
SYNC_LOOKAHEAD_SPEED_M_S = 0.25
WITHIN = 0.10  # "within 10 % of its 1 kHz tracking"
REFERENCE_RATE_HZ = 1000.0
RATE_LAYOUT = 4  # the layout whose rate threshold the summary names


@dataclasses.dataclass(frozen=True)
class StudyJob:
    """One episode of the study: which arm, which row, which layout (with any lookahead it overrides), the
    rung to run and the seed. Frozen and picklable, so a process pool runs it."""

    arm: str
    dimension: str
    value: Any
    base_rung: str
    layout_index: int  # 1-based
    rung: Rung
    seed: int = 0
    lookahead_s: float | tuple[float, ...] | None = None  # None = the study layout's own
    fields: tuple[tuple[str, Any], ...] = ()  # extra coordinates for the row (way, belt speed, calibration)

    @property
    def point(self) -> tuple[str, str, str, str, float | tuple[float, ...], str]:
        ff_model, ff_at, fb_model, fb_at, lookahead, interpolation = STUDY_POINTS[self.layout_index - 1]
        if self.lookahead_s is not None:
            lookahead = self.lookahead_s
        return ff_model, ff_at, fb_model, fb_at, lookahead, interpolation

    def spec(self) -> Any:
        return layout_spec(*self.point, arm=self.arm)


# ── the rungs ──────────────────────────────────────────────────────────────────────────────────────────────


def base_rung(arm: str) -> Rung:
    from robot_controllers.bench.rungs import rung

    if arm == "ur5e":
        from robot_controllers.bench.ur5e import ur5e_rungs

        return {r.id: r for r in ur5e_rungs()}[BASE_RUNG[arm]]
    if arm == "planar_arm":
        return rung(BASE_RUNG[arm])
    raise ValueError(f"unknown arm {arm!r}; known: {ARMS}")


def at_rate(rung: Rung, rate_hz: float) -> Rung:
    """`rung` itself at its own rate, else `rate_variant(rung, rate_hz)`."""
    from robot_controllers.bench.rungs import rate_variant

    return rung if float(rung.timing.rate_hz) == float(rate_hz) else rate_variant(rung, rate_hz)


def stagger_delays_s(joint_count: int) -> tuple[float, ...]:
    """d_j = 1.5 + 2.2·j ms for each joint: one adapter serving its boards in turn, ~2.2 ms per board."""
    return tuple((STAGGER_FIRST_MS + STAGGER_STEP_MS * j) / 1000.0 for j in range(joint_count))


def _joint_count(arm: str) -> int:
    if arm == "ur5e":
        from robot_controllers.bench.ur5e import UR5E_JOINTS

        return len(UR5E_JOINTS)
    from robot_controllers.bench.arm import JOINT_NAMES

    return len(JOINT_NAMES)


def _layout4_at(arm: str) -> Callable[[float | tuple[float, ...]], Any]:
    ff_model, ff_at, fb_model, fb_at, _, interpolation = STUDY_POINTS[3]
    return lambda lookahead: layout_spec(ff_model, ff_at, fb_model, fb_at, lookahead, interpolation, arm=arm)


def plan_study(
    arms: Sequence[str] = ARMS,
    dimensions: Sequence[str] = DIMENSIONS,
    *,
    calibrate: Callable[..., tuple[float, ...]] | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[StudyJob]:
    """Every episode of the study, in row order. `calibrate` (default `metrics.lag.calibrate_lookahead`) runs
    the stagger rows' one calibration episode per (arm, rate) here, in this process, before any pool."""
    from robot_controllers.bench.rungs import bus_variant, rate_variant, speed_variant, sync_variant

    if calibrate is None:
        from robot_controllers.metrics.lag import calibrate_lookahead as calibrate
    jobs: list[StudyJob] = []
    every = range(1, len(STUDY_POINTS) + 1)
    for arm in arms:
        base = base_rung(arm)
        name = base.id
        for dimension in dimensions:
            if dimension == "rate":
                for rate in RATES_HZ:
                    variant = rate_variant(base, rate)
                    jobs += [StudyJob(arm, "rate", rate, name, i, variant) for i in every]
            elif dimension == "speed":
                for move in MOVES_S[arm]:
                    variant = speed_variant(base, move)
                    jobs += [StudyJob(arm, "speed", move, name, i, variant) for i in every]
            elif dimension == "rate_speed":
                for rate in RATES_HZ:
                    for move in MOVES_S[arm]:
                        variant = rate_variant(speed_variant(base, move), rate)
                        jobs += [
                            StudyJob(arm, "rate_speed", [rate, move], name, i, variant)
                            for i in RATE_SPEED_LAYOUTS
                        ]
            elif dimension == "lookahead":
                for rate in LOOKAHEAD_RATES_HZ:
                    variant = at_rate(base, rate)
                    for index in LOOKAHEAD_LAYOUTS:
                        jobs += [
                            StudyJob(arm, "lookahead", [rate, ms], name, index, variant,
                                     lookahead_s=ms / 1000.0)
                            for ms in LOOKAHEADS_MS
                        ]
            elif dimension == "jitter":
                for jitter in JITTER_MS:
                    variant = bus_variant(at_rate(base, 100.0), BUS_DELAY_S, jitter / 1000.0)
                    jobs += [
                        StudyJob(arm, "jitter", jitter, name, JITTER_LAYOUT, variant, seed=seed)
                        for seed in JITTER_SEEDS
                    ]
            elif dimension == "stagger":
                delays = stagger_delays_s(_joint_count(arm))
                for rate in STAGGER_RATES_HZ:
                    variant = bus_variant(at_rate(base, rate), delays)
                    if progress:
                        progress(f"  calibrating the lookahead on {variant.id} (one run at 0 ms)")
                    calibrated = tuple(float(v) for v in calibrate(_layout4_at(arm), variant, seed=0))
                    ways = {
                        "miscalibrated": None,
                        "mean": float(statistics.fmean(delays)),
                        "true": delays,
                        "calibrated": calibrated,
                    }
                    for way, lookahead in ways.items():
                        fields: tuple[tuple[str, Any], ...] = (("way", way), ("true_delay_ms", _ms(delays)))
                        if way == "calibrated":
                            fields += (("calibrated_lookahead_ms", _ms(calibrated)),)
                        jobs.append(
                            StudyJob(arm, "stagger", [rate, way], name, 4, variant, lookahead_s=lookahead,
                                     fields=fields)
                        )
            elif dimension == "sync":
                if arm != "planar_arm":
                    continue  # the sync rung is the planar arm's (it rides the planar belt)
                sync_base = sync_variant(SYNC_LOOKAHEAD_SPEED_M_S).id  # R10.BeltSync@0.25m/s
                for speed in SYNC_SPEEDS_M_S:
                    variant = sync_variant(speed)
                    belt = (("belt_speed_m_s", speed),)
                    jobs += [StudyJob(arm, "sync", speed, sync_base, i, variant, fields=belt) for i in every]
                variant = sync_variant(SYNC_LOOKAHEAD_SPEED_M_S)
                belt = (("belt_speed_m_s", SYNC_LOOKAHEAD_SPEED_M_S),)
                for index in LOOKAHEAD_LAYOUTS:
                    jobs += [
                        StudyJob(arm, "sync", [SYNC_LOOKAHEAD_SPEED_M_S, ms], sync_base, index, variant,
                                 lookahead_s=ms / 1000.0, fields=belt)
                        for ms in LOOKAHEADS_MS
                    ]
            else:
                raise ValueError(f"unknown dimension {dimension!r}; known: {DIMENSIONS}")
    return jobs


# ── one row ────────────────────────────────────────────────────────────────────────────────────────────────


def _ms(value: float | tuple[float, ...]) -> float | list[float]:
    """Seconds as milliseconds, rounded to 1 ns so 8.1 ms reads 8.1, not 8.100000000000001."""
    if np.ndim(value) == 0:
        return round(1000.0 * float(value), 6)
    return [round(1000.0 * float(v), 6) for v in value]


def run_job(job: StudyJob) -> dict[str, Any]:
    """One episode -> its row (every metric, passed, the failed gates' labels, the per-joint lag). Top level
    so a process pool can run it."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from robot_controllers.bench.loop import run_episode
    from robot_controllers.metrics.episode import run_result
    from robot_controllers.metrics.lag import estimate_lag_s

    started = time.perf_counter()
    spec = job.spec()
    log = run_episode(spec, job.rung, seed=job.seed)
    result = run_result(log, job.rung, trace=False)
    per_joint = estimate_lag_s(log, per_joint=True)
    timing = job.rung.timing
    *_, lookahead, interpolation = job.point
    row: dict[str, Any] = {
        "arm": job.arm,
        "dimension": job.dimension,
        "value": job.value,
        "base_rung": job.base_rung,
        "rung": job.rung.id,
        "layout": STUDY_LAYOUTS[job.layout_index - 1],
        "layout_index": job.layout_index,
        "controller": spec.name,
        "seed": job.seed,
        "rate_hz": float(timing.rate_hz),
        "move_s": job.rung.params.get("move_s"),
        "peak_joint_speed_deg_s": job.rung.params.get("peak_joint_speed_deg_s"),
        "peak_tool_speed_m_s": job.rung.params.get("peak_tool_speed_m_s"),
        "lookahead_ms": _ms(lookahead),
        "interpolation": interpolation,
        "command_delay_ms": _ms(timing.command_delay_s),
        "jitter_ms": 1000.0 * float(timing.command_delay_jitter_s),
        **{key: value for key, value in job.fields},
        "metrics": result.metrics,
        "passed": result.passed,
        "failed_gates": [
            f"{g['metric']} {g['op']} {g['value']:g}" for g in result.gates if not g["passed"]
        ],
        "per_joint_lag_ms": [None if v is None else _ms(v) for v in per_joint],
        "joint_names": list(log.joint_names),
        "wall_s": round(time.perf_counter() - started, 3),
    }
    return row


# ── the summary ────────────────────────────────────────────────────────────────────────────────────────────


def _p95(row: Mapping[str, Any]) -> float | None:
    return row["metrics"].get("tracking_p95_deg")


def _rows(
    rows: Sequence[Mapping[str, Any]], arm: str, dimension: str, layout_index: int | None = None
) -> list[Mapping[str, Any]]:
    return [
        r for r in rows
        if r["arm"] == arm and r["dimension"] == dimension
        and (layout_index is None or r["layout_index"] == layout_index)
    ]


def _key(value: Any) -> str:
    return f"{float(value):g}"


def rate_threshold(by_rate: Mapping[float, float | None], within: float = WITHIN) -> dict[str, Any]:
    """The lowest rate from which every higher rate tracks within `within` of the 1 kHz tracking."""
    reference = by_rate.get(REFERENCE_RATE_HZ)
    if reference is None:
        return {"rate_hz": None, "why": "no 1 kHz run to compare with"}
    limit = reference * (1.0 + within)
    threshold = None
    for rate in sorted(by_rate, reverse=True):
        value = by_rate[rate]
        if value is None or value > limit:
            break
        threshold = rate
    tracked = ", ".join(f"{rate:g} Hz {by_rate[rate]:.3g} deg" for rate in sorted(by_rate) if by_rate[rate])
    if threshold is None:
        why = f"no rate tracks within {100 * within:g} % of the 1 kHz run ({reference:.3g} deg); {tracked}"
    else:
        why = (
            f"tracking p95 within {100 * within:g} % of the 1 kHz run ({reference:.3g} deg, limit "
            f"{limit:.3g} deg) from {threshold:g} Hz up; {tracked}"
        )
    return {"rate_hz": threshold, "reference_1khz_deg": reference, "limit_deg": limit, "why": why}


def lookahead_curve_summary(curve: Mapping[float, float]) -> dict[str, Any]:
    """The lookahead (ms) with the lowest tracking on a swept curve, and what ±1 ms around it costs (linear on
    the swept grid; None where ±1 ms falls off it)."""
    points = sorted((la, p95) for la, p95 in curve.items() if p95 is not None)
    if not points:
        return {"best_lookahead_ms": None}
    grid = np.array([la for la, _ in points])
    values = np.array([p95 for _, p95 in points])
    best = int(np.argmin(values))
    best_la, best_p95 = float(grid[best]), float(values[best])

    def at(la: float) -> float | None:
        return float(np.interp(la, grid, values)) if grid[0] <= la <= grid[-1] else None

    minus, plus = at(best_la - 1.0), at(best_la + 1.0)
    sides = [v - best_p95 for v in (minus, plus) if v is not None]
    return {
        "best_lookahead_ms": best_la,
        "best_tracking_p95_deg": best_p95,
        "tracking_at_minus_1ms_deg": minus,
        "tracking_at_plus_1ms_deg": plus,
        "cost_of_1ms_error_deg": float(np.mean(sides)) if sides else None,
        "curve_deg": {_key(la): p95 for la, p95 in points},
    }


def summarize_study(rows: Sequence[Mapping[str, Any]], arms: Sequence[str] = ARMS) -> dict[str, Any]:
    """Per arm: the rate threshold, the lookahead answers, the stagger calibration, jitter, speed and sync."""
    out: dict[str, Any] = {}
    for arm in arms:
        if not any(r["arm"] == arm for r in rows):
            continue
        summary: dict[str, Any] = {"base_rung": BASE_RUNG[arm]}
        # rate: every layout's tracking per rate, and layout 4's threshold
        rate_rows = _rows(rows, arm, "rate")
        if rate_rows:
            table = {
                name: {_key(r["value"]): _p95(r) for r in rate_rows if r["layout"] == name}
                for name in STUDY_LAYOUTS
            }
            layout4 = {float(r["value"]): _p95(r) for r in rate_rows if r["layout_index"] == RATE_LAYOUT}
            summary["rate"] = {
                "layout": STUDY_LAYOUTS[RATE_LAYOUT - 1],
                **rate_threshold(layout4),
                "tracking_p95_deg": table,
                "passed": {
                    name: {_key(r["value"]): r["passed"] for r in rate_rows if r["layout"] == name}
                    for name in STUDY_LAYOUTS
                },
            }
        speed_rows = _rows(rows, arm, "speed")
        if speed_rows:
            summary["speed"] = {
                "tracking_p95_deg": {
                    name: {_key(r["value"]): _p95(r) for r in speed_rows if r["layout"] == name}
                    for name in STUDY_LAYOUTS
                },
                "peak_joint_speed_deg_s": {_key(r["value"]): r["peak_joint_speed_deg_s"] for r in speed_rows},
                "peak_tool_speed_m_s": {_key(r["value"]): r["peak_tool_speed_m_s"] for r in speed_rows},
            }
        grid_rows = _rows(rows, arm, "rate_speed")
        if grid_rows:
            summary["rate_speed"] = {
                name: {
                    f"{r['value'][0]:g}Hz/{r['value'][1]:g}s": _p95(r)
                    for r in grid_rows
                    if r["layout"] == name
                }
                for name in STUDY_LAYOUTS
                if any(r["layout"] == name for r in grid_rows)
            }
        lookahead_rows = _rows(rows, arm, "lookahead")
        if lookahead_rows:
            answers = []
            for index in LOOKAHEAD_LAYOUTS:
                for rate in LOOKAHEAD_RATES_HZ:
                    curve = {
                        float(r["value"][1]): _p95(r)
                        for r in lookahead_rows
                        if r["layout_index"] == index and float(r["value"][0]) == rate
                    }
                    if curve:
                        layout = STUDY_LAYOUTS[index - 1]
                        answers.append({"layout": layout, "rate_hz": rate, **lookahead_curve_summary(curve)})
            summary["lookahead"] = answers
        stagger_rows = _rows(rows, arm, "stagger")
        if stagger_rows:
            stagger = []
            for rate in STAGGER_RATES_HZ:
                here = {r["way"]: r for r in stagger_rows if float(r["rate_hz"]) == rate}
                if not here:
                    continue
                true_ms = next(iter(here.values()))["true_delay_ms"]
                entry: dict[str, Any] = {
                    "rate_hz": rate,
                    "true_delay_ms": true_ms,
                    "tracking_p95_deg": {way: _p95(row) for way, row in here.items()},
                    "lookahead_ms": {way: row["lookahead_ms"] for way, row in here.items()},
                }
                calibrated = here.get("calibrated", {}).get("calibrated_lookahead_ms")
                if calibrated is not None:
                    error = [round(c - t, 6) for c, t in zip(calibrated, true_ms, strict=True)]
                    entry.update(
                        calibrated_lookahead_ms=calibrated,
                        calibration_error_ms=error,
                        calibration_max_abs_error_ms=max(abs(e) for e in error),
                    )
                stagger.append(entry)
            summary["stagger"] = stagger
        jitter_rows = _rows(rows, arm, "jitter")
        if jitter_rows:
            jitter_summary = []
            for jitter in JITTER_MS:
                values = [_p95(r) for r in jitter_rows if float(r["value"]) == jitter and _p95(r) is not None]
                if values:
                    jitter_summary.append(
                        {
                            "jitter_ms": jitter,
                            "tracking_p95_deg_mean": statistics.fmean(values),
                            "tracking_p95_deg_min": min(values),
                            "tracking_p95_deg_max": max(values),
                            "seeds": len(values),
                        }
                    )
            summary["jitter"] = jitter_summary
        sync_rows = _rows(rows, arm, "sync")
        if sync_rows:
            per_layout = {}
            for name in STUDY_LAYOUTS:
                cells = {}
                for r in sync_rows:
                    if r["layout"] != name or isinstance(r["value"], list):
                        continue
                    cells[f"{float(r['value']):.2f}"] = {
                        "sync_timing_ms": r["metrics"].get("sync_timing_ms"),
                        "sync_error_p95_mm": r["metrics"].get("sync_error_p95_mm"),
                        "passed": r["passed"],
                    }
                per_layout[name] = cells
            sweeps = []
            for index in LOOKAHEAD_LAYOUTS:
                curve = {
                    float(r["value"][1]): r["metrics"]
                    for r in sync_rows
                    if isinstance(r["value"], list) and r["layout_index"] == index
                }
                if not curve:
                    continue
                timed = {
                    la: m["sync_timing_ms"] for la, m in curve.items() if m.get("sync_timing_ms") is not None
                }
                errors = {
                    la: m["sync_error_p95_mm"]
                    for la, m in curve.items()
                    if m.get("sync_error_p95_mm") is not None
                }
                on_time = min(timed, key=lambda la: abs(timed[la])) if timed else None
                sweeps.append(
                    {
                        "layout": STUDY_LAYOUTS[index - 1],
                        "belt_speed_m_s": SYNC_LOOKAHEAD_SPEED_M_S,
                        "lookahead_closest_to_on_time_ms": on_time,
                        "lookahead_lowest_error_ms": min(errors, key=errors.get) if errors else None,
                        "sync_timing_ms": {_key(la): v for la, v in sorted(timed.items())},
                        "sync_error_p95_mm": {_key(la): v for la, v in sorted(errors.items())},
                    }
                )
            summary["sync"] = {"per_layout": per_layout, "lookahead_at_0.25m/s": sweeps}
        out[arm] = summary
    return out


# ── the run and the ledger ─────────────────────────────────────────────────────────────────────────────────


def _arm_record(arm: str) -> dict[str, Any]:
    from robot_controllers.bench.run import _jsonable, rung_record

    record: dict[str, Any] = {"base_rung": rung_record(base_rung(arm))}
    if arm == "ur5e":
        from robot_controllers.bench.ur5e import UR5E_TRUTH, planar_shoulder_bandwidth, ur5e_board_config
        from robot_controllers.bench.ur5e import ur5e_ctc_bandwidth_hz as ctc

        wn, zeta = planar_shoulder_bandwidth()
        record.update(
            board=_jsonable(ur5e_board_config()),
            truth=_jsonable(UR5E_TRUTH),
            ctc_bandwidth_hz=ctc(),
            board_bandwidth={"natural_frequency_rad_s": wn, "damping_ratio": zeta},
        )
    else:
        from robot_controllers.bench.board import BoardConfig
        from robot_controllers.bench.layouts import ctc_record

        record.update(board=_jsonable(BoardConfig()), ctc=ctc_record())
    return record


def run_study(
    *,
    arms: Sequence[str] = ARMS,
    dimensions: Sequence[str] = DIMENSIONS,
    workers: int = 1,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """The study-1 document: generated, layouts, arms, rungs, rows, summary."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from robot_controllers.bench.run import code_version, robot_msgs_version, rung_record

    started = time.perf_counter()
    jobs = plan_study(arms, dimensions, progress=progress)
    if progress:
        what = f"{', '.join(arms)} ({', '.join(dimensions)})"
        progress(f"study: {len(jobs)} episodes on {what}, {workers} worker(s)")
    rows: list[dict[str, Any]] = []

    def report(row: dict[str, Any]) -> None:
        rows.append(row)
        if progress and (len(rows) % 25 == 0 or len(rows) == len(jobs)):
            progress(f"  [{len(rows)}/{len(jobs)}] {time.perf_counter() - started:.0f} s")

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for row in pool.map(run_job, jobs, chunksize=4):
                report(row)
    else:
        for job in jobs:
            report(run_job(job))

    sha, dirty = code_version()
    import mujoco

    rungs: dict[str, Any] = {}
    for job in jobs:
        if job.rung.id not in rungs:
            rungs[job.rung.id] = {**rung_record(job.rung), "plant": job.rung.plant}
    wall_s = time.perf_counter() - started
    return {
        "schema": SCHEMA,
        "generated": {
            "commit": sha,
            "dirty": dirty,
            "robot_msgs": robot_msgs_version(),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "date": _dt.date.today().isoformat(),
            "workers": workers,
            "cost_numbers_contended": workers > 1,
            "episodes": len(rows),
            "wall_s": round(wall_s, 1),
            "load_average_1m": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
            "arms": list(arms),
            "dimensions": list(dimensions),
        },
        "layouts": [
            {
                "index": index,
                "name": name,
                **dict(
                    zip(
                        (
                            "feedforward_model",
                            "feedforward_at",
                            "feedback_model",
                            "feedback_at",
                            "lookahead_s",
                            "interpolation",
                        ),
                        point,
                        strict=True,
                    )
                ),
            }
            for index, (name, point) in enumerate(zip(STUDY_LAYOUTS, STUDY_POINTS, strict=True), start=1)
        ],
        "arms": {arm: _arm_record(arm) for arm in arms},
        "rungs": rungs,
        "rows": rows,
        "summary": summarize_study(rows, arms),
    }


def _plain(value: Any) -> Any:
    """JSON-safe values (NaN/inf -> None, numpy -> Python)."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_study(document: Mapping[str, Any], out_dir: Path | None = None) -> Path:
    """results/study/results.json. One row per line (the rest indented), so a diff shows which rows moved."""
    from robot_controllers.bench.run import RESULTS_DIR

    path = (out_dir or RESULTS_DIR) / STUDY_SLUG / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = _plain(document)
    head = {key: value for key, value in document.items() if key != "rows"}
    text = json.dumps({**head, "rows": "__ROWS__"}, indent=1, allow_nan=False)
    rows = ",\n".join(
        "  " + json.dumps(row, allow_nan=False, separators=(",", ":")) for row in document["rows"]
    )
    text = text.replace('"rows": "__ROWS__"', '"rows": [\n' + rows + "\n ]")
    path.write_text(text + "\n")
    return path
