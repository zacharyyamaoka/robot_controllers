"""The grader: every config of the population on every scenario of every arm, one ledger row each (CONTRACT amendment
5, §5.7).

    robot-controllers grade --workers 2      # -> results/grade/results.json (schema robot-controllers/grade-1)

    from robot_controllers.grade import run_scenario
    row = run_scenario("P4s", base_scenario("bam_fivebar"))    # one row: metrics, passed, failed gates, calibration

Zach, 2026-11-20: "a bunch of different controller/driver configurations (that's the population) and we basically
want to grade them on all of these scenarios ... I can tell you exactly what the best controller is ... perhaps if you
want to really optimize for less impact, or better collision detection."

- **A row** (`run_scenario`): the scenario's coordinates (`Scenario.fields`), the config and its controller name,
  the lookahead it ran, every metric, passed / the failed gates' labels, the calibration it read (both numbers of
  §5.2: the best lookahead and the friction-aware bus delay, next to the datasheet and the twin's true bus) and
  `source` "sim". A collision row runs the same config on the same scenario without the stop first (§5.6's
  threshold) and records that run's largest residual.
- **The objectives** (`OBJECTIVES`, §5.6): `accuracy` (tool_p95_mm, then tracking_p95_deg), `impact`
  (collision_peak_nm, planar also contact_peak_n), `collision_detection` (collision_detect_ms, then collision_margin),
  `cost` (host_duty_pct, then wire_bytes_per_s); or a dict of weights over them, scored on normalised ranks.
  `OBJECTIVE_KINDS` says on which scenarios each is graded (accuracy and cost on the paths; impact on the collision
  and the belt; collision detection on the collision). Only passing rows rank: a config that fails a gate in a
  scenario cannot be recommended for it.
- **The summary** per arm: the best config per (scenario, objective) with the configs within 10 % of it, the
  headline at base conditions per objective, the collision numbers per config, the calibrations, and a grade card
  per config (how often it is best, how often within 10 %).

Machine rules (arm_motion_bench CONTRACT v3.8, bound on every lane): `arm_powered()` reads
`/home/bam/arm_motion_bench/outputs/ARM_POWERED` and `rackq list`; while either says the arm is live the runner
submits nothing new, lets the episodes in hand finish, and polls until it is clear. Rows land in a partial file as
they finish, so a stopped run resumes where it was (`resume=True`).

WHY no traces: ~1400 rows of 200 Hz series would be ~70 MB; the ledger keeps the numbers. WHY the cost numbers are
flagged at workers > 1: host_duty_pct is measured CPU time, and episodes side by side contend.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import socket
import subprocess
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any

from robot_controllers.bench.population import POPULATION, calibration_for, config, config_spec, lookahead_for
from robot_controllers.bench.scenarios import Scenario, grid
from robot_controllers.metrics.names import BY_NAME

SCHEMA = "robot-controllers/grade-1"
GRADE_SLUG = "grade"
OBJECTIVES: Mapping[str, tuple[str, ...]] = {
    "accuracy": ("tool_p95_mm", "tracking_p95_deg"),
    "impact": ("collision_peak_nm", "contact_peak_n"),
    "collision_detection": ("collision_detect_ms", "collision_margin"),
    "cost": ("host_duty_pct", "wire_bytes_per_s"),
}
OBJECTIVE_KINDS: Mapping[str, tuple[str, ...]] = {
    "accuracy": ("path",),
    "impact": ("collision", "contact"),
    "collision_detection": ("collision",),
    "cost": ("path",),
}
WITHIN = 0.10  # "best or within 10 %" on the objective's first measured metric
ARM_POWERED_FLAG = Path("/home/bam/arm_motion_bench/outputs/ARM_POWERED")
POLL_S = 120.0  # how often a paused run re-checks the arm
CHECK_EVERY_S = 30.0  # the arm-powered check is cached this long between submissions
# The real runs the ledger imports (§5.9), when they are on this machine: (run folder, its bundle, what it ran).
# WHY "like P5": fullarm_jtc ran the plan-time model column with the Coulomb held (arm_motion_bench modelff) on the
# boards' PD, the reference read 5.5 ms ahead for every joint: P5's layout with a flat, not a calibrated, lookahead.
REAL_RUNS: tuple[tuple[str, str, str], ...] = (
    (
        "/home/bam/arm_motion_bench/outputs/runs_real/bam_fb/fullarm_jtc_model_ff_la_restload/workspace-v2/dev/"
        "run_jtc-20261121T021137Z-3610132",
        "/home/bam/.local/state/rack/submissions/1120-191044-1d2b/bundle",
        "P5 (model + Coulomb at plan time, board PD) with a flat 5.5 ms lookahead, at speed 0.5, 58.8 Hz",
    ),
)
ROW_METRICS_DROPPED = ("steps_done", "steps_total", "sequence_time_s", "reached_error_deg")  # JointCommandSequence only


# ── the machine rule ───────────────────────────────────────────────────────────────────────────────────────


def arm_powered() -> str | None:
    """Why the real arm is live (the ARM_POWERED flag, or a RUNNING rackq job), or None when it is not."""
    if ARM_POWERED_FLAG.exists():
        return f"{ARM_POWERED_FLAG} exists"
    try:
        listing = subprocess.run(
            ["rackq", "list"], capture_output=True, text=True, timeout=60, check=False
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    running = [line.strip() for line in listing.splitlines() if line.strip().startswith("RUNNING")]
    return f"rackq: {running[0][:120]}" if running else None


# ── one row ────────────────────────────────────────────────────────────────────────────────────────────────


def _round(value: Any) -> Any:
    if isinstance(value, float):
        return float(f"{value:.6g}") if math.isfinite(value) else None
    return value


def _calibration_record(calibration: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The calibration as the ledger keeps it (ms lists, no tuples of seconds)."""
    if calibration is None:
        return None
    return {k: v for k, v in calibration.items() if not k.endswith("_s")}


def run_scenario(
    config_id: str,
    scenario: Scenario,
    *,
    calibration: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """One ledger row: `config_id` on `scenario` (a collision scenario also runs its free twin for the threshold).
    `calibration` defaults to `calibration_for(scenario)` (run once per arm, bus and rate, cached)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from robot_controllers.bench.loop import run_episode
    from robot_controllers.metrics import collision
    from robot_controllers.metrics.episode import run_result

    started = time.perf_counter()
    member = config(config_id)
    calibration = calibration if calibration is not None else calibration_for(scenario)
    spec = config_spec(config_id, scenario, calibration)
    lookahead = lookahead_for(config_id, scenario, calibration)
    rung = scenario.rung()
    factory = scenario.plant_factory()
    free_log = None
    threshold = None
    if scenario.kind == "collision":
        free_scenario = scenario.free()
        free_log = run_episode(spec, free_scenario.rung(), seed=seed, plant_factory=factory)
        residual = collision.log_residual(free_log)
        threshold = None if residual is None else collision.THRESHOLD_FACTOR * float(residual.max())
    log = run_episode(spec, rung, seed=seed, plant_factory=factory)
    result = run_result(log, rung, trace=False, free=free_log)
    metrics = {k: _round(v) for k, v in result.metrics.items() if k not in ROW_METRICS_DROPPED}
    count = scenario.joints
    lookahead_ms = [round(1e3 * v, 6) for v in ([lookahead] * count if isinstance(lookahead, float) else lookahead)]
    row: dict[str, Any] = {
        **scenario.fields(),
        "joints": count,
        "config": config_id,
        "controller": spec.name,
        "feedforward": f"{member.ff_model}@{member.ff_at}",
        "feedback": f"{member.fb_model}@{member.fb_at}",
        "runs_on": spec.runs_on,
        "interpolation": member.interpolation,
        "lookahead_mode": member.lookahead_mode,
        "lookahead_ms": lookahead_ms,
        "seed": int(seed),
        "metrics": metrics,
        "passed": bool(result.passed),
        "failed_gates": [
            f"{g['metric']} {g['op']} {g['value']:g} "
            f"({'n/a' if g['measured'] is None else format(g['measured'], '.4g')})"
            for g in result.gates
            if not g["passed"]
        ],
        "calibration": _calibration_record(calibration) if member.needs_calibration else None,
        "source": "sim",
        "wall_s": round(time.perf_counter() - started, 3),
    }
    if scenario.kind == "collision":
        events = [e for e in log.events if e.get("kind") == "stop_touch"]
        row["collision"] = {
            "threshold": _round(threshold),
            "free_scenario": scenario.free().id,
            "free_tracking_p95_deg": _round(
                run_result(free_log, scenario.free().rung(), trace=False).metrics.get("tracking_p95_deg")
            )
            if free_log is not None
            else None,
            "stop_joint": rung.params.get("stop_joint"),
            "stop_angle_deg": None
            if rung.params.get("stop_angle_rad") is None
            else round(math.degrees(float(rung.params["stop_angle_rad"])), 4),
            "first_touch_s": None if not events else _round(float(events[0]["t"])),
        }
    return row


# ── ranking ────────────────────────────────────────────────────────────────────────────────────────────────


def _signed(metric: str, value: Any) -> float:
    """A sort key: lower is better; a missing value sorts last."""
    if value is None or not math.isfinite(float(value)):
        return math.inf
    return -float(value) if BY_NAME[metric].better == "higher" else float(value)


def objective_metrics(objective: str, rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """The objective's metrics measured on these rows, in its order (impact: whichever of its two the scenario
    has)."""
    names = OBJECTIVES[objective]
    measured = tuple(n for n in names if any(r["metrics"].get(n) is not None for r in rows))
    return measured


def graded(objective: str, kind: str) -> bool:
    return kind in OBJECTIVE_KINDS[objective]


def rank(rows: Sequence[Mapping[str, Any]], objective: str | Mapping[str, float]) -> list[Mapping[str, Any]]:
    """The passing rows of one scenario, best first, on `objective` (a name of OBJECTIVES, or {name: weight} over
    them, scored on each one's rank normalised to 0-1). Rows whose first measured metric is missing are dropped."""
    passing = [r for r in rows if r["passed"]]
    if isinstance(objective, str):
        names = objective_metrics(objective, passing)
        if not names:
            return []
        passing = [r for r in passing if r["metrics"].get(names[0]) is not None]
        return sorted(passing, key=lambda r: (*(_signed(n, r["metrics"].get(n)) for n in names), r["config"]))
    weights = {name: float(w) for name, w in objective.items() if float(w) > 0.0}
    if not weights or any(name not in OBJECTIVES for name in weights):
        raise ValueError(f"weights must be over {tuple(OBJECTIVES)}, got {objective!r}")
    score: dict[str, float] = {r["config"]: 0.0 for r in passing}
    total = 0.0
    for name, weight in weights.items():
        order = rank(passing, name)
        if not order:
            continue
        total += weight
        span = max(1, len(order) - 1)
        positions = {r["config"]: i / span for i, r in enumerate(order)}
        for cid in score:
            score[cid] += weight * positions.get(cid, 1.0)
    if total == 0.0:
        return []
    return sorted(passing, key=lambda r: (score[r["config"]] / total, r["config"]))


def within(rows: Sequence[Mapping[str, Any]], objective: str, fraction: float = WITHIN) -> list[str]:
    """The passing configs whose first measured metric is within `fraction` of the best one's."""
    order = rank(rows, objective)
    if not order:
        return []
    name = objective_metrics(objective, order)[0]
    best = float(order[0]["metrics"][name])
    slack = fraction * abs(best)
    if BY_NAME[name].better == "higher":
        return [r["config"] for r in order if float(r["metrics"][name]) >= best - slack - 1e-12]
    return [r["config"] for r in order if float(r["metrics"][name]) <= best + slack + 1e-12]


# ── the summary ────────────────────────────────────────────────────────────────────────────────────────────


def _best_entry(rows: Sequence[Mapping[str, Any]], objective: str) -> dict[str, Any] | None:
    order = rank(rows, objective)
    if not order:
        return None
    names = objective_metrics(objective, order)
    top = order[0]
    return {
        "config": top["config"],
        "metric": names[0],
        "value": top["metrics"][names[0]],
        "values": {n: top["metrics"].get(n) for n in names},
        "within_10pct": within(rows, objective),
        "ranking": [r["config"] for r in order],
        "passing": sum(1 for r in rows if r["passed"]),
        "of": len(rows),
    }


def _headline_scenario(scenarios: Sequence[str], rows_by: Mapping[str, list], objective: str) -> str | None:
    """The scenario at base conditions where `objective` is graded: the base for accuracy and cost, the [event]
    collision (or the belt) for impact and collision detection."""
    for sid in scenarios:
        rows = rows_by.get(sid) or []
        if rows and graded(objective, rows[0]["kind"]) and (
            rows[0]["axis"] == "base" or (rows[0]["axis"] == "event" and objective_metrics(objective, rows))
        ):
            return sid
    return None


def summarize_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One arm's summary (the module docstring)."""
    scenarios: list[str] = []
    rows_by: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if row["scenario"] not in rows_by:
            scenarios.append(row["scenario"])
            rows_by[row["scenario"]] = []
        rows_by[row["scenario"]].append(row)
    configs = [c for c in POPULATION if any(r["config"] == c for r in rows)]
    best: dict[str, dict[str, Any]] = {}
    for sid in scenarios:
        kind = rows_by[sid][0]["kind"]
        entries = {o: _best_entry(rows_by[sid], o) for o in OBJECTIVES if graded(o, kind)}
        best[sid] = {o: e for o, e in entries.items() if e is not None}
    headline: dict[str, Any] = {}
    for objective in OBJECTIVES:
        sid = _headline_scenario(scenarios, rows_by, objective)
        entry = best.get(sid, {}).get(objective) if sid else None
        headline[objective] = None if entry is None else {"scenario": sid, **entry}
    cards: dict[str, Any] = {}
    for cid in configs:
        mine = [r for r in rows if r["config"] == cid]
        per: dict[str, dict[str, int]] = {}
        for objective in OBJECTIVES:
            graded_here = [sid for sid in scenarios if objective in best[sid]]
            per[objective] = {
                "graded": len(graded_here),
                "best": sum(1 for sid in graded_here if best[sid][objective]["config"] == cid),
                "within_10pct": sum(1 for sid in graded_here if cid in best[sid][objective]["within_10pct"]),
            }
        cards[cid] = {
            "rows": len(mine),
            "passed": sum(1 for r in mine if r["passed"]),
            "failed_scenarios": [r["scenario"] for r in mine if not r["passed"]],
            "objectives": per,
            "best": sum(v["best"] for v in per.values()),
            "within_10pct": sum(v["within_10pct"] for v in per.values()),
            "graded": sum(v["graded"] for v in per.values()),
        }
    collision = {}
    for sid in scenarios:
        first = rows_by[sid][0]
        if first["kind"] == "collision" and first["axis"] == "event":
            collision = {
                "scenario": sid,
                "stop_joint": (first.get("collision") or {}).get("stop_joint"),
                "configs": {
                    r["config"]: {
                        "collision_detect_ms": r["metrics"].get("collision_detect_ms"),
                        "collision_peak_nm": r["metrics"].get("collision_peak_nm"),
                        "collision_margin": r["metrics"].get("collision_margin"),
                        "threshold": (r.get("collision") or {}).get("threshold"),
                        "passed": r["passed"],
                    }
                    for r in rows_by[sid]
                },
            }
            break
    calibrations: dict[str, Any] = {}
    for row in rows:
        cal = row.get("calibration")
        if cal:
            key = f"{row['host_rate_hz']:g}Hz·bus={row['bus']['label']}"
            calibrations.setdefault(key, cal)
    return {
        "base_scenario": scenarios[0] if scenarios else None,
        "scenarios": scenarios,
        "headline": headline,
        "best": best,
        "cards": cards,
        "collision": collision,
        "calibrations": calibrations,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    arms_seen: list[str] = []
    for row in rows:
        if row.get("source", "sim") == "sim" and row["arm"] not in arms_seen:
            arms_seen.append(row["arm"])
    return {arm: summarize_arm([r for r in rows if r["arm"] == arm and r.get("source", "sim") == "sim"])
            for arm in arms_seen}


# ── the run ────────────────────────────────────────────────────────────────────────────────────────────────


def _job(job: tuple[str, Scenario, Mapping[str, Any] | None]) -> dict[str, Any]:
    config_id, scenario, calibration = job
    return run_scenario(config_id, scenario, calibration=calibration)


def _calibration_job(key: tuple[str, Any, float]) -> tuple[tuple[str, Any, float], dict[str, Any]]:
    from robot_controllers.bench.population import scenario_calibration

    return key, scenario_calibration(*key)


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return row["arm"], row["scenario"], row["config"]


def default_out_dir() -> Path:
    from robot_controllers.bench.run import RESULTS_DIR

    return RESULTS_DIR / GRADE_SLUG


def read_partial(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line cut by a kill mid-write: that row runs again
    return rows


def plan_grade(arms: Sequence[str], configs: Sequence[str] = POPULATION) -> list[tuple[str, Scenario]]:
    """Every (config, scenario) of the grade, arm by arm, scenario by scenario."""
    return [(cid, scenario) for arm in arms for scenario in grid(arm) for cid in configs]


def run_grade(
    arms: Sequence[str] | None = None,
    *,
    workers: int = 2,
    configs: Sequence[str] = POPULATION,
    out_dir: Path | None = None,
    resume: bool = True,
    wait_for_arm: bool = True,
    progress: Callable[[str], None] | None = print,
    powered: Callable[[], str | None] = arm_powered,
) -> dict[str, Any]:
    """Grade `configs` on every scenario of `arms` (default: every registered arm) and return the grade-1 document
    (`write_grade` writes it). Rows go to `<out_dir>/partial.jsonl` as they finish; with `resume` a row already there
    is not run again. While `powered()` says the arm is live nothing new starts: with `wait_for_arm` the run polls
    every POLL_S until it is clear, else it returns early with `generated.complete = False`."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from robot_controllers.bench.arms import arms as registered

    say = progress or (lambda _text: None)
    arm_names = list(arms) if arms else [a.name for a in registered()]
    out = out_dir or default_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    partial = out / "partial.jsonl"
    done = {_row_key(r): r for r in read_partial(partial)} if resume else {}
    if not resume and partial.exists():
        partial.unlink()
    started = time.perf_counter()
    plan = plan_grade(arm_names, configs)
    todo = [(cid, s) for cid, s in plan if (s.arm, s.id, cid) not in done]
    say(f"grade: {len(arm_names)} arms, {len(plan)} rows ({len(plan) - len(todo)} already in {partial.name}), "
        f"{max(1, workers)} workers")

    last_check = [-math.inf, None]

    def live() -> str | None:
        now = time.monotonic()
        if now - last_check[0] >= CHECK_EVERY_S:
            last_check[0], last_check[1] = now, powered()
        return last_check[1]

    def hold(in_flight: Iterable[Future]) -> bool:
        """Wait while the arm is live; False = give up (wait_for_arm off)."""
        reason = live()
        if reason is None:
            return True
        say(f"  arm powered ({reason}): letting {len(list(in_flight))} episode(s) finish, then pausing")
        if not wait_for_arm:
            return False
        while reason is not None:
            time.sleep(POLL_S)
            last_check[0] = -math.inf
            reason = live()
            if reason is not None:
                say(f"  still powered ({reason}); next check in {POLL_S:g} s")
        say("  arm clear: resuming")
        return True

    complete = True
    with partial.open("a") as sink, ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        # the calibrations every calibrated config reads, once per (arm, bus, rate)
        keys = []
        for _cid, scenario in todo:
            key = (scenario.arm, scenario.bus, scenario.host_rate_hz)
            if key not in keys:
                keys.append(key)
        calibrations: dict[tuple[str, Any, float], dict[str, Any]] = {}
        pending: set[Future] = set()
        queue = list(keys)
        while queue or pending:
            while queue and len(pending) < max(1, workers):
                if not hold(pending):
                    queue.clear()
                    complete = False
                    break
                pending.add(pool.submit(_calibration_job, queue.pop(0)))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                key, record = future.result()
                calibrations[key] = record
                say(f"  calibrated {key[0]} at {key[2]:g} Hz on {record['rung'].split('·bus=')[1].split('·')[0]}: "
                    f"best lookahead {record['best_lookahead_ms']} ms, bus delay {record['bus_delay_ms']} ms")
        jobs = [
            (cid, s, calibrations[(s.arm, s.bus, s.host_rate_hz)])
            for cid, s in todo
            if (s.arm, s.bus, s.host_rate_hz) in calibrations
        ]
        complete = complete and len(jobs) == len(todo)
        pending = set()
        count = 0
        while (jobs and complete) or pending:
            while jobs and complete and len(pending) < max(1, workers):
                if not hold(pending):
                    complete = False
                    break
                pending.add(pool.submit(_job, jobs.pop(0)))
            if not pending:
                break
            finished, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                row = future.result()
                done[_row_key(row)] = row
                sink.write(json.dumps(row, allow_nan=False) + "\n")
                sink.flush()
                count += 1
                if count % 25 == 0 or not jobs and not pending:
                    say(f"  [{len(done)}/{len(plan)}] {row['config']} @ {row['scenario']}: "
                        f"{'pass' if row['passed'] else 'FAIL ' + '; '.join(row['failed_gates'])} ({row['wall_s']} s)")
    wall = time.perf_counter() - started
    order = {(s.arm, s.id, cid): i for i, (cid, s) in enumerate(plan)}
    rows = sorted((r for k, r in done.items() if k in order), key=lambda r: order[_row_key(r)])
    complete = complete and len(rows) == len(plan)
    return grade_document(rows, arm_names, workers=workers, wall_s=wall, complete=complete, configs=configs,
                          real_rows=[r for r in real_rows(say) if r["arm"] in arm_names])


def real_rows(progress: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """`source: real` rows for the REAL_RUNS on this machine (a missing one is skipped and said)."""
    from robot_controllers.real.import_fullarm import real_row

    out = []
    for run_dir, bundle_dir, ran in REAL_RUNS:
        if not (Path(run_dir).exists() and Path(bundle_dir).exists()):
            if progress:
                progress(f"  real run not on this machine: {run_dir}")
            continue
        row = real_row(run_dir, bundle_dir)
        row["ran"] = ran
        row["config_like"] = "P5"
        out.append(row)
    return out


def grade_document(
    rows: Sequence[Mapping[str, Any]],
    arm_names: Sequence[str],
    *,
    workers: int,
    wall_s: float,
    complete: bool,
    configs: Sequence[str] = POPULATION,
    real_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    from robot_controllers.bench.arms import arm as registered
    from robot_controllers.bench.population import CONFIGS
    from robot_controllers.bench.run import code_version, robot_msgs_version

    sha, dirty = code_version()
    import mujoco

    all_rows = [*rows, *real_rows]
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
            "wall_s": round(wall_s, 1),
            "episodes": sum(2 if r["kind"] == "collision" else 1 for r in rows),
            "rows": len(rows),
            "complete": complete,
            "load_average_1m": os.getloadavg()[0] if hasattr(os, "getloadavg") else None,
        },
        "arms": [
            {"name": a, "title": registered(a).title, "joints": list(registered(a).joint_names),
             "real_station": registered(a).real_station}
            for a in arm_names
        ],
        "population": [
            {
                "config": cid,
                "feedforward": f"{CONFIGS[cid].ff_model}@{CONFIGS[cid].ff_at}",
                "feedback": f"{CONFIGS[cid].fb_model}@{CONFIGS[cid].fb_at}",
                "lookahead_mode": CONFIGS[cid].lookahead_mode,
                "interpolation": CONFIGS[cid].interpolation,
                "title": CONFIGS[cid].title,
            }
            for cid in configs
        ],
        "objectives": {name: {"metrics": list(metrics), "kinds": list(OBJECTIVE_KINDS[name])}
                       for name, metrics in OBJECTIVES.items()},
        "rule": "per (scenario, objective): among the configs that pass every gate of the scenario, the best on the "
        "objective's metrics in order (a missing one last); 'within 10 %' = within 10 % of the best on the first "
        "measured one. A collision row's threshold is 1.25 x the same config's largest residual on the same "
        "scenario without the stop.",
        "rows": all_rows,
        "summary": summarize(rows),
    }


def write_grade(document: Mapping[str, Any], out_dir: Path | None = None, *, drop_partial: bool = True) -> Path:
    """results/grade/results.json; the partial file goes once a complete ledger is written."""
    out = out_dir or default_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "results.json"
    # WHY one row per line: the ledger stays a few MB and a re-grade's diff reads row by row
    head = json.dumps({k: v for k, v in document.items() if k != "rows"}, indent=1, allow_nan=False)
    rows = ",\n".join(json.dumps(r, separators=(",", ":"), allow_nan=False) for r in document["rows"])
    path.write_text(head[: -len("\n}")] + ',\n "rows": [\n' + rows + "\n ]\n}\n")
    partial = out / "partial.jsonl"
    if drop_partial and document["generated"].get("complete") and partial.exists():
        partial.unlink()
    return path


def load_ledger(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or default_out_dir() / "results.json").read_text())
