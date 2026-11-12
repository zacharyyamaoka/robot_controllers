"""Log results-1 documents to Weights & Biases: one project per black box, one run per (controller, rung, seed).

    env -u VIRTUAL_ENV uv run --frozen --directory /home/bam/robot_controllers --extra tracking robot-controllers wandb --views --verify

CONTRACT.md section 6. The results JSON is the ledger (the record); W&B only ever holds a copy of it, so nothing here
re-scores: every number, gate verdict, rank and recommendation is the ledger's own (bam_trackio BLACK_BOX_TEMPLATE
rule 4, "adapters translate and never re-score").

- `records(results)` turns one box's results document into plain dicts, one per run, with no wandb import (tested).
- `push(records)` writes them: a new id makes a run; an existing id (the same numbers) gets only the keys it lacks or
  holds differently. `retire(records)` marks the box's older runs of the same (controller, rung, seed)
  `evaluation_state = superseded` (never deleted).
- `build_views(results_by_box)` builds per box a Scoreboard, Performance vs cost, Contact and Timing report and one
  saved view per rung, plus the cross-box `Controllers` hub report, updated in place through `wandb_views.json`.
- `verify(results_by_box)` reads it all back through the W&B API. `log_weave(results)` adds the Weave leaderboard
  through bam_trackio's `bbo.py`, loaded by file.

wandb, wandb_workspaces and weave are imported only inside the functions that talk to W&B (the `tracking` extra), so
importing this module, building records and `--dry-run` need none of them. Patterns and traps carried over from
sim2real `tracking/wandb_log.py` and `tracking/rigs.py` (2026-11-11), each marked WHY where it is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from robot_controllers.core import HUB_PROJECT, WANDB_ENTITY
from robot_controllers.metrics.names import BY_NAME

SCHEMA = "robot-controllers/results-1"
REPO = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO / "results"
REGISTRY = Path(__file__).resolve().parent / "wandb_views.json"  # report and view URLs: a rebuild updates in place
WANDB_DIR = Path.home() / ".cache" / "robot_controllers" / "wandb"  # W&B's local run files: a cache, never the repo
# WHY loaded by file, never imported (sim2real rigs.py, 2026-11-11): bbo.py imports only the standard library at module
# level; importing the bam_trackio package pulls trackio, the pick ladder and the trajgen bench. Override for a test.
BBO = Path(os.environ.get("BAM_TRACKIO_BBO", "/home/bam/bam_ws/src/dev/bam_trackio/bam_trackio/bbo.py"))

NAME_LIMIT_BYTES = 128  # W&B's run display name limit
X_AXIS = "t_s"  # the history's time axis: every trace series is logged against simulated time, in seconds
# A trace series sampled on its own clock names it here; everything else is on the plant clock `t` (CONTRACT §5).
TRACE_TIME_BASES = {"contact_fn_n": "contact_t", "compute_us": "tick_t"}
TRACE_TIME_KEYS = frozenset({"t", "contact_t", "tick_t"})
_HISTORY_NAMES = {"tool_x_mm": "tool_mm/x", "tool_z_mm": "tool_mm/z", "contact_fn_n": "contact/fn_n",
                  "compute_us": "cost/compute_us"}

# ─────────────────────────────────────────────────────────────────────────────────────────── pure: records


def metrics_digest(metrics: Mapping[str, Any]) -> str:
    """sha1 of the run's metrics JSON (sorted keys): the content part of the run id."""
    return hashlib.sha1(json.dumps(dict(metrics), sort_keys=True, default=str).encode()).hexdigest()


def run_id(box_slug: str, controller: str, rung: str, seed: int, digest: str) -> str:
    """`sha1("robot_controllers|box|controller|rung|seed|digest")[:16]` (CONTRACT §6), box = the box slug.

    WHY the metrics digest is in the id: identical numbers resume the same run (a re-log adds nothing), new numbers make
    a new run and the old one is superseded, so W&B keeps every evaluation the ledger ever had.
    """
    text = "|".join(("robot_controllers", box_slug, controller, rung, str(int(seed)), digest))
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def fit_bytes(text: str, limit: int = NAME_LIMIT_BYTES) -> str:
    """`text` cut to `limit` UTF-8 bytes on a character boundary, with an explicit ellipsis when cut (never a silent clip)."""
    raw = text.encode()
    if len(raw) <= limit:
        return text
    ellipsis = "…"
    return raw[: limit - len(ellipsis.encode())].decode("utf-8", errors="ignore") + ellipsis


def run_name(controller: str, rung: str, seed: int = 0) -> str:
    """`"<controller> @ <rung>"` (≤ 128 bytes). A seed other than 0 is appended, so seeds are told apart in tables."""
    name = f"{controller} @ {rung}" + (f" · seed {seed}" if seed else "")
    return fit_bytes(name)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _number(value: Any) -> str:
    """A gate number as a person reads it: ints as ints, whole floats with one decimal ("1.0"), else 4 significant digits."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if value.is_integer() and abs(value) < 1e15:
        return f"{value:.1f}"
    return f"{value:.4g}"


def gate_text(row: Mapping[str, Any]) -> str:
    """`gate/<metric>` text from one RunResult gate row: "pass 0.62 <= 1.0", "FAIL 72.4 <= 60.0" or "not measured"."""
    measured, note = row.get("measured"), row.get("note") or ""
    if measured is None or (isinstance(measured, float) and not math.isfinite(measured)):
        return "not measured" if note in ("", "not measured") else f"not measured ({note})"
    verdict = "pass" if row.get("passed") else "FAIL"
    text = f"{verdict} {_number(measured)} {row.get('op')} {_number(row.get('value'))}"
    return f"{text} ({note})" if note else text


def history_key(trace_key: str) -> str:
    """The W&B history name of a trace key: "q_deg.shoulder" -> "q_deg/shoulder" (W&B sections panels by the prefix)."""
    if trace_key in _HISTORY_NAMES:
        return _HISTORY_NAMES[trace_key]
    if "." in trace_key:
        family, joint = trace_key.split(".", 1)
        return f"{family}/{joint}"
    return f"trace/{trace_key}"


def history_rows(trace: Mapping[str, Sequence[Any]]) -> tuple[list[dict[str, float]], list[str]]:
    """The trace as W&B history rows on one time axis, each series only at its own sample times.

    Returns (rows sorted by time, each `{"t_s": time, <series>: value, ...}`; trace keys that could not be placed). A
    None or non-finite value is dropped (W&B charts skip them anyway, and a NaN in history breaks the run's summary
    sort: sim2real `_clean`); a row with no value left is not logged. WHY one merged axis rather than one step per base:
    W&B's step must increase, and the plant (200 Hz), contact (1 kHz window) and host-tick clocks interleave.
    """
    by_time: dict[float, dict[str, float]] = {}
    skipped: list[str] = []
    for key, values in trace.items():
        if key in TRACE_TIME_KEYS:
            continue
        times = trace.get(TRACE_TIME_BASES.get(key, "t"))
        if not isinstance(values, (list, tuple)) or not times or len(times) != len(values):
            skipped.append(key)
            continue
        name = history_key(key)
        for time, value in zip(times, values, strict=True):
            if _finite(value) and _finite(time):
                by_time.setdefault(round(float(time), 9), {})[name] = float(value)
    rows = [{X_AXIS: time, **values} for time, values in sorted(by_time.items())]
    return rows, sorted(skipped)


def _config_value(value: Any) -> Any:
    """A config value W&B can filter on: scalars and lists of scalars as they are, anything nested as sorted JSON text."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict) or (isinstance(value, (list, tuple)) and any(isinstance(v, (dict, list, tuple)) for v in value)):
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, tuple):
        return list(value)
    return value


def _flat(prefix: str, mapping: Mapping[str, Any] | None) -> dict[str, Any]:
    return {f"{prefix}/{key}": _config_value(value) for key, value in (mapping or {}).items()}


def project_for(results: Mapping[str, Any]) -> str:
    """The box's W&B project: `Box.wandb_project` from the registry; a ledger that names another project is refused."""
    from robot_controllers.registry import box_by_slug

    box = results["box"]
    try:
        project = box_by_slug(box["slug"]).wandb_project
    except KeyError:  # a box the registry does not know yet: trust the ledger
        return box["wandb_project"]
    if box.get("wandb_project") not in (None, project):
        raise ValueError(f"results name W&B project {box['wandb_project']!r}, but Box.wandb_project is {project!r}")
    return project


def _rank(summary: Mapping[str, Any], controller: str, rung: str) -> int | None:
    """The controller's objective rank on the rung from the ledger's summary (`per_controller[c]["rank"][rung]`, as
    metrics.summary.summarize writes it); None when it has none. Never computed here."""
    mine = (summary.get("per_controller") or {}).get(controller) or {}
    ranks = mine.get("rank")
    rank = ranks.get(rung) if isinstance(ranks, Mapping) else None
    return int(rank) if _finite(rank) else None


def records(results: Mapping[str, Any], *, sha: str = "unknown", dirty: bool = False,
            project: str | None = None) -> list[dict[str, Any]]:
    """The W&B runs one box's results document makes, as plain dicts (no wandb needed), one per RunResult.

    Each: {id, project, name, group, job_type, tags, notes, config, summary, history, skipped_series}. `sha` / `dirty`
    are the logging checkout's (`logger_commit`); `commit` / `dirty` in the config are the ledger's own `generated`
    block, the code that made the numbers. `project` overrides the box's project (a scratch project).
    """
    if results.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} document: schema {results.get('schema')!r}")
    box, generated, summary = results["box"], results.get("generated") or {}, results.get("summary") or {}
    project = project or project_for(results)
    rungs = {rung["id"]: rung for rung in results.get("rungs", [])}
    controllers = {spec["name"]: spec for spec in results.get("controllers", [])}
    per_rung = summary.get("per_rung") or {}
    out = []
    for result in results.get("results", []):
        controller, rung_id, seed = result["controller"], result["rung"], int(result.get("seed", 0))
        rung, spec = rungs.get(rung_id, {"id": rung_id}), controllers.get(controller, {"name": controller})
        metrics = dict(result.get("metrics") or {})
        digest = metrics_digest(metrics)
        objective_metric = rung.get("objective", "")
        timing = rung.get("timing") or {}
        config = {
            "schema": SCHEMA, "box": box["name"], "box_slug": box["slug"], "input_type": box["input"],
            "output_type": box["output"], "controller": controller, "controller_title": spec.get("title", controller),
            "runs_on": spec.get("runs_on", ""), "variant_of": spec.get("variant_of") or "",
            **_flat("param", spec.get("config")),
            "rung": rung_id, "rung_title": rung.get("title", rung_id), "rung_family": rung.get("family", ""),
            "rung_order": rung.get("order"), "rung_duration_s": rung.get("duration_s"),
            **_flat("rung_param", rung.get("params")), **_flat("timing", timing), **_flat("scene", rung.get("scene")),
            "host_rate_hz": timing.get("rate_hz"), "objective_metric": objective_metric,
            "minimize": rung.get("minimize", True), "seed": seed, "digest": digest[:12],
            "plant": generated.get("plant", ""), "commit": generated.get("commit", sha), "dirty": bool(generated.get("dirty", dirty)),
            "robot_msgs": generated.get("robot_msgs", ""), "mujoco": generated.get("mujoco", ""),
            "python": generated.get("python", ""), "results_date": generated.get("date", ""),
            "logger_commit": sha, "logger_dirty": bool(dirty), "evaluation_state": "latest",
        }
        gates = list(result.get("gates") or [])
        passed = bool(result.get("passed"))
        objective = result.get("objective")
        run_summary: dict[str, Any] = {name: metrics.get(name) for name in BY_NAME}
        run_summary.update({key: value for key, value in metrics.items() if key not in run_summary})
        run_summary = {key: (value if not (isinstance(value, float) and not math.isfinite(value)) else None)
                       for key, value in run_summary.items()}
        run_summary.update({f"gate/{row['metric']}": gate_text(row) for row in gates})
        best = (per_rung.get(rung_id) or {}).get("best")
        run_summary.update({
            # WHY ints for pass flags (sim2real's passes_all): W&B bar panels and filters handle 0/1, not booleans
            "passed": int(passed), "objective": objective if _finite(objective) else None,
            "objective_metric": objective_metric, "gates_passed": sum(1 for row in gates if row.get("passed")),
            "gates_total": len(gates), "best_on_rung": int(best == controller),
        })
        rank = _rank(summary, controller, rung_id)
        if rank is not None:
            run_summary["objective_rank"] = rank
        rows, skipped = history_rows(result.get("trace") or {})
        # WHY the KPIs are logged in history once as well (sim2real, 2026-11-11): W&B's bar panel looked metrics up in
        # history and found no runs when they were only in the summary
        kpis = {key: value for key, value in run_summary.items()
                if _finite(value) and not key.startswith("gate/") and key not in ("objective_rank",)}
        failed = [f"{row['metric']} {row['op']} {_number(row['value'])}" for row in gates if not row.get("passed")]
        out.append({
            "id": run_id(box["slug"], controller, rung_id, seed, digest),
            "project": project,
            "name": run_name(controller, rung_id, seed),
            "group": controller,
            "job_type": "rung",
            "tags": sorted({rung.get("family") or "unknown", "sim", "latest"}),
            "notes": f"{spec.get('title', controller)} on {rung.get('title', rung_id)}: "
                     + ("passed every gate" if passed else ("fails " + ", ".join(failed)) if failed else "did not pass"),
            "config": config,
            "summary": run_summary,
            "history": rows + ([kpis] if kpis else []),
            "skipped_series": skipped,
        })
    return out


def compact(record: Mapping[str, Any]) -> dict[str, Any]:
    """A record without its history rows (their count instead): what --dry-run prints."""
    out = {key: value for key, value in record.items() if key != "history"}
    out["history_rows"] = len(record.get("history", []))
    return out


def code_version(repo: Path = REPO) -> tuple[str, bool]:
    """(HEAD sha, whether src/ has uncommitted changes) of this checkout; ("unknown", False) outside git."""
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "src"], capture_output=True,
                                    text=True, check=True).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def browser_url(url: str) -> str:
    """A W&B URL a person can click: forge.coreweave.com rewritten to wandb.ai, spaces and "·" percent-encoded.

    WHY the rewrite (sim2real / bam_trackio, 2026-11-11): wandb-workspaces 0.4.13 returns forge.coreweave.com/<entity>/...,
    which 404s; wandb.ai redirects to the right place. WHY the encoding: these project names hold spaces, and a raw
    space ends a markdown link in a report.
    """
    url = re.sub(r"^https://forge\.coreweave\.com/(?:wandb/)?", "https://wandb.ai/", url)
    return quote(url, safe=":/?=&%#+@,;~-._")


def project_url(entity: str, project: str) -> str:
    return browser_url(f"https://wandb.ai/{entity}/{project}")


def _clean(values: Mapping[str, Any]) -> dict[str, Any]:
    """Drop None and non-finite numbers (sim2real: a NaN in history breaks the run's summary sort)."""
    return {k: v for k, v in values.items() if v is not None and not (isinstance(v, float) and not math.isfinite(v))}


def _differs(stored: Any, value: Any) -> bool:
    """Whether a run's stored value is not `value` (numbers to 1e-9: W&B hands back ints and floats alike)."""
    if _finite(value) and _finite(stored):
        return abs(float(stored) - float(value)) > 1e-9
    return stored != value


def summary_patch(summary: Mapping[str, Any], have: Mapping[str, Any]) -> dict[str, Any]:
    """The summary keys an existing run lacks or holds differently (None / NaN never sent: they would never match)."""
    return {key: value for key, value in _clean(summary).items() if key not in have or _differs(have[key], value)}


def to_supersede(batch: Sequence[Mapping[str, Any]], runs: Iterable[Any]) -> list[Any]:
    """The latest runs (objects with .id and .config) of the batch's (box, controller, rung, seed) keys that are not the
    batch's own runs: an older evaluation of the same thing, to mark superseded."""
    keys = {(r["config"]["box_slug"], r["config"]["controller"], r["config"]["rung"], int(r["config"]["seed"])) for r in batch}
    ids = {r["id"] for r in batch}
    out = []
    for run in runs:
        config = run.config
        if config.get("evaluation_state") != "latest":
            continue
        key = (config.get("box_slug"), config.get("controller"), config.get("rung"), int(config.get("seed") or 0))
        if key in keys and run.id not in ids:
            out.append(run)
    return out


# ─────────────────────────────────────────────────────────────────────────────────────────── W&B: writes


def push(batch: Iterable[Mapping[str, Any]], *, entity: str = WANDB_ENTITY, project: str | None = None) -> list[dict]:
    """Write records as W&B runs (resume="allow"). A new id makes a run with its whole history; an existing id holds
    these numbers already, so it gets only the summary keys it lacks or holds differently (and its config refreshed).
    Returns one {id, url, name, project, resumed, patched} per record."""
    import wandb

    WANDB_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=60)
    logged = []
    for record in batch:
        target = project or record["project"]
        try:
            have = dict(api.run(f"{entity}/{target}/{record['id']}").summary)
        except Exception:  # wandb raises CommError / ValueError for a run that does not exist yet
            have = None
        run = wandb.init(entity=entity, project=target, id=record["id"], name=record["name"], group=record["group"],
                         job_type=record["job_type"], tags=record["tags"], notes=record["notes"], config=record["config"],
                         resume="allow", reinit="create_new", allow_val_change=True, dir=str(WANDB_DIR),
                         settings=wandb.Settings(silent=True, console="off"))
        summary = _clean(record["summary"])
        if have is None:
            run.define_metric(X_AXIS)
            for prefix in sorted({key.split("/", 1)[0] for row in record["history"] for key in row if "/" in key}):
                run.define_metric(f"{prefix}/*", step_metric=X_AXIS)
            for row in record["history"]:
                row = _clean(row)
                if row:
                    run.log(row)
            patched: list[str] = []
        else:
            # WHY only what is missing or changed: the id is the numbers' content id, so the run already holds them;
            # re-logging the trace would add steps W&B drops (a step must increase) and duplicate every point
            summary = summary_patch(record["summary"], have)
            patched = sorted(summary)
        if summary:
            run.summary.update(summary)
        logged.append({"id": run.id, "url": browser_url(run.url), "name": record["name"], "project": target,
                       "resumed": have is not None, "patched": patched})
        run.finish()
    return logged


def retire(batch: Sequence[Mapping[str, Any]], *, entity: str = WANDB_ENTITY, project: str | None = None) -> int:
    """Mark every latest run of the same (box, controller, rung, seed) that is not in `batch` as superseded: config
    `evaluation_state = superseded` and tag "superseded" instead of "latest". Never deletes. Returns how many changed."""
    import wandb

    if not batch:
        return 0
    api = wandb.Api(timeout=60)
    changed = 0
    for target in sorted({project or r["project"] for r in batch}):
        for slug in sorted({r["config"]["box_slug"] for r in batch}):
            found = api.runs(f"{entity}/{target}", filters={"config.evaluation_state": "latest", "config.box_slug": slug})
            for run in to_supersede(batch, found):
                run.config["evaluation_state"] = "superseded"
                run.tags = sorted({*run.tags, "superseded"} - {"latest"})
                run.update()
                changed += 1
    return changed


# ─────────────────────────────────────────────────────────────────────────────────────────── W&B: views (pure parts)


def view_name(text: str) -> str:
    """A saved-view name wandb-workspaces accepts: it refuses symbol characters (Unicode So/Sk/Sm/Sc: "→", "×", "✓"),
    so arrows become "to" and other symbols are dropped."""
    text = text.replace("→", "to").replace("×", "x")
    kept = "".join(ch for ch in text if not unicodedata.category(ch).startswith(("So", "Sk", "Sm", "Sc", "Cs")))
    return re.sub(r"\s+", " ", kept).strip()


def _unit(metric: str) -> str:
    unit = BY_NAME[metric].unit if metric in BY_NAME else ""
    return {"deg": "°", "us": " µs", "": "", "count": "", "%": " %"}.get(unit, f" {unit}")


def _seed_groups(results: Mapping[str, Any]) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for result in results.get("results", []):
        groups.setdefault((result["controller"], result["rung"]), []).append(result)
    return groups


def scoreboard_markdown(results: Mapping[str, Any]) -> str:
    """Controller × rung: pass/fail and the objective, best per rung in bold, from the ledger (nothing re-scored)."""
    rungs = sorted(results.get("rungs", []), key=lambda r: r.get("order", 0))
    summary = results.get("summary") or {}
    per_rung, per_controller = summary.get("per_rung") or {}, summary.get("per_controller") or {}
    groups = _seed_groups(results)
    header = "| Controller | Runs on | " + " | ".join(f"{r.get('order', '')}. {r['id']}" for r in rungs) + " | Rungs passed |"
    lines = [header, "|" + "---|" * (len(rungs) + 3)]
    for spec in results.get("controllers", []):
        name = spec["name"]
        cells = []
        for rung in rungs:
            runs = groups.get((name, rung["id"]), [])
            if not runs:
                cells.append("not run")
                continue
            values = [r.get("objective") for r in runs if _finite(r.get("objective"))]
            value = statistics.median(values) if values else None
            shown = f"{_number(value)}{_unit(rung.get('objective', ''))}" if value is not None else "not measured"
            ok = all(r.get("passed") for r in runs)
            fails = sorted({g["metric"] for r in runs for g in r.get("gates", []) if not g.get("passed")})
            cell = f"✓ {shown}" if ok else f"✗ {shown} · fails {', '.join(fails)}"
            if len(runs) > 1:
                cell += f" ({sum(1 for r in runs if r.get('passed'))}/{len(runs)} seeds)"
            if (per_rung.get(rung["id"]) or {}).get("best") == name:
                cell = f"**{cell}**"
            cells.append(cell)
        mine = per_controller.get(name) or {}
        passed = f"{mine.get('rungs_passed', '?')}/{mine.get('rungs_total', len(rungs))}"
        lines.append(f"| {name} | {spec.get('runs_on', '')} | " + " | ".join(cells) + f" | {passed} |")
    return "\n".join(lines)


def gates_markdown(results: Mapping[str, Any]) -> str:
    """The ruler: every rung's gates and objective."""
    parts = []
    for rung in sorted(results.get("rungs", []), key=lambda r: r.get("order", 0)):
        gates = "; ".join(f"`{g['metric']} {g['op']} {_number(g['value'])}` ({g.get('why', '')})" for g in rung.get("gates", []))
        way = "lower" if rung.get("minimize", True) else "higher"
        parts.append(f"- **{rung.get('order', '')}. {rung.get('title', rung['id'])}** (`{rung['id']}`): {rung.get('description', '')} "
                     f"Objective `{rung.get('objective', '')}`, {way} is better. Gates: {gates or 'none'}.")
    return "\n".join(parts)


def recommendation_markdown(results: Mapping[str, Any]) -> str:
    recommendation = (results.get("summary") or {}).get("recommendation") or {}
    name = recommendation.get("controller") or recommendation.get("name")
    if not name:
        return "No recommendation in the ledger yet."
    return f"**Recommended: `{name}`**. {recommendation.get('why', '')}"


def pareto_markdown(results: Mapping[str, Any]) -> str:
    """The ledger's Pareto sets (per belt rung: R1 tracking, that rung's contact max, compute p99)."""
    pareto = (results.get("summary") or {}).get("pareto") or {}
    lines = []
    for rung, entry in pareto.items():
        front = entry.get("front") or [] if isinstance(entry, Mapping) else list(entry or [])
        among = entry.get("among") if isinstance(entry, Mapping) else None
        note = {"passed": " (among the controllers that passed the rung)", "all": " (nobody passed: among all)"}.get(among, "")
        lines.append(f"- `{rung}`: " + (", ".join(f"`{name}`" for name in front) if front else "empty") + note)
    return "\n".join(lines) or "No Pareto sets in the ledger."


def view_plan(results_by_box: Mapping[str, Mapping[str, Any]], *, project: str | None = None,
              hub: str = HUB_PROJECT) -> list[dict[str, str]]:
    """Every report and saved view build_views makes: [{key, kind, project, title}], no wandb needed (--dry-run)."""
    plan = []
    for slug, results in results_by_box.items():
        target = project or project_for(results)
        stem = f"{results['box']['input']} to {results['box']['output']}"
        for key, title in (("scoreboard", "Scoreboard"), ("performance-vs-cost", "Performance vs cost"),
                           ("contact", "Contact"), ("timing", "Timing")):
            plan.append({"key": f"{slug}/{key}", "kind": "report", "project": target, "title": f"{stem} · {title}"})
        for rung in sorted(results.get("rungs", []), key=lambda r: r.get("order", 0)):
            plan.append({"key": f"{slug}/rung/{rung['id']}", "kind": "view", "project": target,
                         "title": view_name(f"{rung.get('order', '')}. {rung['id']} · {rung.get('title', '')}")})
    plan.append({"key": "hub/controllers", "kind": "report", "project": project or hub,
                 "title": "Controllers · every black box"})
    return plan


# ─────────────────────────────────────────────────────────────────────────────────────────── W&B: views


def _by_box(results: Any) -> dict[str, Mapping[str, Any]]:
    """A mapping slug -> results document, from a mapping or a list of documents."""
    if isinstance(results, Mapping) and "schema" in results:
        results = [results]
    if isinstance(results, Mapping):
        return dict(results)
    return {doc["box"]["slug"]: doc for doc in results}


def _belt_rungs(results: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [r for r in sorted(results.get("rungs", []), key=lambda r: r.get("order", 0))
            if (r.get("scene") or {}).get("belt") or r.get("family") == "contact"]


def _joints(results: Mapping[str, Any]) -> list[str]:
    joints = []
    for result in results.get("results", []):
        for key in (result.get("trace") or {}):
            if key.startswith("err_deg.") and key[8:] not in joints:
                joints.append(key[8:])
    return joints or ["shoulder", "elbow"]


def _save_report(wr: Any, entity: str, project: str, url: str | None, title: str, description: str, blocks: list) -> str:
    """Save a report, in place when `url` names one (rebuilt from its id: a slug can hold characters from_url misreads)."""
    report = None
    if url:
        try:
            report_id = unquote(url).rsplit("--", 1)[-1]
            report = wr.Report.from_url(f"https://wandb.ai/{entity}/{project}/reports/report--{report_id}")
            report.title, report.description, report.blocks = title, description, blocks
        except Exception:  # deleted in the app: make it again
            report = None
    if report is None:
        report = wr.Report(entity=entity, project=project, title=title, description=description, blocks=blocks)
    report.width = "fluid"
    report.save()
    return browser_url(report.url)


def _save_view(ws: Any, entity: str, project: str, url: str | None, name: str, sections: list, settings: Any) -> str:
    view = None
    if url:
        try:
            # WHY unquote: Workspace.from_url splits the path into entity / project as written
            view = ws.Workspace.from_url(unquote(url))
            view.name, view.sections, view.runset_settings = name, sections, settings
        except Exception:  # deleted in the app, or a URL from_url cannot parse: make it again
            view = None
    if view is None:
        view = ws.Workspace(entity=entity, project=project, name=name, sections=sections, runset_settings=settings)
    view.save()
    return browser_url(view.url)


PINNED = ["config:controller", "config:rung", "summary:passed", "summary:objective", "summary:gates_passed",
          "summary:compute_us_p99", "summary:tracking_p95_deg", "summary:contact_max_n", "config:runs_on"]

INTRO = ("A **black box** is every controller that takes the same input message and emits a ServoCommand, so any one can be "
         "swapped for any other. Each controller runs every **rung** of the box's ladder (simplest first) on the planar-arm "
         "twin; a rung has pass/fail **gates** and one **objective** to rank on. One W&B run per (controller, rung, seed); the "
         "run id is the numbers' content id, so a re-run with identical numbers updates in place and new numbers supersede "
         "the old run (filter `evaluation_state = latest`). The record is `results/<box>/results.json` in robot_controllers; "
         "W&B holds a copy. Angles are in degrees.")


def build_views(results_by_box: Any, *, entity: str = WANDB_ENTITY, registry: Path = REGISTRY,
                project: str | None = None, hub: str = HUB_PROJECT) -> dict[str, dict[str, str]]:
    """Per box: the Scoreboard, Performance vs cost, Contact and Timing reports and one saved view per rung; in `hub`:
    the cross-box Controllers report. Saved in place through `registry` (never duplicated). `project` puts everything in
    one project (a scratch run). Returns {"views": {key: url}, "reports": {key: url}}."""
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws
    from wandb_workspaces import expr

    boxes = _by_box(results_by_box)
    known = json.loads(registry.read_text()) if registry.exists() else {}
    urls: dict[str, dict[str, str]] = {"views": dict(known.get("views", {})), "reports": dict(known.get("reports", {}))}
    layout = wr.Layout
    latest = expr.Config("evaluation_state") == "latest"
    titles = {item["key"]: item["title"] for item in view_plan(boxes, project=project, hub=hub)}

    def bars(metrics: list[str], *, width: int = 8) -> list:
        """One horizontal bar chart per metric (a bar per run), `24 // width` to a row."""
        per_row = 24 // width
        return [wr.BarPlot(title=f"{m} ({BY_NAME[m].unit})" if m in BY_NAME else m, metrics=[m], orientation="h",
                           layout=layout(x=i % per_row * width, y=i // per_row * 8, w=width, h=8))
                for i, m in enumerate(metrics)]

    for slug, results in boxes.items():
        target = project or project_for(results)
        mine = [latest, expr.Config("box_slug") == slug]
        rungs = sorted(results.get("rungs", []), key=lambda r: r.get("order", 0))
        joints = _joints(results)

        def runset(name: str, *extra: Any, target: str = target, mine: list = mine) -> Any:
            return wr.Runset(entity=entity, project=target, name=name, filters=[*mine, *extra], pinned_columns=PINNED)

        # ── Scoreboard: controller × rung pass table
        blocks = [wr.MarkdownBlock(text=INTRO), wr.H2(text="Recommendation"),
                  wr.MarkdownBlock(text=recommendation_markdown(results)),
                  wr.H2(text="Controller × rung"),
                  wr.MarkdownBlock(text=scoreboard_markdown(results) + "\n\n✓ passed every gate of the rung · ✗ failed one "
                                   "(named) · the number is the rung's objective · **bold** = best among the controllers that "
                                   "passed (the ledger's per-rung best)."),
                  wr.H2(text="Every run"),
                  wr.PanelGrid(runsets=[runset("latest")],
                               panels=[wr.BarPlot(title="Gates passed per run", metrics=["gates_passed"], orientation="h",
                                                  layout=layout(x=0, y=0, w=12, h=10)),
                                       wr.BarPlot(title="Passed every gate (1 = yes)", metrics=["passed"], orientation="h",
                                                  layout=layout(x=12, y=0, w=12, h=10))]),
                  wr.H2(text="The ruler: rungs, gates, objectives"), wr.MarkdownBlock(text=gates_markdown(results))]
        key = f"{slug}/scoreboard"
        urls["reports"][key] = _save_report(wr, entity, target, urls["reports"].get(key), titles[key],
                                            f"{results['box']['name']}: which controller passes which rung", blocks)

        # ── Performance vs cost: tracking and contact against compute per tick
        free = [r for r in rungs if r.get("objective") == "tracking_p95_deg"]
        belt = _belt_rungs(results)
        blocks = [wr.MarkdownBlock(text="Performance against cost: each point is one run, x = the 99th percentile wall time "
                                   "of one controller tick (µs, single BLAS thread), y = how well it did. Lower-left is "
                                   "better on both. The ledger's Pareto sets (R1 tracking, the rung's contact max, compute "
                                   "p99) per belt rung:\n\n" + pareto_markdown(results))]
        if free:
            blocks += [wr.H2(text="Tracking vs compute"),
                       wr.PanelGrid(runsets=[runset("tracking rungs", expr.Config("rung").isin([r["id"] for r in free]))],
                                    panels=[wr.ScatterPlot(title="Tracking p95 (°) vs compute p99 (µs)",
                                                           x=wr.SummaryMetric("compute_us_p99"),
                                                           y=wr.SummaryMetric("tracking_p95_deg"),
                                                           layout=layout(x=0, y=0, w=24, h=12))])]
        if belt:
            blocks += [wr.H2(text="Contact vs compute"),
                       wr.PanelGrid(runsets=[runset("belt rungs", expr.Config("rung").isin([r["id"] for r in belt]))],
                                    panels=[wr.ScatterPlot(title="Contact max (N) vs compute p99 (µs)",
                                                           x=wr.SummaryMetric("compute_us_p99"),
                                                           y=wr.SummaryMetric("contact_max_n"),
                                                           layout=layout(x=0, y=0, w=24, h=12))])]
        blocks += [wr.H2(text="Cost per tick"),
                   wr.PanelGrid(runsets=[runset("latest")],
                                panels=bars(["compute_us_p50", "compute_us_p99", "command_fields", "wire_bytes_per_tick"],
                                            width=12))]
        key = f"{slug}/performance-vs-cost"
        urls["reports"][key] = _save_report(wr, entity, target, urls["reports"].get(key), titles[key],
                                            f"{results['box']['name']}: tracking and contact against compute per tick", blocks)

        # ── Contact: peak and dwell force on the belt rungs
        blocks = [wr.MarkdownBlock(text="The belt rungs: the tool comes down onto a moving belt, presses, dwells and lifts. "
                                   "`contact_peak_n` is the transient (30 ms after first touch), `dwell_force_*` the "
                                   "quasi-static press against the rung's target force (ISO/TS 15066's two phases).")]
        for rung in belt:
            blocks += [wr.H2(text=f"{rung.get('order', '')}. {rung.get('title', rung['id'])}"),
                       wr.PanelGrid(runsets=[runset(rung["id"], expr.Config("rung") == rung["id"])],
                                    panels=bars(["contact_peak_n", "contact_max_n", "dwell_force_mean_n",
                                                 "dwell_force_error_n", "dwell_bounces", "dwell_contact_fraction"])
                                    + [wr.LinePlot(title="Normal force on the tool (N)", x=X_AXIS, y=["contact/fn_n"],
                                                   layout=layout(x=0, y=16, w=24, h=10))])]
        if not belt:
            blocks.append(wr.MarkdownBlock(text="This box has no belt rungs."))
        key = f"{slug}/contact"
        urls["reports"][key] = _save_report(wr, entity, target, urls["reports"].get(key), titles[key],
                                            f"{results['box']['name']}: peak and dwell force on the belt rungs", blocks)

        # ── Timing: the stamped log, black box
        blocks = [wr.MarkdownBlock(text="Timing from the command stamps alone (sent time, state stamp), so it is the same "
                                   "measurement on the twin and the rack: period jitter, missed cycles, sensor-to-command "
                                   "latency, stalls, schedule slip and board watchdog trips."),
                  wr.PanelGrid(runsets=[runset("latest")],
                               panels=bars(["latency_p99_ms", "period_jitter_p99_ms", "missed_cycles", "worst_stall_ms",
                                            "schedule_slip_ms", "watchdog_trips"])
                               + [wr.LinePlot(title="Compute per tick (µs)", x=X_AXIS, y=["cost/compute_us"],
                                              layout=layout(x=0, y=16, w=24, h=10))])]
        key = f"{slug}/timing"
        urls["reports"][key] = _save_report(wr, entity, target, urls["reports"].get(key), titles[key],
                                            f"{results['box']['name']}: timing from the stamped log", blocks)

        # ── one saved view per rung
        for rung in rungs:
            gate_metrics = [g["metric"] for g in rung.get("gates", [])]
            headline = [m for m in dict.fromkeys([rung.get("objective", ""), *gate_metrics]) if m]
            traces = [wr.LinePlot(title="Joint error (°)", x=X_AXIS, y=[f"err_deg/{j}" for j in joints],
                                  layout=layout(x=0, y=0, w=12, h=8)),
                      wr.LinePlot(title="Joint position vs reference (°)", x=X_AXIS,
                                  y=[f"{kind}/{j}" for j in joints for kind in ("q_deg", "ref_deg")],
                                  layout=layout(x=12, y=0, w=12, h=8)),
                      wr.LinePlot(title="Motor torque (N·m)", x=X_AXIS, y=[f"tau_nm/{j}" for j in joints],
                                  layout=layout(x=0, y=8, w=12, h=8)),
                      wr.LinePlot(title="Tool (mm)", x=X_AXIS, y=["tool_mm/x", "tool_mm/z"], layout=layout(x=12, y=8, w=12, h=8))]
            if rung in belt:
                traces.append(wr.LinePlot(title="Normal force on the tool (N)", x=X_AXIS, y=["contact/fn_n"],
                                          layout=layout(x=0, y=16, w=24, h=8)))
            sections = [ws.Section(name="Objective and gates", panels=bars(headline), is_open=True),
                        ws.Section(name="Traces", panels=traces, is_open=True)]
            settings = ws.RunsetSettings(filters=[*mine, expr.Config("rung") == rung["id"]], pinned_columns=PINNED)
            key = f"{slug}/rung/{rung['id']}"
            urls["views"][key] = _save_view(ws, entity, target, urls["views"].get(key), titles[key], sections, settings)

    # ── the hub: one report across every box
    target = project or hub
    blocks = [wr.MarkdownBlock(text=INTRO)]
    for slug, results in boxes.items():
        box_project = project or project_for(results)
        links = " · ".join(f"[{name}]({urls['reports'][f'{slug}/{key}']})" for key, name in (
            ("scoreboard", "Scoreboard"), ("performance-vs-cost", "Performance vs cost"), ("contact", "Contact"),
            ("timing", "Timing")))
        blocks += [wr.H2(text=results["box"]["name"]),
                   wr.MarkdownBlock(text=f"Project [{box_project}]({project_url(entity, box_project)}) · {links}\n\n"
                                    + recommendation_markdown(results) + "\n\n" + scoreboard_markdown(results)),
                   wr.PanelGrid(runsets=[wr.Runset(entity=entity, project=box_project, name=results["box"]["input"],
                                                   filters=[latest, expr.Config("box_slug") == slug], pinned_columns=PINNED)],
                                panels=[wr.BarPlot(title="Compute p99 per run (µs)", metrics=["compute_us_p99"], orientation="h",
                                                   layout=layout(x=0, y=0, w=12, h=10)),
                                        wr.BarPlot(title="Gates passed per run", metrics=["gates_passed"], orientation="h",
                                                   layout=layout(x=12, y=0, w=12, h=10))])]
    urls["reports"]["hub/controllers"] = _save_report(wr, entity, target, urls["reports"].get("hub/controllers"),
                                                      titles["hub/controllers"],
                                                      "Every black box: its recommendation, scoreboard and reports", blocks)
    registry.write_text(json.dumps(urls, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    return urls


# ─────────────────────────────────────────────────────────────────────────────────────────── W&B: checks


def verify(results_by_box: Any, *, entity: str = WANDB_ENTITY, registry: Path = REGISTRY, project: str | None = None,
           hub: str = HUB_PROJECT, views: bool = True) -> list[str]:
    """Read every box back through the W&B API: exactly the ledger's runs are latest, each run's passed / objective /
    gate lines match the ledger, a rung filter returns that rung's controllers, one run's trace is in history, and every
    report and saved view in the registry exists. Returns the failures (empty = all good) and prints what it checked."""
    import wandb

    api = wandb.Api(timeout=60)
    failures: list[str] = []
    boxes = _by_box(results_by_box)
    urls = json.loads(registry.read_text()) if registry.exists() else {}
    for slug, results in boxes.items():
        target = project or project_for(results)
        path = f"{entity}/{target}"
        batch = {record["id"]: record for record in records(results, project=target)}
        runs = list(api.runs(path, filters={"config.evaluation_state": "latest", "config.box_slug": slug}))
        print(f"{path}: {len(runs)} latest runs for {slug} (the ledger has {len(batch)})")
        got = {run.id for run in runs}
        failures += [f"{slug}: run {batch[i]['name']} ({i}) missing" for i in sorted(set(batch) - got)]
        failures += [f"{slug}: latest run {i} is not in the ledger (should be superseded)" for i in sorted(got - set(batch))]
        for run in runs:
            record = batch.get(run.id)
            if record is None:
                continue
            want = record["summary"]
            for key in ("passed", "objective", "gates_passed"):
                if want.get(key) is not None and _differs(run.summary.get(key), want[key]):
                    failures.append(f"{record['name']}: {key} W&B {run.summary.get(key)!r}, ledger {want[key]!r}")
            lines = {k for k in want if k.startswith("gate/")}
            missing = sorted(lines - set(run.summary.keys()))
            if missing:
                failures.append(f"{record['name']}: no {', '.join(missing)}")
        for rung in sorted(results.get("rungs", []), key=lambda r: r.get("order", 0)):
            on_rung = list(api.runs(path, filters={"config.evaluation_state": "latest", "config.box_slug": slug,
                                                   "config.rung": rung["id"]}))
            want_names = sorted(r["name"] for r in batch.values() if r["config"]["rung"] == rung["id"])
            print(f"  filter rung={rung['id']}: {sorted(r.name for r in on_rung)}")
            if sorted(r.name for r in on_rung) != want_names:
                failures.append(f"{slug}: rung {rung['id']} filter gave {sorted(r.name for r in on_rung)}, want {want_names}")
        sample = next((r for r in runs if r.id in batch and len(batch[r.id]["history"]) > 1), None)
        if sample is not None:
            want_rows = sum(1 for row in batch[sample.id]["history"] if X_AXIS in row)
            got_rows = sum(1 for _ in sample.scan_history(keys=[X_AXIS], page_size=1000))
            print(f"  history of {sample.name}: {got_rows} rows on {X_AXIS} (the ledger's trace makes {want_rows})")
            if got_rows != want_rows:
                failures.append(f"{sample.name}: {got_rows} history rows on {X_AXIS}, want {want_rows}")
        in_project = {str(report.id) for report in api.reports(path)}
        for key, url in sorted(urls.get("reports", {}).items()):
            if key.startswith(f"{slug}/") and unquote(url).rsplit("--", 1)[-1] not in in_project:
                failures.append(f"report {key}: {url} is not in {path}")
    hub_url = urls.get("reports", {}).get("hub/controllers")
    if hub_url:
        hub_path = f"{entity}/{project or hub}"
        if unquote(hub_url).rsplit("--", 1)[-1] not in {str(r.id) for r in api.reports(hub_path)}:
            failures.append(f"report hub/controllers: {hub_url} is not in {hub_path}")
    if views and urls.get("views"):
        import wandb_workspaces.workspaces as ws

        for key, url in sorted(urls["views"].items()):
            if not any(key.startswith(f"{slug}/") for slug in boxes):
                continue
            try:
                ws.Workspace.from_url(unquote(url))
            except Exception as error:  # noqa: BLE001 - any failure to load is the finding
                failures.append(f"view {key}: {url} does not load ({type(error).__name__}: {error})")
    print(f"  registry: {len(urls.get('reports', {}))} reports, {len(urls.get('views', {}))} saved views")
    return failures


# ─────────────────────────────────────────────────────────────────────────────────────────── Weave (bbo)


def load_bbo(path: Path = BBO) -> Any:
    """bam_trackio's bbo.py as a private module object (never the bam_trackio package)."""
    spec = importlib.util.spec_from_file_location("robot_controllers_tracking_bbo", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load bbo.py from {path}")
    module = importlib.util.module_from_spec(spec)
    # WHY registered before exec (sim2real 090231d): bbo's dataclasses, under `from __future__ import annotations`,
    # look their module up in sys.modules to resolve string annotations; an unregistered module fails at class creation
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def weave_groups(bbo: Any, results: Mapping[str, Any], batch: Sequence[Mapping[str, Any]],
                 urls: Mapping[str, str] | None = None) -> dict[str, tuple[list, list[str]]]:
    """Every (controller, rung) as a bbo.ScoredRun, grouped by rung (the environment) with its declared scorers.

    Black box = the controller (its version = config + commit), environment = the rung id (fixed identity: its scorers
    are its gates, never the run), one episode per seed. The run id attribute is the W&B runs' content ids, so the same
    numbers are logged once."""
    urls = urls or {}
    rungs = {rung["id"]: rung for rung in results.get("rungs", [])}
    controllers = {spec["name"]: spec for spec in results.get("controllers", [])}
    by_key: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in batch:
        by_key.setdefault((record["config"]["controller"], record["config"]["rung"]), []).append(record)
    groups: dict[str, tuple[list, list[str]]] = {}
    for (controller, rung_id), seeds in sorted(by_key.items(), key=lambda kv: (rungs.get(kv[0][1], {}).get("order", 0), kv[0])):
        rung = rungs.get(rung_id, {})
        scorers = ["objective", "passed"] + [f"gate_{g['metric']}" for g in rung.get("gates", [])]
        episodes = []
        for record in seeds:
            summary = record["summary"]
            scores: dict[str, bool | float] = {"passed": bool(summary["passed"])}
            if _finite(summary.get("objective")):
                scores["objective"] = float(summary["objective"])
            for gate in rung.get("gates", []):
                scores[f"gate_{gate['metric']}"] = str(summary.get(f"gate/{gate['metric']}", "")).startswith("pass")
            episodes.append(bbo.Episode(inputs={"seed": record["config"]["seed"]},
                                        output={"wandb_run": record["id"], "wandb_url": urls.get(record["id"], "")},
                                        scores=scores))
        numbers: dict[str, list[float]] = {}
        for record in seeds:
            for key, value in record["summary"].items():
                if _finite(value) and not key.startswith("gate/"):
                    numbers.setdefault(key, []).append(float(value))
        summary = {key: statistics.median(values) for key, values in numbers.items()}
        summary["passed"] = int(all(record["summary"]["passed"] for record in seeds))
        summary["seeds"] = len(seeds)
        spec = controllers.get(controller, {})
        content = hashlib.sha1("|".join(sorted(record["id"] for record in seeds)).encode()).hexdigest()[:16]
        groups.setdefault(rung_id, ([], scorers))[0].append(bbo.ScoredRun(
            black_box=controller,
            version={"controller": controller, "box": results["box"]["name"], "runs_on": spec.get("runs_on", ""),
                     "config": spec.get("config", {}), "commit": seeds[0]["config"]["commit"]},
            environment=rung_id, episodes=episodes, summary=summary,
            attributes={"run_id": content, "results_commit": seeds[0]["config"]["commit"]}))
    return groups


def log_weave(results: Mapping[str, Any], *, entity: str = WANDB_ENTITY, project: str | None = None,
              bbo_path: Path = BBO, urls: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The box's Weave evaluations (bbo.log_evaluations, one per rung) and leaderboard (bbo.publish_leaderboard):
    controllers × rungs × (objective, passed, compute p99). Returns {"evaluations": {rung: ref}, "leaderboards": {name: ref}}."""
    bbo = load_bbo(bbo_path)
    # WHY no deletion (sim2real rigs.py): bbo deletes evaluation calls whose end was dropped before it retries; nothing
    # in Zach's W&B is deleted by this logger, so an unended call stays (it holds no summary, no leaderboard reads it)
    bbo.delete_unended = lambda client: 0
    target = project or project_for(results)
    batch = records(results, project=target)
    entity_project = f"{entity}/{target}"
    refs: dict[str, str] = {}
    for _rung_id, (runs, scorers) in weave_groups(bbo, results, batch, urls).items():
        refs.update(bbo.log_evaluations(entity_project, runs, scorers=scorers))
    rungs = sorted(results.get("rungs", []), key=lambda r: r.get("order", 0))
    boards: dict[str, str] = {}
    for minimize in sorted({bool(r.get("minimize", True)) for r in rungs}, reverse=True):
        environments = [r["id"] for r in rungs if bool(r.get("minimize", True)) == minimize and r["id"] in refs]
        if not environments:
            continue
        metrics = [bbo.Metric("Objective", "objective", minimize=minimize), bbo.Metric("Passed every gate", "passed"),
                   bbo.Metric("Compute p99 (µs)", "compute_us_p99", minimize=True)]
        stem = f"{results['box']['input']} to {results['box']['output']}"
        name = f"{stem} · leaderboard" + ("" if len({bool(r.get('minimize', True)) for r in rungs}) == 1
                                          else (" · lower is better" if minimize else " · higher is better"))
        boards[name] = bbo.publish_leaderboard(entity_project, name, f"Controllers on the {results['box']['name']} ladder: "
                                               "each rung's objective, whether it passed every gate, compute per tick.",
                                               refs, metrics, environments)
    return {"evaluations": refs, "leaderboards": boards}


# ─────────────────────────────────────────────────────────────────────────────────────────── CLI


def load_results(paths: Sequence[Path]) -> dict[Path, dict[str, Any]]:
    out = {}
    for path in paths:
        document = json.loads(Path(path).read_text())
        if document.get("schema") != SCHEMA:
            raise SystemExit(f"{path}: not a {SCHEMA} document (schema {document.get('schema')!r})")
        out[Path(path)] = document
    return out


def write_back(path: Path, document: dict[str, Any], *, entity: str, project: str, logged: Sequence[Mapping[str, Any]] = (),
               reports: Mapping[str, str] | None = None) -> None:
    """Record the W&B links in the ledger: `wandb.project_url`, `wandb.reports`, each RunResult's `wandb_url`."""
    by_id = {item["id"]: item["url"] for item in logged}
    slug = document["box"]["slug"]
    for result in document.get("results", []):
        digest = metrics_digest(result.get("metrics") or {})
        identity = run_id(slug, result["controller"], result["rung"], int(result.get("seed", 0)), digest)
        if identity in by_id:
            result["wandb_url"] = by_id[identity]
    block = document.setdefault("wandb", {"project_url": None, "reports": {}})
    block["project_url"] = project_url(entity, project)
    if reports is not None:
        block["reports"] = {key.split("/", 1)[1]: url for key, url in reports.items() if key.startswith(f"{slug}/")
                            and "/rung/" not in key}
        if reports.get("hub/controllers"):
            block["reports"]["hub"] = reports["hub/controllers"]
    path.write_text(json.dumps(document, indent=1, allow_nan=False, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robot-controllers wandb", description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, nargs="+", help="results-1 JSON files (default: results/*/results.json)")
    parser.add_argument("--views", action="store_true", help="build or update the reports and saved views")
    parser.add_argument("--verify", action="store_true", help="read every box back through the W&B API and check it")
    parser.add_argument("--weave", action="store_true", help="also log Weave evaluations and the leaderboard (bbo.py)")
    parser.add_argument("--allow-dirty", action="store_true", help="log even when the code or the results are uncommitted")
    parser.add_argument("--dry-run", action="store_true", help="print the records and project names; no network")
    parser.add_argument("--no-push", action="store_true", help="skip logging runs (only --views / --verify / --weave)")
    parser.add_argument("--no-write-back", action="store_true", help="leave the results files untouched")
    parser.add_argument("--project", help="put every box and the hub in this one project (a scratch run)")
    parser.add_argument("--hub", default=HUB_PROJECT, help=f"the cross-box project (default {HUB_PROJECT!r})")
    parser.add_argument("--entity", default=WANDB_ENTITY)
    parser.add_argument("--registry", type=Path, help="where report / view URLs are kept (default: beside this file; "
                        "with --project, a per-project file in ~/.cache/robot_controllers)")
    args = parser.parse_args(argv)

    paths = args.results or sorted(RESULTS_DIR.glob("*/results.json"))
    if not paths:
        parser.error(f"no results files (looked in {RESULTS_DIR}/*/results.json); pass --results")
    documents = load_results(paths)
    registry = args.registry or (REGISTRY if not args.project else
                                 WANDB_DIR.parent / f"wandb_views.{re.sub(r'[^A-Za-z0-9]+', '-', args.project).strip('-')}.json")
    sha, dirty = code_version()

    if args.dry_run:
        print(json.dumps({"entity": args.entity, "hub": args.project or args.hub, "registry": str(registry),
                          "logger_commit": sha, "logger_dirty": dirty}, ensure_ascii=False))
        for path, document in documents.items():
            batch = records(document, sha=sha, dirty=dirty, project=args.project)
            print(json.dumps({"results": str(path), "project": batch[0]["project"] if batch else project_for(document),
                              "runs": len(batch)}, ensure_ascii=False))
            for record in batch:
                print(json.dumps(compact(record), ensure_ascii=False, default=str))
        boxes = {doc["box"]["slug"]: doc for doc in documents.values()}
        for item in view_plan(boxes, project=args.project, hub=args.hub):
            print(json.dumps(item, ensure_ascii=False))
        return 0

    # WHY refuse a dirty tree (sim2real, 2026-11-11): every evaluation in W&B must be reproducible from a commit, both
    # the code that made the numbers (the ledger's `generated.dirty`) and the code that shapes the runs (this checkout)
    unclean = [str(p) for p, d in documents.items() if (d.get("generated") or {}).get("dirty")]
    if not args.no_push and (dirty or unclean) and not args.allow_dirty:
        reasons = ([f"src/ has uncommitted changes (HEAD {sha[:9]})"] if dirty else []) + (
            [f"results made from a dirty tree: {', '.join(unclean)}"] if unclean else [])
        parser.error("; ".join(reasons) + ". Commit and re-run, or pass --allow-dirty")

    logged_by_path: dict[Path, list[dict]] = {}
    weave_failures: list[str] = []
    for path, document in documents.items():
        target = args.project or project_for(document)
        if not args.no_push:
            batch = records(document, sha=sha, dirty=dirty, project=args.project)
            logged = push(batch, entity=args.entity, project=args.project)
            retired = retire(batch, entity=args.entity, project=args.project)
            fresh = sum(1 for item in logged if not item["resumed"])
            patched = sorted({key for item in logged for key in item["patched"]})
            print(f"{target}: {len(logged)} runs ({fresh} new, {len(logged) - fresh} already there"
                  + (f"; patched {', '.join(patched)}" if patched else "") + f"), {retired} superseded")
            logged_by_path[path] = logged
        if args.weave:
            urls = {item["id"]: item["url"] for item in logged_by_path.get(path, [])}
            try:
                print(json.dumps(log_weave(document, entity=args.entity, project=args.project, urls=urls), indent=1))
            except Exception as error:  # noqa: BLE001 - WHY: Weave is optional; its failure must not stop views / verify
                weave_failures.append(f"{target}: Weave failed: {type(error).__name__}: {error}")
                print(weave_failures[-1], file=sys.stderr)
    reports = None
    if args.views:
        built = build_views({d["box"]["slug"]: d for d in documents.values()}, entity=args.entity, registry=registry,
                            project=args.project, hub=args.hub)
        reports = built["reports"]
        for kind, items in built.items():
            for key, url in sorted(items.items()):
                print(f"{kind[:-1]:6s} {key:52s} {url}")
    if not args.no_write_back and (logged_by_path or reports is not None):
        for path, document in documents.items():
            write_back(path, document, entity=args.entity, project=args.project or project_for(document),
                       logged=logged_by_path.get(path, ()), reports=reports)
    if args.verify:
        failures = verify({d["box"]["slug"]: d for d in documents.values()}, entity=args.entity, registry=registry,
                          project=args.project, hub=args.hub)
        print("VERIFY: " + ("PASS" if not failures else "FAIL\n  " + "\n  ".join(failures)))
        return 1 if failures or weave_failures else 0
    return 1 if weave_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
