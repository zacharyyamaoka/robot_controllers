"""Run every built controller of a box on every rung of its ladder and write the results JSON (the ledger).

    env -u VIRTUAL_ENV uv run --frozen --directory /home/bam/robot_controllers --extra bench robot-controllers run

WHY sequential by default: compute time per tick is one of the scored costs, and episodes running side by side on a
shared machine contend for cores and inflate it. `workers > 1` is for quick looks; its cost numbers are flagged.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import os
import platform
import socket
import subprocess
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from robot_controllers.core import Box, ControllerSpec
from robot_controllers.registry import BOXES, controller, in_box

REPO = Path(__file__).resolve().parents[3]
# WHY the checkout test (a wheel install has no repo): the committed ledger lives in the checkout's results/; from an
# installed wheel REPO is site-packages' parent, so results go to ./results (or ROBOT_CONTROLLERS_RESULTS) instead.
RESULTS_DIR = (
    Path(os.environ["ROBOT_CONTROLLERS_RESULTS"]) if os.environ.get("ROBOT_CONTROLLERS_RESULTS")
    else REPO / "results" if (REPO / "pyproject.toml").is_file()
    else Path.cwd() / "results"
)
SCHEMA = "robot-controllers/results-1"


def code_version(repo: Path = REPO) -> tuple[str, bool]:
    """(HEAD sha, whether src/ has uncommitted changes)."""
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout.strip()

    return git("rev-parse", "--short=12", "HEAD") or "unknown", bool(git("status", "--porcelain", "--", "src"))


def robot_msgs_version() -> str:
    import robot_msgs

    root = Path(robot_msgs.__file__).resolve().parents[2]
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True,
                         check=False).stdout.strip()
    return f"{robot_msgs.__version__} @ {sha or 'installed'}"


def _jsonable(value: Any) -> Any:
    """Config / scene / timing values as JSON (tuples -> lists, dataclasses -> dicts, NaN -> None)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def rung_record(rung: Any) -> dict[str, Any]:
    return {
        "id": rung.id,
        "title": rung.title,
        "family": rung.family,
        "order": rung.order,
        "description": rung.description,
        "duration_s": rung.duration_s,
        "params": _jsonable(dict(rung.params)),
        "timing": _jsonable(rung.timing),
        "scene": _jsonable(rung.scene),
        "gates": [{"metric": g.metric, "op": g.op, "value": g.value, "why": g.why} for g in rung.gates],
        "objective": rung.objective,
        "minimize": rung.minimize,
    }


def controller_record(spec: ControllerSpec) -> dict[str, Any]:
    config = spec.make_config() if spec.config_target else None
    return {
        "name": spec.name,
        "title": spec.title,
        "runs_on": spec.runs_on,
        "summary": spec.summary,
        "variant_of": spec.variant_of,
        "prior_art": list(spec.prior_art),
        "config": _jsonable(config) if config is not None else {},
    }


def _episode(job: tuple[str, str, str, int, bool]) -> dict[str, Any]:
    """One (controller, rung, seed) -> RunResult dict. Top level so a process pool can run it."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    box_slug, controller_name, rung_id, seed, trace = job
    from robot_controllers.bench.loop import run_episode
    from robot_controllers.bench.rungs import ladder
    from robot_controllers.metrics.episode import run_result
    from robot_controllers.registry import box_by_slug

    rung = next(r for r in ladder(box_by_slug(box_slug)) if r.id == rung_id)
    started = time.perf_counter()
    log = run_episode(controller(controller_name), rung, seed=seed)
    result = run_result(log, rung, trace=trace).to_dict()
    result["wall_s"] = round(time.perf_counter() - started, 3)
    return result


def run_box(
    box: Box,
    *,
    controllers: Sequence[str] | None = None,
    rungs: Sequence[str] | None = None,
    seeds: Sequence[int] = (0,),
    workers: int = 1,
    trace: bool = True,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """The results-1 document for one box."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from robot_controllers.bench.rungs import ladder
    from robot_controllers.metrics.summary import summarize

    specs = [s for s in in_box(box, built_only=True) if controllers is None or s.name in controllers]
    ladder_rungs = [r for r in ladder(box) if rungs is None or r.id in rungs]
    jobs = [(box.slug, s.name, r.id, seed, trace) for s in specs for r in ladder_rungs for seed in seeds]
    if progress:
        progress(f"{box.name}: {len(specs)} controllers x {len(ladder_rungs)} rungs x {len(seeds)} seeds = {len(jobs)} episodes")
    results: list[dict[str, Any]] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for job, result in zip(jobs, pool.map(_episode, jobs), strict=True):
                results.append(result)
                if progress:
                    progress(f"  {job[1]} @ {job[2]}: {'pass' if result['passed'] else 'FAIL'} ({result['wall_s']} s)")
    else:
        for job in jobs:
            result = _episode(job)
            results.append(result)
            if progress:
                progress(f"  {job[1]} @ {job[2]}: {'pass' if result['passed'] else 'FAIL'} ({result['wall_s']} s)")

    sha, dirty = code_version()
    import mujoco

    document = {
        "schema": SCHEMA,
        "box": {"slug": box.slug, "name": box.name, "input": box.input_name, "output": box.output_name,
                "wandb_project": box.wandb_project},
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
        },
        "rungs": [rung_record(r) for r in ladder_rungs],
        "controllers": [controller_record(s) for s in specs],
        "results": results,
        "summary": summarize(box, ladder_rungs, results),
        "wandb": {"project_url": None, "reports": {}},
    }
    return document


def write_results(document: dict[str, Any], out_dir: Path = RESULTS_DIR) -> Path:
    """results/<box-slug>/results.json, keeping any W&B links an earlier logging pass wrote."""
    path = out_dir / document["box"]["slug"] / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            previous = json.loads(path.read_text())
            if previous.get("wandb", {}).get("project_url"):
                document["wandb"] = previous["wandb"]
        except (json.JSONDecodeError, OSError):
            pass
    path.write_text(json.dumps(document, indent=1, allow_nan=False) + "\n")
    return path


def boxes_with_ladders() -> list[Box]:
    from robot_controllers.bench.rungs import ladder

    return [b for b in BOXES if in_box(b, built_only=True) and ladder(b)]
