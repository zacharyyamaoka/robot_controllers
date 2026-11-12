"""The per-box summary of a results-1 document (CONTRACT §4): scoreboard per controller, best per rung, the
performance-vs-cost Pareto set per belt rung, and the one recommended controller with a one-line why.

`summarize(box, rungs, results)` takes the box's rungs (bench.types.Rung, or their results-1 dicts) and the
runs (RunResult or `RunResult.to_dict()` rows). Several seeds of one (controller, rung) collapse to one cell:
it passes only if every seed passed, a gate counts as passed only if it passed in every seed, and numbers
(objective, metrics) are the median over seeds. A rung a controller did not run counts as failed, with all of
its gates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PARETO_METRICS = ("tracking_p95_deg", "contact_max_n", "compute_us_p99")  # all lower-is-better


@dataclass
class _Cell:
    """One controller on one rung, over its seeds."""

    ran: bool
    passed: bool
    gates_passed: int
    gates_total: int
    objective: float | None
    metrics: dict[str, float | None] = field(default_factory=dict)


def _get(item: Any, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)


def _rung_gate_count(rung: Any) -> int:
    return len(_get(rung, "gates", ()) or ())


def _is_belt_rung(rung: Any) -> bool:
    scene = _get(rung, "scene")
    belt = _get(scene, "belt", False) if scene is not None else False
    return bool(belt) or _get(rung, "family") == "contact"


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    return float(np.median(finite)) if finite else None


def _as_row(result: Any) -> dict[str, Any]:
    return result.to_dict() if hasattr(result, "to_dict") else dict(result)


def _cell(rows: Sequence[Mapping[str, Any]], rung: Any) -> _Cell:
    if not rows:
        return _Cell(
            ran=False, passed=False, gates_passed=0, gates_total=_rung_gate_count(rung), objective=None
        )
    gate_lists = [list(row.get("gates") or []) for row in rows]
    gates_total = max(len(gates) for gates in gate_lists)
    gates_passed = sum(
        1
        for index in range(gates_total)
        if all(index < len(gates) and gates[index].get("passed") for gates in gate_lists)
    )
    metric_names = set().union(*(row.get("metrics", {}).keys() for row in rows))
    return _Cell(
        ran=True,
        passed=all(bool(row.get("passed")) for row in rows),
        gates_passed=gates_passed,
        gates_total=gates_total,
        objective=_median(row.get("objective") for row in rows),
        metrics={name: _median(row.get("metrics", {}).get(name) for row in rows) for name in metric_names},
    )


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """a is no worse than b on every (lower-is-better) axis and strictly better on one."""
    return all(x <= y for x, y in zip(a, b, strict=True)) and any(x < y for x, y in zip(a, b, strict=True))


def pareto_front(points: Mapping[str, Sequence[float]]) -> list[str]:
    """Names of the non-dominated points (every axis lower-is-better), in the input order."""
    return [
        name
        for name, point in points.items()
        if not any(dominates(other, point) for other_name, other in points.items() if other_name != name)
    ]


def summarize(box: Any, rungs: Sequence[Any], results: Sequence[Any]) -> dict[str, Any]:
    """The results-1 `summary`: per_controller, per_rung, pareto, recommendation."""
    rung_list = list(rungs)
    rung_ids = [str(_get(rung, "id")) for rung in rung_list]
    rows = [_as_row(result) for result in results]
    controllers: list[str] = []
    for row in rows:
        if row["controller"] not in controllers:
            controllers.append(row["controller"])
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_cell.setdefault((row["controller"], row["rung"]), []).append(row)
    cells = {
        (name, rung_id): _cell(by_cell.get((name, rung_id), []), rung)
        for name in controllers
        for rung_id, rung in zip(rung_ids, rung_list, strict=True)
    }

    per_controller: dict[str, dict[str, Any]] = {}
    for name in controllers:
        own = [cells[(name, rung_id)] for rung_id in rung_ids]
        per_controller[name] = {
            "gates_passed": sum(c.gates_passed for c in own),
            "gates_total": sum(c.gates_total for c in own),
            "rungs_passed": sum(1 for c in own if c.passed),
            "rungs_total": len(rung_ids),
            "objective": {rung_id: cells[(name, rung_id)].objective for rung_id in rung_ids},
            "rank": {},
            "passed": {rung_id: cells[(name, rung_id)].passed for rung_id in rung_ids},
            # WHY the median over rungs: one controller's cost hardly depends on the rung, and a median
            # ignores the one rung where the machine happened to be busy.
            "compute_us_p99": _median(c.metrics.get("compute_us_p99") for c in own if c.ran),
            "missing": [rung_id for rung_id in rung_ids if not cells[(name, rung_id)].ran],
        }

    def cost_of(name: str) -> float:
        value = per_controller[name]["compute_us_p99"]
        return float("inf") if value is None else value

    per_rung: dict[str, dict[str, Any]] = {}
    for rung_id, rung in zip(rung_ids, rung_list, strict=True):
        minimize = bool(_get(rung, "minimize", True))
        sign = 1.0 if minimize else -1.0
        measured = [name for name in controllers if cells[(name, rung_id)].objective is not None]
        # WHY passers rank first: the rank then agrees with `best` (a controller that failed the gates can
        # have the "best" number by not doing the task, e.g. 0 N by never touching the belt).
        ranked = sorted(
            measured, key=lambda n: (not cells[(n, rung_id)].passed, sign * cells[(n, rung_id)].objective)
        )
        for position, name in enumerate(ranked):
            key = (not cells[(name, rung_id)].passed, cells[(name, rung_id)].objective)
            previous = ranked[position - 1] if position else None
            if previous is not None and key == (
                not cells[(previous, rung_id)].passed,
                cells[(previous, rung_id)].objective,
            ):
                per_controller[name]["rank"][rung_id] = per_controller[previous]["rank"][rung_id]
            else:
                per_controller[name]["rank"][rung_id] = position + 1
        for name in controllers:
            per_controller[name]["rank"].setdefault(rung_id, None)
        passers = [name for name in measured if cells[(name, rung_id)].passed]
        best = (
            min(passers, key=lambda n: (sign * cells[(n, rung_id)].objective, cost_of(n), n))
            if passers
            else None
        )
        per_rung[rung_id] = {
            "title": _get(rung, "title"),
            "objective": _get(rung, "objective"),
            "minimize": minimize,
            "best": best,
            "best_objective": cells[(best, rung_id)].objective if best else None,
            "passed": [name for name in controllers if cells[(name, rung_id)].passed],
            "values": {name: cells[(name, rung_id)].objective for name in controllers},
        }

    pareto = _pareto(rung_list, rung_ids, controllers, cells, per_controller)
    return {
        "per_controller": per_controller,
        "per_rung": per_rung,
        "pareto": pareto,
        "recommendation": _recommendation(controllers, per_controller),
    }


def _free_space_rung_id(rung_list: Sequence[Any], rung_ids: Sequence[str]) -> str | None:
    """R1: the free-space rung lowest on the ladder (the tracking axis of every Pareto plot)."""
    free = [
        (int(_get(rung, "order", 0) or 0), rung_id)
        for rung, rung_id in zip(rung_list, rung_ids, strict=True)
        if _get(rung, "family") == "free_space"
    ]
    if free:
        return min(free)[1]
    starts_r1 = [rung_id for rung_id in rung_ids if rung_id.startswith("R1.")]
    return starts_r1[0] if starts_r1 else None


def _pareto(rung_list, rung_ids, controllers, cells, per_controller) -> dict[str, Any]:
    """Per belt rung: the non-dominated controllers on (R1 tracking_p95_deg, that rung's contact_max_n,
    compute_us_p99). WHY among the rung's passers when there are any: a controller that failed the belt gates
    (never touched: 0 N) would sit on the front for not doing the task; with no passer, it is over all."""
    free_space = _free_space_rung_id(rung_list, rung_ids)
    out: dict[str, Any] = {}
    for rung, rung_id in zip(rung_list, rung_ids, strict=True):
        if not _is_belt_rung(rung):
            continue
        points = []
        for name in controllers:
            tracking = cells[(name, free_space)].metrics.get("tracking_p95_deg") if free_space else None
            points.append(
                {
                    "controller": name,
                    "tracking_p95_deg": tracking,
                    "contact_max_n": cells[(name, rung_id)].metrics.get("contact_max_n"),
                    "compute_us_p99": per_controller[name]["compute_us_p99"],
                    "passed": cells[(name, rung_id)].passed,
                }
            )
        measured = {
            p["controller"]: tuple(p[m] for m in PARETO_METRICS)
            for p in points
            if all(p[m] is not None for m in PARETO_METRICS)
        }
        passers = {name: point for name, point in measured.items() if cells[(name, rung_id)].passed}
        among = "passed" if passers else "all"
        front = pareto_front(passers if passers else measured)
        for point in points:
            point["front"] = point["controller"] in front
        out[rung_id] = {
            "axes": [
                {"rung": free_space, "metric": "tracking_p95_deg"},
                {"rung": rung_id, "metric": "contact_max_n"},
                {"rung": None, "metric": "compute_us_p99"},
            ],
            "points": points,
            "front": front,
            "among": among,
        }
    return out


def _recommendation(
    controllers: Sequence[str], per_controller: Mapping[str, Mapping[str, Any]]
) -> dict | None:
    """Most rungs passed, then most gates passed, then the lowest compute_us_p99 (then the name, for a stable
    pick), with a one-line why that names those numbers and what separated it from the runner-up."""
    if not controllers:
        return None

    def key(name: str) -> tuple:
        row = per_controller[name]
        compute = row["compute_us_p99"]
        return (
            -row["rungs_passed"],
            -row["gates_passed"],
            float("inf") if compute is None else compute,
            name,
        )

    ordered = sorted(controllers, key=key)
    pick = per_controller[ordered[0]]
    why = (
        f"{ordered[0]} passes {pick['rungs_passed']}/{pick['rungs_total']} rungs and "
        f"{pick['gates_passed']}/{pick['gates_total']} gates at {_us(pick['compute_us_p99'])} p99 compute"
    )
    if len(ordered) > 1:
        runner = per_controller[ordered[1]]
        if runner["rungs_passed"] != pick["rungs_passed"]:
            why += f"; next is {ordered[1]} with {runner['rungs_passed']}/{runner['rungs_total']} rungs"
        elif runner["gates_passed"] != pick["gates_passed"]:
            why += (
                f"; {ordered[1]} passes as many rungs but {runner['gates_passed']}/{runner['gates_total']} "
                "gates"
            )
        else:
            why += f"; {ordered[1]} ties on rungs and gates at {_us(runner['compute_us_p99'])}"
    return {
        "controller": ordered[0],
        "rungs_passed": pick["rungs_passed"],
        "rungs_total": pick["rungs_total"],
        "gates_passed": pick["gates_passed"],
        "gates_total": pick["gates_total"],
        "compute_us_p99": pick["compute_us_p99"],
        "why": why + ".",
    }


def _us(value: float | None) -> str:
    return "unmeasured" if value is None else f"{value:.1f} µs"
