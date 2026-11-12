"""Matplotlib figures, rendered at build time to WebP: one per (controller, rung) run, and the Pareto view per box.

Run figures: joint positions in degrees with the reference dashed, the tracking error, and on belt rungs the
contact force twice (the impact zoom with the 30 ms transient window shaded, and the whole run). Colours are the
dataviz reference palette in its fixed slot order (blue, orange, aqua, yellow, magenta); identity is never colour
alone: every series is in the legend or labelled directly.
"""

from __future__ import annotations

import io
import math
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from model import BoxInfo, RungInfo, fmt, free_space_rung, short_name  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8b8b94", "#e7e6e2"
SHADE_TRANSIENT = "#2a78d6"
SHADE_EVENT = "#a1a1aa"
DPI = 150
WIDTH_IN = 7.6

plt.rcParams.update({
    "font.family": ["DejaVu Sans"], "font.size": 8.5, "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 8.5, "axes.labelcolor": INK2, "axes.edgecolor": "#c9c8c2",
    "axes.linewidth": 0.8, "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "legend.fontsize": 8, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "lines.linewidth": 1.4,
})


def to_webp(figure) -> tuple[bytes, tuple[int, int]]:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=DPI)
    plt.close(figure)
    out = io.BytesIO()
    picture = Image.open(buffer)
    picture.save(out, "WEBP", quality=82, method=6)
    return out.getvalue(), picture.size


def _clean(values: list) -> list[float]:
    return [math.nan if v is None else float(v) for v in values]


def _events(run: dict, kind: str) -> list[float]:
    return [float(e["t"]) for e in run.get("events") or [] if isinstance(e, dict) and e.get("kind") == kind and "t" in e]


def _windows(run: dict, start_kind: str, end_kind: str) -> list[tuple[float, float]]:
    return list(zip(_events(run, start_kind), _events(run, end_kind), strict=False))


def _gate_value(rung: RungInfo, metric: str) -> float | None:
    for gate in rung.gates:
        if gate.get("metric") == metric:
            try:
                return float(gate["value"])
            except (KeyError, TypeError, ValueError):
                return None
    return None


def run_figure(run: dict, rung: RungInfo, joint_names: list[str]) -> tuple[bytes, tuple[int, int]] | None:
    """The run's plot, or None when its trace has nothing to draw."""
    trace = run.get("trace") or {}
    if not trace.get("t"):
        return None
    t = _clean(trace["t"])
    joints = [j for j in joint_names if f"q_deg.{j}" in trace] or sorted(
        k.split(".", 1)[1] for k in trace if k.startswith("q_deg."))
    has_error = any(f"err_deg.{j}" in trace for j in joints)
    has_torque = any(f"tau_nm.{j}" in trace for j in joints)
    contact = _clean(trace.get("contact_fn_n") or [])
    contact_t = _clean(trace.get("contact_t") or trace.get("t") or [])
    has_contact = bool(contact) and len(contact) == len(contact_t) and max((c for c in contact if not math.isnan(c)), default=0) > 0

    rows = ["q"] + (["err"] if has_error else ["tau"] if has_torque else []) + (["contact"] if has_contact else [])
    heights = {"q": 1.0, "err": 0.8, "tau": 0.8, "contact": 0.95}
    figure = plt.figure(figsize=(WIDTH_IN, 0.45 + sum(1.75 * heights[r] for r in rows)), layout="constrained")
    grid = figure.add_gridspec(len(rows), 2, height_ratios=[heights[r] for r in rows], width_ratios=[1, 2.2])
    stalls = _windows(run, "stall_start", "stall_end")
    pushes = _windows(run, "push_start", "push_end")
    replans = _events(run, "replan") + _events(run, "refused")

    def decorate(axis, label_events: bool) -> None:
        for start, end in stalls:
            axis.axvspan(start, end, color=SHADE_EVENT, alpha=0.18, lw=0, label="host stall" if label_events else None)
            label_events = False
        for start, end in pushes:
            axis.axvspan(start, end, color=SERIES[3], alpha=0.18, lw=0, label="push" if label_events else None)
            label_events = False
        for index, when in enumerate(replans):
            axis.axvline(when, color=MUTED, lw=0.6, ls=":", label="new trajectory" if (label_events and index == 0) else None)

    shared = None
    for row_index, row in enumerate(rows):
        if row == "contact":
            break
        axis = figure.add_subplot(grid[row_index, :], sharex=shared)
        shared = shared or axis
        if row == "q":
            for index, joint in enumerate(joints):
                color = SERIES[index % len(SERIES)]
                axis.plot(t, _clean(trace[f"q_deg.{joint}"]), color=color, label=f"{joint} measured")
                if f"ref_deg.{joint}" in trace:
                    axis.plot(t, _clean(trace[f"ref_deg.{joint}"]), color=color, ls=(0, (4, 2.5)), lw=1.1, alpha=0.9,
                              label=f"{joint} reference")
            axis.set_ylabel("joint (°)")
            axis.set_title("Joint position: measured solid, reference dashed")
            decorate(axis, True)
            axis.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
        elif row == "err":
            for index, joint in enumerate(joints):
                if f"err_deg.{joint}" in trace:
                    axis.plot(t, _clean(trace[f"err_deg.{joint}"]), color=SERIES[index % len(SERIES)], lw=1.1, label=joint)
            limit = _gate_value(rung, "tracking_p95_deg")
            if limit is not None:
                for sign in (1, -1):
                    axis.axhline(sign * limit, color=INK2, lw=0.8, ls="--", label="gate ± p95 limit" if sign == 1 else None)
            p95 = (run.get("metrics") or {}).get("tracking_p95_deg")
            axis.set_title(f"Tracking error (reference − measured), p95 {fmt(p95)}°" if p95 is not None else "Tracking error")
            axis.set_ylabel("error (°)")
            decorate(axis, False)
            axis.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
        elif row == "tau":
            for index, joint in enumerate(joints):
                if f"tau_nm.{joint}" in trace:
                    axis.plot(t, _clean(trace[f"tau_nm.{joint}"]), color=SERIES[index % len(SERIES)], lw=1.1, label=joint)
            axis.set_title("Motor torque")
            axis.set_ylabel("torque (N·m)")
            decorate(axis, False)
            axis.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
        if row_index == len(rows) - 1 or rows[row_index + 1] == "contact":
            axis.set_xlabel("time (s)")
        else:
            axis.tick_params(labelbottom=False)

    if has_contact:
        row_index = rows.index("contact")
        touches = _events(run, "touch")
        touch = touches[0] if touches else next((tt for tt, f in zip(contact_t, contact, strict=True) if f > 0), contact_t[0])
        zoom = figure.add_subplot(grid[row_index, 0])
        inside = [(tt, f) for tt, f in zip(contact_t, contact, strict=True) if touch - 0.020 <= tt <= touch + 0.080]
        if inside:
            zoom.plot([(tt - touch) * 1000 for tt, _ in inside], [f for _, f in inside], color=SERIES[0], lw=1.3)
        zoom.axvspan(0, 30, color=SHADE_TRANSIENT, alpha=0.14, lw=0, label="30 ms transient window")
        peak_gate = _gate_value(rung, "contact_peak_n")
        if peak_gate is not None:
            zoom.axhline(peak_gate, color=INK2, lw=0.8, ls="--", label=f"gate {fmt(peak_gate)} N")
        peak = (run.get("metrics") or {}).get("contact_peak_n")
        zoom.set_title(f"Impact, peak {fmt(peak)} N" if peak is not None else "Impact")
        zoom.set_xlabel("ms after first touch")
        zoom.set_ylabel("normal force (N)")
        zoom.set_xlim(-20, 80)
        zoom.legend(loc="center right", fontsize=7.5)
        whole = figure.add_subplot(grid[row_index, 1])
        whole.plot(contact_t, contact, color=SERIES[0], lw=1.0, label="normal force")
        whole.axvspan(touch, touch + 0.030, color=SHADE_TRANSIENT, alpha=0.14, lw=0)
        dwell_s = rung.params.get("dwell_s")
        target = rung.params.get("press_target_n")
        if isinstance(dwell_s, (int, float)):
            whole.axvspan(touch, touch + float(dwell_s), color=SERIES[2], alpha=0.08, lw=0, label="dwell")
        if isinstance(target, (int, float)):
            whole.axhline(float(target), color=SERIES[2], lw=0.9, ls="--", label=f"target press {fmt(target)} N")
        decorate(whole, False)
        whole.set_title("Contact force over the run")
        whole.set_xlabel("time (s)")
        whole.legend(loc="upper right")
    return to_webp(figure)


def pareto_figure(box: BoxInfo) -> tuple[bytes, tuple[int, int]] | None:
    """Per belt rung: R1 tracking vs that rung's peak contact, the Pareto set filled; then compute per controller."""
    if not box.pareto_points:
        return None
    order = [c.name for c in box.built]
    colors = {name: SERIES[i % len(SERIES)] for i, name in enumerate(order)}
    rung_ids = [r.id for r in box.rungs if r.id in box.pareto_points]
    panels = len(rung_ids) + 1
    columns = 3 if panels > 4 else panels
    rows = math.ceil(panels / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(10.5, 3.05 * rows), layout="constrained", squeeze=False)
    flat = [a for row in axes for a in row]
    anchor = free_space_rung(box.rungs)
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    for axis, rung_id in zip(flat, rung_ids, strict=False):
        rung = box.rung(rung_id)
        points = box.pareto_points[rung_id]
        front = set(box.pareto.get(rung_id) or [])
        for name, (track, contact, _compute) in points.items():
            on_front = name in front
            passed = bool((box.controller(name).runs.get(rung_id) or {}).get("passed"))
            index = order.index(name)
            axis.scatter([track], [contact], s=62, marker=markers[index % len(markers)], zorder=3,
                         facecolor=colors[name] if on_front else "white", edgecolor=colors[name], linewidth=1.6)
            if not passed:  # WHY a mark, not a text label: five controllers' labels collide where the numbers agree
                axis.scatter([track], [contact], s=26, marker="x", color=INK, linewidth=1.3, zorder=4)
        # WHY no gate line here: the y axis is contact_max_n (the whole run), the gate is on contact_peak_n (the
        # 30 ms transient); drawing one on the other's axis would read as a verdict it is not.
        axis.set_title(rung.title if rung else rung_id)
        axis.set_xlabel(f"{anchor.id if anchor else 'R1'} tracking p95 (°)")
        axis.set_ylabel("largest contact force (N)")
        axis.margins(x=0.25, y=0.2)
    cost_axis = flat[len(rung_ids)]
    cost_axis.grid(axis="y", visible=False)
    names = [n for n in order if box.controller(n).cost.get("compute_us_p99") is not None]
    values = [box.controller(n).cost["compute_us_p99"] for n in names]
    positions = list(range(len(names)))[::-1]
    cost_axis.barh(positions, values, color=[colors[n] for n in names], height=0.6)
    for position, value in zip(positions, values, strict=True):
        cost_axis.annotate(f"{fmt(value)} µs", (value, position), xytext=(4, 0), textcoords="offset points",
                           va="center", fontsize=7.5, color=INK2)
    cost_axis.set_yticks(positions, [short_name(n) for n in names])
    cost_axis.set_xlabel("compute per tick, p99 (µs; median over rungs)")
    cost_axis.set_title("Compute cost")
    cost_axis.margins(x=0.3)
    for axis in flat[len(rung_ids) + 1:]:
        axis.set_visible(False)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], ls="", marker=markers[i % len(markers)], markersize=7.5, markerfacecolor=colors[n],
                      markeredgecolor=colors[n], label=short_name(n)) for i, n in enumerate(order)]
    handles += [Line2D([], [], ls="", marker="o", markersize=7.5, markerfacecolor="white", markeredgecolor=MUTED,
                       label="hollow: beaten on all three axes"),
                Line2D([], [], ls="", marker="x", markersize=6.5, color=INK, label="failed the rung's gates")]
    figure.legend(handles=handles, loc="outside upper center", ncol=min(len(handles), 4), fontsize=8.5)
    return to_webp(figure)
