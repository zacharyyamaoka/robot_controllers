"""The amendment-4 study's sections of the layouts report: rate, speed, lookahead calibration, the bus, sync, UR5e.

Imported by docs/build_layouts_report.py. Every chart and table number is read from results/study/results.json
(schema robot-controllers/study-1, CONTRACT §4.6), mostly from its own `summary`, so the page and the ledger can't
disagree. The few numbers that are not in the ledger (the UR5e Coulomb check) are labelled as the dispatcher's.
"""

from __future__ import annotations

import html
import io
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "results" / "study" / "results.json"

L1 = "ff=none@none+fb=board_gains@board+la=5.5ms"
L2 = "ff=trajectory@trajectory+fb=board_gains@board"
L3 = "ff=trajectory@trajectory+fb=board_gains@board+la=5.5ms"
L4 = "ff=trajectory@trajectory+fb=board_gains@board+la=5.5ms+interp=secant_mid_ff"
L5 = "ff=rigid_body@plan+fb=pd@split"
L6 = "ff=rigid_body@host_measured+fb=ctc@host"
L7 = "ff=rigid_body@plan+fb=pd@host"
LAYOUTS = (L1, L2, L3, L4, L5, L6, L7)
SHORT = {
    L1: "no feed-forward (board PD + lookahead)",
    L2: "JTC as built",
    L3: "JTC + lookahead",
    L4: "JTC + lookahead + secant velocity (v1)",
    L5: "host spring + board damper",
    L6: "computed torque on the host",
    L7: "PD on the host",
}
COLOUR = {L1: "#a1a1aa", L2: "#f97316", L3: "#3b82f6", L4: "#16a34a", L5: "#d97706", L6: "#9333ea", L7: "#dc2626"}
ARMS = (("planar_arm", "planar twin, R2 path"), ("ur5e", "UR5e, U2 path"))
RIG_RATE_HZ = 50.0


def esc(text: object) -> str:
    return html.escape(str(text))


def num(value: object, digits: int = 3) -> str:
    if value is None:
        return "—"
    number = float(value)
    if not math.isfinite(number):
        return "—"
    if abs(number) >= 100:
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.1f}"
    if abs(number) >= 1:
        return f"{number:.2f}"
    return f"{number:.{digits}g}"


def svg(fig) -> str:
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    text = buffer.getvalue()
    return "<figure class='chart'>" + text[text.index("<svg"):] + "</figure>"


def style(ax) -> None:
    ax.grid(True, color="#d4d4d8", lw=0.6, alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


class Study:
    def __init__(self, document: dict) -> None:
        self.document = document
        self.rows = document["rows"]
        self.summary = document["summary"]

    def rows_for(self, arm: str, dimension: str, **where) -> list[dict]:
        return [r for r in self.rows if r["arm"] == arm and r["dimension"] == dimension
                and all(r.get(k) == v for k, v in where.items())]

    def rate(self, arm: str, layout: str, rate_hz: float, key: str = "tracking_p95_deg"):
        block = self.summary[arm]["rate"][key if key == "tracking_p95_deg" else "tracking_p95_deg"]
        return block.get(layout, {}).get(f"{rate_hz:g}")

    def rate_passed(self, arm: str, layout: str, rate_hz: float) -> bool:
        return bool(self.summary[arm]["rate"]["passed"].get(layout, {}).get(f"{rate_hz:g}"))


def load() -> Study | None:
    return Study(json.loads(STUDY.read_text())) if STUDY.exists() else None


# ── charts ─────────────────────────────────────────────────────────────────────────────────────────────────


def rate_chart(study: Study) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for ax, (arm, title) in zip(axes, ARMS, strict=True):
        block = study.summary[arm]["rate"]
        for layout in LAYOUTS:
            curve = block["tracking_p95_deg"].get(layout)
            if not curve:
                continue
            rates = sorted(curve, key=float)
            xs = [float(r) for r in rates]
            ys = [curve[r] for r in rates]
            ok = [bool(block["passed"][layout][r]) for r in rates]
            ax.plot(xs, ys, "-", color=COLOUR[layout], lw=2.6 if layout == L4 else 1.3, label=SHORT[layout],
                    zorder=4 if layout == L4 else 2)
            ax.scatter([x for x, g in zip(xs, ok, strict=True) if g], [y for y, g in zip(ys, ok, strict=True) if g],
                       color=COLOUR[layout], s=20, zorder=5)
            ax.scatter([x for x, g in zip(xs, ok, strict=True) if not g],
                       [y for y, g in zip(ys, ok, strict=True) if not g], facecolor="white", edgecolor=COLOUR[layout],
                       s=24, zorder=5)
        ax.axvline(RIG_RATE_HZ, color="#71717a", ls="--", lw=1)
        ax.annotate("the rig today: 7 boards\non one fdcanusb", (RIG_RATE_HZ, 0.97), xycoords=("data", "axes fraction"),
                    fontsize=7, color="#52525b", va="top", xytext=(4, 0), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([10, 25, 50, 100, 200, 500, 1000], ["10", "25", "50", "100", "200", "500", "1k"])
        ax.set_xlabel("host rate (Hz)")
        ax.set_ylabel("tracking p95, worst joint (°, log)")
        ax.set_title(title, fontsize=10)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Tracking against host rate. Filled = passes every gate; hollow = fails one (at 1 kHz every layout "
                 "fails only 'latency ≤ one period': the 1.5 ms reply is longer than 1 ms)", fontsize=9)
    return svg(fig)


def speed_chart(study: Study) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))
    for ax, (arm, title) in zip(axes, ARMS, strict=True):
        for layout in LAYOUTS:
            rows = sorted(study.rows_for(arm, "speed", layout=layout), key=lambda r: r["peak_joint_speed_deg_s"])
            if not rows:
                continue
            xs = [r["peak_joint_speed_deg_s"] for r in rows]
            ys = [r["metrics"]["tracking_p95_deg"] for r in rows]
            ax.plot(xs, ys, "-", color=COLOUR[layout], lw=2.6 if layout == L4 else 1.3, label=SHORT[layout],
                    zorder=4 if layout == L4 else 2)
            for x, y, row in zip(xs, ys, rows, strict=True):
                ax.scatter([x], [y], s=20, zorder=5, color=COLOUR[layout] if row["passed"] else "white",
                           edgecolor=COLOUR[layout])
        if arm == "ur5e":
            ax.axvline(180.0, color="#71717a", ls=":", lw=1)
            ax.annotate("UR5e rated\njoint speed", (180.0, 0.97), xycoords=("data", "axes fraction"), fontsize=7,
                        color="#52525b", va="top", xytext=(4, 0), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_xlabel("peak joint speed of the path (°/s)")
        ax.set_ylabel("tracking p95 (°, log)")
        ax.set_title(f"{title}, host at 100 Hz", fontsize=10)
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.12))
    return svg(fig)


def lookahead_chart(study: Study) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))
    for ax, (arm, title) in zip(axes, ARMS, strict=True):
        for entry in study.summary[arm]["lookahead"]:
            curve = entry["curve_deg"]
            xs = sorted((float(k) for k in curve))
            ys = [curve[f"{x:g}"] for x in xs]
            secant = "interp=secant" in entry["layout"]
            rate = entry["rate_hz"]
            ax.plot(xs, ys, "-o" if rate == 100 else "--o", ms=3, lw=1.8,
                    color=COLOUR[L4] if secant else COLOUR[L3],
                    label=f"{'secant' if secant else 'instantaneous'} velocity, {rate:g} Hz "
                    f"(best {entry['best_lookahead_ms']:g} ms)")
        ax.axvline(5.5, color="#71717a", ls=":", lw=1)
        ax.annotate("bus delay\n5.5 ms", (5.5, 0.97), xycoords=("data", "axes fraction"), fontsize=7,
                    color="#52525b", va="top", xytext=(3, 0), textcoords="offset points")
        ax.set_xlabel("lookahead (ms)")
        ax.set_ylabel("tracking p95 (°)")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(bottom=0)
        style(ax)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    return svg(fig)


def jitter_chart(study: Study) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for (arm, title), colour in zip(ARMS, ("#16a34a", "#0891b2"), strict=True):
        entries = study.summary[arm]["jitter"]
        xs = [float(e["jitter_ms"]) for e in entries]
        mean = [e["tracking_p95_deg_mean"] for e in entries]
        low = [e["tracking_p95_deg_min"] for e in entries]
        high = [e["tracking_p95_deg_max"] for e in entries]
        ax.plot(xs, mean, "-o", color=colour, ms=3.5, label=f"{title} (mean of 5 seeds)")
        ax.fill_between(xs, low, high, color=colour, alpha=0.15, lw=0)
    ax.set_xlabel("bus jitter: each command's delay is 5.5 ms ± this (uniform, ms)")
    ax.set_ylabel("tracking p95 (°)")
    ax.set_title("v1 (lookahead 5.5 ms + secant velocity) at 100 Hz with a jittery bus; band = best to worst seed",
                 fontsize=9)
    ax.set_ylim(bottom=0)
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    return svg(fig)


def sync_chart(study: Study) -> str:
    per_layout = study.summary["planar_arm"]["sync"]["per_layout"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8))
    for layout in LAYOUTS:
        block = per_layout.get(layout)
        if not block:
            continue
        speeds = sorted(block, key=float)
        xs = [float(s) for s in speeds]
        axes[0].plot(xs, [block[s]["sync_error_p95_mm"] for s in speeds], "-o", ms=3.5, color=COLOUR[layout],
                     lw=2.6 if layout == L4 else 1.3, label=SHORT[layout])
        axes[1].plot(xs, [block[s]["sync_timing_ms"] for s in speeds], "-o", ms=3.5, color=COLOUR[layout],
                     lw=2.6 if layout == L4 else 1.3)
    axes[0].axhline(5.0, color="#71717a", ls=":", lw=1)
    axes[0].annotate("gate: 5 mm", (0.85, 5.0), fontsize=7, color="#52525b", xytext=(2, 3), textcoords="offset points")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("tool's distance from the item, p95 (mm, log)")
    axes[1].axhline(0.0, color="#71717a", lw=0.8)
    axes[1].set_ylabel("timing: how far behind the item (ms)")
    axes[1].set_yscale("symlog", linthresh=2.0)
    for ax in axes:
        ax.set_xlabel("belt speed (m/s)")
        style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("R10: the tool keeps pace with a point riding the belt, 50 mm above it (planar twin, 100 Hz)",
                 fontsize=10)
    return svg(fig)


# ── tables ─────────────────────────────────────────────────────────────────────────────────────────────────


def rate_table(study: Study) -> str:
    rates = (10, 25, 50, 100, 200, 500)
    head = "".join(f"<th>{r} Hz</th>" for r in rates)
    body = []
    for arm, title in ARMS:
        for layout in (L2, L3, L4, L6, L7):
            cells = []
            for rate in rates:
                value = study.rate(arm, layout, rate)
                cls = "ok" if study.rate_passed(arm, layout, rate) else "bad"
                cells.append(f"<td class='{cls}'>{num(value)}</td>")
            mark = " <b>(v1)</b>" if layout == L4 else ""
            body.append(f"<tr><th>{esc(title.split(',')[0])}</th><th>{esc(SHORT[layout].replace(' (v1)', ''))}{mark}"
                        f"</th>{''.join(cells)}</tr>")
    return (f"<div class='scroll'><table class='grid'><thead><tr><th>arm</th><th>layout</th>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div><p class='muted'>Tracking p95 (°), worst joint. Green = "
            f"passes every gate on that rung; red = fails one.</p>")


def grid_table(study: Study, arm: str, layout: str) -> str:
    rows = study.rows_for(arm, "rate_speed", layout=layout)
    if not rows:
        return ""
    cells = {(r["rate_hz"], r["move_s"]): r for r in rows}
    rates = sorted({r["rate_hz"] for r in rows})
    moves = sorted({r["move_s"] for r in rows}, reverse=True)
    speed = {r["move_s"]: r["peak_joint_speed_deg_s"] for r in rows}
    head = "".join(f"<th>{m:g} s<br><span class='muted'>{speed[m]:.0f}°/s</span></th>" for m in moves)
    body = []
    for rate in rates:
        tds = []
        for move in moves:
            row = cells.get((rate, move))
            if row is None:
                tds.append("<td>—</td>")
                continue
            cls = "ok" if row["passed"] else "bad"
            tds.append(f"<td class='{cls}'>{num(row['metrics']['tracking_p95_deg'])}</td>")
        body.append(f"<tr><th>{rate:g} Hz</th>{''.join(tds)}</tr>")
    return (f"<h3>{esc(SHORT[layout])}</h3><div class='scroll'><table class='grid'><thead><tr><th>host rate ╲ move time"
            f"</th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


def stagger_table(study: Study) -> str:
    body = []
    for arm, title in ARMS:
        for entry in study.summary[arm]["stagger"]:
            p95 = entry["tracking_p95_deg"]
            la = entry["lookahead_ms"]

            def fmt(value):
                return "/".join(f"{v:g}" for v in value) if isinstance(value, list) else f"{value:g}"

            cells = "".join(f"<td>{num(p95[way])}°<br><span class='muted'>{esc(fmt(la[way]))} ms</span></td>"
                            for way in ("miscalibrated", "mean", "true", "calibrated"))
            body.append(f"<tr><th>{esc(title.split(',')[0])}</th><th>{entry['rate_hz']:g} Hz</th>{cells}</tr>")
    return ("<div class='scroll'><table><thead><tr><th>arm</th><th>host</th><th>one lookahead, 5.5 ms<br>"
            "<span class='muted'>(the twin's number)</span></th><th>one lookahead, the mean delay</th><th>per joint, "
            "the true bus delays</th><th>per joint, calibrated from one run</th></tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


def sync_table(study: Study) -> str:
    per_layout = study.summary["planar_arm"]["sync"]["per_layout"]
    speeds = sorted(per_layout[L2], key=float)
    head = "".join(f"<th>{float(s):g} m/s</th>" for s in speeds)
    body = []
    for layout in (L2, L3, L4, L6, L7):
        cells = "".join(
            f"<td class='{'ok' if per_layout[layout][s]['passed'] else 'bad'}'>"
            f"{num(per_layout[layout][s]['sync_error_p95_mm'])} mm<br><span class='muted'>"
            f"{num(per_layout[layout][s]['sync_timing_ms'])} ms</span></td>" for s in speeds)
        body.append(f"<tr><th>{esc(SHORT[layout])}</th>{cells}</tr>")
    return (f"<div class='scroll'><table class='grid'><thead><tr><th>layout</th>{head}</tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div><p class='muted'>Each cell: the tool's p95 distance from the item, and "
            "how far behind it the tool runs, as time (positive = behind). At slow belts a fixed offset of a fraction of a "
            "millimetre reads as several ms, so read the mm.</p>")


def ur_table(study: Study) -> str:
    body = []
    for layout in LAYOUTS:
        planar = study.rate("planar_arm", layout, 100)
        ur = study.rate("ur5e", layout, 100)
        planar50 = study.rate("planar_arm", layout, 50)
        ur50 = study.rate("ur5e", layout, 50)
        body.append(f"<tr><th>{esc(SHORT[layout])}</th><td>{num(planar)}</td><td>{num(planar50)}</td>"
                    f"<td>{num(ur)}</td><td>{num(ur50)}</td></tr>")
    return ("<div class='scroll'><table class='grid'><thead><tr><th>layout</th><th>planar R2, 100 Hz</th><th>planar R2, "
            "50 Hz</th><th>UR5e U2, 100 Hz</th><th>UR5e U2, 50 Hz</th></tr></thead><tbody>" + "".join(body)
            + "</tbody></table></div>")


# ── the page's top: the v1 recipe and the study's sections ─────────────────────────────────────────────────


def _row(study: Study, arm: str, dimension: str, **where) -> dict | None:
    rows = study.rows_for(arm, dimension, **where)
    return rows[0] if rows else None


def _metric(study: Study, arm: str, dimension: str, key: str, **where):
    row = _row(study, arm, dimension, **where)
    return None if row is None else row["metrics"].get(key)


def top(study: Study, media: str) -> str:
    s = study.summary
    p = lambda layout, rate, arm="planar_arm": study.rate(arm, layout, rate)  # noqa: E731
    tool = lambda layout, rate, arm="ur5e": _metric(study, arm, "rate", "tool_p95_mm", layout=layout, rate_hz=float(rate))  # noqa: E731
    sync = s["planar_arm"]["sync"]["per_layout"]
    jitter = {arm: {float(e["jitter_ms"]): e for e in s[arm]["jitter"]} for arm, _ in ARMS}
    la = {arm: {(e["layout"], e["rate_hz"]): e for e in s[arm]["lookahead"]} for arm, _ in ARMS}
    stag = {arm: {e["rate_hz"]: e for e in s[arm]["stagger"]} for arm, _ in ARMS}
    fastest = {arm: max(study.rows_for(arm, "speed", layout=L4), key=lambda r: r["peak_joint_speed_deg_s"])
               for arm, _ in ARMS}
    ur_hero = (REPO / "reports" / media / "hero-ur5e.mp4").exists()
    ur_video = (f"<figure><video src='{media}/hero-ur5e.mp4' poster='{media}/hero-ur5e.jpg' autoplay muted loop "
                f"playsinline controls></video><figcaption class='muted'>UR5e on U2 (fast moves, 0.45 s each) at the "
                f"pick cycle's 50 Hz, 2× slow motion, real bench episodes (the meshes are Menagerie's; the physics is the "
                f"vendored bare model). The trace is the tool's distance from where the plan puts it. Left: today's JTC. "
                f"Middle: + 5.5 ms lookahead. Right: + the secant velocity, the v1 recipe. "
                f"<a href='{media}/hero-ur5e.gif'>GIF</a> · made by <code>docs/ur5e_hero_video.py</code></figcaption>"
                f"</figure>") if ur_hero else ""
    v1_la_ur = la["ur5e"][(L4, 100.0)]
    v1_la_pl = la["planar_arm"][(L4, 100.0)]
    return f"""
<div class="verdict">
<p><b>Yes: v1 is feed-forward at plan time and feedback on the board, plus two free fixes on the host. Sample the plan
one bus delay ahead (lookahead), and send the <i>secant</i> velocity (the straight line to the next sample) instead of the
instantaneous one.</b> At the rig's 50 Hz that takes the planar twin from {num(p(L2, 50))}° to {num(p(L4, 50))}° p95 on
fast moves, and the UR5e from {num(p(L2, 50, 'ur5e'))}° to {num(p(L4, 50, 'ur5e'))}° (tool error {num(tool(L2, 50))} →
{num(tool(L4, 50))} mm). Nothing extra runs on the host and nothing extra crosses the bus.</p>
<ul>
<li><b>Why 50 Hz:</b> the bus, not Python or physics. One fdcanusb takes ~15.5 ms to exchange with all seven boards, so a
10 ms tick can't hold it (2026-11-13, bam_ws PEP 0048). With the v1 recipe 50 Hz costs little: the planar twin is at
{num(p(L4, 50))}° vs {num(p(L4, 100))}° at 100 Hz; the UR5e is at {num(p(L4, 50, 'ur5e'))}° vs {num(p(L4, 100, 'ur5e'))}°.</li>
<li><b>Feedback on the host can't run at 50 Hz:</b> PD on the host is at {num(p(L7, 50))}° and computed torque on the host
at {num(p(L6, 50))}° on the planar twin. Its loop carries one host period plus the 7 ms round trip; the board's carries
0.5 ms.</li>
<li><b>Lookahead calibrates from one run.</b> One run with no lookahead gives the delay by lining the motion up with the
plan. The tracking curve is a V with its bottom at the delay: best {v1_la_pl['best_lookahead_ms']:g} ms on the planar
twin and {v1_la_ur['best_lookahead_ms']:g} ms on the UR5e (bus 5.5 ms); 1 ms off costs
{num(min(v1_la_pl['cost_of_1ms_error_deg'], v1_la_ur['cost_of_1ms_error_deg']))}–{num(max(v1_la_pl['cost_of_1ms_error_deg'], v1_la_ur['cost_of_1ms_error_deg']))}°. On a bus that serves joints in turn,
calibrate per joint: on the UR5e that gave {num(stag['ur5e'][50.0]['tracking_p95_deg']['calibrated'])}°, against
{num(stag['ur5e'][50.0]['tracking_p95_deg']['miscalibrated'])}° with one 5.5 ms lookahead.</li>
<li><b>Reliability:</b> exact for a constant delay. With ±1 ms of bus jitter the v1 recipe is at
{num(jitter['planar_arm'][1.0]['tracking_p95_deg_mean'])}° (planar, worst seed
{num(jitter['planar_arm'][1.0]['tracking_p95_deg_max'])}°); at ±3 ms, {num(jitter['planar_arm'][3.0]['tracking_p95_deg_mean'])}°.
Still far better than no lookahead ({num(p(L2, 100))}°).</li>
<li><b>Catching a moving item:</b> on a belt at 0.25 m/s the tool runs {num(sync[L2]['0.25']['sync_timing_ms'])} ms behind
the item as built ({num(sync[L2]['0.25']['sync_error_p95_mm'])} mm p95) and {num(sync[L4]['0.25']['sync_timing_ms'])} ms behind with
v1 ({num(sync[L4]['0.25']['sync_error_p95_mm'])} mm). At 1 m/s: {num(sync[L2]['1.00']['sync_error_p95_mm'])} mm, failing
the 5 mm gate, against {num(sync[L4]['1.00']['sync_error_p95_mm'])} mm.</li>
<li><b>Speed:</b> v1 passes every speed tried, up to {fastest['planar_arm']['peak_joint_speed_deg_s']:.0f}°/s on the
planar twin and {fastest['ur5e']['peak_joint_speed_deg_s']:.0f}°/s on the UR5e (2.5× its rated joint speed). There it
tracks {num(fastest['planar_arm']['metrics']['tracking_p95_deg'])}° and {num(fastest['ur5e']['metrics']['tracking_p95_deg'])}°,
and the boards start touching their torque caps ({num(fastest['planar_arm']['metrics']['saturation_share'])} % and
{num(fastest['ur5e']['metrics']['saturation_share'])} % of the time).</li>
<li><b>On the UR5e the floor is friction.</b> v1 sits at ~0.34° at every rate because the trajectory's torque column
carries no dry friction. The dispatcher's check, not in the ledger: a Coulomb term halves U1's tool error
(1.40 → 0.73 mm) and cuts the low-frequency feedback share from 16 % to 5 %, but it over-compensates wrist_3 at reversals.</li>
</ul>
<p class="muted" style="margin-top:8px">Sim only: the planar MuJoCo twin and a UR5e (Menagerie links, bam_pid_tunning's
friction and rotor values) on the same moteus-like board model and bus. {len(study.rows)} episodes in
<code>results/study/results.json</code>.</p>
</div>
{ur_video}

<h2>Why 50 Hz, and how the rate changes things</h2>
<p>50 Hz is the rate the real pick cycle runs at, and it was set by the bus. The 2026-11-13 wrist run asked for 100 Hz and
was refused: four boards took 9.3 ms p95 per exchange on one fdcanusb, and all seven take ~15.5 ms (p95 16.4 ms,
2026-11-18). A 10 ms tick can't hold that, so the tick became 20 ms (bam_ws <code>docs/peps/0048-full-arm-td4-pans-j1.md</code>,
<code>full_arm.py:77-80</code>). A second adapter is the way up if it's ever needed; nobody has measured it. One more
caveat: the twin's 5.5 ms down-delay is a fitted number, not a timed one.</p>
<p>The board closes its PD at 4 kHz whatever the host does, so a board-side law only needs the host to carry the
reference. Between two commands the board walks its setpoint forward at the velocity it was sent. Send the instantaneous
velocity and the setpoint drifts off a curving path within each tick; send the secant and it lands exactly on the next
sample. That one change is what makes 50 Hz nearly as good as 100 Hz.</p>
{rate_chart(study)}
{rate_table(study)}

<h2>Speed: how far each layout carries</h2>
{speed_chart(study)}
<details><summary>Rate × speed grids (tracking p95 °, green = passes every gate)</summary>
{grid_table(study, 'planar_arm', L4)}{grid_table(study, 'planar_arm', L3)}{grid_table(study, 'planar_arm', L6)}
{grid_table(study, 'ur5e', L4)}{grid_table(study, 'ur5e', L3)}{grid_table(study, 'ur5e', L6)}
</details>

<h2>Lookahead: why it works, how far, and how to calibrate it</h2>
<p><b>Why it works.</b> The bus delay is outside the board's loop: it only delays the setpoint. The plan is known ahead,
so the host sends the setpoint meant for "now + 5.5 ms" and it arrives on time. It also stops the model torque from
arriving late, which is why the as-built JTC ({num(p(L2, 100))}°) barely beats no feed-forward at all. For feedback on
the host the delay is <i>inside</i> the loop, so advancing the reference only compares it with an even staler measurement.
That needs state prediction (a Smith predictor), or the loop moved into the drive.</p>
<p><b>How far.</b> The bottom of the V is at the delay. With the secant velocity the curve is flatter and its bottom
holds at 50 Hz too.</p>
{lookahead_chart(study)}
<p><b>Calibrating it per joint.</b> On one bus the boards are served in turn, so each joint's command lands at a
different time. The study gave each joint a delay of 1.5 + 2.2·j ms (a one-adapter bus at ~2.2 ms per board) and tried
four lookaheads. "Calibrated" is the one-run procedure: run once with no lookahead, line each joint's motion up with its
plan, and use those shifts.</p>
{stagger_table(study)}
<p class="muted">On the UR5e the calibrated values overshoot the bus on wrist_1 and wrist_2 by ~5–6 ms and still track
best. The estimate measures the joint's <i>total</i> lag, its friction-driven servo lag included, and compensates that too.
That's a reason to calibrate from logs rather than from a bus spec.</p>

<h2>How reliable: a jittery bus</h2>
<p>A fixed lookahead cancels the <i>average</i> delay; jitter leaves roughly speed × jitter. Industrial buses remove the
jitter instead (EtherCAT latches every setpoint on a shared clock edge). On the twin the v1 recipe degrades gently:</p>
{jitter_chart(study)}
<p class="muted">The UR5e line is flat because its worst joint is a wrist held back by dry friction (~0.34°), which hides
the jitter. The jitter is drawn per command and rounded to whole 0.25 ms board ticks, so a seed replays exactly (the
builder's choice; jitter under a tick has no effect).</p>

<h2>Catching a moving item</h2>
<p>Timing error is position error: mm = belt speed × ms. Two delays have to be cancelled. One is on the command path,
which is this lookahead. The other is on the perception path, from camera timestamp to plan, where the item is
predicted forward from a timestamped detection. Both need one shared clock. For scale, ABB's conveyor tracking specifies
±2 mm at 150 mm/s.</p>
{sync_chart(study)}
{sync_table(study)}

<h2>The UR5e next to the planar twin</h2>
<p>The same seven layouts, the same board model and bus, on a 6-joint arm with ~10× the inertia. The board gains have the
twin shoulder's bandwidth (ωn 39.7 rad/s, ζ 0.5) on each joint's own inertia, and torque caps 150/28 N·m. The top three are
the same on both arms: v1, then the JTC with lookahead, then computed torque on the host. On the heavier UR5e even PD on the
host holds at 100 Hz (0.71°). At the rig's 50 Hz, on both arms, both fully host-side laws fail by saturating the boards,
and only layouts with the feedback on the board, or at least its damper there, pass.</p>
{ur_table(study)}

<h2>Is lookahead standard practice?</h2>
<div class='scroll'><table><thead><tr><th>System</th><th>What it does</th><th>Same idea?</th></tr></thead><tbody>
<tr><th>ros2_controllers JTC</th><td>samples the command at <code>time + update_period</code> since PR #1306 (Nov 2024);
tolerances still checked at <code>time</code></td><td class='ok'>yes, one period; no bus-latency setting</td></tr>
<tr><th>Beckhoff TwinCAT NC "Time Compensation"</th><td>drive and encoder dead-time compensation from the bus's DC
shifts plus delays you set</td><td class='ok'>yes, built into the motion stack</td></tr>
<tr><th>Siemens SINAMICS DSC</th><td>the position loop runs in the drive; the bus dead time is set automatically and the
feed-forward is "symmetrised" to match</td><td class='ok'>yes, and it's our layout answer too: feedback in the
drive</td></tr>
<tr><th>EtherCAT CiA 402 CSP</th><td>the master plans, the drive closes the loops, velocity/torque offsets go as
feed-forward, setpoints latch on a shared clock edge</td><td class='ok'>yes (time-aligned by the bus)</td></tr>
<tr><th>UR <code>servoj</code> <code>lookahead_time</code></th><td>projects the robot's <i>own</i> position forward by its
velocity: a damping term that adds lag (a user measured ~34 ms)</td><td class='bad'>no, despite the name</td></tr>
</tbody></table></div>
<p class="muted">Sources and the theory (preview control, ZPETC, the Smith predictor) in the vault note
<code>World Model/Chunks &amp; Concepts/C - Reference Lookahead (Command Delay Compensation).md</code>.</p>
"""
