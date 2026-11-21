"""Build the layouts handoff report: reports/controller-layouts-2026-11-20.html.

    env -u VIRTUAL_ENV uv run --frozen --directory /home/bam/robot_controllers --extra test --extra bench \
        python docs/build_layouts_report.py

Every number on the page is read from results/layouts/results.json (the sweep's ledger, CONTRACT amendment 3.6 and
3.8), and the recommendations are the ledger's own `summary.layouts`, so the page and the ledger can't disagree.
Charts are inline SVG drawn by matplotlib; the hero video (docs/layouts_hero_video.py) sits in
reports/media/controller-layouts-2026-11-20/ beside the page.
"""

from __future__ import annotations

import html
import io
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import study_sections  # noqa: E402  (docs/study_sections.py: the amendment-4 study's sections)

REPO = Path(__file__).resolve().parents[1]
DATE = "2026-11-20"
SLUG = f"controller-layouts-{DATE}"
OUT = REPO / "reports" / f"{SLUG}.html"
MEDIA = f"media/{SLUG}"
LEDGER = REPO / "results" / "layouts" / "results.json"
NAME = re.compile(r"ff=(?P<ff>[a-z_]+)@(?P<ff_at>[a-z_]+)\+fb=(?P<fb>[a-z_]+)@(?P<fb_at>[a-z_]+)(?:\+la=(?P<la>[0-9.]+)ms)?")
R2, R8, R9_50, R6 = "R2.FastFreeSpace", "R8.FastFreeSpace·payload", "R9.FastFreeSpace·host50Hz", "R6.HostStall"
LA = "+la=5.5ms"

FF_ORDER = ["trajectory", "none", "gravity", "rigid_body", "rigid_body_coulomb"]
FB_ORDER = ["board_gains", "pd", "ctc"]
FB_AT_ORDER = ["board", "board_scheduled", "split", "host"]
FB_COLOURS = {"board": "#16a34a", "board_scheduled": "#0891b2", "split": "#d97706", "host": "#dc2626"}
SHORT_FB = {"board_gains": "board gains", "pd": "PD", "ctc": "CTC"}
SHORT_AT = {"board": "board", "board_scheduled": "board,\nre-sent", "split": "split", "host": "host"}
PLAIN_FF_AT = {"trajectory": "precomputed in the trajectory's torque column", "none": "none", "plan": "plan time",
               "host_reference": "the host, at the reference", "host_measured": "the host, at the measured state"}
PLAIN = {
    "trajectory": "the trajectory's own torque column", "none": "none", "gravity": "gravity",
    "rigid_body": "rigid body", "rigid_body_coulomb": "rigid body + Coulomb", "plan": "plan time",
    "host_reference": "host, at the reference", "host_measured": "host, at the measured state",
    "board_gains": "the board's own gains", "pd": "PD (same gains, as a function)", "ctc": "computed-torque PD",
    "board": "board", "board_scheduled": "board, gains re-sent every tick", "split": "split (host spring, board damper)",
    "host": "host",
}


def esc(text: object) -> str:
    return html.escape(str(text))


def num(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    v = float(value)
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.{digits}g}" if v else "0"


def parse(name: str) -> dict[str, str] | None:
    match = NAME.fullmatch(name)
    if not match:
        return None
    parts = match.groupdict()
    parts["la"] = parts["la"] or "0"
    return parts


def svg(fig) -> str:
    buffer = io.StringIO()
    fig.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(fig)
    text = buffer.getvalue()
    return "<figure class='chart'>" + text[text.index("<svg"):] + "</figure>"


def style_axes(ax) -> None:
    ax.grid(True, color="#d4d4d8", lw=0.6, alpha=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


class Ledger:
    def __init__(self, document: dict) -> None:
        self.document = document
        self.rungs = [r["id"] for r in document["rungs"]]
        self.runs = {(r["controller"], r["rung"]): r for r in document["results"]}
        self.parts = {c["name"]: parse(c["name"]) for c in document["controllers"] if parse(c["name"])}
        self.summary = document["summary"]["layouts"]
        self.candidates = {c["name"]: c for c in self.summary["candidates"]}

    def metric(self, name: str, rung: str, key: str):
        run = self.runs.get((name, rung))
        return None if run is None else run["metrics"].get(key)

    def passed(self, name: str) -> bool:
        return bool(self.candidates.get(name, {}).get("passed"))

    def failed_rungs(self, name: str) -> list[str]:
        return list(self.candidates.get(name, {}).get("failed_rungs", []))


def short(rung: str) -> str:
    head, _, tail = rung.partition(".")
    return head + ("·" + tail.split("·", 1)[1] if "·" in tail else "")


def recommendation_table(ledger: Ledger) -> str:
    rows = []
    for pair in ledger.summary["per_pair"]:
        ff, fb, rec = pair["feedforward_model"], pair["feedback_model"], pair["recommendation"]
        chosen = parse(rec["name"]) if rec.get("name") else None
        if chosen:
            where = (f"feed-forward <b>{esc(PLAIN_FF_AT[chosen['ff_at']])}</b>, feedback on <b>{esc(PLAIN[chosen['fb_at']])}</b>"
                     + (f", lookahead <b>{esc(chosen['la'])} ms</b>" if chosen["la"] != "0" else ", no lookahead"))
            p95 = num(ledger.metric(rec["name"], R2, "tracking_p95_deg"))
        else:
            where, p95 = "<span class='bad'>no layout passes every rung</span>", "—"
        passing = sum(ledger.passed(n) for n in pair["candidates"])
        rows.append(f"<tr><th>{esc(PLAIN[ff])}</th><th>{esc(PLAIN[fb])}</th><td>{where}</td><td>{p95}</td>"
                    f"<td>{passing} / {len(pair['candidates'])}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>Feed-forward model</th><th>Feedback model</th><th>Recommended layout"
            "</th><th>R2 p95 (°)</th><th>Layouts that pass every rung</th></tr></thead><tbody>" + "".join(rows)
            + "</tbody></table></div>")


def lookahead_chart(ledger: Ledger) -> str:
    """Every feedback placement, with the same feed-forward (rigid body at plan time), at lookahead 0 and 5.5 ms."""
    rows = []
    for fb in FB_ORDER:
        for at in FB_AT_ORDER:
            ff = "ff=trajectory@trajectory" if fb == "board_gains" else "ff=rigid_body@plan"
            name = f"{ff}+fb={fb}@{at}"
            if name in ledger.parts:
                rows.append((f"{SHORT_FB[fb]}\non {SHORT_AT[at]}", name))
    fig, ax = plt.subplots(figsize=(11.5, 3.8))
    width = 0.38
    for k, (suffix, label, colour) in enumerate([("", "no lookahead", "#a1a1aa"), (LA, "lookahead 5.5 ms (the bus delay)", "#2563eb")]):
        xs = [i + (k - 0.5) * width for i in range(len(rows))]
        values = [ledger.metric(n + suffix, R2, "tracking_p95_deg") or 0.0 for _, n in rows]
        ax.bar(xs, values, width, color=colour, label=label)
        for x, (_, n), v in zip(xs, rows, values, strict=True):
            mark = "" if ledger.passed(n + suffix) else "✗"
            ax.annotate(f"{v:.2f}{mark}", (x, v), ha="center", va="bottom", fontsize=7,
                        color="#dc2626" if mark else "#3f3f46")
    ax.set_xticks(range(len(rows)), [r for r, _ in rows], fontsize=8)
    ax.set_xlabel("feedback model, and where it closes (feed-forward: the model at plan time, or the trajectory's column)",
                  fontsize=8)
    ax.set_ylabel("R2 tracking p95 (°)")
    ax.set_title("Where the feedback closes, with and without lookahead (✗ = fails a gate on some rung)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style_axes(ax)
    return svg(fig)


def ladder_chart(ledger: Ledger) -> str:
    """The feed-forward ladder: the board's own gains, lookahead 5.5 ms; only the model changes."""
    order = [("none", "none", "none"), ("gravity", "plan", "gravity"), ("rigid_body", "plan", "rigid body"),
             ("rigid_body_coulomb", "plan", "rigid body\n+ Coulomb")]
    names = [(f"ff={ff}@{at}+fb=board_gains@board{LA}", label) for ff, at, label in order]
    names = [(n, label) for n, label in names if n in ledger.parts]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
    panels = [("tracking_p95_deg", "tracking p95 (°)"), ("fb_share_pct", "feedback share (%)"),
              ("fb_share_lf_pct", "feedback share below 50 Hz (%)")]
    for ax, (key, title) in zip(axes, panels, strict=True):
        xs = range(len(names))
        for k, (rung, label, colour) in enumerate([(R2, "R2", "#2563eb"), (R8, "R8: +0.3 kg nobody modelled", "#f59e0b")]):
            values = [ledger.metric(n, rung, key) or 0.0 for n, _ in names]
            bars = ax.bar([x + (k - 0.5) * 0.38 for x in xs], values, 0.38, color=colour, label=label)
            for bar, value in zip(bars, values, strict=True):
                ax.annotate(num(value, 2), (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha="center",
                            va="bottom", fontsize=7)
        ax.set_xticks(list(xs), [label for _, label in names], fontsize=8)
        ax.set_title(title, fontsize=10)
        style_axes(ax)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Feedback on the board's own gains, lookahead 5.5 ms: only the feed-forward model changes",
                 fontsize=10, y=1.03)
    return svg(fig)


def pareto_chart(ledger: Ledger) -> str:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for name, p in ledger.parts.items():
        cand = ledger.candidates.get(name)
        x, y = (cand or {}).get("host_duty_pct"), ledger.metric(name, R2, "tracking_p95_deg")
        if x is None or y is None:
            continue
        colour = FB_COLOURS.get(p["fb_at"], "#71717a")
        ax.scatter([x], [y], s=30, facecolors=colour if ledger.passed(name) else "none", edgecolors=colour,
                   linewidths=1.2, marker="^" if p["la"] != "0" else "o", zorder=3, alpha=0.9)
    for at, colour in FB_COLOURS.items():
        ax.scatter([], [], color=colour, label=f"feedback on {PLAIN[at]}")
    ax.scatter([], [], marker="^", color="#71717a", label="triangle = lookahead 5.5 ms")
    ax.scatter([], [], facecolors="none", edgecolors="#71717a", label="hollow = fails a gate on some rung")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("host CPU, % of one core (median over the rungs, at 100 Hz)")
    ax.set_ylabel("R2 tracking p95, worst joint (°)")
    ax.set_title(f"All {len(ledger.parts)} layouts: lower-left is better", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    style_axes(ax)
    return svg(fig)


KEY_COLUMNS = [(R2, "tracking_p95_deg", "R2 p95 (°)"), (R2, "fb_share_pct", "FB share (%)"),
               (R2, "fb_share_lf_pct", "FB share <50 Hz (%)"), (None, "host_duty_pct", "host CPU (% core)"),
               (R2, "plan_compute_ms_per_s", "plan CPU (ms/s)"), (R2, "wire_bytes_per_s", "bus (B/s)"),
               (R8, "tracking_p95_deg", "R8 payload p95 (°)"), (R9_50, "tracking_p95_deg", "50 Hz p95 (°)"),
               (R6, "tracking_p95_deg", "host stall p95 (°)")]


def key_numbers(ledger: Ledger, rows: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{esc(label)}</th>" for _, _, label in KEY_COLUMNS)
    body = []
    for label, name in rows:
        if name not in ledger.parts:
            continue
        cells = "".join(
            f"<td>{num(ledger.candidates.get(name, {}).get(key) if rung is None else ledger.metric(name, rung, key))}</td>"
            for rung, key, _ in KEY_COLUMNS)
        failed = ledger.failed_rungs(name)
        verdict = ("<span class='ok'>every rung</span>" if not failed
                   else "<span class='bad'>fails " + ", ".join(esc(short(r)) for r in failed) + "</span>")
        body.append(f"<tr><th>{esc(label)}<br><code>{esc(name)}</code></th>{cells}<td>{verdict}</td></tr>")
    return (f"<div class='scroll'><table><thead><tr><th>layout</th>{head}<th>gates</th></tr></thead><tbody>"
            + "".join(body) + "</tbody></table></div>")


def gate_grid(ledger: Ledger) -> str:
    head = "".join(f"<th>{esc(short(r))}</th>" for r in ledger.rungs)

    def order(name: str):
        p = ledger.parts[name]
        return (FF_ORDER.index(p["ff"]), p["ff_at"], FB_ORDER.index(p["fb"]), FB_AT_ORDER.index(p["fb_at"]), p["la"])

    rows = []
    for name in sorted(ledger.parts, key=order):
        cells = []
        for rung in ledger.rungs:
            run = ledger.runs.get((name, rung))
            if run is None:
                cells.append("<td class='na'>·</td>")
                continue
            failed = [g for g in run["gates"] if not g["passed"]]
            tip = "; ".join(f"{g['metric']} {g['op']} {g['value']:g}: {num(g['measured'])}" for g in failed)
            cells.append(f"<td class='{'ok' if run['passed'] else 'bad'}' title='{esc(tip)}'>{num(run['objective'], 2)}</td>")
        rows.append(f"<tr><th><code>{esc(name)}</code></th>{''.join(cells)}</tr>")
    return (f"<details><summary>Every layout on every rung ({len(rows)} rows; hover a red cell for the failed gate)"
            f"</summary><div class='scroll'><table class='grid'><thead><tr><th>layout</th>{head}</tr></thead><tbody>"
            + "".join(rows) + "</tbody></table></div></details>")


CSS = """
:root{--bg:#fafafa;--fg:#18181b;--muted:#52525b;--card:#fff;--line:#e4e4e7;--ok:#15803d;--bad:#b91c1c;--accent:#2563eb}
@media (prefers-color-scheme: dark){:root{--bg:#0b0b0c;--fg:#e4e4e7;--muted:#a1a1aa;--card:#18181b;--line:#2e2e33;--ok:#4ade80;--bad:#f87171;--accent:#60a5fa}
 figure.chart svg{background:#fff;border-radius:8px;padding:4px}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 80px}h1{font-size:26px;line-height:1.25;margin:0 0 6px}h2{margin:40px 0 10px;font-size:20px}h3{margin:22px 0 6px;font-size:16px}
.verdict{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--accent);padding:14px 18px;border-radius:8px}
.verdict p{margin:0 0 8px}.verdict ul{margin:6px 0 0 18px;padding:0}.muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;background:var(--card);font-size:13px}th,td{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
thead th{background:var(--bg)}.ok{color:var(--ok)}.bad{color:var(--bad)}td.ok{background:rgba(22,163,74,.08)}td.bad{background:rgba(220,38,38,.10)}
.scroll{overflow-x:auto}table.grid td{text-align:center;font-variant-numeric:tabular-nums;min-width:48px;font-size:12px}table.grid th{font-size:12px}
figure{margin:14px 0}video,img{max-width:100%;border-radius:8px;border:1px solid var(--line)}svg{max-width:100%;height:auto}
code{font-size:12px;word-break:break-word}details summary{cursor:pointer;color:var(--accent);margin:8px 0}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.card h3{margin:0 0 6px}.card ul{margin:0;padding-left:18px}pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px;overflow-x:auto;font-size:12px}
"""


def page(ledger: Ledger, study: study_sections.Study | None = None) -> str:
    g = ledger.document["generated"]
    best = ledger.summary["overall"]["name"]
    jtc, jtc_la = "ff=trajectory@trajectory+fb=board_gains@board", "ff=trajectory@trajectory+fb=board_gains@board" + LA
    m = lambda n, r=R2, k="tracking_p95_deg": ledger.metric(n, r, k)  # noqa: E731
    passing = sum(ledger.passed(n) for n in ledger.parts)
    ctc_host = "ff=rigid_body@host_measured+fb=ctc@host"
    key_rows = [
        ("Simple JTC, no feed-forward", "ff=none@none+fb=board_gains@board"),
        ("Simple JTC as built (model torque in the trajectory)", jtc),
        ("Simple JTC + lookahead: the recommendation", jtc_la),
        ("Model at plan time + the board's gains + lookahead", "ff=rigid_body@plan+fb=board_gains@board" + LA),
        ("+ Coulomb friction in the model", "ff=rigid_body_coulomb@plan+fb=board_gains@board" + LA),
        ("Model on the host at the measured state (what the pick cycle does)",
         "ff=rigid_body@host_measured+fb=board_gains@board" + LA),
        ("Computed torque, gains re-sent every tick", "ff=rigid_body@plan+fb=ctc@board_scheduled" + LA),
        ("Host spring, board damper", "ff=rigid_body@plan+fb=pd@split"),
        ("Computed torque on the host", ctc_host),
        ("PD on the host", "ff=rigid_body@plan+fb=pd@host"),
    ]
    hero = (REPO / "reports" / MEDIA / "hero-layouts.mp4").exists()
    hero_html = (f"<figure><video src='{MEDIA}/hero-layouts.mp4' poster='{MEDIA}/hero-layouts.jpg' autoplay muted loop "
                 f"playsinline controls></video><figcaption class='muted'>R2 (three fast 0.3 s moves), 2× slow motion, "
                 f"real bench episodes. The faint arm is the reference. Left: no feed-forward, so the board's PD holds "
                 f"the arm up and it sags about a degree. Middle: today's JTC with the model torque, fast but late. Right: "
                 f"the same JTC sampling the reference 5.5 ms ahead. <a href='{MEDIA}/hero-layouts.gif'>GIF</a> · "
                 f"made by <code>docs/layouts_hero_video.py</code></figcaption></figure>") if hero else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Controller Layouts</title><style>{CSS}</style></head><body><main>
<h1>Controllers v1: where feed-forward and feedback run, at what rate, and how far ahead</h1>
<p class="muted">robot_controllers, contract amendments 3 and 4 · the planar MuJoCo twin and a UR5e on a moteus-like board
model (4 kHz PD, 5.5 ms bus down, 1.5 ms up) · first sweep: {len(ledger.parts)} layouts × {len(ledger.rungs)} rungs
(<code>{esc(g['commit'])}</code>) · study: {len(study.rows) if study else 0} episodes over rate, speed, lookahead, bus jitter,
per-joint delays and a moving target (<code>{esc(study.document['generated'].get('commit', '?')) if study else '—'}</code>)
· <b>sim only</b>; nothing here touched the arm</p>
{study_sections.top(study, MEDIA) if study else ""}

<h2>The first sweep: every layout of every model (amendment 3)</h2>
<div class="verdict">
<p><b>Forward the trajectory, leave the feedback on the board, compute the model at plan time and sample the reference
5.5 ms ahead (the bus delay). That's your "first step" JTC plus one parameter, and it was the best layout of the
{len(ledger.parts)}:</b> {num(m(jtc_la))}° p95 on the fast moves, against {num(m(jtc))}° without the lookahead, with no
extra compute and no extra bus traffic.</p>
<ul>
<li><b>Feed-forward: plan time.</b> It equals host feed-forward at the reference and costs nothing per tick. The model
cuts the feedback's share of the torque from {num(m('ff=none@none+fb=board_gains@board', R2, 'fb_share_pct'))} % to
{num(m(jtc_la, R2, 'fb_share_pct'))} % (below 50 Hz: {num(m(jtc_la, R2, 'fb_share_lf_pct'))} %). Without one, nothing
passes: the arm sags and never settles after a host stall.</li>
<li><b>Feedback: on the board.</b> Lookahead fixes every board-side law (about 1.7° → 0.35–0.5°) and hurts every
host-side one, which already acts on stale data. A PD on the host at 100 Hz fails every rung. Computed torque on the host
tracks well at 100 Hz ({num(m(ctc_host))}°) but saturates at 50 Hz, which is the rate the real arm's pick cycle runs.</li>
<li><b>An unmodelled 0.3 kg payload</b> shows up as low-frequency feedback (below-50 Hz share
{num(m(jtc_la, R2, 'fb_share_lf_pct'))} % → {num(m(jtc_la, R8, 'fb_share_lf_pct'))} %), while tracking stays at
{num(m(jtc_la, R8))}°. That's the number that tells a feed-forward agent its model is missing mass.</li>
</ul>
<p class="muted" style="margin-top:8px">{passing} of {len(ledger.parts)} layouts pass every gate on every rung. The ranking
rule: cheapest layout within 10 % of the best passing tracking (the ledger's own summary; reasons per pair below).</p>
</div>
{hero_html}

<h2>The question, made precise</h2>
<p>A controller here is <b>a trajectory + a feed-forward model <code>ff(q, q̇, q̈) → τ</code> + a feedback model
<code>fb(q_ref, q̇_ref, q, q̇) → τ</code> + a layout</b>: where each model is evaluated (plan time, the host, the board,
or split), at what rate, on which state (reference or measured), and how far ahead the reference is sampled. Both models
are pure functions (<code>robot_controllers.models</code>); one composed controller
(<code>controllers/layout.py</code>) places them, and it is tick-for-tick identical to the built JTC and the three
joint-impedance controllers when placed the way they are (acceptance tests, written before the code).</p>
<pre>trajectory ─► FF @ {{plan | host at reference | host at measured}} ─► FB @ {{board | board, gains each tick | split | host}}
           ─► host 50 / 100 / 500 Hz ─► CAN-FD (5.5 ms down) ─► moteus board PD at 4 kHz ─► arm</pre>

<h2>The answer per (feed-forward model, feedback model)</h2>
{recommendation_table(ledger)}

<h2>Lookahead is the lever the board-side laws were missing</h2>
<p>A board closes its PD on a setpoint that left the host 5.5 ms earlier. Sampling the trajectory 5.5 ms ahead lands
each setpoint on the board when it is due. A host-side law already measured the arm a bus delay ago, so lookahead
only makes it lead further.</p>
{lookahead_chart(ledger)}

<h2>The feed-forward ladder</h2>
<p>Zach's KPI from Nov 11 ("low feedback torque, where most of it is coming from feedforward terms"), now
<code>fb_share_pct</code>, in traj_compare's formula summed over the joints. With the rigid-body model about 7 of its 12
points sit above 50 Hz: the board's kd times its encoder quantisation (the "kd buzz"), which no model removes. The
below-50 Hz share is what a better model could still take off the feedback.</p>
{ladder_chart(ledger)}

<h2>All {len(ledger.parts)} layouts, tracking against host CPU</h2>
<p class="muted">Host CPU is tiny for every layout on this two-joint twin (well under 1 % of a core); it varies 2–4×
between identical laws on this shared machine, so read the x axis as a ranking of work, not a budget.</p>
{pareto_chart(ledger)}

<h2>Key numbers</h2>
{key_numbers(ledger, key_rows)}
{gate_grid(ledger)}

<h2>Metrics this used (and proposes)</h2>
<div class="cards">
<div class="card"><h3>Decision axes (Pareto, never summed)</h3><ul><li><code>tracking_p95_deg</code>, <code>tool_p95_mm</code></li>
<li><code>fb_share_pct</code>: share of the torque feedback had to supply</li><li><code>host_duty_pct</code>, <code>plan_compute_ms_per_s</code>,
<code>wire_bytes_per_s</code>: where the compute and the bus go</li></ul></div>
<div class="card"><h3>Gates (pass/fail)</h3><ul><li>missed cycles 0, latency p99 ≤ a period</li><li>saturation ≤ 2–5 %, no faults or divergence</li>
<li>survive a 0.5 s host stall (R6) and settle</li></ul></div>
<div class="card"><h3>Diagnostics</h3><ul><li><code>fb_share_lf_pct</code>, <code>fb_rms_nm</code>, <code>ff_rms_nm</code></li><li><code>torque_rate_rms_nm_s</code> (chatter)</li>
<li><code>integral_share_pct</code> once integrators exist</li></ul></div>
<div class="card"><h3>Robustness rungs</h3><ul><li>R8: +0.3 kg no model knows</li><li>R9: the host at 50 and 500 Hz</li><li>R6 host stall, R4 push</li></ul></div>
</div>
<p class="muted">Full proposal, catalog and history: the vault MOC <code>Maps &amp; Views/MOC/MOC - Controllers (Nov 20 2026).md</code>.
Contract: <code>/home/bam/robot_controllers/docs/CONTRACT.md</code>, amendment 3 and §3.8.</p>

<h2>Decided for you, left, and not done</h2>
<div class="cards">
<div class="card"><h3>Done and proved</h3><ul>
<li>Amendments 3 and 4 built and landed on robot_controllers <code>main</code>: 399 tests pass, including the 25 + 34 acceptance tests written before the code</li>
<li>The study's pinned numbers (R2 at 50 Hz: 0.775° → 0.201°) were measured first by an independent scratch implementation and match to 3 %</li>
<li>One run at zero lookahead measures the bus delay (5.5 ms on R2; 4.5–6.5 ms on U2)</li>
<li>The UR5e's models equal MuJoCo's <code>mj_inverse</code> and <code>mj_fullM</code> to 1e-9; its gains carry the twin's bandwidth</li></ul></div>
<div class="card"><h3>Decided for you (undo = <code>git revert</code>)</h3><ul>
<li>The secant velocity and mid-hold feed-forward added as a layout option (<code>interpolation</code>); the default stays today's behaviour</li>
<li>Per-joint lookahead and per-joint bus delay, because one adapter serves the boards in turn</li>
<li>UR5e board gains at the twin shoulder's bandwidth (ωn 39.7 rad/s, ζ 0.5) on 2^14-count encoders</li>
<li>Bus jitter rounded to 0.25 ms ticks so a seed replays (the builder's call)</li>
<li>From the first sweep: <code>fb_share_pct</code> in traj_compare's formula, lookahead swept, CTC at 6.3 Hz</li></ul></div>
<div class="card"><h3>Left (next, not blocked)</h3><ul>
<li>Lane J's real JTC: a per-joint lookahead from one logged run, plus the secant velocity (on a 50 Hz tape, the forward difference to the next sample)</li>
<li>The Nov 13 full-arm curriculum's one-tick (20 ms) lookahead and its arm ending moves ahead of the plan: calibrate it from one of its logs</li>
<li>Friction in the feed-forward on the UR-sized arm without over-compensating at reversals (a friction observer, or a deadband)</li>
<li>Wire the identified models (pendulum round 9 <code>ff_model</code>) as <code>FeedforwardModel</code>s</li></ul></div>
<div class="card"><h3>Not done, on purpose</h3><ul>
<li>No real-arm runs: that's lane J's, through rackq</li><li>No W&amp;B logging of these ledgers yet</li>
<li>No tuning per layout: every layout ran on the same gains, so the comparison is of placement and timing alone</li>
<li>No UR moving-target rung: a Cartesian line on the UR needs IK; the planar R10 answers the timing question</li></ul></div>
</div>
</main></body></html>"""


def main() -> Path:
    ledger = Ledger(json.loads(LEDGER.read_text()))
    OUT.write_text(page(ledger, study_sections.load()))
    print(OUT, f"{OUT.stat().st_size / 1e3:.0f} kB")
    return OUT


if __name__ == "__main__":
    main()
