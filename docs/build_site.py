"""Build the public landing page (GitHub Pages) from the ledgers: what robot_controllers answers, how it grades, what it found.

    env -u VIRTUAL_ENV uv run --frozen --directory /home/bam/robot_controllers --extra bench \
        python docs/build_site.py --out /path/to/site/checkout

Every number is read from `results/grade/results.json` (the grade, CONTRACT amendment 5) and `results/study/results.json`
(amendment 4); `recommend()` is called on the ledger for the example answers, so the page and the code can't disagree.
WHY a separate, site-only public repo: GitHub Pages needs a public repo on this account, and the code stays private
until Zach says otherwise. Nothing here may name a private path, a person or a secret: the page is public.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
GRADE = REPO / "results" / "grade" / "results.json"
STUDY = REPO / "results" / "study" / "results.json"
REFERENCES = REPO / "docs" / "references.json"
MEDIA = REPO / "reports" / "media" / "controller-layouts-2026-11-20"

ARM_TITLES = {
    "planar_arm": "Planar twin (2 joints)",
    "ur5e": "UR5e (6 joints)",
    "bam_fivebar": "BAM five-bar arm (6 joints)",
    "pendulum": "1-DOF pendulum",
}
CONFIG_TEXT = {
    "P1": ("none", "board PD", "calibrated"),
    "P2": ("the trajectory's torque column", "board PD", "none"),
    "P3": ("model, at plan time", "board PD", "datasheet"),
    "P4": ("model, at plan time", "board PD", "calibrated"),
    "P5": ("model + friction, at plan time", "board PD", "calibrated"),
    "P6": ("model, on the host at the measured state", "board PD", "calibrated"),
    "P7": ("model, at plan time", "computed-torque gains, re-sent each tick", "calibrated"),
    "P8": ("model, at plan time", "spring on the host, damper on the board", "none"),
    "P9": ("model, on the host at the measured state", "computed torque on the host", "none"),
    "P10": ("model, at plan time", "PD on the host", "none"),
}
OBJECTIVE_TEXT = {
    "accuracy": "accuracy (tool error)",
    "impact": "impact (peak torque into a stop)",
    "collision_detection": "collision detection (time to notice)",
    "cost": "cost (host CPU, bus bytes)",
}
UNITS = {"tool_p95_mm": "mm", "tracking_p95_deg": "°", "collision_peak_nm": "N·m", "collision_detect_ms": "ms",
         "host_duty_pct": "% of a core", "wire_bytes_per_s": "B/s"}


def esc(text: Any) -> str:
    return html.escape(str(text))


def num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    v = float(value)
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.{digits}g}"


def config_name(cid: str) -> str:
    base = cid.rstrip("s")
    ff, fb, la = CONFIG_TEXT[base]
    secant = cid.endswith("s")
    return f"{ff} · {fb} · lookahead {la}" + (" · secant velocity" if secant else "")


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def real_table(grade: Mapping[str, Any], real_rows: Sequence[Mapping[str, Any]]) -> str:
    """Each real run beside the twin's run of the config it ran, at the base conditions (sim rows of the same arm)."""
    base = {(r["arm"], r["config"]): r["metrics"] for r in grade["rows"]
            if r.get("source", "sim") == "sim" and r.get("axis") == "base"}
    body = []
    for r in real_rows:
        like = r.get("config_like")
        twin = base.get((r["arm"], like), {}) if like else {}
        m = r["metrics"]
        body.append(f"<tr><th>{esc(ARM_TITLES.get(r['arm'], r['arm']))}</th><td>{esc(r.get('ran', ''))}</td>"
                    f"<td>{num(m.get('tool_p95_mm'))} mm · {num(m.get('tracking_p95_deg'))}°</td>"
                    f"<td>{esc(like or '—')}: {num(twin.get('tool_p95_mm'))} mm · {num(twin.get('tracking_p95_deg'))}°</td></tr>")
    return ("<p>The first real run is in the ledger, scored by the same metrics as the twin. Every other number on this page is "
            "simulation; more real runs go through the rigs' owners and land here.</p><div class='scroll'><table class='grid'>"
            "<thead><tr><th>arm</th><th>what ran</th><th>real (tool p95 · joint p95)</th><th>its twin config at the base "
            "conditions</th></tr></thead><tbody>" + "".join(body) + "</tbody></table></div><p class='muted'>The twin's base "
            "has a model 10 % off and the datasheet bus; the real run had its own identified model and the rig's bus, so read "
            "this as the same order, not a match.</p>")


def facts(grade: Mapping[str, Any]) -> dict[str, str]:
    """The findings' numbers, read from the ledger so the page and the ledger can't disagree."""
    base: dict[tuple[str, str], Mapping[str, Any]] = {}
    for r in grade["rows"]:
        if r.get("source", "sim") == "sim" and r.get("axis") == "base":
            base[(r["arm"], r["config"])] = r["metrics"]
    tool = lambda arm, c: base.get((arm, c), {}).get("tool_p95_mm")  # noqa: E731
    col = {arm: grade["summary"][arm]["collision"]["configs"] for arm in grade["summary"]}
    det = lambda arm, c: col[arm].get(c, {}).get("collision_detect_ms")  # noqa: E731
    arms = [a for a in ARM_TITLES if a in grade["summary"]]
    board_best = {a: min(v for c, v in ((c, tool(a, c)) for c in ("P3s", "P4s", "P5s", "P7", "P7s")) if v is not None) for a in arms}
    p9 = {a: tool(a, "P9") for a in arms}
    p10_sat = {a: base.get((a, "P10"), {}).get("saturation_share") for a in arms}
    fast = [det(a, c) for a in arms for c in ("P4s", "P5s", "P7s") if det(a, c) is not None]
    nofeed = [det(a, "P1") for a in arms if det(a, "P1") is not None]
    host = [det(a, "P9") for a in arms if det(a, "P9") is not None]
    cal = grade["summary"].get("pendulum", {}).get("calibrations", {}).get("50Hz·bus=datasheet", {})
    return {
        "host_text": (
            " At the rig's 50 Hz a driver-closed config beat computed torque on the host on every arm (best "
            + ", ".join(f"{ARM_TITLES[a].split(' (')[0]} {num(board_best[a])} vs {num(p9[a])} mm" for a in arms)
            + "). PD on the host saturated the drivers (up to "
            + f"{num(max(v for v in p10_sat.values() if v is not None))} % of the time)."
        ),
        "secant_text": (
            " The plan is known ahead, so each setpoint is sent early by the time it takes to reach the driver, and the velocity"
            " sent is the straight line to the next sample (a moteus walks its setpoint at that velocity between commands)."
            f" On the UR5e twin that takes the trajectory-forwarding JTC from {num(tool('ur5e', 'P4'))} to"
            f" {num(tool('ur5e', 'P4s'))} mm at the tool; on the planar twin from {num(tool('planar_arm', 'P4'))} to"
            f" {num(tool('planar_arm', 'P4s'))} mm."
        ),
        "p7_text": (
            " Re-sending computed-torque gains (diag M(q)·ωn²) with every command, still closed by the driver, was the most"
            f" accurate on three of four arms: the five-bar goes from {num(tool('bam_fivebar', 'P4s'))} mm (fixed gains) to"
            f" {num(tool('bam_fivebar', 'P7'))} mm."
        ),
        "friction_text": (
            " The five-bar's drives are friction-bound: adding Coulomb friction to the plan-time model takes it from"
            f" {num(tool('bam_fivebar', 'P4s'))} to {num(tool('bam_fivebar', 'P5s'))} mm, where the secant velocity alone did"
            " nothing."
        ),
        "collision_text": (
            f" With a feed-forward and the feedback on the driver, a hard stop is noticed in {num(min(fast))}–{num(max(fast))} ms"
            f" on every arm; with no feed-forward it takes {num(min(nofeed))}–{num(max(nofeed))} ms, and with computed torque on"
            f" the host {num(min(host))}–{num(max(host))} ms. Less torque left for feedback in free motion makes a hit stand out."
        ),
        "cal_text": (
            " One run with no lookahead measures each joint's delay; the friction-aware fit is the one to look ahead by. On a"
            " single high-friction joint the joint's own lag swamps the bus (the pendulum reads"
            f" {', '.join(f'{v:g}' for v in cal.get('bus_delay_ms', []))} ms on a"
            f" {', '.join(f'{v:g}' for v in cal.get('datasheet_ms', []))} ms bus), so there the datasheet value, or a lookahead"
            " sweep on the real arm, is the one to trust."
        ),
    }


# ── sections ───────────────────────────────────────────────────────────────────────────────────────────────


def recommend_examples(grade_path: Path) -> list[tuple[str, str, Any]]:
    """(the question in words, the code, the Recommendation) for three setups, computed on the ledger."""
    from robot_controllers.calibration import datasheet_bus
    from robot_controllers.models import DriverCaps
    from robot_controllers.recommend import Setup, recommend

    cases = [
        ("BAM's five-bar on a moteus bus at 50 Hz, a rigid-body model that's 10 % off; best accuracy",
         "Setup(arm='bam_fivebar', host_rate_hz=50, bus=datasheet_bus(6), feedforward='rigid_body', model_error=0.1)",
         Setup(arm="bam_fivebar", host_rate_hz=50.0, bus=datasheet_bus(6), feedforward="rigid_body", model_error=0.1),
         "accuracy"),
        ("The same arm, but you care most about noticing a collision",
         "... objective='collision_detection'",
         Setup(arm="bam_fivebar", host_rate_hz=50.0, bus=datasheet_bus(6), feedforward="rigid_body", model_error=0.1),
         "collision_detection"),
        ("A UR5e-sized arm whose driver can't close a PD loop itself, 100 Hz host",
         "Setup(arm='ur5e', host_rate_hz=100, ..., driver_caps=DriverCaps(board_pd=False))",
         Setup(arm="ur5e", host_rate_hz=100.0, bus=datasheet_bus(6), feedforward="rigid_body", model_error=0.1,
               driver_caps=DriverCaps(board_pd=False)),
         "accuracy"),
    ]
    out = []
    for words, code, setup, objective in cases:
        try:
            answer = recommend(setup, objective=objective, ledger=str(grade_path))
        except TypeError:
            answer = recommend(setup, objective=objective)
        out.append((words, code + (f", objective='{objective}'" if objective != "accuracy" and "objective" not in code else ""),
                    answer, objective))
    return out


OBJECTIVE_METRICS = {"accuracy": ("tool_p95_mm", "tracking_p95_deg"), "impact": ("collision_peak_nm",),
                     "collision_detection": ("collision_detect_ms", "collision_peak_nm"),
                     "cost": ("host_duty_pct", "wire_bytes_per_s")}
METRIC_WORDS = {"tool_p95_mm": "tool error p95", "tracking_p95_deg": "joint error p95", "collision_peak_nm": "impact",
                "collision_detect_ms": "noticed in", "host_duty_pct": "host CPU", "wire_bytes_per_s": "bus"}


def answer_block(words: str, code: str, answer: Any, objective: str = "accuracy") -> str:
    expected = getattr(answer, "expected", {}) or {}
    shown = ", ".join(f"{METRIC_WORDS[k]} {num(expected[k])} {UNITS.get(k, '')}" for k in OBJECTIVE_METRICS[objective]
                      if expected.get(k) is not None)
    config = getattr(answer, "config", None)
    return (f"<div class='qa'><p class='q'>{esc(words)}</p><pre><code>recommend({esc(code)})</code></pre>"
            f"<p class='a'><b>{esc(config or 'no config fits')}</b>"
            f"{' — ' + esc(config_name(config)) if config else ''}</p>"
            f"<p class='muted'>{esc(short_why(getattr(answer, 'why', '')))}{(' · expected: ' + esc(shown)) if shown else ''}</p>"
            f"<details><summary class='muted'>the full reasoning</summary><p class='muted'>{esc(getattr(answer, 'why', ''))}</p>"
            f"</details></div>")


def short_why(why: str) -> str:
    """The first clause of recommend()'s reason (it goes on to list runners-up and real-safety notes)."""
    head = why.split(";")[0].split(" (the nearest")[0]
    return head if len(head) <= 260 else head[:257].rstrip() + "…"


def population_table() -> str:
    rows = "".join(
        f"<tr><th>{c}</th><td>{esc(ff)}</td><td>{esc(fb)}</td><td>{esc(la)}</td></tr>"
        for c, (ff, fb, la) in CONFIG_TEXT.items()
    )
    return ("<div class='scroll'><table><thead><tr><th>config</th><th>feed-forward</th><th>feedback</th>"
            "<th>lookahead</th></tr></thead><tbody>" + rows + "</tbody></table></div><p class='muted'>P2–P7 also run "
            "with the secant velocity (suffix <code>s</code>): 16 configs.</p>")


def headline_table(grade: Mapping[str, Any]) -> str:
    summary = grade["summary"]
    arms = [a for a in ARM_TITLES if a in summary]
    head = "".join(f"<th>{esc(ARM_TITLES[a])}</th>" for a in arms)
    body = []
    for objective, label in OBJECTIVE_TEXT.items():
        cells = []
        for arm in arms:
            entry = summary[arm]["headline"].get(objective)
            if not entry:
                cells.append("<td>—</td>")
                continue
            unit = UNITS.get(entry["metric"], "")
            cells.append(f"<td><b>{esc(entry['config'])}</b><br><span class='muted'>{num(entry['value'])} {esc(unit)}"
                         f" · {entry['passing']}/{entry['of']} pass</span></td>")
        body.append(f"<tr><th>{esc(label)}</th>{''.join(cells)}</tr>")
    return (f"<div class='scroll'><table class='grid'><thead><tr><th>best at the rig's conditions</th>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def cards_table(grade: Mapping[str, Any]) -> str:
    summary = grade["summary"]
    arms = [a for a in ARM_TITLES if a in summary]
    configs = list(next(iter(summary.values()))["cards"])
    head = "".join(f"<th>{esc(ARM_TITLES[a].split(' (')[0])}</th>" for a in arms)
    body = []
    for cid in configs:
        cells = []
        for arm in arms:
            card = summary[arm]["cards"].get(cid)
            if not card:
                cells.append("<td>—</td>")
                continue
            share = card["within_10pct"] / max(1, card["graded"])
            passed = card["passed"] / max(1, card["rows"])
            cells.append(f"<td style='--w:{100 * share:.0f}%'><span class='bar'></span>{card['best']} best · "
                         f"{card['within_10pct']} within 10 %<br><span class='muted'>passes {card['passed']}/{card['rows']}"
                         f"</span></td>" if passed >= 0 else "")
        body.append(f"<tr><th>{cid}</th>{''.join(cells)}</tr>")
    return (f"<div class='scroll'><table class='cards'><thead><tr><th>config</th>{head}</tr></thead><tbody>"
            f"{''.join(body)}</tbody></table></div><p class='muted'>Across every scenario and objective an arm was graded "
            f"on: how often each config was the best, or within 10 % of it, and how many scenarios it passed every gate in."
            f"</p>")


def calibration_table(grade: Mapping[str, Any]) -> str:
    body = []
    for arm, title in ARM_TITLES.items():
        cals = grade["summary"].get(arm, {}).get("calibrations", {})
        cal = cals.get("50Hz·bus=datasheet") or next(iter(cals.values()), None)
        if not cal:
            continue
        fmt = lambda v: "/".join(f"{x:g}" for x in v) if isinstance(v, list) else f"{v:g}"  # noqa: E731
        body.append(f"<tr><th>{esc(title)}</th><td>{esc(fmt(cal['datasheet_ms']))}</td>"
                    f"<td>{esc(fmt(cal['bus_delay_ms']))}</td><td>{esc(fmt(cal['best_lookahead_ms']))}</td></tr>")
    return ("<div class='scroll'><table class='grid'><thead><tr><th>arm (50 Hz)</th><th>datasheet delay (ms, per joint)"
            "</th><th>calibrated bus delay</th><th>best lookahead</th></tr></thead><tbody>" + "".join(body)
            + "</tbody></table></div>")


def collision_table(grade: Mapping[str, Any]) -> str:
    body = []
    for arm, title in ARM_TITLES.items():
        col = grade["summary"].get(arm, {}).get("collision") or {}
        configs = col.get("configs", {})
        if not configs:
            continue
        pick = [c for c in ("P1", "P2", "P2s", "P4", "P4s", "P9") if c in configs]
        cells = "".join(f"<td>{num(configs[c]['collision_detect_ms'])} ms<br><span class='muted'>margin "
                        f"{num(configs[c]['collision_margin'])}</span></td>" for c in pick)
        body.append(f"<tr><th>{esc(title)}</th>{cells}</tr>")
    head = "".join(f"<th>{c}</th>" for c in ("P1 no FF", "P2", "P2s", "P4", "P4s", "P9 host CTC"))
    return (f"<div class='scroll'><table class='grid'><thead><tr><th>time to notice a hard stop</th>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def references_html(refs: Mapping[str, Any]) -> str:
    out = []
    for group in refs["groups"]:
        items = []
        for item in group["items"]:
            extra = "".join(f" · <a href='{esc(u)}'>{esc(n)}</a>" for n, u in item.get("extra", []))
            items.append(f"<li><a href='{esc(item['url'])}'><b>{esc(item['name'])}</b></a>{extra}<br>"
                         f"<span class='muted'>{esc(item['what'])}</span></li>")
        out.append(f"<div class='refgroup'><h3>{esc(group['name'])}</h3><ul>{''.join(items)}</ul></div>")
    return f"<div class='refs'>{''.join(out)}</div><p class='muted'>Links checked on {esc(refs['verified'])}.</p>"


DIAGRAM = """<svg viewBox="0 0 860 210" role="img" aria-label="Where each part of a controller can run">
<style>.b{fill:var(--card);stroke:var(--line);stroke-width:1.5}.t{fill:var(--fg);font:600 14px system-ui}.s{fill:var(--muted);font:12px system-ui}.a{stroke:var(--muted);stroke-width:1.5;marker-end:url(#h)}.k{fill:var(--accent)}</style>
<defs><marker id="h" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0L10,5L0,10z" fill="var(--muted)"/></marker></defs>
<rect class="b" x="10" y="30" width="150" height="110" rx="10"/><text class="t" x="24" y="56">Plan time</text><text class="s" x="24" y="78">trajectory</text><text class="s" x="24" y="96">feed-forward model</text><text class="s" x="24" y="114">(once per sample)</text>
<rect class="b" x="215" y="30" width="170" height="110" rx="10"/><text class="t" x="229" y="56">Host loop</text><text class="s" x="229" y="78">25–200 Hz</text><text class="s" x="229" y="96">sample ahead (lookahead)</text><text class="s" x="229" y="114">FF or FB live, if placed here</text>
<rect class="b" x="440" y="30" width="150" height="110" rx="10"/><text class="t" x="454" y="56">Bus</text><text class="s" x="454" y="78">CAN-FD, boards in turn</text><text class="s" x="454" y="96">datasheet vs measured</text><text class="s" x="454" y="114">delay per joint</text>
<rect class="b" x="645" y="30" width="200" height="110" rx="10"/><text class="t" x="659" y="56">Driver (board)</text><text class="s" x="659" y="78">PD at kHz, 0.5 ms delay</text><text class="s" x="659" y="96">walks the setpoint at the</text><text class="s" x="659" y="114">sent velocity between commands</text>
<line class="a" x1="160" y1="85" x2="213" y2="85"/><line class="a" x1="385" y1="85" x2="438" y2="85"/><line class="a" x1="590" y1="85" x2="643" y2="85"/>
<text class="s" x="10" y="180">A controller = a trajectory + a feed-forward model + a feedback model + a layout: where each runs, how often, and how far ahead.</text>
<text class="s" x="10" y="200">The grade varies the layout (the population) against the conditions it will meet (the scenarios).</text></svg>"""


CSS = """
:root{--bg:#fbfbfa;--fg:#17181a;--muted:#5d6168;--card:#fff;--line:#e3e4e6;--accent:#1d6fe0;--ok:#167a3e;--bad:#b42318;--bar:rgba(29,111,224,.14)}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0e0f11;--fg:#e8e9eb;--muted:#9da3ab;--card:#16181b;--line:#2a2d32;--accent:#6aa6ff;--ok:#5bd08a;--bad:#ff7a6b;--bar:rgba(106,166,255,.18)}}
:root[data-theme=dark]{--bg:#0e0f11;--fg:#e8e9eb;--muted:#9da3ab;--card:#16181b;--line:#2a2d32;--accent:#6aa6ff;--ok:#5bd08a;--bad:#ff7a6b;--bar:rgba(106,166,255,.18)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:40px 16px 96px}h1{font-size:34px;line-height:1.15;margin:0 0 10px;letter-spacing:-.01em}
h2{font-size:23px;margin:56px 0 12px}h3{font-size:16px;margin:18px 0 6px}p{margin:8px 0}.lede{font-size:19px;color:var(--muted);max-width:780px}
.muted{color:var(--muted);font-size:14px}a{color:var(--accent)}code,pre{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;overflow-x:auto;margin:8px 0}
details summary{cursor:pointer}.qa{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:12px 0}.qa .q{font-weight:600;margin:0}.qa .a{margin:6px 0 2px}
table{border-collapse:collapse;width:100%;background:var(--card);font-size:14px}th,td{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}
thead th{background:var(--bg)}.scroll{overflow-x:auto;margin:10px 0}table.grid td{font-variant-numeric:tabular-nums}
table.cards td{position:relative;font-variant-numeric:tabular-nums}table.cards td .bar{position:absolute;left:0;top:0;bottom:0;width:var(--w);background:var(--bar);z-index:0}
.finding{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:14px 0}.finding div{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.finding b{display:block;margin-bottom:4px}svg{width:100%;height:auto;margin:10px 0}video{width:100%;border-radius:12px;border:1px solid var(--line)}
.refs{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:8px 24px}.refgroup ul{padding-left:18px;margin:4px 0}.refgroup li{margin:6px 0}
.badge{display:inline-block;font-size:12px;border:1px solid var(--line);border-radius:999px;padding:1px 9px;margin-right:6px;color:var(--muted)}
footer{margin-top:64px;color:var(--muted);font-size:13px}
"""


def page(grade: Mapping[str, Any], study: Mapping[str, Any] | None, refs: Mapping[str, Any], examples, media: bool,
         internal: str = "") -> str:
    f = facts(grade)
    summary = grade["summary"]
    arms = [a for a in ARM_TITLES if a in summary]
    real_rows = [r for r in grade["rows"] if r.get("source") == "real"]
    sim_rows = [r for r in grade["rows"] if r.get("source", "sim") == "sim"]
    commit = esc(grade.get("generated", {}).get("commit", ""))
    video = ("<video src='media/hero-ur5e.mp4' poster='media/hero-ur5e.jpg' autoplay muted loop playsinline controls></video>"
             "<p class='muted'>A UR5e twin on fast moves at 50 Hz, the same law three ways: as built, + lookahead, + secant "
             "velocity. The trace is the tool's error.</p>") if media else ""
    real = (real_table(grade, real_rows) if real_rows else "<p><b>Every number on this page is simulation so far.</b> Real runs on the five-bar and the pendulum "
            "go through the rigs' owners and land in the same ledger; the importer for the real arm's logs already scores "
            "them like twin episodes.</p>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>robot_controllers</title><meta name="description" content="Tell it your driver, your models and your rates; it tells you the controller.">
<style>{CSS}</style></head><body><main>
<p><span class="badge">simulation, first real run in</span><span class="badge">{len(sim_rows)} graded episodes</span><span class="badge">{len(arms)} arms</span></p>
<h1>Tell it your driver, your models and your rates. It tells you the controller.</h1>
<p class="lede">robot_controllers treats a robot controller as a trajectory plus a feed-forward model plus a feedback model, placed by a
layout: where each runs (plan time, the host, or the servo driver), how often, and how far ahead it samples. It grades a population
of layouts against the conditions a real arm meets, and answers the question with a lookup.</p>
{internal}
{''.join(answer_block(*e) for e in examples)}

<h2>The idea</h2>
{DIAGRAM}

<h2>What it found</h2>
<div class="finding">
<div><b>Feedback stays on the driver.</b>{f['host_text']}</div>
<div><b>Look ahead by the bus delay, and send the secant velocity.</b>{f['secant_text']}</div>
<div><b>Re-send inertia-scaled gains each tick.</b>{f['p7_text']}</div>
<div><b>Model the friction where it dominates.</b>{f['friction_text']}</div>
<div><b>A good model notices collisions sooner.</b>{f['collision_text']}</div>
<div><b>Calibrate the delay, but know its limit.</b>{f['cal_text']}</div>
</div>
{video}

<h2>The answer per arm</h2>
<p>At the rig's conditions: 50 Hz host, the datasheet bus, a model 10 % off. One row per objective.</p>
{headline_table(grade)}

<h2>The population</h2>
{population_table()}

<h2>The scenarios</h2>
<p>Around that base, one axis at a time, on fixed target trajectories per arm: trajectory speed (slow, nominal, fast); host rate (25,
50, 100, 200 Hz); how wrong the feed-forward model is (0, 10, 30 %); how soft or stiff the driver's gains are (×0.5, ×1, ×2); the bus
(ideal, datasheet, as measured on the rig, with jitter); a collision with a hard stop; and the host rate × model error grid the lookup
interpolates over. Metrics: tracking and tool error, saturation, the share of torque feedback supplies, host CPU, bus bytes, impact
torque, and time to detect a collision.</p>

<h2>How each config did</h2>
{cards_table(grade)}

<h2>Bus timing: the datasheet, a calibration, and a drift alarm</h2>
<p>The nominal delay comes from the adapter's and driver's datasheets. One run with no lookahead measures each joint's real delay;
a live monitor then watches it and raises an alert when a joint drifts by more than 2 ms. On a slow joint the motor's own friction
lag reads as delay, so a calibration reports two numbers: the bus delay (what's compared with the datasheet) and the best
lookahead (what the controller uses).</p>
{calibration_table(grade)}

<h2>Collision detection</h2>
{collision_table(grade)}
<p class="muted">The detector is the feedback torque over the driver's cap, low-passed, against 1.25× its own peak in the same run
without the stop. Margin = how far over that threshold the hit goes.</p>

<h2>Real validation</h2>
{real}

<h2>Related work and references</h2>
{references_html(refs)}

<footer>Built from the grade ledger at <code>{commit}</code>. Arms: {esc(', '.join(ARM_TITLES[a] for a in arms))}.</footer>
</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=GRADE, help="a grade ledger (default: the committed one)")
    parser.add_argument("--internal", type=Path, default=None,
                        help="an HTML fragment (the decision record) placed under the lede: the internal report, never the site")
    args = parser.parse_args()
    grade, study, refs = load(args.ledger), load(STUDY), load(REFERENCES)
    if grade is None or refs is None:
        raise SystemExit(f"need {args.ledger} and {REFERENCES}")
    examples = recommend_examples(args.ledger)
    args.out.mkdir(parents=True, exist_ok=True)
    media = (MEDIA / "hero-ur5e.mp4").exists()
    if media:
        (args.out / "media").mkdir(exist_ok=True)
        for name in ("hero-ur5e.mp4", "hero-ur5e.jpg"):
            shutil.copy2(MEDIA / name, args.out / "media" / name)
    internal = args.internal.read_text() if args.internal else ""
    (args.out / "index.html").write_text(page(grade, study, refs, examples, media, internal))
    (args.out / ".nojekyll").write_text("")
    print(args.out / "index.html", f"{(args.out / 'index.html').stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
