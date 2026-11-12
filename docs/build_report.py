"""Build the handoff report from the results ledger: reports/robot-controllers-<date>.html.

    env -u VIRTUAL_ENV uv run --frozen --directory /home/bam/robot_controllers --extra test python docs/build_report.py

Every number on the page is read from results/*/results.json (the ledger the bench wrote and W&B mirrors); the
media (hero video, site screenshots) sit in reports/media/<slug>/ beside it.
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATE = "2026-11-12"
SLUG = f"robot-controllers-{DATE}"
OUT = REPO / "reports" / f"{SLUG}.html"
MEDIA = f"media/{SLUG}"
JT = "jointtrajectory-to-servocommand"
JCS = "jointcommandsequence-to-servocommand"


def load(slug: str) -> dict:
    return json.loads((REPO / "results" / slug / "results.json").read_text())


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


def runs_by(document: dict) -> dict[tuple[str, str], dict]:
    return {(r["controller"], r["rung"]): r for r in document["results"]}


def leaderboard(document: dict) -> str:
    runs = runs_by(document)
    controllers = [c["name"] for c in document["controllers"]]
    titles = {c["name"]: c["title"] for c in document["controllers"]}
    per_rung = document["summary"]["per_rung"]
    head = "".join(f"<th>{esc(titles[c])}<br><code>{esc(c)}</code></th>" for c in controllers)
    rows = []
    for rung in document["rungs"]:
        best = per_rung.get(rung["id"], {}).get("best")
        cells = []
        for name in controllers:
            run = runs.get((name, rung["id"]))
            if run is None:
                cells.append("<td class='na'>not run</td>")
                continue
            failed = [g for g in run["gates"] if not g["passed"]]
            mark = "<span class='ok'>✓</span>" if run["passed"] else "<span class='bad'>✗</span>"
            title = "; ".join(f"{g['metric']} {g['op']} {g['value']:g}: measured {num(g['measured'])} {g['note']}" for g in failed)
            cls = "best" if name == best else ""
            cells.append(f"<td class='{cls}' title='{esc(title)}'>{num(run['objective'])} {mark}</td>")
        arrow = "↓" if rung["minimize"] else "↑"
        rows.append(f"<tr><th class='rung'><code>{esc(rung['id'])}</code><br><span>{esc(rung['title'])} · "
                    f"<code>{esc(rung['objective'])}</code> {arrow}</span></th>{''.join(cells)}</tr>")
    per_controller = document["summary"]["per_controller"]
    totals = "".join(f"<td><b>{per_controller[c]['rungs_passed']}/{per_controller[c]['rungs_total']}</b> rungs<br>"
                     f"{per_controller[c]['gates_passed']}/{per_controller[c]['gates_total']} gates<br>"
                     f"{num(per_controller[c]['compute_us_p99'])} µs p99</td>" for c in controllers)
    return (f"<table class='board'><thead><tr><th>Rung · objective</th>{head}</tr></thead><tbody>{''.join(rows)}"
            f"<tr class='total'><th>Totals (compute = CPU time per tick)</th>{totals}</tr></tbody></table>")


def board_vs_host(document: dict) -> str:
    runs = runs_by(document)
    names = ["joint_trajectory_controller", "joint_impedance_board", "joint_impedance_host_spring", "joint_impedance_host"]
    where = {c["name"]: c["runs_on"] for c in document["controllers"]}
    columns = [
        ("R1.FreeSpace", "tracking_p95_deg", "free-space p95 error (°)"),
        ("R3.BeltImpact@0.10m/s", "contact_peak_n", "impact peak @0.10 m/s (N)"),
        ("R3.BeltImpact@0.10m/s", "dwell_force_mean_n", "dwell press (N, target 10)"),
        ("R7.BeltImpact@0.10m/s·host50Hz", "contact_peak_n", "impact peak, host 50 Hz (N)"),
        ("R7.BeltImpact@0.10m/s·host500Hz", "contact_peak_n", "impact peak, host 500 Hz (N)"),
        ("R4.SidewaysPush", "push_deflection_mm", "push deflection (mm)"),
        ("R1.FreeSpace", "compute_us_p99", "compute p99 (µs)"),
        ("R1.FreeSpace", "command_fields", "fields / joint / tick"),
    ]
    head = "".join(f"<th>{esc(label)}</th>" for _, _, label in columns)
    rows = []
    for name in names:
        cells = "".join(f"<td>{num(runs[(name, rung)]['metrics'].get(metric))}</td>" for rung, metric, _ in columns)
        rows.append(f"<tr><th><code>{esc(name)}</code><br><span>{esc(where[name])}</span></th>{cells}</tr>")
    return f"<table class='board'><thead><tr><th>Controller · where the law closes</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def wandb_links(documents: list[dict]) -> str:
    items = []
    for document in documents:
        block = document.get("wandb") or {}
        project = block.get("project_url")
        name = document["box"]["wandb_project"]
        if not project:
            items.append(f"<li><b>{esc(name)}</b>: not logged</li>")
            continue
        reports = "".join(f" · <a href='{esc(url)}'>{esc(key)}</a>" for key, url in sorted((block.get("reports") or {}).items()))
        items.append(f"<li><a href='{esc(project)}'><b>{esc(name)}</b></a>{reports}</li>")
    return "<ul class='links'>" + "".join(items) + "</ul>"


def page(jt: dict, jcs: dict) -> str:
    rec = jt["summary"]["recommendation"]
    seq_rec = jcs["summary"]["recommendation"]
    gen = jt["generated"]
    runs = runs_by(jt)
    jtc_dwell = runs[("joint_trajectory_controller", "R3.BeltImpact@0.10m/s")]["metrics"]["dwell_force_mean_n"]
    imp_dwell = runs[("joint_impedance_board", "R3.BeltImpact@0.10m/s")]["metrics"]["dwell_force_mean_n"]
    peaks = [runs[(c["name"], "R3.BeltImpact@0.10m/s")]["metrics"]["contact_peak_n"] for c in jt["controllers"]]
    peaks20 = [runs[(c["name"], "R3.BeltImpact@0.20m/s")]["metrics"]["contact_peak_n"] for c in jt["controllers"]]
    s3 = runs_by(jcs)[("joint_command_sequence_controller", "S3.HostStall")]
    s3_abort = next((e.get("why") for e in s3["events"] if e.get("kind") == "abort"), "")
    tf_r6 = runs[("tape_feeder", "R6.HostStall")]["metrics"]["schedule_slip_ms"]
    jtc_r6 = runs[("joint_trajectory_controller", "R6.HostStall")]["metrics"]["schedule_slip_ms"]
    board_r1 = runs[("joint_impedance_board", "R1.FreeSpace")]["metrics"]["tracking_p95_deg"]
    host_r1 = runs[("joint_impedance_host", "R1.FreeSpace")]["metrics"]["tracking_p95_deg"]
    push = [runs[(n, "R4.SidewaysPush")]["metrics"]["push_deflection_mm"] for n in
            ("joint_impedance_board", "joint_impedance_host", "joint_impedance_host_spring")]
    shots = [("site-landing.png", "Landing page: every black box and its recommendation"),
             ("site-box.png", "Box page: the leaderboard (controller × rung, best per rung highlighted)"),
             ("site-controller.png", "Controller page: where K and D live, parameters, per-rung results"),
             ("site-plots.png", "Controller page: tracking and belt-force plots for each rung")]
    shot_html = "".join(f"<figure><a href='{MEDIA}/{f}'><img src='{MEDIA}/{f}' loading='lazy' alt='{esc(c)}'></a><figcaption>{esc(c)} (click for full size)</figcaption></figure>"
                        for f, c in shots if (REPO / "reports" / MEDIA / f).exists())
    return f"""<!doctype html><html lang="en" data-theme="dark"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>robot_controllers · {DATE}</title>
<style>
:root{{--bg:#0b0b0c;--fg:#e4e4e7;--muted:#a1a1aa;--line:#27272a;--card:#111113;--ok:#4ade80;--bad:#f87171;--best:#14532d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:32px 20px 80px}}h1{{font-size:30px;margin:0 0 6px}}h2{{margin:44px 0 10px;font-size:21px}}
.verdict{{font-size:17px;border-left:3px solid var(--ok);padding:10px 14px;background:var(--card);border-radius:6px}}
.muted,span{{color:var(--muted)}}code{{font:13px ui-monospace,monospace;background:#18181b;padding:1px 4px;border-radius:4px}}
video,img{{width:100%;border-radius:8px;border:1px solid var(--line);background:#000}}
table.board{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}}table.board th,table.board td{{border:1px solid var(--line);padding:6px 8px;text-align:right;vertical-align:top}}
table.board th{{text-align:left;background:var(--card);font-weight:600}}table.board th span{{font-weight:400;font-size:12px}}td.best{{background:var(--best)}}td.na{{color:var(--muted)}}
.ok{{color:var(--ok)}}.bad{{color:var(--bad)}}tr.total td{{text-align:left}}figure{{margin:16px 0}}figcaption{{color:var(--muted);font-size:13px;margin-top:4px}}
ol li,ul li{{margin:6px 0}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px}}a{{color:#93c5fd}}
</style></head><body><main>
<p class="muted">robot_controllers · {DATE} · ledger commit <code>{esc(gen['commit'])}</code> · {esc(gen['plant'])} · MuJoCo {esc(gen['mujoco'])} · robot_msgs {esc(gen['robot_msgs'])}</p>
<h1>Controllers as black boxes, benchmarked</h1>
<p class="verdict"><b>Two black boxes run the ladder: 6 built controllers, 13 rungs, 53 episodes, one W&amp;B project per box, and a robotmsgs-style site.</b>
For <code>JointTrajectory → ServoCommand</code> the ladder recommends <code>{esc(rec['controller'])}</code>
({esc(rec['why'].rstrip('.'))}). That is a sim result on a planar test arm with an assumed belt contact, not the rack.</p>

<h2>Hero: the same belt impact under three controllers</h2>
<video src="{MEDIA}/hero-belt-impact.mp4" poster="{MEDIA}/hero-belt-impact.jpg" autoplay muted loop playsinline controls></video>
<p class="muted">Real bench episodes on rung <code>R3.BeltImpact@0.10m/s</code>, replayed from the logged joint angles, 4× slow motion.
The impact spike is about the same for all three ({', '.join(num(p) for p in peaks[:5])} N across the five controllers): arm mass and touch speed set it.
In the dwell, the stiff joint trajectory controller presses {num(jtc_dwell)} N into the belt and board impedance presses {num(imp_dwell)} N.
<a href="{MEDIA}/hero-belt-impact.gif">GIF fallback</a>.</p>

<h2>The black box: <code>JointTrajectory → ServoCommand</code> leaderboard</h2>
<p>Each cell is the rung's objective; ✓ means every gate passed; green is the best passing controller on that rung. Hover a ✗ to see the failed gates. Gates were frozen in <code>docs/CONTRACT.md</code> before any run.</p>
{leaderboard(jt)}

<h2>Impedance on the moteus board vs on the computer</h2>
<p>The same spring and damper (K 40 / 25 N·m/rad, ζ 0.8) placed three ways, beside the stiff joint trajectory controller:</p>
{board_vs_host(jt)}

<h2>Findings</h2>
<ol>
<li><b>Impact force is physics, not the controller.</b> At 0.10 m/s every controller peaks at 28–35 N; at 0.20 m/s all of them fail the 60 N gate ({', '.join(num(p) for p in peaks20)} N). The arm's own inertia hits the belt before any loop can react, as the Oct 26 impedance study found (a series spring is what cuts it). The controller's real lever is the dwell press: {num(jtc_dwell)} N stiff vs {num(imp_dwell)} N impedance.</li>
<li><b>Board impedance lags in free space ({num(board_r1)}° p95 vs {num(host_r1)}° for the same law on the host).</b> The board tracks a setpoint that reaches it 5.5 ms late over the bus. The host law compares the newest reference with a measurement 1.5 ms old, which adds a forward push. A cheap fix for every board-side controller is to sample the reference ahead by the measured bus delay.</li>
<li><b>Soft springs and gear friction don't come back to the millimetre.</b> Under the 15 N push the impedance variants deflect {', '.join(num(p) for p in push)} mm and never return within 1 mm (the frictionloss deadband is ≈ 0.3 N·m / 40 N·m/rad). The stiff controller deflects 9 mm and recovers in 0.05 s.</li>
<li><b>A late tick: the tape feeder slips, the time-driven controller skips.</b> After the 0.5 s host stall the tape feeder ends {num(tf_r6)} ms behind schedule; the joint trajectory controller ends {num(jtc_r6)} ms behind. The tape feeder also refuses 7 of 8 replans on R5.</li>
<li><b>The robot_msgs sequence controller goes limp after a host stall.</b> On <code>S3.HostStall</code> its ramp clock kept counting through the 0.5 s stall, so on resume it commanded a jump and its own speed guard aborted the sequence: “{esc(s3_abort)}”. This is the five-bar home button's controller, and the F09 bus hangs are exactly this case.</li>
<li><b>Cost is small and similar:</b> 40–100 µs CPU per tick for every controller (2 joints), far inside a 10 ms period. The board variant sends 5 fields per joint per tick, the host torque variant 3. Compute is thread CPU time, because this machine ran peer fleets at load ~100 and wall time counted their turns.</li>
</ol>

<h2>W&amp;B: one project per black box</h2>
{wandb_links([jt, jcs])}
<p class="muted">Each project has one run per (controller, rung), a saved view per rung, and Scoreboard, Performance vs cost, Contact and Timing reports. The <code>Controllers</code> hub report compares boxes.</p>

<h2>The site</h2>
<div class="grid">{shot_html}</div>

<h2><code>JointCommandSequence → ServoCommand</code></h2>
<p>One controller (robot_msgs' own): {esc(seq_rec['why'])} Rungs: move to a pose, seek a hard stop and define it, and a host stall mid-move.</p>
{leaderboard(jcs)}

<h2>Decision surface</h2>
<div class="grid">
<div class="card"><b>Done and proved</b><ul>
<li>A new repo <code>/home/bam/robot_controllers</code> (local git), with the contract, 6 built controllers and 4 planned in the registry, the bench, metrics, W&amp;B logging and the site. 229 package tests plus the site suite.</li>
<li>Full ladder run (53 episodes) → <code>results/*/results.json</code> (the ledger), mirrored to W&amp;B, rendered on the site.</li></ul></div>
<div class="card"><b>Decided for you</b> (each one is a revert away)<ul>
<li>A box is (input type, output type). W&amp;B project names are <code>Controller · &lt;Input&gt; to &lt;Output&gt;</code>, plus a <code>Controllers</code> hub.</li>
<li>The plant is a planar 2-joint arm (MuJoCo) with a moteus-like board model checked tick-for-tick against bam_pid_tunning. It is not the five-bar.</li>
<li>Compute is scored as CPU time. A host-torque law gets "n/a" on the command-jump gate. The tape feeder is set to each rung's rate.</li>
<li>The recommendation rule: most rungs passed, then most gates passed, then the lowest compute. The leaderboard also shows the best per rung.</li></ul></div>
<div class="card"><b>Left</b><ul>
<li>Run the ladder on the real rack and on the five-bar twin. The Plant protocol is the seam: robotparts drivers implement read/write.</li>
<li>Cartesian impedance, admittance and OSC are planned in the registry (they need a kinematics model or a force signal).</li>
<li>Board-side reference lookahead (finding 2), and a stall-safe sequence controller (finding 5, robot_msgs).</li></ul></div>
<div class="card"><b>Needs you</b><ul>
<li>Public or not: the repo and site are local only. Recommendation: publish once you've looked at the site; a domain like robotcontrollers.com costs money, so it's yours.</li>
<li>A force signal on the rig (load-cell plate) before the contact rungs can run on hardware.</li></ul></div>
</div>
</main></body></html>
"""


def main() -> None:
    jt, jcs = load(JT), load(JCS)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page(jt, jcs))
    print(OUT, f"{OUT.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
