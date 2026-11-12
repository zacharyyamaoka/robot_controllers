"""HTML for every page of the robot_controllers site. Pure functions from model data to strings.

The chrome (shell, sidebar, code blocks, headings) is robotmsgs.com's, with the sidebar taking its groups as data:
[(label, [(title, href, key, badge)])].
"""

from __future__ import annotations

import html
import json
import re

from highlight import highlight
from model import (
    BOARD_RATE_HZ,
    OP_TEXT,
    BoxInfo,
    ControllerInfo,
    RungInfo,
    fmt,
    intro,
    metric,
    metrics_by_family,
    unit_text,
    value_text,
    wandb_project_url,
    with_unit,
)
from robot_controllers.bench.types import HostTiming

SidebarItem = tuple[str, str, str, "str | None"]  # (title, href, key, badge)
SidebarGroup = tuple[str, list[SidebarItem]]

RUNS_ON_TEXT = {"board": "on the board", "host": "on the host", "split": "split host / board"}


def esc(text: object) -> str:
    return html.escape(str(text))


def inline(text: str) -> str:
    """Escape text and turn its ``code`` / `code` spans (reST or Markdown) into <code>."""
    escaped = esc(text)
    escaped = re.sub(r"``([^`]+)``", r"<code>\1</code>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)


# --------------------------------------------------------------------------------- components


def code_block(code: str, language: str = "python", *, wrap: bool = False, label: str = "") -> str:
    cls = "code wrap" if wrap else "code"
    return (
        f'<div class="{cls}">'
        f'<button class="copy" type="button" aria-label="Copy {esc(label or "code")}">Copy</button>'
        f'<pre><code>{highlight(code, language)}</code></pre></div>'
    )


def heading(level: int, text: str, anchor: str) -> str:
    return (
        f'<h{level} id="{anchor}"><a class="anchor" href="#{anchor}" '
        f'aria-label="Link to {esc(text)}">{esc(text)}</a></h{level}>'
    )


def badge(text: str, kind: str = "") -> str:
    return f'<span class="badge{" " + kind if kind else ""}">{esc(text)}</span>'


def metric_link(name: str) -> str:
    return f'<a class="mlink" href="/metrics/#m-{esc(name)}"><code>{esc(name)}</code></a>'


def io_chips(box: BoxInfo) -> str:
    (input_name, input_url), (output_name, output_url) = box.input_link, box.output_link
    return (
        f'<span class="io"><a class="chip" href="{input_url}" rel="noopener" title="robot_msgs type page">{esc(input_name)}</a>'
        f'<span class="io-arrow" aria-label="to">→</span>'
        f'<a class="chip" href="{output_url}" rel="noopener" title="robot_msgs type page">{esc(output_name)}</a></span>'
    )


def gate_text(gate: dict, *, measured: bool) -> str:
    op = OP_TEXT.get(str(gate.get("op")), str(gate.get("op")))
    limit = fmt(gate.get("value"))
    if not measured:
        return f"{gate.get('metric')} {op} {limit}"
    if gate.get("measured") is None:
        return f"{gate.get('metric')} not measured ({op} {limit})"
    return f"{gate.get('metric')} {fmt(gate.get('measured'))} {op} {limit}"


def gate_chip(gate: dict, *, measured: bool) -> str:
    """One gate as a chip: the rule, or the rule with its measured value and verdict (✓ / ✗, never colour alone)."""
    why = str(gate.get("why") or "")
    note = str(gate.get("note") or "")
    title = " · ".join(p for p in (why, note) if p)
    if not measured:
        return f'<span class="gate" title="{esc(title)}">{esc(gate_text(gate, measured=False))}</span>'
    passed = bool(gate.get("passed"))
    mark = "✓" if passed else "✗"
    verdict = "pass" if passed else "fail"
    return (f'<span class="gate {verdict}" title="{esc(title or verdict)}" data-metric="{esc(gate.get("metric"))}">'
            f'<span class="mark" aria-label="{verdict}">{mark}</span> {esc(gate_text(gate, measured=True))}'
            f'{" · " + esc(note) if note and note != "not measured" else ""}</span>')


def fixture_banner(boxes: list[BoxInfo]) -> str:
    if not any(b.fixture for b in boxes):
        return ""
    return ('<div class="banner" role="note"><strong>Synthetic numbers.</strong> These results are the site\'s test '
            'fixture, not a bench run: the layout is real, the values are invented.</div>')


def sidebar_groups(boxes: list[BoxInfo]) -> list[SidebarGroup]:
    groups: list[SidebarGroup] = [("Overview", [("Home", "/", "home", None), ("Rungs", "/rungs/", "rungs", None),
                                                ("Metrics", "/metrics/", "metrics", None)])]
    for box in boxes:
        items: list[SidebarItem] = [("Leaderboard", box.path, box.slug, None)]
        items += [(c.title, c.path, c.name, None if c.built else "planned") for c in box.controllers]
        groups.append((box.name, items))
    return groups


def shell(
    *,
    title: str,
    description: str,
    path: str,
    body: str,
    asset_version: str,
    groups: list[SidebarGroup],
    section: str = "",
    current: str = "",
    sidebar: bool = True,
    toc: list[tuple[str, str, int]] | None = None,
    stamp: str = "",
    robots: str = "",
) -> str:
    nav_items = [("Home", "/", "home"), ("Rungs", "/rungs/", "rungs"), ("Metrics", "/metrics/", "metrics")]
    nav_links = "".join(
        f'<a href="{href}"{" aria-current=\"page\"" if section == key else ""}>{label}</a>'
        for label, href, key in nav_items
    )

    side = sidebar_html(groups, current, full=sidebar, nav_links=nav_links)
    toc_html = ""
    if toc:
        items = "".join(
            f'<li class="l{level}"><a href="#{anchor}">{esc(label)}</a></li>'
            for anchor, label, level in toc
        )
        toc_html = (
            f'<aside class="toc" aria-label="On this page"><div class="toc-in">'
            f'<h2 class="toc-h">On this page</h2><ul>{items}</ul></div></aside>'
        )
    layout = "docs" if sidebar else "wide"
    return f"""<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<meta name="color-scheme" content="dark light">{f'<meta name="robots" content="{robots}">' if robots else ""}
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}">
<link rel="icon" href="/assets/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap">
<script>try{{var t=localStorage.getItem("theme");if(t==="light"||t==="dark")document.documentElement.dataset.theme=t}}catch(e){{}}</script>
<link rel="stylesheet" href="/assets/site.css?v={asset_version}">
<link rel="stylesheet" href="/assets/rc.css?v={asset_version}">
</head>
<body class="{layout}">
<a class="skip" href="#main">Skip to content</a>
<header class="topbar">
  <div class="topbar-in">
    <button class="icon-btn menu-btn" type="button" aria-label="Open menu" aria-expanded="false" aria-controls="sidebar">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
    </button>
    <a class="brand" href="/"><svg width="20" height="20" viewBox="0 0 32 32" aria-hidden="true"><rect x="3" y="3" width="26" height="26" rx="6" fill="none" stroke="currentColor" stroke-width="2.5"/><path d="M9 16h11m-4-5 5 5-5 5" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg><span>robot_controllers</span></a>
    <nav class="topnav" aria-label="Main">{nav_links}</nav>
    <div class="spacer"></div>
    <button class="icon-btn theme-btn" type="button" aria-label="Toggle light and dark theme">
      <svg class="i-sun" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
      <svg class="i-moon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z"/></svg>
    </button>
  </div>
</header>
<div class="shell">
{side}
<main id="main" class="main" tabindex="-1">
{body}
<footer class="foot"><span>{esc(stamp)}</span><span><a href="https://robotmsgs.com/" rel="noopener">robot_msgs</a> · MIT license · generated from the registry and the results ledger, not written per controller</span></footer>
</main>
{toc_html}
</div>
<div class="scrim" hidden></div>
<script src="/assets/site.js?v={asset_version}" defer></script>
</body>
</html>
"""


def sidebar_html(groups: list[SidebarGroup], current: str, *, full: bool, nav_links: str) -> str:
    """The first group (Overview) always shows; the rest are filtered by the search box."""
    drawer = f'<nav class="drawer-nav" aria-label="Site">{nav_links}</nav>'
    if not full:
        return f'<aside id="sidebar" class="sidebar lite">{drawer}</aside>'
    parts = []
    for index, (label, items) in enumerate(groups):
        rows = "".join(
            f'<li data-name="{esc((title + " " + key).lower())}"><a href="{href}"'
            f'{" aria-current=\"page\"" if current == key else ""}>{esc(title)}'
            f'{(" " + badge(item_badge, "plan")) if item_badge else ""}</a></li>'
            for title, href, key, item_badge in items
        )
        filterable = " data-group" if index > 0 else ""
        parts.append(f'<div class="sb-group"{filterable}><h2 class="sb-h">{esc(label)}</h2><ul>{rows}</ul></div>')
    return (
        f'<aside id="sidebar" class="sidebar">{drawer}'
        '<div class="search"><input id="search" type="search" placeholder="Search controllers…" '
        'aria-label="Filter controllers" autocomplete="off" spellcheck="false"><kbd>/</kbd></div>'
        '<nav class="sb-nav" aria-label="Controllers">'
        + "".join(parts)
        + '<p class="sb-empty" hidden>No controller matches.</p></nav></aside>'
    )


# ------------------------------------------------------------------------------------- home


def recommendation_html(box: BoxInfo, *, compact: bool = False) -> str:
    if not box.built:
        return '<p class="muted">No built controller yet.</p>'
    if not box.has_results:
        return '<p class="muted rec-none">Not run yet: no results for this box, so no recommendation.</p>'
    rec = box.recommendation
    if not rec:
        return '<p class="muted rec-none">No recommendation: the results have no scored runs.</p>'
    info = box.controller(rec["controller"])
    name = (f'<a href="{info.path}">{esc(info.title)}</a> <code>{esc(info.name)}</code>' if info
            else f'<code>{esc(rec["controller"])}</code>')
    why = f" — {inline(rec['why'])}" if rec.get("why") else ""
    if compact:
        return f'<p class="rec-line"><span class="rec-k">Recommended</span> {name}{why}</p>'
    return (f'<div class="callout rec" data-recommendation="{esc(rec["controller"])}"><p class="callout-k">Recommended</p>'
            f'<p>{name}{why}</p></div>')


def home(boxes: list[BoxInfo], **ctx) -> str:
    words = intro()
    pillars = "".join(
        f'<div class="pillar"><h2>{inline(title)}</h2><p>{inline(body)}</p></div>' for title, body in words.pillars
    )
    cards = []
    for box in boxes:
        rows = "".join(
            f'<li><a href="{c.path}">{esc(c.title)}</a> {badge(RUNS_ON_TEXT[c.spec.runs_on])}'
            f'{"" if c.built else " " + badge("planned", "plan")}</li>'
            for c in box.controllers
        )
        built = len(box.built)
        cards.append(
            f'<section class="boxcard" aria-labelledby="bx-{box.slug}">'
            f'<h3 id="bx-{box.slug}"><a href="{box.path}">{esc(box.name)}</a></h3>'
            f'<p class="boxcard-io">{io_chips(box)}</p>'
            f'<p class="muted small">{built} built · {len(box.controllers) - built} planned'
            f'{f" · {len(box.rungs)} rungs" if box.rungs else ""}</p>'
            f'<ul class="boxcard-list">{rows}</ul>{recommendation_html(box, compact=True)}</section>'
        )
    featured = next((c for b in boxes if b.recommendation for c in b.built if c.name == b.recommendation["controller"]
                     and c.example and c.example.ok), None)
    featured = featured or next((c for b in boxes for c in b.built if c.example and c.example.ok), None)
    featured = featured or next((c for b in boxes for c in b.built if c.example), None)
    example = ""
    if featured and featured.example:
        output = (f'<p class="out-label">Output</p>{code_block(featured.example.output, "text", wrap=True, label="example output")}'
                  if featured.example.ok else
                  f'<p class="muted small">Not run in this build: {esc(featured.example.error or "the class is not importable yet")}.</p>')
        example = (
            f'<section class="tour" aria-labelledby="ex-h"><h2 id="ex-h">Same thread, a dozen lines</h2>'
            f'<p class="muted">Build a controller from its input message, then tick it from your own loop: read the drive, '
            f'tick, send. No thread, no clock, no hardware inside. This is <a href="{featured.path}">{esc(featured.title)}</a>, '
            f'run when this page was built.</p>{code_block(featured.example.code, label="example")}{output}</section>'
        )
    install = code_block("pip install -e ./robot_controllers", "shell", label="install command")
    deps = ", ".join(f"<code>{esc(d)}</code>" for d in words.dependencies) or "numpy and robot_msgs"
    first = next((b for b in boxes if b.has_results), next((b for b in boxes if b.built), boxes[0]))
    body = f"""<div class="home">
<section class="hero" aria-labelledby="hero-h">
  <p class="eyebrow">{esc(words.eyebrow)}</p>
  <h1 id="hero-h">robot_controllers <span class="dash">—</span> {esc(words.tagline.rstrip("."))}</h1>
  <p class="lead">{inline(words.lead)}</p>
  <div class="cta">
    <a class="btn primary" href="{first.path}">{esc(first.name)} leaderboard</a>
    <a class="btn" href="/rungs/">The rung ladder</a>
  </div>
  <div class="install">
    {install}
    <p class="note">Not on PyPI yet: install from a checkout. Runtime dependencies: {deps}.</p>
  </div>
</section>
{fixture_banner(boxes)}
<section class="pillars rc-pillars" aria-label="Principles">
  {pillars}
</section>

<section class="boxes" aria-labelledby="bx-h">
  <h2 id="bx-h">Black boxes</h2>
  <p class="muted">A controller here is anything that reads joint state and emits a <code>ServoCommand</code> every tick. Controllers that take the same input message and emit the same output are one black box: they swap for each other, so they run the same rungs and land on one leaderboard.</p>
  <div class="boxgrid">{"".join(cards)}</div>
</section>
{example}
</div>"""
    return shell(
        title=f"robot_controllers — {words.tagline.rstrip('.')}", description=words.tagline,
        path="/", body=body, section="home", sidebar=False, groups=sidebar_groups(boxes), **ctx,
    )  # fmt: skip


# ------------------------------------------------------------------------------------- box page


def objective_label(rung: RungInfo) -> str:
    info = metric(rung.objective)
    arrow = "↓" if rung.minimize else "↑"
    unit = f" ({unit_text(info.unit)})" if info and unit_text(info.unit) else ""
    return f'{metric_link(rung.objective)}<span class="unit">{esc(unit)} {arrow}</span>'


def leaderboard(box: BoxInfo) -> str:
    built = box.built
    best_tag = '<span class="best-tag">best</span>'
    head = "".join(f'<th scope="col"><a href="{c.path}">{esc(c.title)}</a></th>' for c in built)
    rows = []
    for rung in box.rungs:
        cells = []
        best = box.best.get(rung.id)
        for info in built:
            run = info.runs.get(rung.id)
            if run is None:
                cells.append(f'<td class="lb-cell none" data-controller="{esc(info.name)}" data-rung="{esc(rung.id)}">not run</td>')
                continue
            passed = bool(run.get("passed"))
            gates = run.get("gates") or []
            gates_passed = sum(bool(g.get("passed")) for g in gates)
            classes = ["lb-cell", "pass" if passed else "fail"] + (["best"] if best == info.name else [])
            failed = [g.get("metric") for g in gates if not g.get("passed")]
            title = f"{gates_passed}/{len(gates)} gates" + (f"; failed: {', '.join(map(str, failed))}" if failed else "")
            title += "; best on this rung" if best == info.name else ""
            mark = "✓" if passed else "✗"
            cells.append(
                f'<td class="{" ".join(classes)}" data-controller="{esc(info.name)}" data-rung="{esc(rung.id)}" '
                f'data-value="{esc(json.dumps(run.get("objective")))}" data-passed="{str(passed).lower()}" title="{esc(title)}">'
                f'<span class="num">{esc(fmt(run.get("objective")))}</span> <span class="mark" aria-label="{"pass" if passed else "fail"}">{mark}</span>'
                f'{best_tag if best == info.name else ""}</td>'
            )
        rows.append(
            f'<tr><th scope="row"><a href="/rungs/#{rung.anchor}"><code>{esc(rung.id)}</code></a>'
            f'<span class="sub">{esc(rung.title)}</span><span class="obj">{objective_label(rung)}</span></th>{"".join(cells)}</tr>'
        )

    def totals(label: str, values: list[str], kind: str) -> str:
        return (f'<tr class="totals"><th scope="row">{label}</th>'
                + "".join(f'<td class="lb-total" data-total="{kind}" data-controller="{esc(c.name)}">{v}</td>'
                          for c, v in zip(built, values, strict=True)) + "</tr>")

    rungs_passed = [f"{sum(bool(r.get('passed')) for r in c.runs.values())}/{len(c.runs)}" if c.runs else "not run" for c in built]
    gates_passed = []
    for c in built:
        gates = [g for r in c.runs.values() for g in r.get("gates") or []]
        gates_passed.append(f"{sum(bool(g.get('passed')) for g in gates)}/{len(gates)}" if gates else "not run")
    compute = [with_unit(c.cost.get("compute_us_p99"), "us") if c.runs else "not run" for c in built]
    rows.append(totals("Rungs passed", rungs_passed, "rungs"))
    rows.append(totals("Gates passed", gates_passed, "gates"))
    rows.append(totals("Compute p99", compute, "compute"))
    return (f'<div class="table-wrap lb-wrap"><table class="lb"><thead><tr><th scope="col">Rung · objective</th>'
            f'{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def cost_table(box: BoxInfo) -> str:
    rows = []
    for info in box.built:
        c = info.cost
        rows.append(
            f'<tr data-controller="{esc(info.name)}"><th scope="row"><a href="{info.path}">{esc(info.title)}</a></th>'
            f'<td>{esc(RUNS_ON_TEXT[info.spec.runs_on])}</td>'
            f'<td class="num">{esc(with_unit(c.get("compute_us_p50"), "us"))}</td>'
            f'<td class="num">{esc(with_unit(c.get("compute_us_p99"), "us"))}</td>'
            f'<td class="num">{esc(with_unit(c.get("compute_us_max"), "us"))}</td>'
            f'<td class="num">{esc(fmt(c.get("command_fields")))}</td>'
            f'<td class="num">{esc(fmt(c.get("wire_bytes_per_tick")))}</td></tr>'
        )
    return ('<div class="table-wrap"><table class="cost"><thead><tr><th>Controller</th><th>Runs</th><th>Compute p50</th>'
            '<th>p99</th><th>max</th><th>Fields / joint</th><th>Bytes / tick</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
            '<p class="muted small">Median over the rungs of each run\'s numbers. Compute is the measured wall time of one '
            '<code>tick()</code> on one BLAS thread; fields and bytes are the ServoCommand fields set per joint and the '
            'estimated CAN-FD payload for all joints (moteus registers).</p>')


def ladder_table(box: BoxInfo) -> str:
    rows = []
    for rung in box.rungs:
        gates = " ".join(gate_chip(g, measured=False) for g in rung.gates) or (
            f'<span class="faint">{esc(rung.gates_text)}</span>' if rung.gates_text else '<span class="faint">none</span>')
        rows.append(
            f'<tr><td class="num">{rung.order}</td><th scope="row"><a href="/rungs/#{rung.anchor}"><code>{esc(rung.id)}</code></a>'
            f'<span class="sub">{esc(rung.title)}</span></th><td class="gates-cell">{gates}</td><td class="obj">{objective_label(rung)}</td></tr>'
        )
    return ('<div class="table-wrap"><table class="ladder"><thead><tr><th>#</th><th>Rung</th><th>Gates</th><th>Objective</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def source_note(box: BoxInfo) -> str:
    if box.rungs_source == "results":
        g = box.generated
        bits = [f"commit <code>{esc(g.get('commit', '?'))}</code>{' (dirty)' if g.get('dirty') else ''}",
                esc(g.get("date", "")), f"plant {esc(g.get('plant', '?'))}", f"MuJoCo {esc(g.get('mujoco', '?'))}",
                f"robot_msgs {esc(g.get('robot_msgs', '?'))}"]
        if g.get("cost_numbers_contended"):
            bits.append("<strong>compute measured with parallel workers: contended</strong>")
        return f'<p class="muted small">From the results ledger: {" · ".join(b for b in bits if b)}.</p>'
    if box.rungs_source == "bench":
        return '<p class="muted small">Ladder from the bench (<code>bench.rungs.ladder</code>); no results yet.</p>'
    if box.rungs_source == "contract":
        return ('<p class="muted small">Ladder read from <code>docs/CONTRACT.md</code> (the frozen ruler): the bench '
                'ladder is not importable in this build and there are no results yet.</p>')
    return ""


def wandb_line(box: BoxInfo) -> str:
    project = esc(box.box.wandb_project)
    if box.wandb_url:
        reports = "".join(f' · <a href="{esc(url)}" rel="noopener">{esc(name)}</a>' for name, url in box.wandb_reports.items())
        return f'<a href="{esc(box.wandb_url)}" rel="noopener" data-wandb>{project}</a>{reports}'
    return f'<span class="faint" data-wandb="not-logged">not logged yet</span> <span class="muted small">(project “{project}”)</span>'


def box_page(box: BoxInfo, boxes: list[BoxInfo], plot_files: dict[str, str], **ctx) -> str:
    toc = [("top", box.name, 2)]
    sections = []
    if box.built:
        toc.append(("recommendation", "Recommendation", 2))
        sections.append(f'<section aria-labelledby="recommendation">{heading(2, "Recommendation", "recommendation")}'
                        f'{recommendation_html(box)}</section>')
    if box.rungs and box.built:
        toc.append(("leaderboard", "Leaderboard", 2))
        if box.has_results:
            board = (leaderboard(box) +
                     '<p class="muted small">Each cell is the rung\'s objective; ✓ passed every gate of the rung, ✗ failed at '
                     'least one (hover for which). <span class="best-tag">best</span> is the best objective among the '
                     'controllers that passed. Values are seed 0.</p>')
        else:
            board = '<p class="muted" data-not-run>Not run yet: no results for this box.</p>'
        sections.append(f'<section aria-labelledby="leaderboard">{heading(2, "Leaderboard", "leaderboard")}'
                        f'{source_note(box)}{board}</section>')
    if "pareto" in plot_files:
        toc.append(("pareto", "Performance vs cost", 2))
        fronts = "".join(
            f'<li><code>{esc(rung_id)}</code>: ' + ", ".join(
                f'<a href="{box.controller(n).path}">{esc(n)}</a>' if box.controller(n) else esc(n) for n in names) + "</li>"
            for rung_id, names in box.pareto.items()
        )
        w, h = plot_files.get("pareto_size", (1575, 920))
        sections.append(
            f'<section aria-labelledby="pareto">{heading(2, "Performance vs cost", "pareto")}'
            f'<p class="muted">Per belt rung: tracking on the free-space rung against the largest contact force on that rung. '
            f'Filled markers are the Pareto set over three axes, all lower-is-better: that tracking, that contact force, and '
            f'compute per tick (last panel). Hollow markers are beaten on all three by another controller; an × marks a run '
            f'that failed the rung\'s gates.</p>'
            f'<figure class="figure"><img src="{plot_files["pareto"]}" alt="Pareto plots of tracking against contact force per belt rung, and compute per controller" '
            f'loading="lazy" decoding="async" width="{w}" height="{h}"></figure>'
            f'<details class="json"><summary>The Pareto sets as a list</summary><ul class="fronts">{fronts}</ul></details></section>'
        )
    if box.has_results and box.built:
        toc.append(("cost", "Cost", 2))
        sections.append(f'<section aria-labelledby="cost">{heading(2, "Cost", "cost")}{cost_table(box)}</section>')
    if box.rungs:
        toc.append(("ladder", "Rung ladder", 2))
        sections.append(f'<section aria-labelledby="ladder">{heading(2, "Rung ladder", "ladder")}'
                        f'<p class="muted">Simplest first. The gates are the ruler: fixed before any result, never moved after. '
                        f'Details on <a href="/rungs/#{box.slug}">the ladder page</a>.</p>{ladder_table(box)}</section>')
    toc.append(("controllers", "Controllers", 2))
    cards = "".join(
        f'<a class="card" href="{c.path}"><span class="card-name">{esc(c.title)}</span>'
        f'<span class="card-sum">{inline(c.spec.summary)}</span>'
        f'<span class="card-tags">{badge(RUNS_ON_TEXT[c.spec.runs_on])}{"" if c.built else badge("planned", "plan")}</span></a>'
        for c in box.controllers
    )
    sections.append(f'<section aria-labelledby="controllers">{heading(2, "Controllers", "controllers")}'
                    f'<div class="grid">{cards}</div></section>')
    lead = (f"Every controller that takes a <code>{esc(box.box.input_name)}</code> and emits a "
            f"<code>{esc(box.box.output_name)}</code>. They swap for each other, so they run the same rungs and are "
            f"ranked on one leaderboard.")
    if not box.built:
        lead += " Nothing here is built yet, so there is no ladder and no result."
    body = f"""<article class="prose wide-prose">
<nav class="crumbs" aria-label="Breadcrumb"><a href="/">Home</a> <span aria-hidden="true">/</span> <span aria-current="page">{esc(box.name)}</span></nav>
<h1 id="top">{esc(box.name)}</h1>
<p class="lead">{lead}</p>
{fixture_banner([box])}
<div class="meta">
  <div class="meta-row"><span class="meta-k">Types</span>{io_chips(box)}</div>
  <div class="meta-row"><span class="meta-k">W&amp;B</span><span>{wandb_line(box)}</span></div>
</div>
{"".join(sections)}
</article>"""
    return shell(
        title=f"{box.name} · robot_controllers", description=f"Controllers that take {box.box.input_name} and emit "
        f"{box.box.output_name}: leaderboard, rungs, cost.", path=box.path, body=body, current=box.slug, toc=toc,
        groups=sidebar_groups(boxes), **ctx,
    )  # fmt: skip


# ------------------------------------------------------------------------------- controller page


def command_fields_sent(info: ControllerInfo) -> list[str] | None:
    """The ServoCommand fields the example's last command set (a finite value on any joint)."""
    run = info.example
    if not run or not run.command:
        return None
    return [name for name, values in run.command.items() if values is not None and any(v is not None for v in values)]


def where_it_runs(info: ControllerInfo) -> tuple[str, str]:
    """(inline SVG of host <-> bus <-> board with K and D placed, a sentence saying the same)."""
    runs_on = info.spec.runs_on
    timing = HostTiming()
    fields = command_fields_sent(info)
    stiffness_sent = None
    if info.example and info.example.command:
        values = info.example.command.get("stiffness")
        stiffness_sent = values is not None and any(v not in (None, 0.0) for v in values)
    k_place = "host" if runs_on in ("host", "split") else "board"
    d_place = "host" if runs_on == "host" else "board"
    if fields is not None:
        bus_label = ", ".join(f.replace("feedforward_effort", "feedforward") for f in fields)
    else:
        bus_label = {"board": "position, velocity, feedforward", "host": "feedforward torque (gains 0)",
                     "split": "velocity, damping, feedforward"}[runs_on]
    if runs_on == "board":
        gains = ("K and D are sent with every setpoint" if stiffness_sent
                 else "K and D are the board's configured gains" if stiffness_sent is False
                 else "K and D close on the board")
        sentence = (f"The spring and the damper close on the servo board at {fmt(BOARD_RATE_HZ / 1000)} kHz: {gains}. "
                    f"The host only samples the input and streams setpoints at {fmt(timing.rate_hz)} Hz.")
        host_note, board_note = "samples the input", "PD closes here"
    elif runs_on == "host":
        sentence = (f"The spring and the damper are computed on the host as a torque, from a state that is already "
                    f"{fmt(timing.reply_delay_s * 1000)} ms old, and land on the board "
                    f"{fmt(timing.command_delay_s * 1000)} ms later; the board's gains are zero, so it only applies the torque.")
        host_note, board_note = "τ = K·Δq + D·Δq̇ + τ_ref", "gains 0: applies τ"
    else:
        sentence = (f"The spring is a host torque at {fmt(timing.rate_hz)} Hz; the damper runs on the board at "
                    f"{fmt(BOARD_RATE_HZ / 1000)} kHz (kp 0, kd > 0), so damping acts without the bus delay.")
        host_note, board_note = "τ = K·Δq + τ_ref", "D closes here"

    lines, line = [], ""
    for word in bus_label.split(", "):
        candidate = f"{line}, {word}" if line else word
        if len(candidate) > 30 and line:
            lines.append(line + ",")
            line = word
        else:
            line = candidate
    lines.append(line)
    bus_lines = "".join(f'<text x="380" y="{94 + 13 * i}" text-anchor="middle" class="bus-f mono">{esc(text)}</text>'
                        for i, text in enumerate(lines[:2]))

    def law(place: str, x: int) -> str:
        chips = []
        if k_place == place:
            chips.append(("K", "spring"))
        if d_place == place:
            chips.append(("D", "damper"))
        out = []
        for index, (letter, word) in enumerate(chips):
            cx = x + index * 104
            out.append(f'<g class="law"><rect x="{cx}" y="104" width="94" height="34" rx="8"/>'
                       f'<text x="{cx + 14}" y="126" class="law-k">{letter}</text>'
                       f'<text x="{cx + 32}" y="126" class="law-w">{word}</text></g>')
        if not chips:
            out.append(f'<text x="{x}" y="126" class="dim">no K, no D</text>')
        return "".join(out)

    svg = f"""<svg class="diagram" viewBox="0 0 760 196" role="img" aria-labelledby="dg-t dg-d" xmlns="http://www.w3.org/2000/svg">
<title id="dg-t">Where {esc(info.title)} runs</title><desc id="dg-d">{esc(sentence)}</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" class="arrowhead"/></marker></defs>
<g class="node{" active" if runs_on in ("host", "split") else ""}"><rect x="8" y="20" width="240" height="150" rx="12"/>
<text x="26" y="48" class="node-h">Host</text><text x="26" y="68" class="node-s">tick() at {fmt(timing.rate_hz)} Hz, this thread</text>
<text x="26" y="90" class="node-s mono">{esc(host_note)}</text>{law("host", 26)}</g>
<g class="bus"><text x="380" y="40" text-anchor="middle" class="node-h">CAN-FD bus</text>
<line x1="256" y1="76" x2="502" y2="76" marker-end="url(#arrow)"/><text x="380" y="68" text-anchor="middle" class="bus-t">ServoCommand · {fmt(timing.command_delay_s * 1000)} ms</text>
{bus_lines}
<line x1="502" y1="144" x2="256" y2="144" marker-end="url(#arrow)"/><text x="380" y="136" text-anchor="middle" class="bus-t">ServoState · {fmt(timing.reply_delay_s * 1000)} ms</text>
<text x="380" y="162" text-anchor="middle" class="bus-f mono">position, velocity, effort</text></g>
<g class="node{" active" if runs_on in ("board", "split") else ""}"><rect x="512" y="20" width="240" height="150" rx="12"/>
<text x="530" y="48" class="node-h">Servo board</text><text x="530" y="68" class="node-s">PD loop at {fmt(BOARD_RATE_HZ / 1000)} kHz → motor</text>
<text x="530" y="90" class="node-s mono">{esc(board_note)}</text>{law("board", 530)}</g>
</svg>"""
    return svg, sentence


def params_table(info: ControllerInfo) -> str:
    if info.params is None:
        return f'<p class="muted" data-params="not-available">{esc(info.params_note or "Parameters not available.")}</p>'
    if not info.params:
        return f'<p class="muted">{esc(info.params_note)}</p>'
    rows = "".join(
        f'<tr><th scope="row"><code>{esc(p.name)}</code>{" " + badge("set by this controller", "sample") if p.set_by_spec else ""}</th>'
        f'<td><code>{esc(p.default_text)}</code></td><td>{esc(p.unit) or "—"}</td>'
        f'<td>{inline(p.doc) if p.doc else "<span class=faint>—</span>"}</td></tr>'
        for p in info.params
    )
    cls = info.spec.config_target or ""
    note = f'<p class="muted small">{esc(info.params_note)}</p>' if info.params_note else ""
    return (f'<p class="muted">The fields of <code>{esc(cls.split(":")[-1])}</code>, with the value this controller runs '
            f'with. Angles are shown in degrees beside the SI value the code uses.</p>{note}'
            f'<div class="table-wrap"><table class="fields params"><thead><tr><th>Field</th><th>Default</th><th>Unit</th>'
            f'<th>Meaning</th></tr></thead><tbody>{rows}</tbody></table></div>')


def example_html(info: ControllerInfo) -> str:
    run = info.example
    if run is None:
        return '<p class="muted">No example for this input type yet.</p>'
    code = code_block(run.code, label="example")
    if run.ok:
        return (f'<p class="muted">Built from its input message and ticked from the caller\'s own loop. Run when this page '
                f'was built; the output is what it printed.</p>{code}<p class="out-label">Output</p>'
                f'{code_block(run.output, "text", wrap=True, label="example output")}')
    reason = run.error or "the class is not importable yet"
    return (f'<p class="muted">Built from its input message and ticked from the caller\'s own loop.</p>{code}'
            f'<p class="muted small" data-example="not-run">Not run in this build: {esc(reason)}.</p>')


def results_table(info: ControllerInfo, box: BoxInfo) -> str:
    rows = []
    for rung in box.rungs:
        run = info.runs.get(rung.id)
        objective = metric(rung.objective)
        unit = objective.unit if objective else ""
        if run is None:
            rows.append(f'<tr data-rung="{esc(rung.id)}"><th scope="row"><a href="/rungs/#{rung.anchor}"><code>{esc(rung.id)}</code></a>'
                        f'<span class="sub">{esc(rung.title)}</span></th><td class="num" data-value="null">not run</td>'
                        f'<td>—</td><td></td><td></td></tr>')
            continue
        passed = bool(run.get("passed"))
        best = box.best.get(rung.id) == info.name
        chips = " ".join(gate_chip(g, measured=True) for g in run.get("gates") or [])
        url = run.get("wandb_url")
        wandb = (f'<a href="{esc(url)}" rel="noopener" data-wandb>run</a>' if isinstance(url, str) and url.startswith("https://")
                 else '<span class="faint" data-wandb="not-logged">not logged yet</span>')
        verdict = (f'<span class="verdict {"pass" if passed else "fail"}"><span class="mark">{"✓" if passed else "✗"}</span> '
                   f'{"pass" if passed else "fail"}</span>{" <span class=best-tag>best</span>" if best else ""}')
        rows.append(
            f'<tr data-rung="{esc(rung.id)}"><th scope="row"><a href="/rungs/#{rung.anchor}"><code>{esc(rung.id)}</code></a>'
            f'<span class="sub">{esc(rung.title)}</span></th>'
            f'<td class="num" data-value="{esc(json.dumps(run.get("objective")))}"><span class="num-v">{esc(fmt(run.get("objective")))}</span>'
            f'<span class="unit"> {esc(unit_text(unit))}</span><span class="sub">{esc(rung.objective)}</span></td>'
            f'<td>{verdict}</td><td class="gates-cell">{chips}</td><td>{wandb}</td></tr>'
        )
    return ('<div class="table-wrap"><table class="results"><thead><tr><th>Rung</th><th>Objective</th><th>Verdict</th>'
            f'<th>Gates</th><th>W&amp;B</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>')


def cost_html(info: ControllerInfo) -> str:
    c = info.cost
    fields = command_fields_sent(info)
    rows = [("Compute per tick, p50", with_unit(c.get("compute_us_p50"), "us")),
            ("Compute per tick, p99", with_unit(c.get("compute_us_p99"), "us")),
            ("Compute per tick, max", with_unit(c.get("compute_us_max"), "us")),
            ("Command fields per joint", fmt(c.get("command_fields"))),
            ("Bus payload per tick", with_unit(c.get("wire_bytes_per_tick"), "bytes"))]
    table = "".join(f'<tr><th scope="row">{esc(k)}</th><td class="num">{esc(v)}</td></tr>' for k, v in rows)
    sent = ""
    if fields is not None:
        sent = (f'<p class="muted small">Fields its command set in the example: '
                + (", ".join(f"<code>{esc(f)}</code>" for f in fields) or "none") + ".</p>")
    if not info.runs:
        return f'<p class="muted" data-not-run>Not run yet: no measured cost.</p>{sent}'
    return (f'<div class="table-wrap"><table class="kv cost-kv"><tbody>{table}</tbody></table></div>'
            f'<p class="muted small">Median over the rungs of each run\'s numbers (seed 0).</p>{sent}')


def self_report_html(info: ControllerInfo) -> str:
    run = info.example
    if not run or not run.ran:
        return '<p class="muted">Not available: the example did not run in this build.</p>'
    parts = [f'<p>Takes a new input while running (<code>replace_input</code>): '
             f'<strong>{"yes" if run.replannable else "no"}</strong>.</p>']
    if run.diagnostics:
        rows = "".join(f'<tr><th scope="row"><code>{esc(k)}</code></th><td><code>{esc(fmt(v) if isinstance(v, (int, float)) else v)}</code></td></tr>'
                       for k, v in run.diagnostics.items())
        parts.append('<p class="muted">Keys of <code>diagnostics()</code> after the example, with their values. Shown beside '
                     'the metrics, never used by them.</p>'
                     f'<div class="table-wrap"><table class="kv"><tbody>{rows}</tbody></table></div>')
    else:
        parts.append('<p class="muted">It has no <code>diagnostics()</code>: everything about it is measured from outside.</p>')
    return "".join(parts)


def plots_html(info: ControllerInfo, box: BoxInfo, plot_files: dict[str, str]) -> str:
    figures = []
    for rung in box.rungs:
        key = f"{info.name}|{rung.id}"
        if key not in plot_files:
            continue
        w, h = plot_files.get(f"{key}|size", (1140, 600))
        figures.append(
            f'<figure class="figure run-fig" id="plot-{rung.anchor}"><figcaption><code>{esc(rung.id)}</code> · {esc(rung.title)}</figcaption>'
            f'<img src="{plot_files[key]}" alt="{esc(info.title)} on {esc(rung.title)}: joint positions and tracking error'
            f'{", contact force" if rung.objective == "contact_max_n" else ""}" loading="lazy" decoding="async" width="{w}" height="{h}"></figure>'
        )
    if not figures:
        return '<p class="muted" data-not-run>No plots: not run yet.</p>'
    return ('<p class="muted">Joint positions in degrees (reference dashed), the tracking error against the rung\'s gate, '
            'and on belt rungs the contact force: the impact with its 30 ms transient window shaded, then the whole run.</p>'
            + "".join(figures))


def controller_page(info: ControllerInfo, box: BoxInfo, boxes: list[BoxInfo], plot_files: dict[str, str], **ctx) -> str:
    spec = info.spec
    toc: list[tuple[str, str, int]] = []
    sections: list[str] = []
    svg, sentence = where_it_runs(info)
    toc.append(("where", "Where it runs", 2))
    sections.append(f'<section aria-labelledby="where">{heading(2, "Where it runs", "where")}<div class="diagram-wrap">{svg}</div>'
                    f'<p class="muted">{esc(sentence)}</p></section>')
    if info.built:
        toc.append(("parameters", "Parameters", 2))
        sections.append(f'<section aria-labelledby="parameters">{heading(2, "Parameters", "parameters")}{params_table(info)}</section>')
        toc.append(("example", "Example", 2))
        sections.append(f'<section aria-labelledby="example">{heading(2, "Example", "example")}{example_html(info)}</section>')
        toc.append(("results", "Results", 2))
        if box.rungs:
            body = results_table(info, box) if box.has_results else '<p class="muted" data-not-run>Not run yet: no results for this box.</p>'
            if not box.has_results:
                body += results_table(info, box)
        else:
            body = '<p class="muted" data-not-run>Not run yet: this box has no ladder.</p>'
        sections.append(f'<section aria-labelledby="results">{heading(2, "Results", "results")}{source_note(box) if box.has_results else ""}{body}</section>')
        toc.append(("plots", "Plots", 2))
        sections.append(f'<section aria-labelledby="plots">{heading(2, "Plots", "plots")}{plots_html(info, box, plot_files)}</section>')
        toc.append(("cost", "Cost", 2))
        sections.append(f'<section aria-labelledby="cost">{heading(2, "Cost", "cost")}{cost_html(info)}</section>')
        toc.append(("self-report", "Self-report", 2))
        sections.append(f'<section aria-labelledby="self-report">{heading(2, "Self-report", "self-report")}{self_report_html(info)}</section>')
    toc.append(("prior-art", "Prior art", 2))
    prior = "".join(f"<li>{inline(p)}</li>" for p in spec.prior_art) or "<li class='faint'>none listed</li>"
    sections.append(f'<section aria-labelledby="prior-art">{heading(2, "Prior art", "prior-art")}<ul class="prior">{prior}</ul></section>')
    if info.built:
        toc.append(("wandb", "W&B", 2))
        links = [(r, run.get("wandb_url")) for r in box.rungs if (run := info.runs.get(r.id))]
        logged = [(r, u) for r, u in links if isinstance(u, str) and u.startswith("https://")]
        if logged:
            items = "".join(f'<li><a href="{esc(u)}" rel="noopener" data-wandb><code>{esc(r.id)}</code></a></li>' for r, u in logged)
            wandb = f'<p>Runs in {wandb_line(box)}:</p><ul class="prior">{items}</ul>'
        else:
            wandb = (f'<p data-wandb="not-logged"><span class="faint">Not logged yet.</span> Its runs go to the W&amp;B project '
                     f'“{esc(box.box.wandb_project)}”, group <code>{esc(info.name)}</code>.</p>')
        sections.append(f'<section aria-labelledby="wandb">{heading(2, "W&B", "wandb")}{wandb}</section>')

    tags = [badge(RUNS_ON_TEXT[spec.runs_on])]
    tags.append(badge("built") if info.built else badge("planned", "plan"))
    if info.example and info.example.ran and info.example.replannable is not None:
        tags.append(badge("replannable" if info.example.replannable else "not replannable"))
    variant = ""
    if spec.variant_of:
        parent = box.controller(spec.variant_of)
        variant = (f'<p class="muted small">Variant of <a href="{parent.path}">{esc(parent.title)}</a>: the same law, split '
                   f'or scheduled differently.</p>') if parent else ""
    if info.variants:
        links = ", ".join(f'<a href="{box.controller(v).path}">{esc(box.controller(v).title)}</a>' for v in info.variants
                          if box.controller(v))
        variant += f'<p class="muted small">Variants: {links}.</p>'
    planned = ""
    if not info.built:
        planned = ('<div class="callout plan" data-planned><p class="callout-k">Not built yet</p><p>This controller is in the '
                   'registry so the list is honest about what is missing. It has no class, no parameters and no results; '
                   'the prior art below is where it would start.</p></div>')
    elif not info.importable:
        planned = (f'<div class="callout plan"><p class="callout-k">Class not importable in this build</p><p><code>{esc(spec.target)}</code>: '
                   f'{esc(info.import_error)}. The spec below is the registry\'s; parameters and the example are not available.</p></div>')
    import_lines = [f"from robot_controllers import controller\n\narm = controller({spec.name!r}).build(message)"]
    meta_rows = [f'<div class="meta-row"><span class="meta-k">Types</span>{io_chips(box)}</div>']
    if info.built:
        meta_rows.append(f'<div class="meta-row"><span class="meta-k">Build</span><div class="meta-v">{code_block(import_lines[0], label="build line")}</div></div>')
        meta_rows.append(f'<div class="meta-row"><span class="meta-k">Class</span><code>{esc(spec.target)}</code></div>')
        if spec.config_target:
            meta_rows.append(f'<div class="meta-row"><span class="meta-k">Config</span><code>{esc(spec.config_target)}</code></div>')
    doc = ""
    if info.class_summary and info.class_summary != spec.summary:
        doc += f'<p class="muted">{inline(info.class_summary)}</p>'
    if info.class_doc:
        doc += f'<details class="doc-rest"><summary>Full docstring</summary><pre class="plain">{esc(info.class_doc)}</pre></details>'
    body = f"""<article class="prose">
<nav class="crumbs" aria-label="Breadcrumb"><a href="/">Home</a> <span aria-hidden="true">/</span> <a href="{box.path}">{esc(box.name)}</a> <span aria-hidden="true">/</span> <span aria-current="page">{esc(info.title)}</span></nav>
<h1 id="top">{esc(info.title)}</h1>
<p class="lead">{inline(spec.summary)}</p>
<p class="tags">{"".join(tags)} <code class="name">{esc(spec.name)}</code></p>
{variant}{doc}{planned}{fixture_banner([box]) if info.built else ""}
<div class="meta">{"".join(meta_rows)}</div>
{"".join(sections)}
</article>"""
    return shell(
        title=f"{info.title} · robot_controllers", description=spec.summary, path=info.path, body=body,
        current=info.name, toc=toc, groups=sidebar_groups(boxes), **ctx,
    )  # fmt: skip


# ------------------------------------------------------------------------------- rungs and metrics


def kv_rows(mapping: dict) -> str:
    rows = []
    for key, value in mapping.items():
        unit = ""
        for suffix, shown in (("_rad_s", "rad/s"), ("_rad", "rad"), ("_m_s", "m/s"), ("_mm", "mm"), ("_n", "N"),
                              ("_s", "s"), ("_deg", "deg"), ("_nm", "N·m"), ("_hz", "Hz")):
            if str(key).endswith(suffix):
                unit = shown
                break
        shown = value if isinstance(value, str) else value_text(value, unit)  # a rung's prose parameter reads as written
        rows.append(f'<tr><th scope="row"><code>{esc(key)}</code></th><td>{esc(shown)}</td></tr>')
    return "".join(rows)


def timing_text(rung: RungInfo) -> str:
    t = rung.timing
    parts = [f"host {fmt(rung.rate_hz)} Hz"]
    if isinstance(t.get("command_delay_s"), (int, float)):
        parts.append(f"command delay {fmt(t['command_delay_s'] * 1000)} ms")
    if isinstance(t.get("reply_delay_s"), (int, float)):
        parts.append(f"reply delay {fmt(t['reply_delay_s'] * 1000)} ms")
    if t.get("jitter_s"):
        parts.append(f"jitter {fmt(float(t['jitter_s']) * 1000)} ms")
    for stall in t.get("stalls") or []:
        try:
            parts.append(f"stall {fmt(float(stall[1]) * 1000)} ms at {fmt(float(stall[0]))} s")
        except (TypeError, ValueError, IndexError):
            pass
    return " · ".join(parts)


def scene_text(rung: RungInfo) -> str:
    s = rung.scene
    if not s:
        return ""
    parts = []
    if s.get("belt"):
        parts.append(f"belt at {fmt(s.get('belt_speed_m_s'))} m/s (contact {fmt(s.get('contact_stiffness_n_m'))} N/m, "
                     f"{fmt(s.get('contact_damping_n_s_m'))} N·s/m)")
    for push in s.get("pushes") or []:
        if isinstance(push, dict):
            parts.append(f"push {fmt(max((abs(float(v)) for v in push.get('force_n') or [0]), default=0))} N for "
                         f"{fmt(push.get('duration_s'))} s at {fmt(push.get('start_s'))} s")
    if s.get("payload_kg"):
        parts.append(f"payload {fmt(s['payload_kg'])} kg")
    if s.get("start_q"):
        parts.append("start " + value_text(list(s["start_q"]), "rad"))
    return " · ".join(parts) or "free space"


def rungs_page(boxes: list[BoxInfo], **ctx) -> str:
    toc = [("rungs", "Rungs", 2)]
    sections = []
    for box in boxes:
        toc.append((box.slug, box.name, 2))
        if not box.rungs:
            sections.append(f'<section aria-labelledby="{box.slug}">{heading(2, box.name, box.slug)}'
                            f'<p class="muted">No ladder yet: {"nothing in this box is built" if not box.built else "no rungs defined"}.</p></section>')
            continue
        cards = []
        for rung in box.rungs:
            gates = "".join(
                f'<li>{gate_chip(g, measured=False)}{" <span class=muted>" + esc(g.get("why")) + "</span>" if g.get("why") else ""}</li>'
                for g in rung.gates) or f'<li class="faint">{esc(rung.gates_text) or "none"}</li>'
            info_rows = [("Objective", objective_label(rung)), ("Timing", esc(timing_text(rung)))]
            if scene_text(rung):
                info_rows.append(("Scene", esc(scene_text(rung))))
            if rung.duration_s:
                info_rows.append(("Duration", f"{esc(fmt(rung.duration_s))} s"))
            if rung.family:
                info_rows.append(("Family", esc(rung.family)))
            info = "".join(f'<tr><th scope="row">{k}</th><td>{v}</td></tr>' for k, v in info_rows)
            params = (f'<details class="json"><summary>Parameters</summary><div class="table-wrap"><table class="kv"><tbody>'
                      f'{kv_rows(rung.params)}</tbody></table></div></details>') if rung.params else ""
            cards.append(
                f'<div class="rung" data-rung="{esc(rung.id)}">{heading(3, f"{rung.order}. {rung.id} · {rung.title}", rung.anchor)}'
                f'<p>{inline(rung.description)}</p><div class="table-wrap"><table class="kv"><tbody>{info}</tbody></table></div>'
                f'<p class="gates-h">Gates</p><ul class="gate-list">{gates}</ul>{params}</div>'
            )
        sections.append(f'<section aria-labelledby="{box.slug}">{heading(2, box.name, box.slug)}{source_note(box)}'
                        f'<p class="muted"><a href="{box.path}">Leaderboard</a> · {len(box.rungs)} rungs, simplest first.</p>'
                        f'{"".join(cards)}</section>')
    body = f"""<article class="prose">
<p class="crumbs">Rungs</p>
<h1 id="rungs">The rung ladder</h1>
<p class="lead">Every box is scored on its own ladder: the simplest rung first, harder ones after. Each rung fixes an input script, a scene, the host timing, its gates and the one objective it ranks controllers on. The gates are the ruler: set before any result and never moved after one, so a controller that fails is a finding.</p>
{"".join(sections)}
</article>"""
    return shell(
        title="Rungs · robot_controllers", description="The rung ladder of every controller box: what each rung does, its gates and objective.",
        path="/rungs/", body=body, section="rungs", current="rungs", toc=toc, groups=sidebar_groups(boxes), **ctx,
    )  # fmt: skip


def metrics_page(boxes: list[BoxInfo], **ctx) -> str:
    used: dict[str, list[tuple[RungInfo, str]]] = {}
    for box in boxes:
        for rung in box.rungs:
            for gate in rung.gates:
                used.setdefault(str(gate.get("metric")), []).append((rung, "gate"))
            used.setdefault(rung.objective, []).append((rung, "objective"))
    toc = [("metrics", "Metrics", 2)]
    sections = []
    for family, members in metrics_by_family():
        toc.append((f"family-{family}", family.capitalize(), 3))
        rows = []
        for m in members:
            seen: dict[str, set[str]] = {}
            for rung, role in used.get(m.name, []):
                seen.setdefault(rung.id, set()).add(role)
            uses = " ".join(
                f'<a class="chip small-chip" href="/rungs/#{next(r.anchor for r, _ in used[m.name] if r.id == rid)}" '
                f'title="{esc(" and ".join(sorted(roles)))}">{esc(rid)}{"★" if "objective" in roles else ""}</a>'
                for rid, roles in seen.items())
            better = {"lower": "↓ lower", "higher": "↑ higher", "none": "context"}[m.better]
            rows.append(
                f'<tr id="m-{esc(m.name)}"><th scope="row"><code>{esc(m.name)}</code></th><td>{esc(unit_text(m.unit) or "—")}</td>'
                f'<td class="nowrap">{better}</td><td>{inline(m.description)}</td><td class="uses">{uses or "<span class=faint>—</span>"}</td></tr>'
            )
        sections.append(
            f'<section aria-labelledby="family-{family}">{heading(2, family.capitalize(), f"family-{family}")}'
            f'<div class="table-wrap"><table class="metrics"><thead><tr><th>Metric</th><th>Unit</th><th>Better</th>'
            f'<th>What it measures</th><th>Used by</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div></section>'
        )
    body = f"""<article class="prose wide-prose">
<p class="crumbs">Metrics</p>
<h1 id="metrics">Metrics</h1>
<p class="lead">Every number a rung reports, the same in every box, with its unit in its name. A metric a run could not measure is “not measured”, never 0, and a gate on it fails. Angles are in degrees. “Used by” lists the rungs that gate on it; ★ marks the rungs it is the objective of.</p>
{"".join(sections)}
</article>"""
    return shell(
        title="Metrics · robot_controllers", description="The metric catalogue: tracking, contact, disturbance, replanning, timing, health, cost and sequence numbers.",
        path="/metrics/", body=body, section="metrics", current="metrics", toc=toc, groups=sidebar_groups(boxes), **ctx,
    )  # fmt: skip


def not_found(boxes: list[BoxInfo], **ctx) -> str:
    body = """<div class="home">
<section class="hero">
  <p class="eyebrow">404</p>
  <h1>No controller lives here</h1>
  <p class="lead">Nothing at this address. Try the rung ladder, or go back to the start.</p>
  <div class="cta"><a class="btn primary" href="/rungs/">The rung ladder</a><a class="btn" href="/">Home</a></div>
</section></div>"""
    return shell(
        title="Not found · robot_controllers", description="Page not found.", path="/404.html",
        body=body, sidebar=False, groups=sidebar_groups(boxes), robots="noindex", **ctx,
    )  # fmt: skip
