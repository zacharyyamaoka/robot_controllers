"""The robot_controllers site: every controller has a page under its box, every number equals the results JSON,
every link resolves, pages stay small, and nothing crashes when results or classes are missing.

    env -u VIRTUAL_ENV uv run --frozen --directory /home/bam/robot_controllers --extra test pytest -q website/tests/test_site.py

The build runs against the committed fixture (tests/fixtures/results, synthetic numbers that say so), or against
another results folder named by RC_SITE_RESULTS. A second build with no results checks the "not run yet" pages.
"""

import dataclasses
import html
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "build"))

import build as site_build  # noqa: E402
import model  # noqa: E402
import pages  # noqa: E402
from robot_controllers.core import ControllerSpec  # noqa: E402
from robot_controllers.registry import BOXES, CONTROLLERS, JOINT_COMMAND_SEQUENCE, JOINT_TRAJECTORY  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "results"
RESULTS = Path(os.environ.get("RC_SITE_RESULTS", FIXTURES))
if RESULTS == FIXTURES and not any(FIXTURES.glob("*/results.json")):
    # WHY generate, not commit: the fixture is 2 MB of synthetic numbers that make_fixtures.py rebuilds byte for
    # byte; git carries the generator, never the regenerable output.
    import runpy

    runpy.run_path(str(ROOT / "tests" / "fixtures" / "make_fixtures.py"), run_name="__main__")
MAX_PAGE_BYTES = 1_500_000
ROBOT_MSGS_WEBSITE = Path("/home/bam/robot_msgs/website")


def plain(text):
    """Visible text: tags dropped, entities decoded."""
    return html.unescape(re.sub(r"<[^>]+>", "", text))


def html_pages(dist):
    return sorted(p for p in dist.rglob("*.html") if "assets" not in p.relative_to(dist).parts)


def load_documents():
    documents = {}
    for box in BOXES:
        path = RESULTS / box.slug / "results.json"
        if path.is_file():
            documents[box.slug] = json.loads(path.read_text())
    return documents


def lowest_seed_runs(document):
    runs = {}
    for row in sorted(document["results"], key=lambda r: r.get("seed") or 0, reverse=True):
        runs[(row["controller"], row["rung"])] = row
    return runs


def leaderboard_cell(page_text, controller, rung):
    pattern = (r'<td class="([^"]*)" data-controller="' + re.escape(html.escape(controller)) + r'" data-rung="'
               + re.escape(html.escape(rung)) + r'"([^>]*)>(.*?)</td>')
    match = re.search(pattern, page_text, re.S)
    return match


class BuiltSite(unittest.TestCase):
    """One build against the fixture (or RC_SITE_RESULTS)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="rc-site-"))
        cls.dist = cls.tmp / "dist"
        cls.boxes = site_build.main(["--results-dir", str(RESULTS), "--dist", str(cls.dist), "--origin", "https://example.test"])
        cls.documents = load_documents()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def read(self, path):
        return (self.dist / path.lstrip("/") / "index.html").read_text() if not path.endswith(".html") else (
            self.dist / path.lstrip("/")).read_text()

    # ── the domain ─────────────────────────────────────────────────────────────────────────────────────────

    def test_the_fixture_is_there_and_has_both_boxes(self):
        self.assertIn(JOINT_TRAJECTORY.slug, self.documents)
        self.assertIn(JOINT_COMMAND_SEQUENCE.slug, self.documents)

    def test_every_registry_controller_has_a_page_under_its_box(self):
        for spec in CONTROLLERS:
            page = self.dist / "controllers" / spec.box.slug / spec.name / "index.html"
            self.assertTrue(page.is_file(), f"no page for {spec.name}")
            box_page = self.read(f"/boxes/{spec.box.slug}/")
            self.assertIn(f'href="/controllers/{spec.box.slug}/{spec.name}/"', box_page, spec.name)
            self.assertIn(html.escape(spec.title), page.read_text())

    def test_planned_controllers_say_not_built_and_list_prior_art(self):
        for spec in (s for s in CONTROLLERS if s.maturity == "planned"):
            text = self.read(f"/controllers/{spec.box.slug}/{spec.name}/")
            self.assertIn("Not built yet", text, spec.name)
            self.assertNotIn('id="parameters"', text, spec.name)
            for art in spec.prior_art:
                self.assertIn(html.escape(art), text, f"{spec.name}: {art}")

    def test_sidebar_has_overview_then_one_group_per_box_in_registry_order(self):
        text = self.read("/rungs/")
        headings = re.findall(r'<h2 class="sb-h">([^<]+)</h2>', text)
        self.assertEqual(headings, ["Overview"] + [html.escape(b.name) for b in BOXES])
        self.assertIn('placeholder="Search controllers', text)
        for spec in (s for s in CONTROLLERS if s.maturity == "planned"):
            self.assertRegex(text, re.escape(html.escape(spec.title)) + r' <span class="badge plan">planned</span>')

    def test_box_pages_link_their_types_on_robotmsgs(self):
        for box in BOXES:
            text = self.read(f"/boxes/{box.slug}/")
            for cls in (box.input_type, box.output_type):
                domain = cls.__module__.split(".")[1]
                self.assertIn(f'href="https://robotmsgs.com/types/{domain}/{cls.__name__}/"', text, box.slug)

    def test_leaderboard_cells_equal_the_results_json(self):
        for slug, document in self.documents.items():
            text = self.read(f"/boxes/{slug}/")
            for (controller, rung), run in lowest_seed_runs(document).items():
                match = leaderboard_cell(text, controller, rung)
                self.assertIsNotNone(match, f"{slug}: no cell for {controller} @ {rung}")
                classes, attributes, inner = match.groups()
                value = re.search(r'data-value="([^"]*)"', attributes).group(1)
                self.assertEqual(json.loads(html.unescape(value)), run["objective"], f"{controller} @ {rung}")
                self.assertEqual(re.search(r'<span class="num">([^<]*)</span>', inner).group(1),
                                 html.escape(model.fmt(run["objective"])), f"{controller} @ {rung}")
                self.assertIn("pass" if run["passed"] else "fail", classes.split(), f"{controller} @ {rung}")

    def test_best_is_the_best_objective_among_runs_that_passed(self):
        teeth = 0
        for slug, document in self.documents.items():
            text = self.read(f"/boxes/{slug}/")
            runs = lowest_seed_runs(document)
            compute = {}
            for (controller, _), run in runs.items():
                compute.setdefault(controller, []).append(run["metrics"]["compute_us_p99"])
            compute = {c: sorted(v)[len(v) // 2] if len(v) % 2 else sum(sorted(v)[len(v) // 2 - 1:len(v) // 2 + 1]) / 2
                       for c, v in compute.items()}
            for rung in document["rungs"]:
                rows = [r for (c, rid), r in runs.items() if rid == rung["id"]]
                passing = [r for r in rows if r["passed"] and r["objective"] is not None]
                sign = 1 if rung["minimize"] else -1
                # ties go to the lower compute p99, then the name
                expected = min(passing, key=lambda r: (sign * r["objective"], compute[r["controller"]], r["controller"]))[
                    "controller"] if passing else None
                every = [r for r in rows if r["objective"] is not None]
                overall = min(every, key=lambda r: sign * r["objective"])["controller"] if every else None
                if overall != expected:
                    teeth += 1  # a rung where ignoring the gates would pick a different controller
                for row in rows:
                    classes = leaderboard_cell(text, row["controller"], rung["id"]).group(1).split()
                    self.assertEqual("best" in classes, row["controller"] == expected, f"{row['controller']} @ {rung['id']}")
        if RESULTS == FIXTURES:
            self.assertGreater(teeth, 0, "the fixture must hold a rung whose overall best failed its gates")

    def test_best_agrees_with_the_ledger_summary(self):
        for slug, document in self.documents.items():
            text = self.read(f"/boxes/{slug}/")
            for rung_id, entry in (document["summary"].get("per_rung") or {}).items():
                for match in re.finditer(r'<td class="([^"]*)" data-controller="([^"]+)" data-rung="'
                                         + re.escape(html.escape(rung_id)) + '"', text):
                    self.assertEqual("best" in match.group(1).split(), match.group(2) == entry["best"], rung_id)

    def test_totals_count_rungs_and_gates_from_the_json(self):
        for slug, document in self.documents.items():
            text = self.read(f"/boxes/{slug}/")
            runs = lowest_seed_runs(document)
            for controller in {c for c, _ in runs}:
                mine = [r for (c, _), r in runs.items() if c == controller]
                rungs = f"{sum(r['passed'] for r in mine)}/{len(mine)}"
                gates = [g for r in mine for g in r["gates"]]
                gate_text = f"{sum(g['passed'] for g in gates)}/{len(gates)}"
                self.assertRegex(text, rf'data-total="rungs" data-controller="{controller}">{re.escape(rungs)}<')
                self.assertRegex(text, rf'data-total="gates" data-controller="{controller}">{re.escape(gate_text)}<')

    def test_recommendation_is_the_summary_one_with_its_why(self):
        for slug, document in self.documents.items():
            recommendation = document["summary"]["recommendation"]
            text = self.read(f"/boxes/{slug}/")
            self.assertIn(f'data-recommendation="{recommendation["controller"]}"', text)
            self.assertIn(recommendation["why"], plain(text))
            self.assertIn(recommendation["why"], plain(self.read("/")))

    def test_pareto_sets_on_the_page_are_the_ledger_ones(self):
        for slug, document in self.documents.items():
            pareto = document["summary"].get("pareto") or {}
            if not pareto:
                continue
            text = self.read(f"/boxes/{slug}/")
            fronts = text.split('<ul class="fronts">', 1)[1].split("</ul>", 1)[0]
            for rung, entry in pareto.items():
                names = entry["front"] if isinstance(entry, dict) else entry
                item = re.search(r"<li><code>" + re.escape(html.escape(rung)) + r"</code>: (.*?)</li>", fronts).group(1)
                self.assertEqual(re.findall(r">([a-z_]+)</a>", item), names, rung)
            self.assertTrue((self.dist / "assets/plots" / slug / "pareto.webp").is_file())

    def test_every_rung_shows_a_value_or_not_run_on_every_built_controller_page(self):
        for box in self.boxes:
            document = self.documents.get(box.slug)
            runs = lowest_seed_runs(document) if document else {}
            for info in box.built:
                text = self.read(info.path)
                for rung in box.rungs:
                    row = re.search(r'<tr data-rung="' + re.escape(html.escape(rung.id)) + r'">(.*?)</tr>', text, re.S)
                    self.assertIsNotNone(row, f"{info.name}: no row for {rung.id}")
                    cell = re.search(r'<td class="num" data-value="([^"]*)">(.*?)</td>', row.group(1), re.S)
                    run = runs.get((info.name, rung.id))
                    if run is None:
                        self.assertEqual(plain(cell.group(2)).strip(), "not run")
                        continue
                    self.assertEqual(json.loads(html.unescape(cell.group(1))), run["objective"])
                    self.assertIn(model.fmt(run["objective"]), plain(cell.group(2)))
                    for gate in run["gates"]:
                        shown = (f"{gate['metric']} {model.fmt(gate['measured'])} {model.OP_TEXT[gate['op']]} "
                                 f"{model.fmt(gate['value'])}" if gate["measured"] is not None else f"{gate['metric']} not measured")
                        self.assertIn(shown, plain(row.group(1)), f"{info.name} @ {rung.id}")

    def test_every_built_controller_page_has_a_wandb_link_or_says_not_logged_yet(self):
        for spec in (s for s in CONTROLLERS if s.maturity == "built"):
            text = self.read(f"/controllers/{spec.box.slug}/{spec.name}/")
            self.assertIn('id="wandb"', text, spec.name)
            self.assertTrue("data-wandb>" in text or "Not logged yet" in text, spec.name)

    def test_every_run_has_a_plot_that_exists_and_is_webp(self):
        for slug, document in self.documents.items():
            for (controller, rung), run in lowest_seed_runs(document).items():
                if not run.get("trace", {}).get("t"):
                    continue
                spec = next(s for s in CONTROLLERS if s.name == controller)
                text = self.read(f"/controllers/{slug}/{spec.name}/")
                src = f"/assets/plots/{slug}/{controller}/{model.slug(rung)}.webp"
                self.assertIn(f'src="{src}"', text, f"{controller} @ {rung}")
                data = (self.dist / src.lstrip("/")).read_bytes()
                self.assertEqual((data[:4], data[8:12]), (b"RIFF", b"WEBP"), src)

    def test_parameters_table_lists_every_config_field(self):
        for spec in (s for s in CONTROLLERS if s.maturity == "built" and s.config_target):
            text = self.read(f"/controllers/{spec.box.slug}/{spec.name}/")
            try:
                config = spec.config_class()
            except Exception:  # noqa: BLE001  not importable: the page must say so instead
                self.assertIn('data-params="not-available"', text, spec.name)
                continue
            for f in dataclasses.fields(config):
                self.assertIn(f"<code>{f.name}</code>", text, f"{spec.name}.{f.name}")
            for name, value in spec.default_config.items():
                self.assertRegex(text, rf"<code>{name}</code> <span class=\"badge sample\">set by this controller</span>")
                self.assertIn(html.escape(repr(value)), text)

    def test_examples_ran_or_say_why_not(self):
        for box in self.boxes:
            for info in box.built:
                text = self.read(info.path)
                if info.importable:
                    self.assertTrue(info.example.ok, f"{info.name}: {info.example.error}")
                    self.assertIn('<p class="out-label">Output</p>', text, info.name)
                    self.assertIn(info.example.output.splitlines()[0], plain(text))
                else:
                    self.assertIn('data-example="not-run"', text, info.name)

    def test_where_it_runs_places_k_and_d_by_runs_on(self):
        expected = {"board": ("board", "board"), "host": ("host", "host"), "split": ("host", "board")}
        for spec in CONTROLLERS:
            text = self.read(f"/controllers/{spec.box.slug}/{spec.name}/")
            svg = text.split('<svg class="diagram"', 1)[1].split("</svg>", 1)[0]
            host, board = svg.split("CAN-FD bus", 1)[0], svg.split("Servo board", 1)[1]
            k_place, d_place = expected[spec.runs_on]
            self.assertIn(">K</text>", host if k_place == "host" else board, spec.name)
            self.assertIn(">D</text>", host if d_place == "host" else board, spec.name)
            self.assertNotIn(">K</text>", board if k_place == "host" else host, spec.name)

    def test_home_lists_every_box_with_its_controllers_and_an_example(self):
        text = self.read("/")
        for box in BOXES:
            self.assertIn(f'href="/boxes/{box.slug}/"', text)
        for spec in CONTROLLERS:
            self.assertIn(f'href="/controllers/{spec.box.slug}/{spec.name}/"', text)
        self.assertIn("controller(", plain(text))
        self.assertIn("arm.tick(state, 0.01)", plain(text))

    def test_fixture_numbers_carry_the_synthetic_banner(self):
        if RESULTS != FIXTURES:
            self.skipTest("not the fixture")
        self.assertIn("Synthetic numbers.", self.read(f"/boxes/{JOINT_TRAJECTORY.slug}/"))
        self.assertIn("Synthetic numbers.", self.read("/"))

    def test_rungs_and_metrics_pages_cover_the_ladder_and_the_catalogue(self):
        from robot_controllers.metrics.names import METRICS

        rungs = self.read("/rungs/")
        for document in self.documents.values():
            for rung in document["rungs"]:
                self.assertIn(f'id="{model.slug(rung["id"])}"', rungs, rung["id"])
                for gate in rung["gates"]:
                    self.assertIn(f"{gate['metric']} {model.OP_TEXT[gate['op']]} {model.fmt(gate['value'])}", plain(rungs))
        metrics = self.read("/metrics/")
        for m in METRICS:
            self.assertIn(f'id="m-{m.name}"', metrics, m.name)

    # ── generic: links, anchors, sizes, landmarks (from robotmsgs.com's tests) ───────────────────────────────

    def test_every_internal_link_resolves(self):
        for page in html_pages(self.dist):
            for href in re.findall(r'(?:href|src)="(/[^"]*)"', page.read_text()):
                path = href.split("#")[0].split("?")[0]
                target = self.dist / path.lstrip("/")
                if path.endswith("/") or path == "":
                    target = target / "index.html"
                self.assertTrue(target.is_file(), f"{page.relative_to(self.dist)}: {href} -> missing")

    def test_in_page_anchors_resolve(self):
        for page in html_pages(self.dist):
            text = page.read_text()
            ids = set(re.findall(r'\bid="([^"]+)"', text))
            for anchor in re.findall(r'href="#([^"]+)"', text):
                self.assertIn(anchor, ids, f"{page.relative_to(self.dist)}: #{anchor}")

    def test_cross_page_anchors_resolve(self):
        for page in html_pages(self.dist):
            for path, anchor in re.findall(r'href="(/[^"#]*)#([^"]+)"', page.read_text()):
                target = self.dist / path.lstrip("/") / "index.html" if path.endswith("/") else self.dist / path.lstrip("/")
                self.assertIn(f'id="{anchor}"', target.read_text(), f"{page.relative_to(self.dist)}: {path}#{anchor}")

    def test_ids_are_unique_per_page(self):
        for page in html_pages(self.dist):
            ids = re.findall(r'\bid="([^"]+)"', page.read_text())
            self.assertEqual(len(ids), len(set(ids)), f"duplicate ids in {page.relative_to(self.dist)}")

    def test_no_page_over_limit(self):
        for page in html_pages(self.dist):
            self.assertLess(page.stat().st_size, MAX_PAGE_BYTES, str(page))

    def test_pages_have_landmarks_and_one_h1(self):
        for page in html_pages(self.dist):
            text = page.read_text()
            self.assertEqual(text.count("<h1"), 1, page.name)
            for landmark in ("<main", "<header", "<nav"):
                self.assertIn(landmark, text, f"{page.relative_to(self.dist)} lacks {landmark}")

    def test_external_links_are_https(self):
        for page in html_pages(self.dist):
            for url in re.findall(r'href="(http[^"]*)"', page.read_text()):
                self.assertTrue(url.startswith("https://"), url)

    def test_404_robots_and_sitemap(self):
        self.assertTrue((self.dist / "404.html").is_file())
        sitemap = (self.dist / "sitemap.xml").read_text()
        self.assertEqual(sitemap.count("<url>"), 3 + len(BOXES) + len(CONTROLLERS))
        self.assertIn("https://example.test/rungs/", sitemap)
        self.assertIn("Sitemap: https://example.test/sitemap.xml", (self.dist / "robots.txt").read_text())

    def test_chrome_is_robotmsgs_byte_for_byte(self):
        if not ROBOT_MSGS_WEBSITE.is_dir():
            self.skipTest("robot_msgs website not checked out here")
        for theirs, ours in (("site/site.css", "site/site.css"), ("site/site.js", "site/site.js"),
                             ("build/highlight.py", "build/highlight.py")):
            self.assertEqual((ROBOT_MSGS_WEBSITE / theirs).read_bytes(), (ROOT / ours).read_bytes(), ours)


class NoResults(unittest.TestCase):
    """With nothing in the results folder every page still builds and says the box was not run."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="rc-site-empty-"))
        (cls.tmp / "results").mkdir()
        cls.dist = cls.tmp / "dist"
        cls.boxes = site_build.main(["--results-dir", str(cls.tmp / "results"), "--dist", str(cls.dist), "--no-examples"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_boxes_and_controllers_say_not_run_yet(self):
        for box in self.boxes:
            text = (self.dist / box.path.lstrip("/") / "index.html").read_text()
            if box.built:
                self.assertIn("Not run yet", text, box.slug)
            for info in box.built:
                page = (self.dist / info.path.lstrip("/") / "index.html").read_text()
                self.assertIn("Not run yet", page, info.name)
                for rung in box.rungs:
                    row = re.search(r'<tr data-rung="' + re.escape(html.escape(rung.id)) + r'">(.*?)</tr>', page, re.S)
                    self.assertIsNotNone(row, f"{info.name}: {rung.id}")
                    self.assertIn("not run", plain(row.group(1)))

    def test_the_ladder_still_shows_without_results(self):
        jt = next(b for b in self.boxes if b.box == JOINT_TRAJECTORY)
        self.assertEqual(len(jt.rungs), 13, jt.rungs_source)  # ten rungs + R8 and both R9 (amendment 3)
        self.assertFalse((self.dist / "assets" / "plots").exists())
        self.assertFalse((self.dist / "sitemap.xml").exists())  # no origin given: no invented domain


class Pieces(unittest.TestCase):
    """Formulas and parsers on their own, each with a case that fails if the formula is wrong."""

    def test_number_format(self):
        self.assertEqual([model.fmt(v) for v in (0.38, 1.6, 15.74, 118.0, 0.000123, 0, None, 9, float("nan"))],
                         ["0.38", "1.60", "15.7", "118", "0.000123", "0", "—", "9", "—"])

    def test_angles_are_shown_in_degrees_beside_si(self):
        self.assertEqual(model.value_text(0.017453292519943295, "rad"), "1° (0.0175 rad)")
        self.assertEqual(model.value_text([40.0, 25.0], "N·m/rad"), "40, 25 N·m/rad (0.698, 0.436 N·m/deg)")
        self.assertEqual(model.with_unit(0.38, "deg"), "0.38°")

    def test_non_dominated_keeps_incomparable_points_and_drops_beaten_ones(self):
        points = {"a": (1.0, 5.0, 1.0), "b": (2.0, 2.0, 2.0), "c": (3.0, 6.0, 3.0), "d": (1.0, 5.0, 1.0)}
        self.assertEqual(model.non_dominated(points), ["a", "b", "d"])  # c is beaten by a on every axis

    def test_best_of_ignores_runs_that_failed_their_gates(self):
        runs = [{"controller": "fast_but_failed", "objective": 1.0, "passed": False},
                {"controller": "passed", "objective": 2.0, "passed": True},
                {"controller": "worse", "objective": 3.0, "passed": True}]
        self.assertEqual(model.best_of(runs, minimize=True), "passed")
        self.assertEqual(model.best_of(runs, minimize=False), "worse")
        self.assertIsNone(model.best_of(runs[:1], minimize=True))

    def test_field_comments_are_read_from_the_source(self):
        comments = model.field_comments(_Commented)
        self.assertEqual(comments["stiffness_nm_rad"], "K per joint (shoulder, elbow)")
        self.assertEqual(comments["damping_ratio"], "zeta: D = 2 zeta sqrt(K I)")
        self.assertNotIn("location", comments)
        self.assertEqual(model.unit_of("stiffness_nm_rad"), "N·m/rad")
        self.assertEqual(model.unit_of("damping_nm_s_rad"), "N·m·s/rad")

    def test_contract_fallback_reads_the_frozen_ladder(self):
        rungs = model.rungs_from_contract(JOINT_TRAJECTORY)
        self.assertEqual([r.id for r in rungs][:3], ["R1.FreeSpace", "R2.FastFreeSpace", "R3.BeltImpact@0.05m/s"])
        self.assertEqual(len(rungs), 10)
        by_id = {r.id: r for r in rungs}
        r1 = {(g["metric"], g["op"], g["value"]) for g in by_id["R1.FreeSpace"].gates}
        self.assertIn(("tracking_p95_deg", "<=", 1.0), r1)
        self.assertIn(("latency_p99_ms", "<=", 10.0), r1)
        same = {g["metric"] for g in by_id["R3.BeltImpact@0.20m/s"].gates}
        self.assertIn("contact_peak_n", same)  # "same" copied the previous rung's gates
        slow = {(g["metric"], g["value"]) for g in by_id["R7.BeltImpact@0.10m/s·host50Hz"].gates}
        self.assertIn(("latency_p99_ms", 20.0), slow)
        self.assertIn(("dwell_bounces", 0.0), slow)  # "rung 4's gates"
        self.assertNotIn("latency_p99_ms", {g["metric"] for g in by_id["R6.HostStall"].gates})
        self.assertEqual(len(model.rungs_from_contract(JOINT_COMMAND_SEQUENCE)), 3)

    def test_wandb_links_point_at_wandb_ai(self):
        self.assertEqual(model.rewrite_wandb("https://abc.forge.coreweave.com/e/p/runs/x"), "https://wandb.ai/e/p/runs/x")

    def test_a_missing_class_degrades_to_the_spec(self):
        boxes = model.collect(None, run_examples=False)
        box = next(b for b in boxes if b.box == JOINT_TRAJECTORY)
        broken = model.ControllerInfo(spec=ControllerSpec(
            name="ghost_controller", title="Ghost controller", box=JOINT_TRAJECTORY, runs_on="host",
            summary="Not importable on purpose.", target="robot_controllers.controllers.nope:Missing",
            config_target="robot_controllers.controllers.nope:MissingConfig"), box_slug=box.slug)
        model.describe_class(broken)
        model.describe_params(broken)
        self.assertFalse(broken.importable)
        self.assertIsNone(broken.params)
        box.controllers.append(broken)
        text = pages.controller_page(broken, box, boxes, {}, asset_version="x", stamp="")
        self.assertIn("Class not importable in this build", text)
        self.assertIn('data-params="not-available"', text)
        self.assertIn("not logged yet", text.lower())

    def test_a_logged_run_links_to_wandb(self):
        boxes = model.collect(FIXTURES, run_examples=False)
        box = next(b for b in boxes if b.box == JOINT_TRAJECTORY)
        info = box.built[0]
        first = box.rungs[0].id
        info.runs[first] = {**info.runs[first], "wandb_url": "https://wandb.ai/e/p/runs/abc"}
        text = pages.controller_page(info, box, boxes, {}, asset_version="x", stamp="")
        self.assertIn('href="https://wandb.ai/e/p/runs/abc" rel="noopener" data-wandb', text)


@dataclasses.dataclass(frozen=True)
class _Commented:
    location: str = "board"
    stiffness_nm_rad: tuple[float, ...] = (40.0, 25.0)  # K per joint (shoulder, elbow)
    # zeta: D = 2 zeta sqrt(K I)
    damping_ratio: float = 0.8


if __name__ == "__main__":
    unittest.main()
