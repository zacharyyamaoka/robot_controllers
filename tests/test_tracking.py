"""W&B tracking (CONTRACT §6): pure record building, views planning and the CLI's offline paths. No network."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest

from robot_controllers.metrics.names import METRICS
from robot_controllers.registry import JOINT_TRAJECTORY
from robot_controllers.tracking import wandb_log as tracking

FIXTURE = Path(__file__).parent / "fixtures" / "results_wandb.json"
JTC, JIH = "joint_trajectory_controller", "joint_impedance_host"
R1, R3 = "R1.FreeSpace", "R3.BeltImpact@0.10m/s"


@pytest.fixture
def results() -> dict:
    return json.loads(FIXTURE.read_text())


def by_name(batch: list[dict]) -> dict[tuple[str, str], dict]:
    return {(r["config"]["controller"], r["config"]["rung"]): r for r in batch}


# ── the module stays importable without the tracking extra ────────────────────────────────────────────────


def test_importing_the_module_and_building_records_never_imports_wandb():
    code = ("import json, sys; from robot_controllers.tracking import wandb_log as t; "
            f"t.records(json.load(open({str(FIXTURE)!r}))); "
            "print(json.dumps(sorted(m for m in ('wandb', 'weave', 'wandb_workspaces') if m in sys.modules)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout) == []


# ── run identity ────────────────────────────────────────────────────────────────────────────────────────────


def test_run_id_is_the_contract_formula_over_the_metrics_json(results):
    """Recomputed by hand from CONTRACT §6: sha1("robot_controllers|box|controller|rung|seed|digest")[:16]."""
    record = by_name(tracking.records(results))[(JTC, R1)]
    metrics = next(r["metrics"] for r in results["results"] if r["controller"] == JTC and r["rung"] == R1)
    digest = hashlib.sha1(json.dumps(metrics, sort_keys=True).encode()).hexdigest()
    text = f"robot_controllers|jointtrajectory-to-servocommand|{JTC}|{R1}|0|{digest}"
    assert record["id"] == hashlib.sha1(text.encode()).hexdigest()[:16]
    assert len(record["id"]) == 16 and int(record["id"], 16) >= 0


def test_ids_are_deterministic_and_do_not_depend_on_the_logging_checkout(results):
    first = [r["id"] for r in tracking.records(results, sha="aaa", dirty=False)]
    second = [r["id"] for r in tracking.records(copy.deepcopy(results), sha="bbb", dirty=True)]
    assert first == second
    assert len(set(first)) == 4


def test_a_changed_metric_changes_only_that_runs_id(results):
    before = {k: r["id"] for k, r in by_name(tracking.records(results)).items()}
    changed = copy.deepcopy(results)
    target = next(r for r in changed["results"] if r["controller"] == JIH and r["rung"] == R3)
    target["metrics"]["contact_peak_n"] = 72.5
    after = {k: r["id"] for k, r in by_name(tracking.records(changed)).items()}
    assert after[(JIH, R3)] != before[(JIH, R3)]
    assert {k: v for k, v in after.items() if k != (JIH, R3)} == {k: v for k, v in before.items() if k != (JIH, R3)}


def test_a_changed_trace_or_gate_text_keeps_the_id(results):
    """Only the metrics are the run's content: a re-run whose trace rounding moved resumes the same run."""
    before = by_name(tracking.records(results))[(JTC, R1)]["id"]
    changed = copy.deepcopy(results)
    target = next(r for r in changed["results"] if r["controller"] == JTC and r["rung"] == R1)
    target["trace"]["tau_nm.shoulder"][0] = 99.0
    assert by_name(tracking.records(changed))[(JTC, R1)]["id"] == before


# ── names, group, tags, project ─────────────────────────────────────────────────────────────────────────────


def test_names_groups_and_tags_follow_the_contract(results):
    for record in tracking.records(results):
        config = record["config"]
        assert record["name"] == f"{config['controller']} @ {config['rung']}"
        assert len(record["name"].encode()) <= 128
        assert record["group"] == config["controller"]
        assert record["job_type"] == "rung"
        assert set(record["tags"]) == {config["rung_family"], "sim", "latest"}


def test_a_long_name_is_cut_to_128_bytes_with_an_ellipsis():
    name = tracking.run_name("x" * 60 + "_controller", "R7.BeltImpact@0.10m/s·host500Hz" + "·" * 40)
    assert len(name.encode()) <= 128
    assert name.endswith("…")
    name.encode().decode("utf-8")  # cut on a character boundary
    assert tracking.run_name("a", "b", seed=2) == "a @ b · seed 2"


def test_the_project_is_the_boxes_wandb_project(results):
    batch = tracking.records(results)
    assert {r["project"] for r in batch} == {JOINT_TRAJECTORY.wandb_project}
    assert {r["project"] for r in tracking.records(results, project="Controller · scratch")} == {"Controller · scratch"}


def test_a_ledger_naming_another_project_is_refused(results):
    results["box"]["wandb_project"] = "Controller · JointTrajectory → ServoCommand"
    with pytest.raises(ValueError, match="Box.wandb_project"):
        tracking.records(results)


def test_a_document_of_another_schema_is_refused(results):
    results["schema"] = "robot-controllers/results-0"
    with pytest.raises(ValueError, match="results-1"):
        tracking.records(results)


# ── config ──────────────────────────────────────────────────────────────────────────────────────────────────


def test_config_carries_box_controller_rung_timing_and_provenance(results):
    record = by_name(tracking.records(results, sha="logger123", dirty=True))[(JIH, R3)]
    config = record["config"]
    assert config["box"] == "JointTrajectory → ServoCommand"
    assert config["box_slug"] == "jointtrajectory-to-servocommand"
    assert (config["input_type"], config["output_type"]) == ("JointTrajectory", "ServoCommand")
    assert (config["controller"], config["runs_on"], config["variant_of"]) == (JIH, "host", "joint_impedance_board")
    assert config["param/location"] == "host"
    assert config["param/stiffness_nm_rad"] == [40.0, 25.0]
    assert config["rung"] == R3 and config["rung_family"] == "contact" and config["rung_order"] == 4
    assert config["rung_param/touch_speed_m_s"] == 0.1
    assert config["timing/rate_hz"] == 100.0 and config["host_rate_hz"] == 100.0
    assert config["scene/belt"] is True
    assert config["objective_metric"] == "contact_max_n" and config["minimize"] is True
    # the numbers' provenance is the ledger's; the logger's own checkout is kept beside it
    assert (config["commit"], config["dirty"]) == ("101230a4f084", False)
    assert (config["logger_commit"], config["logger_dirty"]) == ("logger123", True)
    assert config["plant"] == "planar_arm" and config["robot_msgs"] == "0.2.0 @ fixture"
    assert config["evaluation_state"] == "latest" and config["seed"] == 0
    json.dumps(config)  # everything JSON-able (W&B config)


# ── summary ─────────────────────────────────────────────────────────────────────────────────────────────────


def test_summary_holds_every_metric_and_the_verdict(results):
    batch = by_name(tracking.records(results))
    for record in batch.values():
        assert {m.name for m in METRICS} <= set(record["summary"])
    good, bad = batch[(JTC, R1)]["summary"], batch[(JIH, R3)]["summary"]
    assert good["tracking_p95_deg"] == 0.62 and good["contact_peak_n"] is None  # not measured stays None, never 0
    assert (good["passed"], good["objective"], good["gates_passed"], good["gates_total"]) == (1, 0.62, 6, 6)
    assert (bad["passed"], bad["objective"], bad["gates_passed"], bad["gates_total"]) == (0, 72.4, 6, 7)
    assert good["best_on_rung"] == 1 and bad["best_on_rung"] == 0
    assert good["objective_rank"] == 1 and bad["objective_rank"] == 2


def test_gate_lines_say_pass_fail_or_not_measured(results):
    batch = by_name(tracking.records(results))
    assert batch[(JTC, R1)]["summary"]["gate/tracking_p95_deg"] == "pass 0.62 <= 1.0"
    assert batch[(JIH, R1)]["summary"]["gate/tracking_p95_deg"] == "FAIL 1.84 <= 1.0"
    assert batch[(JIH, R3)]["summary"]["gate/contact_peak_n"] == "FAIL 72.4 <= 60.0"
    assert batch[(JTC, R3)]["summary"]["gate/dwell_contact_fraction"] == "pass 99.2 >= 95.0"
    assert batch[(JTC, R3)]["summary"]["gate/missed_cycles"] == "pass 0.0 == 0.0"
    assert set(k for k in batch[(JTC, R3)]["summary"] if k.startswith("gate/")) == {
        f"gate/{g['metric']}" for g in results["rungs"][1]["gates"]}
    unmeasured = {"metric": "dwell_bounces", "op": "==", "value": 0.0, "measured": None, "passed": False, "note": "not measured"}
    assert tracking.gate_text(unmeasured) == "not measured"
    refused = {"metric": "replans_accepted", "op": ">=", "value": 8, "measured": 0, "passed": False,
               "note": "unsupported: refuses replanning"}
    assert tracking.gate_text(refused) == "FAIL 0 >= 8 (unsupported: refuses replanning)"


def test_gate_text_reads_the_verdict_not_the_comparison():
    """The ledger judged the gate; the text must never re-judge it (a mutation that compares would flip this)."""
    row = {"metric": "tracking_p95_deg", "op": "<=", "value": 1.0, "measured": 0.5, "passed": False, "note": ""}
    assert tracking.gate_text(row).startswith("FAIL")


def test_a_nan_metric_becomes_none_in_the_summary(results):
    results["results"][0]["metrics"]["tracking_rms_deg"] = math.nan
    record = tracking.records(results)[0]
    assert record["summary"]["tracking_rms_deg"] is None
    assert all("tracking_rms_deg" not in row for row in record["history"])


# ── history ─────────────────────────────────────────────────────────────────────────────────────────────────


def test_history_puts_each_series_at_its_own_sample_times(results):
    record = by_name(tracking.records(results))[(JTC, R3)]
    trace = next(r["trace"] for r in results["results"] if r["controller"] == JTC and r["rung"] == R3)
    trace_rows = [row for row in record["history"] if "t_s" in row]
    times = [row["t_s"] for row in trace_rows]
    assert times == sorted(times) and len(times) == len(set(times))
    contact_times = [row["t_s"] for row in trace_rows if "contact/fn_n" in row]
    assert contact_times == trace["contact_t"]  # the 1 kHz window keeps its own clock
    assert [row["t_s"] for row in trace_rows if "cost/compute_us" in row] == trace["tick_t"]
    assert [row["t_s"] for row in trace_rows if "q_deg/shoulder" in row] == trace["t"]
    at_peak = next(row for row in trace_rows if row["t_s"] == 0.022)
    assert at_peak == {"t_s": 0.022, "contact/fn_n": 48.3}  # no plant sample at 22 ms, so nothing else there
    assert record["skipped_series"] == []


def test_history_drops_none_and_nan(results):
    target = next(r for r in results["results"] if r["controller"] == JTC and r["rung"] == R1)
    assert target["trace"]["err_deg.elbow"][0] is None  # the fixture's hole
    target["trace"]["tau_nm.elbow"][3] = math.nan
    record = by_name(tracking.records(results))[(JTC, R1)]
    for row in record["history"]:
        assert all(v is not None and math.isfinite(v) for v in row.values())
    first = next(row for row in record["history"] if row.get("t_s") == 0.0)
    assert "err_deg/elbow" not in first and "err_deg/shoulder" in first
    fourth = next(row for row in record["history"] if row.get("t_s") == 0.015)
    assert "tau_nm/elbow" not in fourth and "tau_nm/shoulder" in fourth


def test_history_ends_with_the_kpis_once(results):
    record = by_name(tracking.records(results))[(JIH, R3)]
    last = record["history"][-1]
    assert "t_s" not in last
    assert last["contact_peak_n"] == 72.4 and last["passed"] == 0 and last["objective"] == 72.4
    assert not any(key.startswith("gate/") for key in last)
    assert sum(1 for row in record["history"] if "contact_peak_n" in row) == 1


def test_a_series_off_its_clock_is_skipped_and_named():
    rows, skipped = tracking.history_rows({"t": [0.0, 0.1], "q_deg.a": [1.0, 2.0], "q_deg.b": [1.0], "compute_us": [3.0]})
    assert skipped == ["compute_us", "q_deg.b"]
    assert rows == [{"t_s": 0.0, "q_deg/a": 1.0}, {"t_s": 0.1, "q_deg/a": 2.0}]
    assert tracking.history_key("tool_x_mm") == "tool_mm/x"
    assert tracking.history_key("err_deg.elbow") == "err_deg/elbow"


# ── what push and retire decide (their W&B calls are exercised by the live smoke run) ───────────────────────


def test_an_existing_run_is_patched_only_with_what_it_lacks_or_holds_differently(results):
    summary = tracking.records(results)[0]["summary"]
    have = {k: v for k, v in summary.items() if v is not None}
    assert tracking.summary_patch(summary, have) == {}
    have["passed"] = 0.0  # W&B hands numbers back as floats: 0.0 vs 1 differs, 1.0 vs 1 does not
    have["tracking_p95_deg"] = 0.62 + 1e-12
    del have["gate/tracking_p95_deg"]
    assert tracking.summary_patch(summary, have) == {"passed": 1, "gate/tracking_p95_deg": "pass 0.62 <= 1.0"}


def test_only_older_latest_runs_of_the_same_controller_rung_and_seed_are_superseded(results):
    batch = tracking.records(results)
    first = batch[0]["config"]

    def run(run_id, **config):
        return SimpleNamespace(id=run_id, config={"evaluation_state": "latest", "box_slug": first["box_slug"],
                                                  "controller": first["controller"], "rung": first["rung"], "seed": 0, **config})

    current, older = run(batch[0]["id"]), run("0123456789abcdef")
    other_seed, other_rung = run("1111111111111111", seed=1), run("2222222222222222", rung="R9.Nope")
    already = run("3333333333333333", evaluation_state="superseded")
    assert tracking.to_supersede(batch, [current, older, other_seed, other_rung, already]) == [older]


# ── views (planned offline) ─────────────────────────────────────────────────────────────────────────────────


def test_view_plan_covers_the_four_reports_every_rung_and_the_hub(results):
    plan = tracking.view_plan({results["box"]["slug"]: results})
    keys = [item["key"] for item in plan]
    slug = results["box"]["slug"]
    assert keys == [f"{slug}/scoreboard", f"{slug}/performance-vs-cost", f"{slug}/contact", f"{slug}/timing",
                    f"{slug}/rung/{R1}", f"{slug}/rung/{R3}", "hub/controllers"]
    assert {i["project"] for i in plan if i["key"] != "hub/controllers"} == {JOINT_TRAJECTORY.wandb_project}
    assert plan[-1]["project"] == "Controllers"
    scratch = tracking.view_plan({slug: results}, project="Controller · scratch")
    assert {i["project"] for i in scratch} == {"Controller · scratch"}


def test_saved_view_names_hold_no_symbol_characters(results):
    """wandb-workspaces refuses a view name with a So/Sk/Sm/Sc character ("→" is Sm)."""
    names = [i["title"] for i in tracking.view_plan({results["box"]["slug"]: results}) if i["kind"] == "view"]
    names.append(tracking.view_name("JointTrajectory → ServoCommand × ✓ R7·host500Hz"))
    for name in names:
        assert not any(unicodedata.category(ch).startswith(("So", "Sk", "Sm", "Sc", "Cs")) for ch in name), name
    assert names[-1] == "JointTrajectory to ServoCommand x R7·host500Hz"


def test_scoreboard_marks_pass_fail_and_bolds_the_ledgers_best(results):
    text = tracking.scoreboard_markdown(results)
    lines = text.splitlines()
    assert lines[0].startswith("| Controller | Runs on | 1. R1.FreeSpace | 4. R3.BeltImpact@0.10m/s")
    jtc = next(line for line in lines if line.startswith(f"| {JTC}"))
    jih = next(line for line in lines if line.startswith(f"| {JIH}"))
    assert "**✓ 0.62°**" in jtc and "**✓ 51.7 N**" in jtc and jtc.endswith("| 2/2 |")
    assert "✗ 1.84° · fails tracking_p95_deg" in jih and "✗ 72.4 N · fails contact_peak_n" in jih and "**" not in jih
    assert "joint_trajectory_controller" in tracking.recommendation_markdown(results)
    assert "`R3.BeltImpact@0.10m/s`: `joint_trajectory_controller`" in tracking.pareto_markdown(results)


def test_browser_urls_are_wandb_ai_and_clickable():
    url = tracking.browser_url("https://forge.coreweave.com/wandb/zach-yamaoka-independent/Controller · scratch?nw=abc")
    assert url == "https://wandb.ai/zach-yamaoka-independent/Controller%20%C2%B7%20scratch?nw=abc"
    assert tracking.browser_url(url) == url  # already encoded: unchanged
    assert tracking.browser_url("https://forge.coreweave.com/e/p/reports/T--Vmlldz==").startswith("https://wandb.ai/e/p/")


def test_the_committed_registry_is_json(results):
    assert isinstance(json.loads(tracking.REGISTRY.read_text()), dict)


# ── Weave grouping (bbo loaded by file) ─────────────────────────────────────────────────────────────────────


def test_bbo_loads_by_file_without_the_bam_trackio_package():
    if not tracking.BBO.exists():
        pytest.skip(f"bam_trackio's bbo.py is not at {tracking.BBO}")
    bbo = tracking.load_bbo()
    assert "bam_trackio" not in sys.modules
    assert sys.modules["robot_controllers_tracking_bbo"] is bbo
    assert bbo.ScoredRun.__dataclass_fields__.keys() >= {"black_box", "environment", "episodes", "summary"}


def test_weave_groups_are_rungs_with_their_gates_as_scorers(results):
    bbo = SimpleNamespace(Episode=lambda **kw: SimpleNamespace(**kw), ScoredRun=lambda **kw: SimpleNamespace(**kw))
    groups = tracking.weave_groups(bbo, results, tracking.records(results))
    assert list(groups) == [R1, R3]
    runs, scorers = groups[R3]
    assert scorers == ["objective", "passed"] + [f"gate_{g['metric']}" for g in results["rungs"][1]["gates"]]
    host = next(r for r in runs if r.black_box == JIH)
    assert host.environment == R3 and host.summary["passed"] == 0 and host.summary["contact_peak_n"] == 72.4
    (episode,) = host.episodes
    assert episode.scores["gate_contact_peak_n"] is False and episode.scores["gate_dwell_bounces"] is True
    assert episode.scores["objective"] == 72.4 and episode.inputs == {"seed": 0}
    again = tracking.weave_groups(bbo, results, tracking.records(results))[R3][0]
    assert [r.attributes["run_id"] for r in again] == [r.attributes["run_id"] for r in runs]  # logged once


# ── CLI offline paths ───────────────────────────────────────────────────────────────────────────────────────


def test_dry_run_prints_projects_and_records_without_wandb():
    code = ("import sys; from robot_controllers.tracking.wandb_log import main; "
            f"rc = main(['--results', {str(FIXTURE)!r}, '--dry-run']); "
            "print('IMPORTED', sorted(m for m in ('wandb', 'weave', 'wandb_workspaces') if m in sys.modules)); sys.exit(rc)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    lines = out.stdout.splitlines()
    assert lines[-1] == "IMPORTED []"
    rows = [json.loads(line) for line in lines[:-1]]
    assert rows[1]["project"] == "Controller · JointTrajectory to ServoCommand" and rows[1]["runs"] == 4
    records = [r for r in rows if "id" in r]
    assert len(records) == 4 and all("history" not in r and r["history_rows"] > 10 for r in records)
    assert {r["key"] for r in rows if "key" in r} >= {"hub/controllers"}


def test_logging_refuses_results_made_from_a_dirty_tree(results, tmp_path, capsys):
    results["generated"]["dirty"] = True
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results))
    with pytest.raises(SystemExit) as stopped:
        tracking.main(["--results", str(path)])
    assert stopped.value.code == 2
    assert "results made from a dirty tree" in capsys.readouterr().err


def test_write_back_records_the_links_in_the_ledger(results, tmp_path):
    path = tmp_path / "results.json"
    batch = by_name(tracking.records(results))
    logged = [{"id": batch[(JTC, R1)]["id"], "url": "https://wandb.ai/e/p/runs/x"}]
    reports = {"jointtrajectory-to-servocommand/scoreboard": "https://wandb.ai/e/p/reports/S--1",
               "jointtrajectory-to-servocommand/rung/R1.FreeSpace": "https://wandb.ai/e/p?nw=1",
               "hub/controllers": "https://wandb.ai/e/Controllers/reports/C--2"}
    tracking.write_back(path, results, entity="zach-yamaoka-independent", project=JOINT_TRAJECTORY.wandb_project,
                        logged=logged, reports=reports)
    saved = json.loads(path.read_text())
    assert saved["wandb"]["project_url"] == ("https://wandb.ai/zach-yamaoka-independent/"
                                             "Controller%20%C2%B7%20JointTrajectory%20to%20ServoCommand")
    assert saved["wandb"]["reports"] == {"scoreboard": "https://wandb.ai/e/p/reports/S--1",
                                         "hub": "https://wandb.ai/e/Controllers/reports/C--2"}
    urls = {(r["controller"], r["rung"]): r["wandb_url"] for r in saved["results"]}
    assert urls[(JTC, R1)] == "https://wandb.ai/e/p/runs/x" and urls[(JIH, R3)] is None
    assert [r["id"] for r in tracking.records(saved)] == [r["id"] for r in tracking.records(results)]  # ids survive
