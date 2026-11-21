"""Lane B2's own tests (CONTRACT amendment 5, §5.4-5.7, §5.9): the collision arithmetic, scenarios, the population,
ranking, recommend on a hand-built ledger, the resumable arm-powered runner, and the real-log import."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from robot_controllers.models import DriverCaps, ModelError
from robot_controllers.timing import BusTiming

REAL_RUN = Path(
    "/home/bam/arm_motion_bench/outputs/runs_real/bam_fb/fullarm_jtc_model_ff_la_restload/workspace-v2/dev/"
    "run_jtc-20261121T021137Z-3610132"
)
REAL_BUNDLE = Path("/home/bam/.local/state/rack/submissions/1120-191044-1d2b/bundle")


# ── collision ────────────────────────────────────────────────────────────────────────────────────────────────


def test_collision_detects_the_first_crossing_after_contact_and_keeps_the_peak_before_it():
    from robot_controllers.metrics.collision import collision_from_arrays

    t = np.arange(10) * 0.001
    residual = np.array([0.9, 0.5, 0.4, 0.6, 0.8, 1.3, 2.0, 1.0, 0.2, 0.1])  # 0.9 before contact never counts
    tau_stop = np.zeros((10, 2))
    tau_stop[3:8, 1] = [-1.0, -5.0, -20.0, -3.0, -1.0]
    out = collision_from_arrays(t, residual, tau_stop, threshold=1.0)
    assert out["collision_detect_ms"] == pytest.approx(2.0)  # contact at 3 ms, crossing at 5 ms
    assert out["collision_peak_nm"] == pytest.approx(20.0)  # |tau_stop| up to and including the detecting sample
    assert out["collision_margin"] == pytest.approx(2.0)


def test_collision_never_detected_is_none_and_the_peak_covers_the_whole_contact():
    from robot_controllers.metrics.collision import collision_from_arrays

    t = np.arange(5) * 0.001
    tau_stop = np.array([[0.0], [2.0], [7.0], [1.0], [0.0]])
    out = collision_from_arrays(t, np.full(5, 0.5), tau_stop, threshold=1.0)
    assert out["collision_detect_ms"] is None
    assert out["collision_peak_nm"] == pytest.approx(7.0)
    assert out["collision_margin"] == pytest.approx(0.5)
    untouched = collision_from_arrays(t, np.full(5, 0.5), np.zeros((5, 1)), threshold=1.0)
    assert all(v is None for v in untouched.values())


def test_the_residual_is_the_low_passed_feedback_over_each_joints_cap():
    from robot_controllers.metrics.collision import residual

    t = np.arange(4000) * 0.00025
    board = np.column_stack([np.full(4000, 2.0), np.full(4000, -3.0)])
    out = residual(t, board, np.zeros_like(board), caps_nm=(10.0, 6.0))
    assert out[-1] == pytest.approx(0.5, rel=1e-6)  # joint 2: |-3| / 6 after the filter settles
    assert out[0] == pytest.approx(0.5)  # the filter starts at the first sample
    with pytest.raises(ValueError):
        residual(t, board, np.zeros_like(board), caps_nm=(10.0,))


# ── scenarios ────────────────────────────────────────────────────────────────────────────────────────────────


def test_scenario_ids_name_every_coordinate_and_the_bus_axis_runs_three_buses():
    from robot_controllers.bench.scenarios import bus_label, grid, rig_bus

    scenarios = grid("planar_arm")
    assert scenarios[0].id == "planar_arm/path@nominal·50Hz·bus=datasheet·ff±10%·gains×1"
    buses = {bus_label(s.bus) for s in scenarios if s.axis == "bus"}
    assert buses == {"5.5ms", "stagger", "stagger±1ms"}
    rig = rig_bus("ur5e")
    np.testing.assert_allclose(np.diff(rig.per_joint(6)), 0.0155 / 7, rtol=1e-12)
    rate_error = [s for s in scenarios if s.axis == "rate_error"]
    assert len(rate_error) == 6  # 4 x 3 less the six points the base, rate and model_error axes already have
    belt = next(s for s in scenarios if s.kind == "contact")
    assert belt.rung().scene.belt and belt.rung().timing.rate_hz == 50.0


def test_a_scenario_rung_runs_its_rate_bus_and_a_collision_gate():
    from robot_controllers.bench.scenarios import grid

    scenarios = grid("ur5e")
    jitter = next(s for s in scenarios if s.axis == "bus" and s.bus.jitter_s)
    rung = jitter.rung()
    assert rung.id == jitter.id and rung.timing.rate_hz == 50.0
    assert rung.timing.command_delay_jitter_s == pytest.approx(0.001)
    assert rung.timing.command_delays(6) == jitter.bus.per_joint(6)
    assert any(g.metric == "latency_p99_ms" and g.value == 20.0 for g in rung.gates)
    collision = next(s for s in scenarios if s.kind == "collision")
    assert any(g.metric == "collision_detect_ms" for g in collision.rung().gates)
    assert collision.free().target == "ur5e/path@nominal"


def test_the_effort_column_carries_the_scenarios_model_error_and_nothing_else():
    from robot_controllers.bench.arms import arm
    from robot_controllers.bench.scenarios import base_scenario

    base = base_scenario("planar_arm")
    exact = dataclasses.replace(base, model_error=ModelError())
    home = np.asarray(arm("planar_arm").home_q)
    given = base.rung().inputs(home)[0][1]
    nominal = exact.rung().inputs(home)[0][1]
    original = base.base_rung().inputs(home)[0][1]
    assert nominal is original  # ModelError() leaves the target's own script
    np.testing.assert_array_equal(given.positions, nominal.positions)
    np.testing.assert_array_equal(given.velocities, nominal.velocities)
    assert not np.allclose(given.effort, nominal.effort)
    expected = arm("planar_arm").feedforward("rigid_body", ModelError(0.1))(
        given.positions, given.velocities, given.accelerations
    )
    np.testing.assert_allclose(given.effort, expected)


def test_gains_scale_the_plants_boards():
    from robot_controllers.bench.scenarios import grid

    soft = next(s for s in grid("planar_arm") if s.axis == "gains" and s.gain_scale == 0.5)
    plant = soft.plant_factory()()
    assert plant.board_config.kp_nm_rad == (150.0, 100.0)
    assert grid("planar_arm")[0].plant_factory() is None


# ── the population ───────────────────────────────────────────────────────────────────────────────────────────


def test_the_lookahead_modes_and_a_nominal_override():
    from robot_controllers.bench.population import CONFIGS, config_spec, lookahead_for
    from robot_controllers.bench.scenarios import base_scenario

    base = base_scenario("planar_arm")
    calibration = {"lookahead_s": (0.004, 0.006)}
    assert lookahead_for("P4", base, calibration) == (0.004, 0.006)
    assert lookahead_for("P1", base, calibration) == (0.004, 0.006)
    assert lookahead_for("P8", base, calibration) == 0.0
    mine = BusTiming((0.002, 0.003), source="assumed")
    assert lookahead_for("P3s", base, calibration, nominal=mine) == (0.002, 0.003)
    spec = config_spec("P7s", base, calibration)
    layout = spec.default_config["layout"]
    assert (layout.feedforward_at, layout.feedback_at, layout.interpolation) == ("plan", "board_scheduled",
                                                                                 "secant_mid_ff")
    assert CONFIGS["P5"].feedforward_needed == "rigid_body_coulomb"
    assert CONFIGS["P2"].feedforward_needed == "trajectory"


def test_calibrated_reads_the_friction_aware_bus_delay_and_the_row_keeps_both():
    from robot_controllers.bench.population import calibration_for, lookahead_for
    from robot_controllers.bench.scenarios import base_scenario

    base = base_scenario("planar_arm")
    record = calibration_for(base)
    assert record["lookahead"] == "bus_delay" and record["lookahead_s"] == record["bus_delay_s"]
    assert lookahead_for("P4s", base) == record["bus_delay_s"]
    assert record["best_lookahead_ms"] and record["datasheet_ms"] == [1.25, 2.5]


# ── ranking and recommend on a hand-built ledger ────────────────────────────────────────────────────────────


def _row(arm, scenario, config, *, kind="path", axis="base", rate=50.0, error=0.1, passed=True, bus_ms=(1.25, 2.5),
         jitter=0.0, **metrics):
    from robot_controllers.bench.population import CONFIGS

    member = CONFIGS[config]
    return {
        "arm": arm, "scenario": scenario, "axis": axis, "target": f"{arm}/path@nominal" if kind == "path"
        else f"{arm}/collision", "kind": kind, "speed": "nominal", "host_rate_hz": rate,
        "bus": {"label": "x", "command_delay_ms": list(bus_ms), "reply_delay_ms": 1.5, "jitter_ms": jitter,
                "source": "datasheet"},
        "model_error": error, "gain_scale": 1.0, "joints": len(bus_ms), "config": config,
        "controller": config, "feedforward": f"{member.ff_model}@{member.ff_at}",
        "feedback": f"{member.fb_model}@{member.fb_at}", "interpolation": member.interpolation,
        "lookahead_mode": member.lookahead_mode, "metrics": metrics, "passed": passed, "failed_gates": [] if passed
        else ["tracking_p95_deg <= 3 (9)"],
        "calibration": {"lookahead_ms": [2.0] * len(bus_ms), "twin_bus_ms": list(bus_ms)}
        if member.lookahead_mode == "calibrated" else None,
        "source": "sim",
    }


def _ledger():
    rows = [
        _row("planar_arm", "base", "P4s", tool_p95_mm=0.98, tracking_p95_deg=0.3, host_duty_pct=0.5),
        _row("planar_arm", "base", "P5", tool_p95_mm=0.9, tracking_p95_deg=0.3, host_duty_pct=0.4),
        _row("planar_arm", "base", "P2", tool_p95_mm=1.05, tracking_p95_deg=0.4, host_duty_pct=0.2),
        _row("planar_arm", "base", "P10", tool_p95_mm=0.5, tracking_p95_deg=0.2, host_duty_pct=0.1, passed=False),
        _row("planar_arm", "base", "P9", tool_p95_mm=3.0, tracking_p95_deg=1.0, host_duty_pct=0.6),
        _row("planar_arm", "fast200", "P2", rate=200.0, tool_p95_mm=0.1, tracking_p95_deg=0.1, host_duty_pct=1.0),
        _row("planar_arm", "col", "P4s", kind="collision", axis="event", collision_detect_ms=2.0,
             collision_margin=30.0, collision_peak_nm=140.0),
        _row("planar_arm", "col", "P1", kind="collision", axis="event", collision_detect_ms=2.0,
             collision_margin=10.0, collision_peak_nm=150.0),
    ]
    return {"rows": rows}


def test_rank_orders_passing_rows_with_ties_broken_by_the_next_metric_and_higher_better_margins():
    from robot_controllers.grade import rank, within

    rows = _ledger()["rows"]
    base = [r for r in rows if r["scenario"] == "base"]
    assert [r["config"] for r in rank(base, "accuracy")] == ["P5", "P4s", "P2", "P9"]  # P10 failed a gate
    assert within(base, "accuracy") == ["P5", "P4s"]  # 0.9 mm x 1.1 = 0.99
    collision = [r for r in rows if r["scenario"] == "col"]
    assert [r["config"] for r in rank(collision, "collision_detection")] == ["P4s", "P1"]  # same ms, margin wins
    assert [r["config"] for r in rank(base, {"accuracy": 1.0, "cost": 1.0})][0] in ("P5", "P2")


def test_recommend_takes_the_nearest_scenario_and_what_the_setup_can_run():
    from robot_controllers.recommend import Setup, recommend

    ledger = _ledger()
    bus = BusTiming((0.00125, 0.0025), source="datasheet")
    answer = recommend(Setup("planar_arm", 50.0, bus, "rigid_body", 0.1), "accuracy", ledger=ledger)
    assert answer.config == "P4s" and answer.scenario == "base"  # P5 needs the Coulomb term
    assert answer.runner_up == "P2" and "P5" in answer.why
    assert tuple(answer.controller.default_config["layout"].lookaheads(2)) == pytest.approx((0.002, 0.002))
    coulomb = recommend(Setup("planar_arm", 50.0, bus, "rigid_body_coulomb", 0.1), "accuracy", ledger=ledger)
    assert coulomb.config == "P5"
    fast = recommend(Setup("planar_arm", 180.0, bus, "rigid_body", 0.1), "accuracy", ledger=ledger)
    assert fast.scenario == "fast200"
    host_only = recommend(Setup("planar_arm", 50.0, bus, "rigid_body", 0.1, DriverCaps(board_pd=False)),
                          ledger=ledger)
    assert host_only.config == "P9" and host_only.controller.default_config["layout"].feedback_at == "host"
    shifted = recommend(Setup("planar_arm", 50.0, BusTiming((0.00225, 0.0035), source="mine"), "rigid_body", 0.1),
                        ledger=ledger)
    assert tuple(shifted.controller.default_config["layout"].lookaheads(2)) == pytest.approx((0.003, 0.003))
    impact = recommend(Setup("planar_arm", 50.0, bus, "rigid_body", 0.1), "impact", ledger=ledger)
    assert impact.config == "P4s" and impact.scenario == "col"


def test_recommend_says_nearest_for_an_arm_it_has_not_graded_and_none_when_nothing_fits():
    from robot_controllers.recommend import Setup, recommend

    ledger = _ledger()
    bus = BusTiming((0.00125, 0.0025), source="datasheet")
    other = recommend(Setup("two_link_x", 50.0, bus, "rigid_body", 0.1, joint_count=2), ledger=ledger)
    assert other.arm == "planar_arm" and "nearest" in other.why
    nothing = recommend(Setup("planar_arm", 50.0, bus, "none", 0.1, DriverCaps(board_pd=False)), ledger=ledger)
    assert nothing.controller is None and nothing.config is None and nothing.why


# ── the runner ───────────────────────────────────────────────────────────────────────────────────────────────


def test_the_runner_pauses_while_the_arm_is_powered_and_resumes_from_its_partial_file(tmp_path, monkeypatch):
    from robot_controllers import grade

    answers = iter(["test: arm powered"])

    def powered():
        return next(answers, None)

    stopped = grade.run_grade(["pendulum"], workers=1, configs=("P2",), out_dir=tmp_path, wait_for_arm=False,
                              progress=None, powered=powered)
    assert stopped["generated"]["complete"] is False and not stopped["rows"]
    monkeypatch.setattr(grade, "POLL_S", 0.01)
    answers = iter(["test: arm powered", "test: still powered"])
    document = grade.run_grade(["pendulum"], workers=1, configs=("P2",), out_dir=tmp_path, progress=None,
                               powered=powered)
    assert document["generated"]["complete"] is True
    assert len([r for r in document["rows"] if r["source"] == "sim"]) == len(grade.grid("pendulum"))
    lines = (tmp_path / "partial.jsonl").read_text().splitlines()
    assert len(lines) == len(grade.grid("pendulum"))
    again = grade.run_grade(["pendulum"], workers=1, configs=("P2",), out_dir=tmp_path, progress=None,
                            powered=lambda: None)
    assert [r["metrics"] for r in again["rows"] if r["source"] == "sim"] == [
        r["metrics"] for r in document["rows"] if r["source"] == "sim"
    ]
    path = grade.write_grade(again, tmp_path)
    assert json.loads(path.read_text())["schema"] == grade.SCHEMA
    assert not (tmp_path / "partial.jsonl").exists()
    assert "pendulum" in again["summary"]


def test_arm_powered_reads_the_flag(tmp_path, monkeypatch):
    from robot_controllers import grade

    flag = tmp_path / "ARM_POWERED"
    monkeypatch.setattr(grade, "ARM_POWERED_FLAG", flag)
    monkeypatch.setattr(grade.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": "not-rdy p3 x\n"})())
    assert grade.arm_powered() is None
    flag.write_text("")
    assert "ARM_POWERED" in grade.arm_powered()
    flag.unlink()
    monkeypatch.setattr(grade.subprocess, "run",
                        lambda *a, **k: type("R", (), {"stdout": "RUNNING  p1  abc  pick  owner=x\n"})())
    assert grade.arm_powered().startswith("rackq: RUNNING")


# ── real logs and requests ───────────────────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not (REAL_RUN.exists() and REAL_BUNDLE.exists()), reason="the 2026-11-20 real run is not here")
def test_the_real_import_keeps_the_tape_the_board_frame_and_the_boards_limit_flags():
    from robot_controllers.real.import_fullarm import import_fullarm_run

    log, rung = import_fullarm_run(REAL_RUN, REAL_BUNDLE)
    assert log.q.shape == (1523, 6) and log.meta["tape_phase"] == 1.0
    assert not np.any(log.board_mode)  # every board ran in position mode
    assert "j2" in log.meta["limit_flags"]  # CAN 12's limit register, not a fault
    assert rung.timing.rate_hz == pytest.approx(58.75, abs=0.1)
    np.testing.assert_allclose(log.tau_fb_board + log.tau_ff, log.tau, rtol=0, atol=1e-12)


def test_a_run_request_names_the_four_real_safe_configs_in_bundle_terms():
    from robot_controllers.real.requests import run_request

    request = run_request("bam_fivebar", ledger={"rows": []})
    assert request["station"] == "bam_fb" and request["rate_hz"] == 50.0
    assert [c["config"] for c in request["configs"]] == ["P2", "P4", "P4s", "P5s"]
    assert [c["effort"] for c in request["configs"]] == ["as_built", "model_at_plan", "model_at_plan",
                                                          "model_at_plan + coulomb"]
    with pytest.raises(ValueError):
        run_request("bam_fivebar", configs=("P10",))
    with pytest.raises(ValueError):
        run_request("planar_arm")
