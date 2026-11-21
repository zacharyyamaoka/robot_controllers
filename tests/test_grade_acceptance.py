"""Acceptance for CONTRACT amendment 5, lane B2: fixed targets, scenarios, the population, collision, the grader,
`recommend()`, and real logs.

Written by the dispatcher BEFORE the build and read-only to the builders (CONTRACT §5.10). The real-log check is pinned
to the dispatcher's own reading of the first real `fullarm_jtc` log (2026-11-20: j2's p95 against the plan's reference at
the JTC's clock, 0.664°), made with a scratch script before any of this code existed.

Zach, 2026-11-20: "once you tell me what driver your using, and the state of your FF/FB and the host rate etc ... I can
tell you exactly what the best controller is you should be using. And perhaps if you want like to really optimize for
less impact, or better collision detection".
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robot_controllers.models import DriverCaps, ModelError

REPO = Path(__file__).resolve().parents[1]
REAL_RUN = Path(
    "/home/bam/arm_motion_bench/outputs/runs_real/bam_fb/fullarm_jtc_model_ff_la_restload/workspace-v2/dev/"
    "run_jtc-20261121T021137Z-3610132"
)
REAL_BUNDLE = Path("/home/bam/.local/state/rack/submissions/1120-191044-1d2b/bundle")
ARMS = ("planar_arm", "ur5e", "bam_fivebar", "pendulum")


def episode(spec, rung, seed=0):
    from robot_controllers.bench.loop import run_episode
    from robot_controllers.metrics.episode import episode_metrics

    log = run_episode(spec, rung, seed=seed)
    return log, episode_metrics(log, rung)


# ── 5.4 targets and scenarios ────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("arm_name", ARMS)
def test_every_arm_has_three_speeds_of_one_path_a_calibration_and_a_collision(arm_name):
    from robot_controllers.bench.arms import arm

    spec = arm(arm_name)
    targets = {t.id: t for t in spec.targets()}
    paths = [targets[f"{arm_name}/path@{s}"] for s in ("slow", "nominal", "fast")]
    trajs = [p.rung().inputs(np.asarray(spec.home_q))[0][1] for p in paths]
    durations = [float(tr.t[-1]) for tr in trajs]
    assert durations[0] > durations[1] > durations[2]
    phase = np.linspace(0.0, 1.0, 41)
    shapes = [
        np.stack([np.interp(phase * float(tr.t[-1]), tr.t, tr.positions[:, j]) for j in range(tr.positions.shape[1])], 1)
        for tr in trajs
    ]
    np.testing.assert_allclose(shapes[0], shapes[2], atol=1e-6)
    assert paths[0].params["peak_joint_speed_deg_s"] < paths[2].params["peak_joint_speed_deg_s"]
    calib = targets[f"{arm_name}/calibration"].rung().inputs(np.asarray(spec.home_q))[0][1]
    assert np.all(np.degrees(np.ptp(calib.positions, axis=0)) >= 5.0)
    assert targets[f"{arm_name}/collision"].kind == "collision"


@pytest.mark.parametrize("arm_name", ARMS)
def test_the_base_scenario_is_the_rig_rate_a_datasheet_bus_and_a_ten_percent_model(arm_name):
    from robot_controllers.bench.scenarios import base_scenario, grid

    base = base_scenario(arm_name)
    assert base.host_rate_hz == 50.0 and base.gain_scale == 1.0 and base.axis == "base"
    assert base.model_error == ModelError(0.1)
    assert base.target == f"{arm_name}/path@nominal"
    assert base.bus.source.startswith("datasheet")
    from robot_controllers.bench.arms import arm

    joints = len(arm(arm_name).joint_names)
    if len(base.bus.per_joint(joints)) > 1:
        assert len(set(base.bus.command_delay_s)) > 1  # staggered: the boards are served in turn
    scenarios = grid(arm_name)
    assert scenarios[0] == base
    assert len({s.id for s in scenarios}) == len(scenarios)
    assert {s.host_rate_hz for s in scenarios} >= {25.0, 50.0, 100.0, 200.0}
    assert {s.model_error.scale for s in scenarios} >= {0.0, 0.1, 0.3}
    assert {s.gain_scale for s in scenarios} >= {0.5, 1.0, 2.0}
    assert {s.target.split("/")[1] for s in scenarios} >= {"path@slow", "path@nominal", "path@fast", "collision"}
    assert any(s.bus.jitter_s > 0 for s in scenarios)
    assert "50Hz" in base.id and "ff±10%" in base.id


# ── 5.5 the population ───────────────────────────────────────────────────────────────────────────────────────


def test_the_population_is_the_contracts_sixteen_and_only_board_side_configs_go_real():
    from robot_controllers.bench.population import POPULATION, REAL_SAFE, config_spec
    from robot_controllers.bench.scenarios import base_scenario

    assert tuple(POPULATION) == (
        "P1", "P2", "P2s", "P3", "P3s", "P4", "P4s", "P5", "P5s", "P6", "P6s", "P7", "P7s", "P8", "P9", "P10",
    )
    assert tuple(REAL_SAFE) == ("P2", "P4", "P4s", "P5s")
    base = base_scenario("planar_arm")
    layouts = {c: config_spec(c, base).default_config["layout"] for c in POPULATION}
    assert layouts["P10"].feedback_at == "host" and layouts["P9"].feedback_at == "host"
    assert layouts["P8"].feedback_at == "split"
    assert layouts["P4s"].interpolation == "secant_mid_ff" and layouts["P4"].interpolation == "tangent"
    assert all(layouts[c].feedback_at in ("board", "board_scheduled") for c in REAL_SAFE)
    np.testing.assert_allclose(layouts["P3"].lookaheads(2), base.bus.per_joint(2), rtol=0, atol=1e-12)  # nominal = the datasheet bus
    assert all(v == 0.0 for v in layouts["P2"].lookaheads(2))


# ── 5.6 collision ────────────────────────────────────────────────────────────────────────────────────────────


def test_collision_metrics_exist_and_a_feedforward_makes_the_stop_stand_out():
    from robot_controllers.bench.population import config_spec
    from robot_controllers.bench.scenarios import grid
    from robot_controllers.metrics.names import info

    for name, unit in (("collision_detect_ms", "ms"), ("collision_peak_nm", "N·m"), ("collision_margin", "")):
        assert info(name).unit == unit
    collision = next(s for s in grid("planar_arm") if s.target.endswith("/collision") and s.axis == "event")
    from robot_controllers.grade import run_scenario

    rows = {c: run_scenario(c, collision) for c in ("P1", "P4s")}
    for row in rows.values():
        detect = row["metrics"]["collision_detect_ms"]
        assert detect is None or detect >= 0.0
        assert row["metrics"]["collision_peak_nm"] is not None and row["metrics"]["collision_peak_nm"] > 0.0
    assert rows["P4s"]["metrics"]["collision_margin"] >= rows["P1"]["metrics"]["collision_margin"]


# ── 5.7 recommend ────────────────────────────────────────────────────────────────────────────────────────────


def _bus(n):
    from robot_controllers.timing import BusTiming

    return BusTiming.from_datasheet(0.0155, 7, slot_order=tuple(range(n)), source="datasheet:test")


def test_recommend_answers_the_rig_question_with_board_feedback_and_a_lookahead():
    from robot_controllers.recommend import Setup, recommend

    answer = recommend(
        Setup(arm="planar_arm", host_rate_hz=50.0, bus=_bus(2), feedforward="rigid_body", model_error=0.1),
        objective="accuracy",
    )
    layout = answer.controller.default_config["layout"]
    assert layout.feedback_at in ("board", "board_scheduled")
    assert min(layout.lookaheads(2)) > 0.0
    assert answer.expected["tracking_p95_deg"] is not None and answer.why and answer.evidence


def test_recommend_respects_what_the_driver_cannot_do_and_says_so_for_an_unknown_arm():
    from robot_controllers.recommend import Setup, recommend

    no_board_pd = recommend(
        Setup(arm="ur5e", host_rate_hz=100.0, bus=_bus(6), feedforward="rigid_body", model_error=0.1,
              driver_caps=DriverCaps(board_pd=False)),
    )
    if no_board_pd.controller is not None:
        assert no_board_pd.controller.default_config["layout"].feedback_at == "host"
    assert no_board_pd.why
    other = recommend(
        Setup(arm="some_new_arm", joint_count=6, host_rate_hz=50.0, bus=_bus(6), feedforward="rigid_body",
              model_error=0.1),
    )
    assert other.controller is not None and "nearest" in other.why.lower()


@pytest.mark.parametrize("objective", ["accuracy", "impact", "collision_detection", "cost"])
def test_every_objective_gives_an_answer_on_the_ledger(objective):
    from robot_controllers.recommend import Setup, recommend

    answer = recommend(
        Setup(arm="bam_fivebar", host_rate_hz=50.0, bus=_bus(6), feedforward="rigid_body", model_error=0.1),
        objective=objective,
    )
    assert answer.controller is not None and answer.config


# ── 5.9 real logs ────────────────────────────────────────────────────────────────────────────────────────────


def test_a_real_fullarm_jtc_log_imports_and_scores_like_a_twin_episode():
    if not (REAL_RUN.exists() and REAL_BUNDLE.exists()):
        pytest.skip("the 2026-11-20 real run is not on this machine")
    from robot_controllers.real.import_fullarm import import_fullarm_run

    log, rung = import_fullarm_run(REAL_RUN, REAL_BUNDLE)
    assert log.joint_names == ("j1", "j2", "c1", "w1", "w2", "w3")
    from robot_controllers.metrics.episode import episode_metrics

    metrics = episode_metrics(log, rung)
    assert metrics["tracking_p95_deg"] == pytest.approx(0.664, rel=0.05)


# ── the ledger ───────────────────────────────────────────────────────────────────────────────────────────────


def test_the_committed_grade_ledger_covers_every_arm_axis_and_config():
    ledger = json.loads((REPO / "results" / "grade" / "results.json").read_text())
    assert ledger["schema"] == "robot-controllers/grade-1"
    rows = [r for r in ledger["rows"] if r.get("source", "sim") == "sim"]
    for arm_name in ARMS:
        axes = {r["axis"] for r in rows if r["arm"] == arm_name}
        assert axes >= {"base", "speed", "rate", "model_error", "gains", "bus", "event"}, (arm_name, axes)
        configs = {r["config"] for r in rows if r["arm"] == arm_name}
        assert len(configs) == 16, (arm_name, sorted(configs))
        assert arm_name in ledger["summary"]
