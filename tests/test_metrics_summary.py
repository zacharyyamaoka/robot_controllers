"""summarize on a small hand-made results list: scoreboard, best per rung, Pareto set, recommendation."""

import pytest
from test_metrics_helpers import make_rung

from robot_controllers.bench.types import Gate, RunResult, Scene
from robot_controllers.metrics.summary import dominates, pareto_front, summarize
from robot_controllers.registry import JOINT_TRAJECTORY

GATE = {"op": "<=", "value": 1.0, "why": "x", "note": ""}

R1 = make_rung(
    "R1.FreeSpace",
    family="free_space",
    order=1,
    gates=(Gate("tracking_p95_deg", "<=", 1.0, "x"), Gate("faults", "==", 0, "x")),
)
R3 = make_rung(
    "R3.BeltImpact@0.10m/s",
    family="contact",
    order=4,
    scene=Scene(belt=True),
    objective="contact_max_n",
    gates=(
        Gate("contact_peak_n", "<=", 60, "x"),
        Gate("dwell_bounces", "==", 0, "x"),
        Gate("dwell_contact_fraction", ">=", 95, "x"),
    ),
)
R4 = make_rung(
    "R4.SidewaysPush",
    family="disturbance",
    order=6,
    objective="push_recovery_s",
    gates=(Gate("push_recovery_s", "<=", 0.5, "x"),),
)
RUNGS = [R1, R3, R4]


def result(controller, rung, passes, objective, *, tracking=None, contact=None, compute=10.0, seed=0):
    gates = [
        {**GATE, "metric": g.metric, "measured": 0.0, "passed": ok}
        for g, ok in zip(rung.gates, passes, strict=True)
    ]
    metrics = {"tracking_p95_deg": tracking, "contact_max_n": contact, "compute_us_p99": compute}
    return RunResult(controller, rung.id, seed, metrics, gates, all(passes), objective)


def results():
    return [
        # A: passes everything, R1 0.5 deg, R3 30 N, cheap
        result("A", R1, [True, True], 0.5, tracking=0.5, compute=10.0),
        result("A", R3, [True, True, True], 30.0, contact=30.0, compute=12.0),
        result("A", R4, [True], 0.20, compute=11.0),
        # B: passes everything, tracks better (0.3) and touches softer (20 N) but costs more
        result("B", R1, [True, True], 0.3, tracking=0.3, compute=40.0),
        result("B", R3, [True, True, True], 20.0, contact=20.0, compute=42.0),
        result("B", R4, [True], 0.30, compute=41.0),
        # C: fails the belt rung by barely touching (5 N, the "best" number) and the push
        result("C", R1, [True, True], 0.8, tracking=0.8, compute=5.0),
        result("C", R3, [True, False, False], 5.0, contact=5.0, compute=5.0),
        result("C", R4, [False], 0.9, compute=5.0),
        # D: dominated by A on every Pareto axis (worse tracking, harder touch, dearer), passes everything
        result("D", R1, [True, True], 0.6, tracking=0.6, compute=15.0),
        result("D", R3, [True, True, True], 35.0, contact=35.0, compute=15.0),
        result("D", R4, [True], 0.25, compute=15.0),
    ]


def test_per_controller_scoreboard():
    summary = summarize(JOINT_TRAJECTORY, RUNGS, results())
    a, c = summary["per_controller"]["A"], summary["per_controller"]["C"]
    assert (a["rungs_passed"], a["rungs_total"], a["gates_passed"], a["gates_total"]) == (3, 3, 6, 6)
    assert (c["rungs_passed"], c["gates_passed"]) == (1, 3)
    assert a["compute_us_p99"] == pytest.approx(11.0)  # median over rungs
    assert a["objective"]["R3.BeltImpact@0.10m/s"] == 30.0
    assert c["passed"] == {"R1.FreeSpace": True, "R3.BeltImpact@0.10m/s": False, "R4.SidewaysPush": False}


def test_per_rung_best_ignores_controllers_that_failed_the_gates():
    per_rung = summarize(JOINT_TRAJECTORY, RUNGS, results())["per_rung"]
    assert per_rung["R3.BeltImpact@0.10m/s"]["best"] == "B"  # C's 5 N is lower but C failed the rung
    assert per_rung["R3.BeltImpact@0.10m/s"]["best_objective"] == 20.0
    assert per_rung["R1.FreeSpace"]["best"] == "B"
    assert per_rung["R4.SidewaysPush"]["best"] == "A"
    assert per_rung["R3.BeltImpact@0.10m/s"]["passed"] == ["A", "B", "D"]


def test_nobody_passing_means_no_best():
    rows = [result("A", R4, [False], 0.9), result("B", R4, [False], 0.7)]
    assert summarize(JOINT_TRAJECTORY, [R4], rows)["per_rung"]["R4.SidewaysPush"]["best"] is None


def test_rank_puts_passers_first_then_orders_by_objective():
    per_controller = summarize(JOINT_TRAJECTORY, RUNGS, results())["per_controller"]
    belt = "R3.BeltImpact@0.10m/s"
    assert {name: per_controller[name]["rank"][belt] for name in "ABCD"} == {"B": 1, "A": 2, "D": 3, "C": 4}


def test_a_maximized_objective_ranks_high_first():
    rung = make_rung(
        "S9.More",
        family="sequence",
        objective="steps_done",
        minimize=False,
        gates=(Gate("faults", "==", 0, "x"),),
    )
    rows = [result("A", rung, [True], 3), result("B", rung, [True], 5)]
    summary = summarize(JOINT_TRAJECTORY, [rung], rows)
    assert summary["per_rung"]["S9.More"]["best"] == "B"
    assert summary["per_controller"]["B"]["rank"]["S9.More"] == 1


def test_pareto_set_on_tracking_contact_and_compute():
    pareto = summarize(JOINT_TRAJECTORY, RUNGS, results())["pareto"]
    assert list(pareto) == ["R3.BeltImpact@0.10m/s"]  # belt rungs only
    belt = pareto["R3.BeltImpact@0.10m/s"]
    assert belt["front"] == ["A", "B"]  # D is dominated by A; C failed the belt rung
    assert belt["among"] == "passed"
    assert [axis["metric"] for axis in belt["axes"]] == [
        "tracking_p95_deg",
        "contact_max_n",
        "compute_us_p99",
    ]
    assert belt["axes"][0]["rung"] == "R1.FreeSpace"
    point_c = next(p for p in belt["points"] if p["controller"] == "C")
    assert point_c == {
        "controller": "C",
        "tracking_p95_deg": 0.8,
        "contact_max_n": 5.0,
        "compute_us_p99": 5.0,
        "passed": False,
        "front": False,
    }


def test_pareto_front_and_dominance():
    assert dominates((1, 1, 1), (1, 2, 1)) and not dominates((1, 1, 1), (1, 1, 1))
    assert not dominates((0, 2, 1), (1, 1, 1))
    assert pareto_front({"a": (1, 2), "b": (2, 1), "c": (2, 2), "d": (1, 2)}) == ["a", "b", "d"]


def test_recommendation_most_rungs_then_gates_then_cheapest():
    recommendation = summarize(JOINT_TRAJECTORY, RUNGS, results())["recommendation"]
    # A, B and D pass every rung and gate; A is the cheapest (11 us vs 15 and 41)
    assert recommendation["controller"] == "A"
    assert recommendation["rungs_passed"] == 3 and recommendation["gates_passed"] == 6
    assert recommendation["compute_us_p99"] == pytest.approx(11.0)
    assert "A passes 3/3 rungs and 6/6 gates at 11.0 µs" in recommendation["why"]
    assert "D ties on rungs and gates at 15.0 µs" in recommendation["why"]
    assert "\n" not in recommendation["why"]


def test_recommendation_prefers_rungs_over_compute():
    rows = [r for r in results() if r.controller in ("B", "C")]
    recommendation = summarize(JOINT_TRAJECTORY, RUNGS, rows)["recommendation"]
    assert recommendation["controller"] == "B"  # 3 rungs at 41 us beats 1 rung at 5 us
    assert "next is C with 1/3 rungs" in recommendation["why"]


def test_recommendation_prefers_rungs_over_gates():
    rows = [
        # X: passes R1 and R4, gets none of R3's gates -> 2 rungs, 3 gates
        result("X", R1, [True, True], 0.5),
        result("X", R3, [False, False, False], 70.0),
        result("X", R4, [True], 0.2),
        # Y: passes only R1 but nearly passes R3 -> 1 rung, 4 gates
        result("Y", R1, [True, True], 0.5),
        result("Y", R3, [True, True, False], 30.0),
        result("Y", R4, [False], 0.9),
    ]
    recommendation = summarize(JOINT_TRAJECTORY, RUNGS, rows)["recommendation"]
    assert recommendation["controller"] == "X"
    assert (recommendation["rungs_passed"], recommendation["gates_passed"]) == (2, 3)


def test_seeds_collapse_to_one_cell_that_passes_only_if_every_seed_did():
    rows = [
        result("A", R4, [True], 0.20, seed=0),
        result("A", R4, [False], 0.60, seed=1),
        result("A", R4, [True], 0.30, seed=2),
    ]
    row = summarize(JOINT_TRAJECTORY, [R4], [r.to_dict() for r in rows])["per_controller"]["A"]
    assert row["rungs_passed"] == 0 and row["gates_passed"] == 0
    assert row["objective"]["R4.SidewaysPush"] == pytest.approx(0.30)  # median over seeds


def test_a_rung_that_was_not_run_counts_as_failed_with_its_gates():
    rows = [r for r in results() if r.controller == "A" and r.rung != "R4.SidewaysPush"]
    row = summarize(JOINT_TRAJECTORY, RUNGS, rows)["per_controller"]["A"]
    assert row["rungs_passed"] == 2 and row["gates_total"] == 6 and row["missing"] == ["R4.SidewaysPush"]
    assert row["rank"]["R4.SidewaysPush"] is None


def test_empty_results():
    summary = summarize(JOINT_TRAJECTORY, RUNGS, [])
    assert summary["recommendation"] is None and summary["per_controller"] == {}
    assert summary["per_rung"]["R1.FreeSpace"]["best"] is None
