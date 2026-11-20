"""The layout sweep's own pieces: which points it sweeps, the CTC bandwidth rule, the recommendation on edge
cases, the per-(FF, FB) summary of a ledger, and one real end-to-end sweep document."""

from __future__ import annotations

import json

import numpy as np
import pytest

from robot_controllers.bench import layouts
from robot_controllers.bench.arm import HOME_Q
from robot_controllers.bench.arm_models import planar_arm_mass_matrix
from robot_controllers.bench.board import BoardConfig


def test_the_sweep_is_every_valid_pair_once_at_both_lookaheads():
    points = layouts.sweep_points()
    # feed-forward: trajectory@trajectory, none@none, 3 models x 3 placements = 11; feedback:
    # board_gains@board, pd and ctc x 4 placements = 9; lookahead 0 and the bus delay (CONTRACT §3.8.2)
    assert len(points) == 11 * 9 * 2 == len(set(points))
    assert {p[4] for p in points} == {0.0, 0.0055}
    names = [layouts.layout_name(*p) for p in points]
    assert len(set(names)) == len(names)
    assert "ff=trajectory@trajectory+fb=board_gains@board" in names  # lookahead 0 names are unchanged
    assert "ff=trajectory@trajectory+fb=board_gains@board+la=5.5ms" in names
    assert "ff=rigid_body_coulomb@host_measured+fb=ctc@board_scheduled+la=5.5ms" in names
    for point in points:
        spec = layouts.layout_spec(*point)
        assert spec.name == layouts.layout_name(*point) and spec.variant_of == "layout_controller"
        assert (
            spec.runs_on
            == {"board": "board", "board_scheduled": "board", "host": "host", "split": "split"}[point[3]]
        )
        layout = spec.make_config().layout
        assert layout.feedforward_at == point[1] and layout.lookahead_s == point[4]
    # four positional arguments are lookahead 0, as the acceptance test calls it
    assert layouts.layout_spec("gravity", "plan", "pd", "split").name == "ff=gravity@plan+fb=pd@split"
    assert (
        layouts.layout_name("gravity", "plan", "pd", "split", 0.008) == "ff=gravity@plan+fb=pd@split+la=8ms"
    )
    with pytest.raises(ValueError):
        layouts.layout_spec("gravity", "plan", "pd", "split", -0.001)


@pytest.mark.parametrize(
    "point",
    [
        ("trajectory", "plan", "pd", "board"),  # the effort column is not a model to evaluate
        ("gravity", "trajectory", "pd", "board"),  # a model must be evaluated somewhere
        ("none", "plan", "pd", "board"),
        ("gravity", "plan", "board_gains", "host"),  # no model to run on the host
        ("gravity", "plan", "board_gains", "split"),
        ("gravity", "plan", "lqr", "board"),
    ],
)
def test_invalid_points_are_refused(point):
    with pytest.raises(ValueError):
        layouts.layout_spec(*point)


def test_ctc_bandwidth_puts_its_home_shoulder_kp_at_the_boards():
    bandwidth = layouts.ctc_bandwidth_hz()
    wn = 2 * np.pi * bandwidth
    kp_home = planar_arm_mass_matrix()(np.asarray(HOME_Q))[0, 0] * wn**2
    assert kp_home == pytest.approx(BoardConfig().kp_nm_rad[0], rel=1e-12)
    record = layouts.ctc_record()
    assert record["home_kp_nm_rad"][0] == pytest.approx(300.0, rel=1e-12)
    assert (
        6.0 < bandwidth < 6.6
    )  # 6.32 Hz on this arm: recorded in the ledger, pinned here so a geometry change shows


def _candidate(name, passed, tracking, duty, plan=0.0, wire=1000.0, rungs_passed=7):
    return {
        "name": name,
        "passed": passed,
        "tracking_p95_deg": tracking,
        "host_duty_pct": duty,
        "plan_compute_ms_per_s": plan,
        "wire_bytes_per_s": wire,
        "rungs_passed": rungs_passed,
        "rungs_total": 7,
        "failed_rungs": [] if passed else ["R6.HostStall"],
    }


def test_recommend_breaks_ties_on_wire_then_name_and_names_the_closest_failure():
    pick = layouts.recommend([_candidate("b", True, 1.0, 1.0, wire=900.0), _candidate("a", True, 1.0, 1.0)])
    assert pick["name"] == "b" and "wire 900 B/s" in pick["why"]
    pick = layouts.recommend([_candidate("b", True, 1.0, 1.0), _candidate("a", True, 1.0, 1.0)])
    assert pick["name"] == "a"
    nobody = layouts.recommend(
        [_candidate("x", False, 0.5, 1.0, rungs_passed=3), _candidate("y", False, 0.9, 1.0, rungs_passed=6)]
    )
    assert nobody["name"] is None and "closest: y (6/7 rungs passed, failed R6.HostStall)" in nobody["why"]
    assert layouts.recommend([])["name"] is None
    # a passing layout with no tracking number cannot be compared and does not count
    assert (
        layouts.recommend([_candidate("z", True, None, 0.1), _candidate("w", True, 2.0, 5.0)])["name"] == "w"
    )
    # the exact 10 % boundary is inside
    assert (
        layouts.recommend([_candidate("best", True, 1.0, 9.0), _candidate("edge", True, 1.1, 1.0)])["name"]
        == "edge"
    )


def _row(controller, rung, passed, tracking, compute_us_p50, rate_hz=100.0):
    return {
        "controller": controller,
        "rung": rung,
        "passed": passed,
        "objective": tracking,
        "metrics": {
            "tracking_p95_deg": tracking,
            "compute_us_p50": compute_us_p50,
            "host_rate_hz": rate_hz,
            "host_duty_pct": compute_us_p50 * rate_hz / 1e4,
            "plan_compute_ms_per_s": 0.0,
            "wire_bytes_per_s": 2800.0,
            "fb_share_pct": 20.0,
        },
        "gates": [
            {"metric": "tracking_p95_deg", "op": "<=", "value": 3.0, "measured": tracking, "passed": passed}
        ],
    }


def test_the_summary_recommends_per_model_pair_from_the_ledger():
    rungs = [{"id": "R1.FreeSpace"}, {"id": "R2.FastFreeSpace"}]
    points = [
        ("rigid_body", "plan", "pd", "board"),
        ("rigid_body", "plan", "pd", "host"),
        ("gravity", "plan", "pd", "board"),
    ]
    controllers = [
        dict(
            zip(("feedforward_model", "feedforward_at", "feedback_model", "feedback_at"), p, strict=True),
            name=layouts.layout_name(*p),
        )
        for p in points
    ]
    names = [c["name"] for c in controllers]
    rungs.append({"id": "R9.FastFreeSpace·host500Hz"})
    results = [
        _row(names[0], "R1.FreeSpace", True, 0.5, 30.0),
        _row(names[0], "R2.FastFreeSpace", True, 1.00, 90.0),  # one noisy rung: the median ignores it
        _row(names[0], "R9.FastFreeSpace·host500Hz", True, 0.3, 20.0, rate_hz=500.0),
        _row(names[1], "R1.FreeSpace", True, 0.4, 60.0),
        _row(names[1], "R2.FastFreeSpace", True, 0.95, 50.0),
        _row(names[1], "R9.FastFreeSpace·host500Hz", True, 0.3, 55.0, rate_hz=500.0),
        _row(names[2], "R1.FreeSpace", False, 4.0, 20.0),
        _row(names[2], "R2.FastFreeSpace", True, 2.0, 20.0),
        _row(names[2], "R9.FastFreeSpace·host500Hz", True, 0.3, 20.0, rate_hz=500.0),
    ]
    summary = layouts.layout_summary({"rungs": rungs, "controllers": controllers, "results": results})
    candidates = {c["name"]: c for c in summary["candidates"]}
    # CONTRACT §3.8.3: the cost is the median over the swept rungs of compute_us_p50, at 100 Hz
    assert candidates[names[0]]["host_duty_pct"] == pytest.approx(30.0 * 100 / 1e4)  # not R2's 0.9 %
    assert candidates[names[1]]["host_duty_pct"] == pytest.approx(55.0 * 100 / 1e4)
    assert "median" in summary["rule"]
    pairs = {(p["feedforward_model"], p["feedback_model"]): p for p in summary["per_pair"]}
    # 1.00 is within 10 % of 0.95 and cheaper over the median: the board layout wins for rigid_body + pd
    assert pairs[("rigid_body", "pd")]["recommendation"]["name"] == names[0]
    assert pairs[("gravity", "pd")]["recommendation"]["name"] is None  # its only layout failed R1
    failed = next(c for c in summary["candidates"] if c["name"] == names[2])
    assert failed["failed_rungs"] == ["R1.FreeSpace"] and failed["rungs_passed"] == 2  # of 3
    assert summary["overall"]["name"] == names[0]


def test_one_real_sweep_document_is_results_1_with_the_layout_summary(tmp_path):
    pytest.importorskip("mujoco")
    document = layouts.run_layouts(
        ff_models=("rigid_body",), fb_models=("board_gains",), rungs=("R1.FreeSpace",), progress=None
    )
    assert document["schema"] == "robot-controllers/results-1" and document["sweep"] == "layouts"
    assert [c["name"] for c in document["controllers"]] == [
        f"ff=rigid_body@{at}+fb=board_gains@board{la}"
        for at in ("plan", "host_reference", "host_measured")
        for la in ("", "+la=5.5ms")
    ]
    assert [c["lookahead_s"] for c in document["controllers"]] == [0.0, 0.0055] * 3
    assert len(document["results"]) == 6 and all(not r["trace"] for r in document["results"])
    config = document["controllers"][0]["config"]
    assert config["layout"] == {"feedforward_at": "plan", "feedback_at": "board", "lookahead_s": 0.0}
    assert config["feedforward"].startswith("planar_arm_feedforward(")  # a stable repr, not an address
    summary = document["summary"]["layouts"]
    assert summary["ctc"]["bandwidth_hz"] == pytest.approx(layouts.ctc_bandwidth_hz())
    assert len(summary["per_pair"]) == 1 and summary["per_pair"][0]["recommendation"]["why"]
    path = layouts.write_layouts(document, tmp_path)
    assert path == tmp_path / "layouts" / "results.json"
    assert json.loads(path.read_text())["summary"]["layouts"]["per_pair"] == summary["per_pair"]


def test_controller_names_take_a_lookahead_suffix_and_nothing_else():
    from robot_controllers.core import ControllerSpec
    from robot_controllers.registry import JOINT_TRAJECTORY

    def spec(name):
        return ControllerSpec(
            name=name, title="t", box=JOINT_TRAJECTORY, runs_on="board", summary="s", maturity="planned"
        )

    for good in ("ff=a@b+fb=c@d", "ff=a@b+fb=c@d+la=5.5ms", "ff=a@b+fb=c@d+la=8ms", "joint_impedance_board"):
        assert spec(good).name == good
    for bad in ("ff=a@b+fb=c@d+la=ms", "ff=a@b+fb=c@d+la=5.5", "ff=a@b+fb=c@d+la=-1ms", "ff=a@b+fb=c@d+x"):
        with pytest.raises(ValueError):
            spec(bad)
