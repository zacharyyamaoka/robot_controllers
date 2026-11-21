"""The study (§4.6) beyond the acceptance file: its plan has the contract's rows, its summary helpers answer
the contract's questions on hand-made numbers, and one real slice runs, summarizes and round-trips through
the ledger writer."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from robot_controllers.bench import study


def _no_calibration(spec_at, rung, seed=0):
    assert callable(spec_at) and seed == 0
    return tuple(0.001 * (j + 1) for j in range(len(rung.timing.command_delay_s)))


def test_the_plan_has_one_job_per_contract_row():
    jobs = study.plan_study(calibrate=_no_calibration)
    counts = Counter((job.arm, job.dimension) for job in jobs)
    for arm, moves in (("planar_arm", 5), ("ur5e", 6)):
        assert counts[(arm, "rate")] == 7 * 7
        assert counts[(arm, "speed")] == moves * 7
        assert counts[(arm, "rate_speed")] == 7 * moves * 5
        assert counts[(arm, "lookahead")] == 2 * 14 * 2
        assert counts[(arm, "jitter")] == 5 * 5
        assert counts[(arm, "stagger")] == 4 * 2
    assert counts[("planar_arm", "sync")] == 4 * 7 + 2 * 14 and ("ur5e", "sync") not in counts
    keys = [(j.arm, j.dimension, json.dumps(j.value), j.layout_index, j.seed) for j in jobs]
    assert len(keys) == len(set(keys))  # one row per (arm, dimension, value, layout, seed)
    stagger = [j for j in jobs if j.arm == "ur5e" and j.dimension == "stagger" and j.value == [50.0, "true"]]
    assert stagger[0].lookahead_s == pytest.approx((0.0015, 0.0037, 0.0059, 0.0081, 0.0103, 0.0125))
    assert stagger[0].rung.timing.command_delay_s == stagger[0].lookahead_s
    calibrated = next(j for j in jobs if j.arm == "planar_arm" and j.value == [100.0, "calibrated"])
    assert calibrated.lookahead_s == (0.001, 0.002) and dict(calibrated.fields)["way"] == "calibrated"
    assert {j.spec().name for j in jobs if j.dimension == "rate"} == set(study.STUDY_LAYOUTS)


def test_the_rate_threshold_is_the_lowest_rate_from_which_every_faster_rate_is_close():
    by_rate = {10.0: 1.0, 25.0: 0.30, 50.0: 0.118, 100.0: 0.105, 200.0: 0.12, 500.0: 0.101, 1000.0: 0.100}
    answer = study.rate_threshold(by_rate)
    assert answer["rate_hz"] == 500.0  # 200 Hz is 20 % off, so 100 Hz does not count however close it is
    assert answer["limit_deg"] == pytest.approx(0.11) and "500 Hz up" in answer["why"]
    assert study.rate_threshold({**by_rate, 500.0: 0.5})["rate_hz"] == 1000.0


def test_the_lookahead_curve_gives_its_minimum_and_the_cost_of_a_millisecond_either_side():
    curve = {ms: 0.1 + 0.3 * abs(ms - 5.5) for ms in study.LOOKAHEADS_MS}
    answer = study.lookahead_curve_summary(curve)
    assert answer["best_lookahead_ms"] == 5.5 and answer["best_tracking_p95_deg"] == pytest.approx(0.1)
    assert answer["cost_of_1ms_error_deg"] == pytest.approx(0.3)
    edge = study.lookahead_curve_summary({0.0: 0.1, 1.0: 0.4, 2.0: 0.7})
    assert edge["tracking_at_minus_1ms_deg"] is None and edge["cost_of_1ms_error_deg"] == pytest.approx(0.3)


def test_a_real_slice_runs_summarizes_and_round_trips(tmp_path):
    pytest.importorskip("mujoco")
    document = study.run_study(arms=("planar_arm",), dimensions=("stagger",), progress=None)
    rows = document["rows"]
    assert document["schema"] == "robot-controllers/study-1" and len(rows) == 8
    expected = {(rate, way) for rate in (50.0, 100.0) for way in study.STAGGER_WAYS}
    assert {tuple(r["value"]) for r in rows} == expected
    true_row = next(r for r in rows if r["value"] == [100.0, "true"])
    assert true_row["lookahead_ms"] == [1.5, 3.7] and true_row["command_delay_ms"] == [1.5, 3.7]
    assert true_row["controller"].endswith("+la=1.5/3.7ms+interp=secant_mid_ff")
    assert set(true_row["metrics"]) >= {"tracking_p95_deg", "tracking_lag_ms", "sync_timing_ms"}
    stagger = document["summary"]["planar_arm"]["stagger"]
    assert [entry["rate_hz"] for entry in stagger] == [50.0, 100.0]
    for entry in stagger:
        assert len(entry["calibrated_lookahead_ms"]) == 2 and entry["calibration_max_abs_error_ms"] <= 1.0
        assert entry["tracking_p95_deg"]["true"] < entry["tracking_p95_deg"]["miscalibrated"]
    path = study.write_study(document, tmp_path)
    again = json.loads(path.read_text())
    assert again["rows"] == json.loads(json.dumps(rows)) and again["summary"]["planar_arm"]["stagger"]
    assert path.read_text().count("\n") > len(rows)  # one row per line
