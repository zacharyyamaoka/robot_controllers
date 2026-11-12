"""episode_metrics covers the whole catalogue, gates judge it, run_result and its trace follow CONTRACT §5."""

import json
import math

import numpy as np
import pytest
from test_metrics_helpers import (
    TOUCH_S,
    belt_log,
    belt_rung,
    make_log,
    make_rung,
    push_log,
    push_rung,
)

from robot_controllers.bench.types import Gate, RunResult
from robot_controllers.metrics.episode import (
    episode_metrics,
    episode_metrics_and_notes,
    evaluate,
    round_significant,
    run_result,
)
from robot_controllers.metrics.names import METRICS


def sequence_case():
    log = make_log(duration_s=1.0, rung="S1.MoveTo")
    log.ref_position = log.ref_tool_pos = None
    log.events = [{"t": 0.5, "kind": "step_done", "outcome": "Reached"}]
    log.status = np.where(log.sent_t >= 0.5, "done", "running").astype(object)
    log.meta["steps_total"] = 1
    return log, make_rung(
        "S1.MoveTo",
        family="sequence",
        objective="sequence_time_s",
        sequence_box=True,
        inputs=lambda q: [(0.0, None)],
    )


CASES = {
    "free_space": lambda: (make_log(), make_rung()),
    "belt": lambda: (belt_log(), belt_rung()),
    "push": lambda: (push_log(), push_rung()),
    "sequence": sequence_case,
}


@pytest.mark.parametrize("case", CASES)
def test_every_catalogue_name_is_present_and_json_safe(case):
    log, rung = CASES[case]()
    metrics = episode_metrics(log, rung)
    assert set(metrics) == {m.name for m in METRICS}
    for name, value in metrics.items():
        assert value is None or type(value) in (int, float), name
        assert value is None or math.isfinite(value), name
    result = run_result(log, rung)
    json.dumps(result.to_dict(), allow_nan=False)  # the results JSON is written with allow_nan=False


def test_families_are_measured_only_where_the_rung_has_them():
    belt = episode_metrics(*CASES["belt"]())
    push = episode_metrics(*CASES["push"]())
    assert belt["contact_max_n"] is not None and belt["push_recovery_s"] is None
    assert push["push_recovery_s"] is not None and push["contact_max_n"] is None
    assert (
        belt["compute_us_p99"] is not None and push["latency_p50_ms"] is not None
    )  # timing and cost: always


def test_a_gate_on_a_missing_number_fails_with_the_reason():
    log, rung = (
        make_log(),
        make_rung(
            gates=(
                Gate("contact_peak_n", "<=", 60, "soft touch"),
                Gate("tracking_p95_deg", "<=", 1.0, "tracks"),
            )
        ),
    )
    metrics, notes = episode_metrics_and_notes(log, rung)
    contact_gate, tracking_gate = evaluate(rung, metrics, notes)
    assert not contact_gate.passed and contact_gate.note == "not measured: no belt on this rung"
    assert tracking_gate.passed and tracking_gate.note == ""
    assert evaluate(rung, metrics)[0].note == "not measured"  # without notes: judge's default


def test_a_refused_replan_fails_the_replan_gate_as_unsupported():
    log = make_log()
    log.events = [{"t": 0.15 * k, "kind": "refused"} for k in range(1, 9)]
    rung = make_rung(
        "R5.RapidReplan", family="replanning", gates=(Gate("replans_accepted", ">=", 8, "replans"),)
    )
    result = run_result(log, rung, trace=False)
    assert not result.passed
    assert result.gates[0]["measured"] == 0 and result.gates[0]["note"].startswith("unsupported")


def test_a_gate_on_an_unknown_metric_is_a_contract_error():
    rung = make_rung(gates=(Gate("tracking_p96_deg", "<=", 1.0, "typo"),))
    with pytest.raises(KeyError):
        evaluate(rung, episode_metrics(make_log(), rung))


def test_run_result_carries_gates_objective_and_events():
    log = make_log()
    log.ref_position = log.ref_position + math.radians(1.5)
    log.events = [{"t": np.float64(0.3), "kind": "touch", "where": np.array([0.3, 0.0, 0.15])}]
    rung = make_rung(
        gates=(Gate("tracking_p95_deg", "<=", 1.0, "tracks"), Gate("faults", "==", 0, "no faults"))
    )
    result = run_result(log, rung)
    assert isinstance(result, RunResult)
    assert result.rung == "R1.FreeSpace" and result.controller == "test_controller"
    assert result.objective == pytest.approx(1.5)
    assert [g["passed"] for g in result.gates] == [False, True]
    assert result.gates[0] == {
        "metric": "tracking_p95_deg",
        "op": "<=",
        "value": 1.0,
        "why": "tracks",
        "applies": "always",
        "measured": pytest.approx(1.5),
        "passed": False,
        "note": "",
    }
    assert not result.passed
    assert result.events == [{"t": 0.3, "kind": "touch", "where": [0.3, 0.0, 0.15]}]


def test_trace_keys_rate_and_rounding():
    log, rung = make_log(), make_rung()
    trace = run_result(log, rung).trace
    expected = {"t", "tool_x_mm", "tool_z_mm", "tick_t", "compute_us"}
    for joint in log.joint_names:
        expected |= {f"q_deg.{joint}", f"ref_deg.{joint}", f"err_deg.{joint}", f"tau_nm.{joint}"}
    assert set(trace) == expected  # no contact keys without a belt
    assert np.allclose(np.diff(trace["t"]), 0.005, atol=1e-3)  # 200 Hz from a 4 kHz plant
    assert len(trace["compute_us"]) == len(trace["tick_t"]) == log.sent_t.size
    assert trace["q_deg.shoulder"][-1] == round_significant(math.degrees(log.q[-1, 0]))
    for values in trace.values():
        for value in values:
            assert value is None or value == float(f"{value:.4g}")
    # the plant follows the reference; what is left is the reference interpolated linearly between 10 ms ticks
    assert max(abs(v) for v in trace["err_deg.elbow"]) < 0.01


def test_the_contact_trace_is_1_khz_around_the_touch_and_200_hz_elsewhere():
    trace = run_result(belt_log(), belt_rung()).trace
    contact_t = np.asarray(trace["contact_t"])
    assert len(contact_t) == len(trace["contact_fn_n"])
    steps = np.diff(contact_t)
    window = (contact_t[1:] > TOUCH_S - 0.019) & (contact_t[1:] < TOUCH_S + 0.079)
    assert np.allclose(steps[window], 0.001, atol=2e-4)
    assert np.allclose(steps[contact_t[1:] < TOUCH_S - 0.025], 0.005, atol=2e-4)
    assert np.allclose(steps[contact_t[1:] > TOUCH_S + 0.085], 0.005, atol=2e-4)
    assert max(trace["contact_fn_n"]) == pytest.approx(50.0)
    assert 40.0 in trace["contact_fn_n"]  # the spike lies inside the 1 kHz window


def test_round_significant():
    assert round_significant(123.456) == 123.5
    assert round_significant(0.000123456) == 0.0001235
    assert round_significant(float("nan")) is None
