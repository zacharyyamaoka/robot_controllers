"""Sequence metrics: step_done events, meta steps_total, the status column and the Reached error."""

import math

import numpy as np
import pytest
from test_metrics_helpers import make_log, make_rung

from robot_controllers.metrics.episode import episode_metrics, episode_metrics_and_notes
from robot_controllers.metrics.sequence import reached_error_deg, steps_done


def sequence_log():
    log = make_log(duration_s=2.0, rung="S1.MoveTo")
    log.ref_position = None
    log.ref_tool_pos = None
    target = np.array([0.5, 0.3])
    log.cmd_position = np.tile(target, (log.sent_t.size, 1))
    # the plant sits 0.4 deg short on the shoulder, 0.1 deg on the elbow
    log.q = np.tile(target - np.radians([0.4, 0.1]), (log.t.size, 1))
    log.status = np.where(log.sent_t >= 1.2 - 1e-9, "done", "running").astype(object)
    log.events = [
        {"t": 0.81, "kind": "step_done", "index": 0, "name": "move", "outcome": "Reached"},
        {"t": 1.2, "kind": "step_done", "index": 1, "name": "hold", "outcome": "Elapsed"},
    ]
    log.meta["steps_total"] = 2
    return log


def sequence_rung():
    return make_rung(
        "S1.MoveTo",
        family="sequence",
        objective="sequence_time_s",
        sequence_box=True,
        inputs=lambda q: [(0.0, None)],
    )


def test_sequence_counts_time_and_reached_error():
    metrics = episode_metrics(sequence_log(), sequence_rung())
    assert metrics["steps_done"] == 2
    assert metrics["steps_total"] == 2
    assert metrics["sequence_time_s"] == pytest.approx(1.2)
    assert metrics["reached_error_deg"] == pytest.approx(0.4)  # worst joint
    assert metrics["tracking_p95_deg"] is None and metrics["schedule_slip_ms"] is None


def test_an_aborted_step_is_not_done_and_the_abort_ends_the_sequence():
    log = sequence_log()
    log.events[1] = {"t": 1.2, "kind": "step_done", "index": 1, "name": "hold", "outcome": "timeout"}
    log.status = np.where(log.sent_t >= 1.2 - 1e-9, "aborted", "running").astype(object)
    metrics = episode_metrics(log, sequence_rung())
    assert metrics["steps_done"] == 1
    assert metrics["sequence_time_s"] == pytest.approx(1.2)
    assert metrics["faults"] == 1


def test_stall_outcomes_name_the_joint_and_still_count():
    events = [
        {"t": 0.5, "kind": "step_done", "outcome": "Stall elbow"},
        {"t": 0.9, "kind": "step_done", "outcome": "guard"},
        {"t": 0.2, "kind": "replan"},
    ]
    assert steps_done(events) == 1


def test_without_a_recorded_target_the_setpoint_comes_from_the_tick_before_the_firing_one():
    # the firing tick (0.81 s) already commands the next step, 5 deg away; the step's own setpoint is the tick
    # before
    log = sequence_log()
    firing = int(np.argmin(np.abs(log.tick_t - 0.81)))
    log.cmd_position = log.cmd_position.copy()
    log.cmd_position[firing:, 0] += math.radians(5.0)
    error, note = reached_error_deg(log.events, log.joint_names, log.tick_t, log.cmd_position, log.t, log.q)
    assert error == pytest.approx(0.4) and note == ""


def test_a_recorded_target_wins_over_the_commanded_setpoint():
    log = sequence_log()
    log.events[0]["target"] = {"shoulder": 0.5 + math.radians(1.0), "elbow": 0.3}
    error, note = reached_error_deg(log.events, log.joint_names, log.tick_t, log.cmd_position, log.t, log.q)
    assert error == pytest.approx(1.4) and note == ""


def test_a_sequence_that_never_finishes_has_no_time_and_says_so():
    log = sequence_log()
    log.status = np.array(["running"] * log.sent_t.size, dtype=object)
    metrics, notes = episode_metrics_and_notes(log, sequence_rung())
    assert metrics["sequence_time_s"] is None and "did not finish" in notes["sequence_time_s"]


def test_a_trajectory_rung_reports_sequence_metrics_as_not_measured():
    metrics = episode_metrics(make_log(), make_rung())
    for name in ("steps_done", "steps_total", "sequence_time_s", "reached_error_deg"):
        assert metrics[name] is None
