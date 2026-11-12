"""Tracking and replanning metrics: known offsets, lags and jumps give known numbers."""

import math

import numpy as np
import pytest
from test_metrics_helpers import make_log, make_rung, planar_fk, smooth_reference

from robot_controllers.metrics.episode import episode_metrics, episode_metrics_and_notes
from robot_controllers.metrics.tracking import (
    command_jump_max_deg,
    reference_end_time,
    settle_time,
    torque_rate_max_nm_s,
)


def offset_reference(offset_deg):
    def reference(t):
        return smooth_reference(t) + np.radians(offset_deg)

    return reference


def test_a_one_degree_reference_offset_is_one_degree_of_tracking_error():
    log = make_log()
    log.ref_position = log.ref_position + math.radians(1.0)
    metrics = episode_metrics(log, make_rung())
    assert metrics["tracking_p95_deg"] == pytest.approx(1.0)
    assert metrics["tracking_rms_deg"] == pytest.approx(1.0)
    assert metrics["tracking_max_deg"] == pytest.approx(1.0)
    assert metrics["final_error_deg"] == pytest.approx(1.0)


def test_the_worst_joint_is_reported_not_the_mean_over_joints():
    log = make_log()
    log.ref_position = log.ref_position + np.radians([1.0, 0.2])
    metrics = episode_metrics(log, make_rung())
    assert metrics["tracking_p95_deg"] == pytest.approx(1.0)  # a mean over joints would say 0.6


def test_error_is_measured_against_the_plant_at_the_sent_time_not_the_delayed_state():
    # The plant follows the reference exactly; the state the controller saw is 20 ms old. Interpolating the
    # plant at sent_t gives zero error; using the state's stamp would give ~slope x 20 ms (about 1 deg here).
    log = make_log(latency_s=0.020)
    metrics = episode_metrics(log, make_rung())
    assert metrics["tracking_max_deg"] == pytest.approx(0.0, abs=1e-6)
    delayed_error = np.degrees(np.abs(smooth_reference(log.sent_t) - smooth_reference(log.state_stamp))).max()
    assert delayed_error > 0.5  # the mutation this test guards against would show up


def test_tool_error_is_the_3d_distance_to_the_reference_forward_kinematics_in_mm():
    log = make_log()
    log.ref_tool_pos = planar_fk(log.ref_position) + np.array([0.0, 0.0, 0.002])
    metrics = episode_metrics(log, make_rung())
    assert metrics["tool_p95_mm"] == pytest.approx(2.0)
    assert metrics["tool_rms_mm"] == pytest.approx(2.0)


def test_no_reference_means_tracking_is_not_measured_not_zero():
    log = make_log()
    log.ref_position = None
    log.ref_tool_pos = None
    metrics = episode_metrics(log, make_rung())
    for name in (
        "tracking_rms_deg",
        "tracking_p95_deg",
        "tracking_max_deg",
        "tool_rms_mm",
        "final_error_deg",
        "settle_time_s",
    ):
        assert metrics[name] is None


def test_reference_end_is_where_the_reference_stops_changing():
    sent = np.arange(0.0, 1.0, 0.01)
    end = reference_end_time(sent, smooth_reference(sent))
    assert 0.59 <= end <= 0.61  # smooth_reference ends at 0.6 s
    assert reference_end_time(sent, np.column_stack([sent, sent])) is None  # never ends


def test_settle_time_of_a_first_order_lag_matches_the_closed_form():
    # after the reference ends at t_end, q = final - A exp(-(t - t_end) / tau): within 0.5 deg after
    # tau * ln(A / 0.5 deg)
    amplitude, tau, t_end = math.radians(5.0), 0.05, 0.6
    t = np.arange(0.0, 1.5, 0.00025)
    final = np.array([0.5, 0.3])
    q = np.tile(final, (t.size, 1))
    after = t >= t_end
    q[after, 0] = final[0] - amplitude * np.exp(-(t[after] - t_end) / tau)
    q[~after, 0] = final[0] - amplitude
    settle, note = settle_time(t_end, final, t, q)
    assert settle == pytest.approx(tau * math.log(5.0 / 0.5), abs=0.0005)
    assert note == ""


def test_never_settling_is_none_with_a_note_not_the_time_left():
    t = np.arange(0.0, 1.0, 0.001)
    q = np.column_stack([np.full(t.size, 0.1), np.zeros(t.size)])
    settle, note = settle_time(0.5, np.zeros(2), t, q)
    assert settle is None and "never settled" in note


def test_settle_through_episode_metrics():
    log = make_log(duration_s=1.2)
    lag = np.radians(2.0) * np.exp(-np.clip(log.t - 0.6, 0.0, None) / 0.04)
    log.q = log.q.copy()
    log.q[:, 1] -= lag
    metrics, notes = episode_metrics_and_notes(log, make_rung())
    expected = 0.04 * math.log(2.0 / 0.5)
    # the reference end is detected on host ticks (10 ms) a tick either side of 0.6 s
    assert metrics["settle_time_s"] == pytest.approx(expected, abs=0.012)


def test_command_jump_is_the_largest_tick_to_tick_change_in_degrees():
    positions = np.zeros((10, 2))
    positions[5:, 1] = math.radians(3.0)
    positions[:, 0] = np.radians(np.arange(10) * 0.5)
    assert command_jump_max_deg(positions) == pytest.approx(3.0)


def test_a_controller_that_never_commands_a_position_has_no_command_jump():
    assert command_jump_max_deg(np.full((10, 2), np.nan)) is None
    log = make_log()
    log.cmd_position = np.full_like(log.cmd_position, np.nan)
    metrics, notes = episode_metrics_and_notes(log, make_rung())
    assert (
        metrics["command_jump_max_deg"] is None
        and "never commands a position" in notes["command_jump_max_deg"]
    )


def test_torque_rate_is_the_largest_derivative():
    t = np.arange(0.0, 0.01, 0.001)
    tau = np.zeros((t.size, 2))
    tau[5:, 0] = 0.5  # a 0.5 N·m step over 1 ms = 500 N·m/s
    assert torque_rate_max_nm_s(t, tau) == pytest.approx(500.0)


def test_replans_are_counted_from_events_and_a_refusal_is_zero_with_a_note():
    log = make_log()
    log.events = [{"t": 0.15 * k, "kind": "replan"} for k in range(1, 9)]
    metrics = episode_metrics(log, make_rung("R5.RapidReplan", family="replanning"))
    assert metrics["replans_accepted"] == 8
    refused = make_log()
    refused.events = [{"t": 0.15 * k, "kind": "refused"} for k in range(1, 9)]
    metrics, notes = episode_metrics_and_notes(refused, make_rung("R5.RapidReplan", family="replanning"))
    assert metrics["replans_accepted"] == 0
    assert notes["replans_accepted"].startswith("unsupported: refused 8 replans")
    assert episode_metrics(make_log(), make_rung())["replans_accepted"] is None  # no replans: not measured
