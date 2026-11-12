"""Health and cost metrics: saturation share, faults, divergence, compute, command fields and wire bytes."""

import numpy as np
import pytest
from test_metrics_helpers import make_log, make_rung

from robot_controllers.metrics.cost import command_fields, compute_metrics, wire_bytes_per_tick
from robot_controllers.metrics.episode import episode_metrics
from robot_controllers.metrics.health import diverged, fault_count, saturation_share


def test_saturation_share_is_the_share_of_samples_with_any_joint_at_its_cap():
    at_cap = np.zeros((100, 2), dtype=bool)
    at_cap[0:10, 0] = True  # joint 0 capped in 10 % of samples
    at_cap[5:15, 1] = True  # joint 1 in another 10 %, overlapping 5
    assert saturation_share(at_cap) == pytest.approx(15.0)  # a per-element mean would say 10


def test_saturation_through_episode_metrics():
    log = make_log()
    log.at_cap = log.at_cap.copy()
    log.at_cap[: log.t.size // 4, 1] = True
    assert episode_metrics(log, make_rung())["saturation_share"] == pytest.approx(25.0, abs=0.1)


def test_faults_are_board_fault_entries_plus_controller_aborts_without_double_counting():
    modes = np.zeros((20, 2), dtype=int)
    modes[3:5, 0] = 1
    modes[10:12, 1] = 1
    status = np.array(["running"] * 6 + ["aborted"] * 4)
    assert fault_count(modes, status) == 3
    assert fault_count(modes, status, abort_events=1) == 3  # the loop's abort event is the same abort
    assert fault_count(np.zeros((5, 2)), np.array(["running", "done", "done"])) == 0


def _divergence_inputs(log):
    return dict(
        q=log.q,
        qd=log.qd,
        tau=log.tau,
        tool_pos=log.tool_pos,
        cmd_position=log.cmd_position,
        cmd_velocity=log.cmd_velocity,
        cmd_feedforward=log.cmd_feedforward,
    )


def test_a_nan_state_or_an_infinite_command_diverges_but_a_nan_command_field_is_just_unset():
    log = make_log()
    assert diverged(**_divergence_inputs(log)) == 0
    log.cmd_position = np.full_like(log.cmd_position, np.nan)  # position None: the host-torque variants
    assert diverged(**_divergence_inputs(log)) == 0
    log.cmd_feedforward = log.cmd_feedforward.copy()
    log.cmd_feedforward[10, 0] = np.inf
    assert diverged(**_divergence_inputs(log)) == 1
    nan_state = make_log()
    nan_state.q = nan_state.q.copy()
    nan_state.q[100, 1] = np.nan
    assert episode_metrics(nan_state, make_rung())["diverged"] == 1
    runaway = make_log()
    runaway.qd = runaway.qd.copy()
    runaway.qd[-1, 0] = 500.0
    assert episode_metrics(runaway, make_rung())["diverged"] == 1


def test_compute_percentiles_in_microseconds():
    out = compute_metrics(np.concatenate([np.full(99, 10e-6), [500e-6]]))
    assert out["compute_us_p50"] == pytest.approx(10.0)
    assert out["compute_us_max"] == pytest.approx(500.0)
    assert 10.0 < out["compute_us_p99"] < 500.0
    assert compute_metrics(np.array([]))["compute_us_p50"] is None


def test_command_fields_and_wire_bytes_for_a_position_velocity_feedforward_command():
    # the JTC sends position, velocity and feed-forward; stiffness, damping and max effort stay None (NaN)
    log = make_log()
    metrics = episode_metrics(log, make_rung())
    assert metrics["command_fields"] == pytest.approx(3.0)
    assert metrics["wire_bytes_per_tick"] == pytest.approx(2 * (2 + 4 * 3))  # 28 bytes
    assert metrics["host_rate_hz"] == pytest.approx(100.0)


def test_a_zero_gain_is_a_field_on_the_bus():
    # host impedance: position NaN, velocity NaN, feed-forward set, stiffness 0 and damping 0 sent explicitly
    ticks = 10
    nan, zero = np.full((ticks, 2), np.nan), np.zeros((ticks, 2))
    fields = command_fields((nan, nan, np.ones((ticks, 2)), zero, zero, nan))
    assert fields == pytest.approx(3.0)  # counting non-zero values instead would say 1
    assert wire_bytes_per_tick(2, fields) == pytest.approx(28.0)
    # a mean over ticks: half the ticks add max_effort
    max_effort = nan.copy()
    max_effort[:5] = 8.0
    assert command_fields((nan, nan, np.ones((ticks, 2)), zero, zero, max_effort)) == pytest.approx(3.5)
