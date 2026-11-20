"""Amendment 3's torque split end to end: the board's split adds up to the applied torque bit for bit, the
loop maps every physics step to the host tick whose command ran, and the decomposition and cost metrics have
known answers."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from robot_msgs.control import ServoCommand
from test_metrics_helpers import make_log, make_rung

from robot_controllers.bench.board import WATCHDOG_MODE, BoardConfig, ServoBoards
from robot_controllers.metrics import cost, decomposition, health
from robot_controllers.metrics.episode import episode_metrics

JOINTS = ("shoulder", "elbow")
DT = 0.00025


# ── the board's split ──────────────────────────────────────────────────────────────────────────────────────


def test_the_board_split_adds_up_to_the_applied_torque_bit_for_bit_and_names_the_command():
    boards = ServoBoards(JOINTS, BoardConfig())
    boards.reset(0.0, [0.0, 0.0])
    command = ServoCommand(
        name=JOINTS,
        position=np.array([0.002, -0.001]),
        velocity=np.array([0.3, -0.2]),
        feedforward_effort=np.array([1.5, -0.7]),
    )
    boards.schedule(command, latch_t=DT, tag=7)
    for k in range(40):
        q = [0.0005 * k, -0.0002 * k]
        tau = list(boards.step(k * DT, q))
        tag = -1 if k == 0 else 7
        assert boards.latched_tag == [tag, tag]
        for j in range(2):
            if not boards.at_cap[j]:
                assert tau[j] == boards.tau_ff[j] + boards.tau_fb[j]  # exact: the board's own summation order
        if k:
            assert boards.tau_ff == [1.5, -0.7]


def test_in_watchdog_mode_the_brake_is_feedback_and_no_feed_forward_is_added():
    config = BoardConfig(watchdog_s=0.01)
    boards = ServoBoards(JOINTS, config)
    boards.reset(0.0, [0.0, 0.0], feedforward=[2.0, 1.0])
    for k in range(80):  # no command: the hold latched at reset times out after 10 ms
        tau = list(boards.step(k * DT, [0.0001 * k, 0.0]))  # 0.4 rad/s: a brake inside the cap
    assert boards.mode == [WATCHDOG_MODE, WATCHDOG_MODE]
    assert (
        boards.tau_ff == [0.0, 0.0] and boards.tau_fb == tau and tau[0] < 0.0
    )  # braking the moving shoulder


# ── the loop's log ─────────────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def r1():
    pytest.importorskip("mujoco")
    from robot_controllers.bench.rungs import rung

    return rung("R1.FreeSpace")


def _log(spec, rung):
    from robot_controllers.bench.loop import run_episode

    return run_episode(spec, rung, seed=0)


def test_every_physics_step_runs_the_command_of_the_host_tick_it_names(r1):
    from robot_controllers.bench.layouts import layout_spec

    log = _log(layout_spec("gravity", "host_reference", "pd", "split"), r1)
    ticks = log.latched_tick
    assert ticks.shape == log.tau.shape and ticks.min() == -1 and ticks.max() == len(log.sent_t) - 1
    assert np.all(np.diff(ticks, axis=0) >= 0)  # commands latch in order
    ran = (ticks >= 0) & (log.board_mode == 0)
    for joint in range(2):
        rows = ticks[ran[:, joint], joint]
        np.testing.assert_array_equal(log.tau_ff[ran[:, joint], joint], log.cmd_feedforward[rows, joint])
        # the first sample each command runs on is at or after its latch time (sent + bus delay)
        first = np.flatnonzero(np.diff(ticks[:, joint], prepend=-2) != 0)
        first = first[ticks[first, joint] >= 0]
        assert np.all(log.t[first] >= log.sent_t[ticks[first, joint]] + 0.0055 - 1e-9)
    # the controller's split is exactly its feed-forward field
    np.testing.assert_allclose(log.split_ff + log.split_fb_host, log.cmd_feedforward, rtol=0, atol=1e-12)
    assert np.abs(log.split_fb_host).max() > 0.0  # the host spring did work
    assert log.meta["plan_compute_s"] == 0.0  # host_reference plans nothing


def test_a_controller_without_last_split_has_no_split_and_its_host_feedback_counts_zero(r1):
    from robot_controllers.registry import controller

    log = _log(controller("joint_trajectory_controller"), r1)
    assert log.split_ff is None and log.split_fb_host is None and "plan_compute_s" not in log.meta
    metrics = episode_metrics(log, r1)
    assert metrics["plan_compute_ms_per_s"] is None  # not reported is not zero
    host = decomposition.host_feedback_per_sample(log.latched_tick, log.board_mode, None)
    assert not host.any()
    assert 0.0 < metrics["fb_share_pct"] < 100.0


def test_host_feedback_moves_from_the_board_term_into_the_split_and_the_metrics_follow_it(r1):
    from robot_controllers.bench.layouts import layout_spec

    exact = dataclasses.replace(r1, timing=dataclasses.replace(r1.timing, include_compute=False))
    board = episode_metrics(_log(layout_spec("rigid_body", "plan", "pd", "board"), exact), exact)
    host_log = _log(layout_spec("rigid_body", "plan", "pd", "host"), exact)
    host = episode_metrics(host_log, exact)
    # on the host the board's PD is off: its term is ~0 and all the feedback is the controller's split
    commanded = (
        host_log.latched_tick[:, 0] >= 0
    )  # before the first command latches, the pre-roll hold's PD runs
    assert np.abs(host_log.tau_fb_board[commanded]).max() < 1e-9
    assert host["fb_rms_nm"] > 0.05 and board["fb_rms_nm"] > 0.05
    assert host["ff_rms_nm"] == pytest.approx(
        board["ff_rms_nm"], rel=0.05
    )  # the same model carried the same ff


def test_plan_compute_is_per_second_of_planned_trajectory(r1):
    from robot_controllers.bench.layouts import layout_spec

    log = _log(layout_spec("rigid_body", "plan", "board_gains", "board"), r1)
    metrics = episode_metrics(log, r1)
    planned_s = log.meta["script"][0]["duration_s"]
    assert metrics["plan_compute_ms_per_s"] == pytest.approx(1e3 * log.meta["plan_compute_s"] / planned_s)
    assert metrics["plan_compute_ms_per_s"] > 0.0
    assert metrics["host_duty_pct"] == pytest.approx(metrics["compute_us_p50"] * 100.0 / 1e4)


# ── the formulas on hand-built arrays ──────────────────────────────────────────────────────────────────────


def test_host_feedback_per_sample_maps_ticks_and_skips_watchdog_and_unlatched_steps():
    split = np.array([[1.0, -1.0], [2.0, np.nan], [3.0, 30.0]])
    ticks = np.array([[-1, -1], [0, 0], [0, 1], [2, 1], [2, 2]])
    modes = np.zeros((5, 2), dtype=int)
    modes[4, 0] = WATCHDOG_MODE
    host = decomposition.host_feedback_per_sample(ticks, modes, split)
    np.testing.assert_array_equal(host, [[0.0, 0.0], [1.0, -1.0], [1.0, 0.0], [3.0, 0.0], [0.0, 30.0]])


def test_decomposition_metrics_have_known_answers():
    tau = np.array([[4.0, 1.0], [4.0, -1.0], [-4.0, 1.0], [-4.0, -1.0]])
    host = np.array([[0.0, 0.5], [0.0, 0.5], [0.0, 0.5], [0.0, 0.5]])
    tau_fb_board = np.array([[1.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    tau_ff = tau - tau_fb_board  # what the board added: model ff + host fb
    out = decomposition.decomposition_metrics(tau, tau_ff, tau_fb_board, host)
    # shoulder: fb rms 1 of 4 = 25 %; elbow: fb rms 0.5 of 1 = 50 % -> the worst joint
    assert out["fb_share_pct"] == pytest.approx(50.0)
    assert out["fb_rms_nm"] == pytest.approx(1.0)
    # the model's feed-forward: shoulder +-3 / +-5 (rms sqrt(17)), elbow +-1 - 0.5 (rms sqrt(1.25))
    assert out["ff_rms_nm"] == pytest.approx(np.sqrt(17.0))
    assert (
        decomposition.decomposition_metrics(
            np.zeros((3, 2)), np.zeros((3, 2)), np.zeros((3, 2)), np.zeros((3, 2))
        )["fb_share_pct"]
        is None
    )


def test_cost_rates_and_torque_rate_rms():
    assert cost.host_duty_pct(25.0, 100.0) == pytest.approx(0.25)  # 25 us every 10 ms = 0.25 % of a core
    assert cost.wire_bytes_per_s(28.0, 500.0) == 14000.0
    assert cost.plan_compute_ms_per_s(0.003, 1.5) == pytest.approx(2.0)
    assert cost.plan_compute_ms_per_s(None, 1.5) is None and cost.host_duty_pct(None, 100.0) is None
    t = np.arange(0.0, 1.0, 0.001)
    tau = np.column_stack([3.0 * t, np.sin(2 * np.pi * 5 * t)])
    # joint 0 ramps at 3 N·m/s; joint 1 at 10 pi cos(.): rms 10 pi / sqrt 2 = 22.2 N·m/s, the worst
    assert health.torque_rate_rms_nm_s(t, tau) == pytest.approx(10 * np.pi / np.sqrt(2), rel=1e-3)


def test_old_logs_without_the_split_say_not_measured():
    log = make_log()
    metrics = episode_metrics(log, make_rung())
    assert metrics["fb_share_pct"] is None and metrics["fb_rms_nm"] is None and metrics["ff_rms_nm"] is None
    assert metrics["host_duty_pct"] == pytest.approx(20.0 * 100.0 / 1e4)
    assert metrics["wire_bytes_per_s"] == pytest.approx(metrics["wire_bytes_per_tick"] * 100.0)
