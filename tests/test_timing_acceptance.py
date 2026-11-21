"""Acceptance for CONTRACT amendment 5, lane A: the arm registry, bus timing, calibration and the drift monitor.

Written by the dispatcher BEFORE the build and read-only to the builders (CONTRACT §5.10). Oracles: MuJoCo on the vendored
UR5e model, the plant's own latch log, and bench episodes on the twin whose bus is known exactly.

Zach, 2026-11-20: "this controller then should register a routine to calibrate the delay ... compare that against reality
and of course if that begins to change a lot we should be alerted".
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from robot_controllers.models import ModelError
from robot_controllers.timing import BusTiming

V1 = ("trajectory", "trajectory", "board_gains", "board")


def spec_at(lookahead, arm="planar_arm"):
    from robot_controllers.bench.layouts import layout_spec

    return layout_spec(*V1, lookahead_s=lookahead, interpolation="secant_mid_ff", arm=arm)


def episode(spec, rung, seed=0):
    from robot_controllers.bench.loop import run_episode
    from robot_controllers.metrics.episode import episode_metrics

    log = run_episode(spec, rung, seed=seed)
    return log, episode_metrics(log, rung)


def latch_delays(log) -> np.ndarray:
    """(K, J) latch time minus send time of each host tick's command on each joint (NaN = never ran)."""
    ticks = np.asarray(log.latched_tick)
    out = np.full((len(log.sent_t), ticks.shape[1]), np.nan)
    for j in range(ticks.shape[1]):
        for k in range(len(log.sent_t)):
            hits = np.flatnonzero(ticks[:, j] == k)
            if hits.size:
                out[k, j] = log.t[hits[0]] - log.sent_t[k]
    return out


@pytest.fixture(scope="module")
def rungs():
    from robot_controllers.bench.rungs import rung

    return rung("R1.FreeSpace"), rung("R2.FastFreeSpace")


# ── 5.1 the registry ─────────────────────────────────────────────────────────────────────────────────────────


def test_the_registry_has_the_planar_arm_and_the_ur5e_in_order():
    from robot_controllers.bench.arms import arm, arms
    from robot_controllers.bench.ur5e import UR5E_JOINTS

    names = [a.name for a in arms()]
    assert names[:2] == ["planar_arm", "ur5e"]
    assert arm("planar_arm").joint_names == ("shoulder", "elbow")
    assert arm("ur5e").joint_names == UR5E_JOINTS
    with pytest.raises(KeyError):
        arm("no_such_arm")


def test_the_registrys_planar_models_are_todays_and_model_error_is_deterministic():
    from robot_controllers.bench.arm_models import RIGID_BODY, planar_arm_feedforward
    from robot_controllers.bench.arms import arm

    planar = arm("planar_arm")
    today = planar_arm_feedforward(RIGID_BODY)
    rng = np.random.default_rng(3)
    states = [(rng.uniform(-2, 2, 2), rng.uniform(-3, 3, 2), rng.uniform(-20, 20, 2)) for _ in range(6)]
    for q, qd, qdd in states:
        np.testing.assert_array_equal(planar.feedforward("rigid_body", None)(q, qd, qdd), today(q, qd, qdd))
        np.testing.assert_array_equal(planar.feedforward("rigid_body", ModelError())(q, qd, qdd), today(q, qd, qdd))
    wrong = planar.feedforward("rigid_body", ModelError(0.3, seed=1))
    again = planar.feedforward("rigid_body", ModelError(0.3, seed=1))
    other = planar.feedforward("rigid_body", ModelError(0.3, seed=2))
    diffs = [np.max(np.abs(wrong(*s) - today(*s))) for s in states]
    assert max(diffs) > 1e-3
    for s in states:
        np.testing.assert_array_equal(wrong(*s), again(*s))
    assert any(not np.array_equal(wrong(*s), other(*s)) for s in states) or ModelError(0.3, seed=1).link_factors(
        2
    ).tolist() == ModelError(0.3, seed=2).link_factors(2).tolist()


def test_ur5e_model_error_scales_the_jointed_bodies_and_friction_not_the_armature():
    import mujoco

    from robot_controllers.bench.arms import arm
    from robot_controllers.bench.ur5e import MODEL_PATH

    error = ModelError(0.3, seed=2)
    oracle = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    oracle.dof_armature[:] = (0.5, 0.5, 0.5, 0.08, 0.08, 0.08)
    oracle.dof_damping[:] = np.array((2.0, 2.0, 2.0, 0.3, 0.3, 0.3)) * error.friction_factor
    oracle.dof_frictionloss[:] = 0.0
    jointed = [b for b in range(oracle.nbody) if oracle.body_jntnum[b] > 0]
    factors = error.link_factors(len(jointed))
    for b, f in zip(jointed, factors, strict=True):
        oracle.body_mass[b] *= f
        oracle.body_inertia[b] *= f
    data = mujoco.MjData(oracle)
    model = arm("ur5e").feedforward("rigid_body", error)
    rng = np.random.default_rng(5)
    for _ in range(5):
        q, qd, qdd = rng.uniform(-2, 2, 6), rng.uniform(-2, 2, 6), rng.uniform(-8, 8, 6)
        data.qpos[:], data.qvel[:], data.qacc[:] = q, qd, qdd
        mujoco.mj_inverse(oracle, data)
        np.testing.assert_allclose(model(q, qd, qdd), data.qfrc_inverse, rtol=0, atol=1e-9)


def test_gain_scale_multiplies_kp_and_kd_only():
    from robot_controllers.bench.arms import arm
    from robot_controllers.bench.board import scaled_board

    board = arm("ur5e").board_config()
    half = scaled_board(board, 0.5)
    np.testing.assert_allclose(half.kp_nm_rad, np.asarray(board.kp_nm_rad) * 0.5, rtol=1e-12)
    np.testing.assert_allclose(half.kd_nm_s_rad, np.asarray(board.kd_nm_s_rad) * 0.5, rtol=1e-12)
    assert half.torque_cap_nm == board.torque_cap_nm and half.rate_hz == board.rate_hz
    assert dataclasses.replace(half, kp_nm_rad=board.kp_nm_rad, kd_nm_s_rad=board.kd_nm_s_rad) == board


# ── 5.2 bus timing ───────────────────────────────────────────────────────────────────────────────────────────


def test_datasheet_timing_staggers_the_boards_in_slot_order():
    bus = BusTiming.from_datasheet(0.014, 7, slot_order=(6, 0, 1, 2, 3, 4), command_latency_s=0.0005, source="x")
    np.testing.assert_allclose(bus.command_delay_s, [0.0145, 0.0025, 0.0045, 0.0065, 0.0085, 0.0105], atol=1e-12)
    assert bus.per_joint(6) == bus.command_delay_s
    with pytest.raises(ValueError):
        bus.per_joint(2)


def test_host_timing_from_a_bus_and_the_lookahead_it_implies():
    from robot_controllers.bench.types import HostTiming
    from robot_controllers.timing import lookahead_from

    bus = BusTiming((0.002, 0.008), reply_delay_s=0.001, jitter_s=0.0005)
    timing = HostTiming.from_bus(bus, 50.0)
    assert timing.rate_hz == 50.0
    assert tuple(timing.command_delay_s) == (0.002, 0.008)
    assert timing.reply_delay_s == 0.001 and timing.command_delay_jitter_s == 0.0005
    assert lookahead_from(bus, 2) == (0.002, 0.008)
    assert lookahead_from(BusTiming(0.0055), 3) == (0.0055, 0.0055, 0.0055)


def test_a_bus_delay_step_changes_the_latch_delay_from_its_time_on(rungs):
    _, r2 = rungs
    timing = dataclasses.replace(r2.timing, command_delay_steps=((1.5, 0.0095),))
    log, _ = episode(spec_at(0.0055), dataclasses.replace(r2, timing=timing))
    delays = latch_delays(log)
    sent = np.asarray(log.sent_t)
    dt = 1.0 / 4000.0
    before = delays[(sent < 1.5 - 0.02) & np.isfinite(delays[:, 0]), 0]
    after = delays[(sent >= 1.5) & np.isfinite(delays[:, 0]), 0]
    assert before.size > 50 and after.size > 50
    assert before.min() >= 0.0055 - 1e-9 and before.max() <= 0.0055 + dt + 1e-9
    assert after.min() >= 0.0095 - 1e-9 and after.max() <= 0.0095 + dt + 1e-9


# ── 5.2 the calibration routine ──────────────────────────────────────────────────────────────────────────────


def test_xcorr_calibration_measures_a_staggered_bus_and_compares_it_with_the_nominal(rungs):
    from robot_controllers.bench.rungs import bus_variant
    from robot_controllers.calibration import calibrate_bus

    _, r2 = rungs
    truth = (0.002, 0.008)
    rung = bus_variant(r2, truth)
    wrong = calibrate_bus(spec_at, rung, nominal=BusTiming(0.0055, source="assumed"), method="xcorr")
    np.testing.assert_allclose(wrong.measured.command_delay_s, truth, atol=0.001)
    np.testing.assert_allclose(wrong.delta_s, np.asarray(wrong.measured.command_delay_s) - 0.0055, atol=1e-12)
    assert wrong.within is False and wrong.method == "xcorr" and wrong.why
    right = calibrate_bus(spec_at, rung, nominal=BusTiming(truth, source="truth"), method="xcorr")
    assert right.within is True
    assert right.measured.source.startswith("calibrated")


def test_sweep_calibration_finds_the_bus_delay_from_tracking_alone(rungs):
    from robot_controllers.calibration import calibrate_bus

    _, r2 = rungs
    result = calibrate_bus(spec_at, r2, nominal=BusTiming(0.0055), method="sweep")
    measured = result.measured.per_joint(2)
    assert measured[0] == measured[1]
    assert abs(measured[0] - 0.0055) <= 0.0015
    assert result.method == "sweep"


# ── 5.2 the live monitor ─────────────────────────────────────────────────────────────────────────────────────


def _feed(monitor, log, chunk_s=0.1):
    events = []
    t = np.asarray(log.t)
    sent = np.asarray(log.sent_t)
    edges = np.arange(0.0, float(t[-1]) + chunk_s, chunk_s)
    for start, end in zip(edges[:-1], edges[1:], strict=True):
        rows = (t >= start) & (t < end)
        ticks = (sent >= start) & (sent < end)
        events += monitor.update(t[rows], np.asarray(log.q)[rows], sent[ticks], np.asarray(log.ref_position)[ticks])
    return events


def test_the_monitor_alerts_when_the_bus_slows_and_stays_quiet_when_it_does_not(rungs):
    from robot_controllers.monitor import LagMonitor

    r1, _ = rungs
    expected = BusTiming(0.0055, source="calibrated:test")

    def monitor():
        return LagMonitor(expected, lookahead_s=0.0055, window_s=1.0, step_s=0.25, alert_s=0.002, confirm=2)

    steady, _ = episode(spec_at(0.0055), r1)
    assert [e for e in _feed(monitor(), steady) if e.kind == "drift"] == []

    slowed_timing = dataclasses.replace(r1.timing, command_delay_steps=((2.0, 0.0095),))
    slowed, _ = episode(spec_at(0.0055), dataclasses.replace(r1, timing=slowed_timing))
    drift = [e for e in _feed(monitor(), slowed) if e.kind == "drift"]
    assert drift, "no alert after the bus slowed from 5.5 to 9.5 ms"
    assert all(e.t > 2.0 for e in drift)
    first = min(drift, key=lambda e: e.t)
    assert first.t <= float(r1.duration_s)
    assert abs(first.measured_s - 0.0095) <= 0.0015 and first.expected_s == pytest.approx(0.0055)
