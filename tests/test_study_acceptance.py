"""Acceptance for CONTRACT amendment 4: host rate, speed, lookahead calibration, a moving target, and a UR5e.

Written by the dispatcher BEFORE the build and read-only to the builder (CONTRACT §4.7). Every check is observable
behaviour: the ServoCommand a tick sends, a latch time the plant logged, a pure function against an independent oracle
(MuJoCo's `mj_inverse` / `mj_fullM` on the vendored model), or a bench episode's metrics. The numbers pinned from bench
episodes (0.775°, 0.223°, 0.201°, 0.122°, the 5.5 ms lag) were measured by the dispatcher on `a07fef1` with a scratch
monkeypatch of `TrajectorySampler.sample` and a scratch lag estimator, independent of the builder's code. A builder
that cannot pass one of these reports FAIL; it never edits this file.

Zach, 2026-11-20: "how that can be calibrated on how far the look ahead should be? … how reliabiliy does look ahead
work?" and "Can you please run examples on the UR robot arm. and also try trajectories at different speeds."
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from robot_msgs.sensor import JointState
from robot_msgs.trajectory import JointTrajectory

from robot_controllers.controllers._sampling import TrajectorySampler
from robot_controllers.controllers.layout import LayoutConfig, LayoutController
from robot_controllers.models import Layout

REPO = Path(__file__).resolve().parents[1]
JOINTS = ("shoulder", "elbow")
SAMPLE_S = 0.004
V1 = ("trajectory", "trajectory", "board_gains", "board")  # the simple JTC
UR5E_ARMATURE = (0.5, 0.5, 0.5, 0.08, 0.08, 0.08)
UR5E_DAMPING = (2.0, 2.0, 2.0, 0.3, 0.3, 0.3)
UR5E_FRICTIONLOSS = (4.0, 4.0, 4.0, 0.8, 0.8, 0.8)
UR5E_HOME = (0.0, -1.2, 1.4, -1.6, -1.57, 0.0)


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────


def min_jerk(duration_s=0.8, rest_s=0.2):
    times = np.round(np.arange(0.0, duration_s + rest_s + SAMPLE_S / 2, SAMPLE_S), 12)
    phase = np.clip(times / duration_s, 0.0, 1.0)[:, None]
    start, delta = np.array([0.1, -0.4]), np.array([0.5, -0.3])
    positions = start + delta * (10 * phase**3 - 15 * phase**4 + 6 * phase**5)
    velocities = delta / duration_s * (30 * phase**2 - 60 * phase**3 + 30 * phase**4)
    acc = delta / duration_s**2 * (60 * phase - 180 * phase**2 + 120 * phase**3)
    return JointTrajectory(t=times, joint_names=JOINTS, positions=positions, velocities=velocities,
                           accelerations=acc, effort=0.7 * acc + np.array([1.0, -0.5]))


def at_rest_state() -> JointState:
    return JointState(name=JOINTS, position=[0.1, -0.4], velocity=[0.0, 0.0])


def episode(spec, rung, seed=0):
    from robot_controllers.bench.loop import run_episode
    from robot_controllers.metrics.episode import episode_metrics

    log = run_episode(spec, rung, seed=seed)
    return log, episode_metrics(log, rung)


def spec(*layout, lookahead_s=0.0, interpolation="tangent", arm="planar_arm"):
    from robot_controllers.bench.layouts import layout_spec

    return layout_spec(*layout, lookahead_s=lookahead_s, interpolation=interpolation, arm=arm)


def latch_times(log) -> np.ndarray:
    """(K, J) the plant time at which each host tick's command first ran on each joint (NaN = never)."""
    ticks = np.asarray(log.latched_tick)
    out = np.full((len(log.sent_t), ticks.shape[1]), np.nan)
    for j in range(ticks.shape[1]):
        column = ticks[:, j]
        for k in range(len(log.sent_t)):
            hits = np.flatnonzero(column == k)
            if hits.size:
                out[k, j] = log.t[hits[0]]
    return out


@pytest.fixture(scope="module")
def r2():
    from robot_controllers.bench.rungs import rung

    return rung("R2.FastFreeSpace")


# ── 4.1 interpolation and per-joint lookahead ──────────────────────────────────────────────────────────────


def test_secant_velocity_lands_the_boards_walk_on_the_next_sample():
    traj, period = min_jerk(), 3 * SAMPLE_S
    sampler = TrajectorySampler(traj)
    controller = LayoutController(
        traj, LayoutConfig(layout=Layout("trajectory", "board", interpolation="secant"), control_period_s=period)
    )
    for k in range(60):
        command, _ = controller.tick(at_rest_state(), 0.0 if k == 0 else period)
        s = k * period
        if s >= sampler.end_time:
            np.testing.assert_allclose(command.velocity, 0.0, atol=0, err_msg=f"tick {k}: at rest past the end")
            continue
        expected = (sampler.sample(s + period)[0] - sampler.sample(s)[0]) / period
        np.testing.assert_allclose(command.velocity, expected, atol=1e-12, err_msg=f"tick {k}")
        np.testing.assert_allclose(command.position, sampler.sample(s)[0], atol=1e-12, err_msg=f"tick {k}")
        np.testing.assert_allclose(command.feedforward_effort, sampler.sample(s)[2], atol=1e-12)


def test_secant_mid_ff_takes_the_feedforward_from_the_middle_of_the_hold():
    traj, period = min_jerk(), 3 * SAMPLE_S
    sampler = TrajectorySampler(traj)
    controller = LayoutController(
        traj,
        LayoutConfig(layout=Layout("trajectory", "board", interpolation="secant_mid_ff"), control_period_s=period),
    )
    for k in range(50):
        command, _ = controller.tick(at_rest_state(), 0.0 if k == 0 else period)
        s = k * period
        if s >= sampler.end_time:
            break
        np.testing.assert_allclose(command.feedforward_effort, sampler.sample(s + period / 2)[2], atol=1e-12)
        np.testing.assert_allclose(command.position, sampler.sample(s)[0], atol=1e-12)


def test_default_interpolation_and_scalar_lookahead_are_todays_commands():
    traj = min_jerk()
    old = LayoutController(traj, LayoutConfig(layout=Layout("trajectory", "board", lookahead_s=2 * SAMPLE_S)))
    new = LayoutController(
        traj,
        LayoutConfig(
            layout=Layout("trajectory", "board", lookahead_s=(2 * SAMPLE_S, 2 * SAMPLE_S), interpolation="tangent"),
            control_period_s=0.01,
        ),
    )
    for k in range(40):
        a, _ = old.tick(at_rest_state(), 0.0 if k == 0 else SAMPLE_S)
        b, _ = new.tick(at_rest_state(), 0.0 if k == 0 else SAMPLE_S)
        for field in ("position", "velocity", "feedforward_effort", "stiffness", "damping"):
            x, y = getattr(a, field), getattr(b, field)
            assert (x is None) == (y is None), field
            if x is not None:
                np.testing.assert_array_equal(np.asarray(x, float), np.asarray(y, float), err_msg=f"{field} tick {k}")


def test_per_joint_lookahead_samples_each_joint_ahead_by_its_own_amount():
    traj = min_jerk()
    controller = LayoutController(
        traj, LayoutConfig(layout=Layout("trajectory", "board", lookahead_s=(SAMPLE_S, 3 * SAMPLE_S)))
    )
    for k in range(40):
        command, _ = controller.tick(at_rest_state(), 0.0 if k == 0 else SAMPLE_S)
        assert command.position[0] == pytest.approx(traj.positions[min(k + 1, len(traj) - 1), 0], abs=1e-9)
        assert command.position[1] == pytest.approx(traj.positions[min(k + 3, len(traj) - 1), 1], abs=1e-9)
        assert command.feedforward_effort[0] == pytest.approx(traj.effort[min(k + 1, len(traj) - 1), 0], abs=1e-9)
        assert command.feedforward_effort[1] == pytest.approx(traj.effort[min(k + 3, len(traj) - 1), 1], abs=1e-9)


def test_a_lookahead_tuple_of_the_wrong_length_is_refused_at_build():
    with pytest.raises(ValueError):
        LayoutController(min_jerk(), LayoutConfig(layout=Layout("trajectory", "board", lookahead_s=(0.001,) * 3)))
    with pytest.raises(ValueError):
        Layout("trajectory", "board", lookahead_s=(0.001, -0.002))
    with pytest.raises(ValueError):
        Layout("trajectory", "board", interpolation="cubic")


def test_layout_names_carry_lookahead_and_interpolation():
    assert spec(*V1, lookahead_s=0.0055, interpolation="secant_mid_ff").name == (
        "ff=trajectory@trajectory+fb=board_gains@board+la=5.5ms+interp=secant_mid_ff"
    )
    assert spec(*V1, lookahead_s=(0.0015, 0.0037)).name == "ff=trajectory@trajectory+fb=board_gains@board+la=1.5/3.7ms"
    assert spec(*V1).name == "ff=trajectory@trajectory+fb=board_gains@board"


@pytest.mark.parametrize(
    ("rate_hz", "interpolation", "p95_deg"),
    [(50, "tangent", 0.775), (50, "secant", 0.223), (50, "secant_mid_ff", 0.201), (100, "secant_mid_ff", 0.122)],
)
def test_interpolation_matches_the_dispatchers_scratch_measurement_on_r2(r2, rate_hz, interpolation, p95_deg):
    from robot_controllers.bench.rungs import rate_variant

    _, metrics = episode(spec(*V1, lookahead_s=0.0055, interpolation=interpolation), rate_variant(r2, rate_hz))
    assert metrics["tracking_p95_deg"] == pytest.approx(p95_deg, rel=0.03)


# ── 4.2 the bus: per-joint delay and jitter ────────────────────────────────────────────────────────────────


def test_defaults_and_their_explicit_spelling_give_identical_episodes(r2):
    exact = dataclasses.replace(r2, timing=dataclasses.replace(r2.timing, include_compute=False))
    spelled = dataclasses.replace(
        exact, timing=dataclasses.replace(exact.timing, command_delay_s=(0.0055, 0.0055), command_delay_jitter_s=0.0)
    )
    log_a, a = episode(spec(*V1, lookahead_s=0.0055), exact)
    log_b, b = episode(spec(*V1, lookahead_s=0.0055), spelled)
    np.testing.assert_array_equal(log_a.q, log_b.q)
    for name in ("tracking_p95_deg", "tracking_rms_deg", "saturation_share", "fb_share_pct"):
        assert a[name] == b[name], name


def test_the_amendment_three_ledger_still_reproduces(r2):
    ledger = json.loads((REPO / "results" / "layouts" / "results.json").read_text())
    want = {
        (row["controller"], row["rung"]): row["metrics"]["tracking_p95_deg"]
        for row in ledger["results"]
        if row["rung"] == "R2.FastFreeSpace"
    }
    for lookahead, name in ((0.0, "ff=trajectory@trajectory+fb=board_gains@board"),
                            (0.0055, "ff=trajectory@trajectory+fb=board_gains@board+la=5.5ms")):
        _, metrics = episode(spec(*V1, lookahead_s=lookahead), r2)
        assert metrics["tracking_p95_deg"] == pytest.approx(want[(name, "R2.FastFreeSpace")], rel=1e-3), name


def test_per_joint_command_delays_latch_each_joint_at_its_own_delay(r2):
    from robot_controllers.bench.rungs import bus_variant

    rung = bus_variant(r2, (0.002, 0.008))
    assert rung.id == "R2.FastFreeSpace·bus2/8ms"
    log, _ = episode(spec(*V1), rung)
    delay = latch_times(log) - np.asarray(log.sent_t)[:, None]
    dt = 1.0 / 4000.0
    for j, d in enumerate((0.002, 0.008)):
        column = delay[:, j][np.isfinite(delay[:, j])]
        assert column.size > 0.9 * len(log.sent_t)
        assert column.min() >= d - 1e-9 and column.max() <= d + dt + 1e-9, (j, column.min(), column.max())


def test_bus_jitter_is_bounded_first_in_first_out_and_seeded(r2):
    from robot_controllers.bench.rungs import bus_variant

    rung = bus_variant(r2, 0.0055, 0.002)
    assert rung.id == "R2.FastFreeSpace·bus5.5ms±2ms"
    log, _ = episode(spec(*V1, lookahead_s=0.0055), rung, seed=3)
    latches = latch_times(log)
    delay = latches - np.asarray(log.sent_t)[:, None]
    dt = 1.0 / 4000.0
    finite = delay[np.isfinite(delay)]
    assert finite.min() >= 0.0055 - 0.002 - 1e-9 and finite.max() <= 0.0055 + 0.002 + dt + 1e-9
    assert np.std(delay[:, 0][np.isfinite(delay[:, 0])]) > 0.0005  # the jitter is really drawn (U(-2, 2) ms: 1.15)
    for j in range(latches.shape[1]):
        column = latches[:, j][np.isfinite(latches[:, j])]
        assert np.all(np.diff(column) >= 0.0), "a command latched before the one sent before it"
    again, _ = episode(spec(*V1, lookahead_s=0.0055), rung, seed=3)
    other, _ = episode(spec(*V1, lookahead_s=0.0055), rung, seed=4)
    np.testing.assert_array_equal(latch_times(again), latches)
    assert not np.array_equal(np.nan_to_num(latch_times(other)), np.nan_to_num(latches))


# ── 4.3 lag, calibration and sync ──────────────────────────────────────────────────────────────────────────


def _synthetic_log(shift_s: float, still_second_joint: bool = False):
    sent_t = np.round(np.arange(0.0, 2.0, 0.01), 12)
    ref = np.stack([np.sin(2 * np.pi * 1.3 * sent_t), 0.5 * np.cos(2 * np.pi * 0.7 * sent_t)], axis=1)
    if still_second_joint:
        ref[:, 1] = 0.25
    t = np.round(np.arange(0.0, 2.0, 0.00025), 12)
    q = np.stack([np.interp(t - shift_s, sent_t, ref[:, j]) for j in range(2)], axis=1)
    return SimpleNamespace(t=t, q=q, sent_t=sent_t, ref_position=ref)


@pytest.mark.parametrize("shift_s", [0.0, 0.00625, 0.0115, -0.003])
def test_estimate_lag_recovers_a_known_shift(shift_s):
    from robot_controllers.metrics.lag import estimate_lag_s

    assert estimate_lag_s(_synthetic_log(shift_s)) == pytest.approx(shift_s, abs=1e-9)
    per_joint = estimate_lag_s(_synthetic_log(shift_s), per_joint=True)
    assert per_joint == pytest.approx((shift_s, shift_s), abs=1e-9)


def test_estimate_lag_skips_a_joint_that_does_not_move():
    from robot_controllers.metrics.lag import estimate_lag_s

    lags = estimate_lag_s(_synthetic_log(0.004, still_second_joint=True), per_joint=True)
    assert lags[0] == pytest.approx(0.004, abs=1e-9) and lags[1] is None


def test_one_run_at_lookahead_zero_measures_the_bus_delay(r2):
    _, bare = episode(spec(*V1), r2)
    _, ahead = episode(spec(*V1, lookahead_s=0.0055), r2)
    assert 5.0 <= bare["tracking_lag_ms"] <= 6.0
    assert abs(ahead["tracking_lag_ms"]) <= 0.5


def test_calibrate_lookahead_recovers_staggered_per_joint_delays(r2):
    from robot_controllers.bench.rungs import bus_variant
    from robot_controllers.metrics.lag import calibrate_lookahead

    truth = (0.002, 0.008)
    rung = bus_variant(r2, truth)
    calibrated = calibrate_lookahead(
        lambda lookahead: spec(*V1, lookahead_s=lookahead, interpolation="secant_mid_ff"), rung
    )
    assert len(calibrated) == 2
    for got, want in zip(calibrated, truth, strict=True):
        assert got == pytest.approx(want, abs=0.001)


def test_the_sync_rung_cruises_at_belt_speed_inside_its_window():
    from robot_controllers.bench.arm import fk
    from robot_controllers.bench.rungs import rung, sync_variant

    sync = rung("R10.BeltSync@0.25m/s")
    assert sync.family == "sync" and sync.objective == "sync_error_p95_mm" and sync.order == 14
    assert sync_variant(0.25).id == sync.id and sync_variant(0.5).id == "R10.BeltSync@0.50m/s"
    start = np.asarray(sync.scene.start_q, dtype=float)
    traj = sync.inputs(start)[0][1]
    x, z = fk(traj.positions)
    t = np.asarray(traj.t)
    window = (t >= sync.params["sync_start_s"] + 1e-6) & (t <= sync.params["sync_end_s"] - 1e-6)
    speed = np.gradient(x, t)[window]
    np.testing.assert_allclose(speed, 0.25, atol=2e-3)
    np.testing.assert_allclose(z[window], 0.20, atol=1e-6)
    assert x[window].min() >= 0.16 - 1e-3 and x[window].max() <= 0.36 + 1e-3


def test_sync_metrics_follow_their_definition_and_lookahead_closes_the_timing_gap():
    from robot_controllers.bench.rungs import rung

    sync = rung("R10.BeltSync@0.25m/s")
    results = {}
    for lookahead in (0.0, 0.0055):
        log, metrics = episode(spec(*V1, lookahead_s=lookahead), sync)
        t = np.asarray(log.t)
        window = (t >= sync.params["sync_start_s"]) & (t <= sync.params["sync_end_s"])
        sent = np.asarray(log.sent_t)
        target = np.stack([np.interp(t[window], sent, np.asarray(log.ref_tool_pos)[:, i]) for i in range(3)], axis=1)
        tool = np.asarray(log.tool_pos)[window]
        timing_ms = 1000.0 * float(np.mean(target[:, 0] - tool[:, 0])) / 0.25
        error_mm = 1000.0 * float(np.percentile(np.linalg.norm(target - tool, axis=1), 95))
        assert metrics["sync_timing_ms"] == pytest.approx(timing_ms, rel=1e-6, abs=1e-9)
        assert metrics["sync_error_p95_mm"] == pytest.approx(error_mm, rel=1e-6, abs=1e-9)
        results[lookahead] = metrics
    assert results[0.0]["sync_timing_ms"] > results[0.0055]["sync_timing_ms"]
    assert abs(results[0.0055]["sync_timing_ms"]) < abs(results[0.0]["sync_timing_ms"])


def test_amendment_four_metrics_are_in_the_catalogue():
    from robot_controllers.metrics.names import info

    for name, unit, better in (("tracking_lag_ms", "ms", "none"), ("sync_error_p95_mm", "mm", "lower"),
                               ("sync_timing_ms", "ms", "none")):
        metric = info(name)
        assert (metric.unit, metric.family, metric.better) == (unit, "tracking", better), name


# ── 4.4 rung variants ──────────────────────────────────────────────────────────────────────────────────────


def _gate_set(rung):
    return {(g.metric, g.op, float(g.value)) for g in rung.gates}


def test_rate_variant_of_r2_at_50_hz_is_r9(r2):
    from robot_controllers.bench.rungs import rate_variant, rung

    variant, r9 = rate_variant(r2, 50), rung("R9.FastFreeSpace·host50Hz")
    assert variant.id == "R2.FastFreeSpace·host50Hz"
    assert variant.timing == r9.timing
    assert _gate_set(variant) == _gate_set(r9)
    assert variant.duration_s == r9.duration_s and variant.params["host_rate_hz"] == 50
    start = np.array([0.0, -math.pi / 2])
    np.testing.assert_array_equal(variant.inputs(start)[0][1].positions, r9.inputs(start)[0][1].positions)
    assert r2.timing.rate_hz == 100.0  # the base is untouched


def test_speed_variant_reproduces_r1_and_r2_and_reports_peak_speeds(r2):
    from robot_controllers.bench.rungs import rung, speed_variant

    start = np.array([0.0, -math.pi / 2])
    r1 = rung("R1.FreeSpace")
    slow, same = speed_variant(r2, 0.8), speed_variant(r2, 0.3)
    assert slow.id == "R2.FastFreeSpace·move800ms"
    np.testing.assert_array_equal(slow.inputs(start)[0][1].positions, r1.inputs(start)[0][1].positions)
    np.testing.assert_array_equal(same.inputs(start)[0][1].positions, r2.inputs(start)[0][1].positions)
    assert _gate_set(slow) == _gate_set(r2) and slow.duration_s == r1.duration_s
    speeds = [speed_variant(r2, m).params["peak_joint_speed_deg_s"] for m in (1.2, 0.8, 0.5, 0.3, 0.2)]
    assert all(b > a for a, b in zip(speeds, speeds[1:])), speeds
    assert speed_variant(r2, 0.3).params["peak_tool_speed_m_s"] > 0.0


# ── 4.5 the UR5e ───────────────────────────────────────────────────────────────────────────────────────────


def _ur5e_oracle():
    import mujoco

    from robot_controllers.bench.ur5e import MODEL_PATH

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    model.dof_armature[:] = UR5E_ARMATURE
    model.dof_damping[:] = UR5E_DAMPING
    model.dof_frictionloss[:] = 0.0
    return mujoco, model, mujoco.MjData(model)


def test_the_vendored_ur5e_is_bam_pid_tunnings_fixture():
    from robot_controllers.bench.ur5e import MODEL_PATH, UR5E_HOME_Q, UR5E_JOINTS

    source = Path("/home/bam/bam_ws/src/core/mdp/setup/bam_pid_tunning/tests/fixtures/ur5e_bare.xml")
    if source.exists():
        assert Path(MODEL_PATH).read_bytes() == source.read_bytes()
    assert (Path(MODEL_PATH).parent / "LICENSE").exists() or (Path(MODEL_PATH).parent / "LICENSE.ur5e").exists()
    assert UR5E_JOINTS == ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint",
                           "wrist_2_joint", "wrist_3_joint")
    assert tuple(UR5E_HOME_Q) == pytest.approx(UR5E_HOME)


def test_ur5e_feedforward_models_match_mj_inverse():
    from robot_controllers.bench.ur5e import ur5e_feedforward

    mujoco, model, data = _ur5e_oracle()
    rng = np.random.default_rng(7)
    rigid, gravity, coulomb = (ur5e_feedforward(t) for t in ("rigid_body", "gravity", "rigid_body_coulomb"))
    batch_q, batch_qd, batch_qdd, want = [], [], [], []
    for _ in range(8):
        q, qd, qdd = rng.uniform(-2, 2, 6), rng.uniform(-2, 2, 6), rng.uniform(-10, 10, 6)
        data.qpos[:], data.qvel[:], data.qacc[:] = q, qd, qdd
        mujoco.mj_inverse(model, data)
        tau = data.qfrc_inverse.copy()
        np.testing.assert_allclose(rigid(q, qd, qdd), tau, rtol=0, atol=1e-9)
        np.testing.assert_allclose(coulomb(q, qd, qdd), tau + np.array(UR5E_FRICTIONLOSS) * np.tanh(qd / 0.01),
                                   rtol=0, atol=1e-9)
        data.qvel[:], data.qacc[:] = 0.0, 0.0
        mujoco.mj_inverse(model, data)
        np.testing.assert_allclose(gravity(q, qd, qdd), data.qfrc_inverse, rtol=0, atol=1e-9)
        batch_q.append(q), batch_qd.append(qd), batch_qdd.append(qdd), want.append(tau)
    np.testing.assert_allclose(rigid(np.array(batch_q), np.array(batch_qd), np.array(batch_qdd)), np.array(want),
                               rtol=0, atol=1e-9)


def test_ur5e_mass_matrix_matches_mj_full_m():
    from robot_controllers.bench.ur5e import ur5e_mass_matrix

    mujoco, model, data = _ur5e_oracle()
    mass = ur5e_mass_matrix()
    for q in (np.array(UR5E_HOME), np.array([0.4, -0.9, 1.1, -1.2, -1.0, 0.3])):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        full = np.zeros((6, 6))
        mujoco.mj_fullM(model, data, full)
        np.testing.assert_allclose(mass(q), full, rtol=0, atol=1e-9)


def test_ur5e_board_gains_have_the_planar_shoulders_bandwidth():
    import mujoco

    from robot_controllers.bench.arm import ARM, HOME_Q, build_model
    from robot_controllers.bench.types import Scene
    from robot_controllers.bench.ur5e import ur5e_board_config

    planar, _ = build_model(Scene(), ARM)
    planar_data = mujoco.MjData(planar)
    planar_data.qpos[:2] = HOME_Q
    mujoco.mj_forward(planar, planar_data)
    planar_mass = np.zeros((planar.nv, planar.nv))
    mujoco.mj_fullM(planar, planar_data, planar_mass)
    m11 = planar_mass[0, 0]
    wn, zeta = math.sqrt(300.0 / m11), 7.5 / (2.0 * math.sqrt(300.0 * m11))
    _, model, data = _ur5e_oracle()
    data.qpos[:] = UR5E_HOME
    mujoco.mj_forward(model, data)
    full = np.zeros((6, 6))
    mujoco.mj_fullM(model, data, full)
    board = ur5e_board_config()
    np.testing.assert_allclose(board.kp_nm_rad, np.diag(full) * wn**2, rtol=1e-9)
    np.testing.assert_allclose(board.kd_nm_s_rad, 2.0 * zeta * wn * np.diag(full), rtol=1e-9)
    assert tuple(board.torque_cap_nm) == (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)
    assert board.encoder_counts_per_rev == 2**14 and board.rate_hz == 4000.0


def test_ur5e_rungs_and_the_v1_layout_tracks_the_normal_speed_path():
    from robot_controllers.bench.ur5e import ur5e_rungs

    rungs = {r.id: r for r in ur5e_rungs()}
    assert set(rungs) == {"U1.FreeSpace", "U2.FastFreeSpace"}
    assert all(r.plant == "ur5e" for r in rungs.values())
    assert (rungs["U1.FreeSpace"].params["move_s"], rungs["U2.FastFreeSpace"].params["move_s"]) == (1.2, 0.45)
    _, metrics = episode(
        spec(*V1, lookahead_s=0.0055, interpolation="secant_mid_ff", arm="ur5e"), rungs["U1.FreeSpace"]
    )
    assert metrics["tracking_p95_deg"] <= 1.0
    assert metrics["saturation_share"] <= 2.0 and metrics["diverged"] == 0 and metrics["faults"] == 0
    assert metrics["tool_p95_mm"] is not None and metrics["tool_p95_mm"] >= 0.0


def test_one_run_at_lookahead_zero_measures_the_bus_delay_on_the_ur5e_too():
    from robot_controllers.bench.ur5e import ur5e_rungs

    fast = {r.id: r for r in ur5e_rungs()}["U2.FastFreeSpace"]
    _, metrics = episode(spec(*V1, arm="ur5e"), fast)
    assert 4.5 <= metrics["tracking_lag_ms"] <= 6.5


# ── 4.6 the study and its ledger ───────────────────────────────────────────────────────────────────────────


def test_the_study_layouts_are_the_contracts_seven():
    from robot_controllers.bench.study import STUDY_LAYOUTS

    assert tuple(STUDY_LAYOUTS) == (
        "ff=none@none+fb=board_gains@board+la=5.5ms",
        "ff=trajectory@trajectory+fb=board_gains@board",
        "ff=trajectory@trajectory+fb=board_gains@board+la=5.5ms",
        "ff=trajectory@trajectory+fb=board_gains@board+la=5.5ms+interp=secant_mid_ff",
        "ff=rigid_body@plan+fb=pd@split",
        "ff=rigid_body@host_measured+fb=ctc@host",
        "ff=rigid_body@plan+fb=pd@host",
    )


def test_the_committed_study_ledger_covers_every_dimension_and_agrees_with_the_scratch_numbers():
    ledger = json.loads((REPO / "results" / "study" / "results.json").read_text())
    assert ledger["schema"] == "robot-controllers/study-1"
    rows = ledger["rows"]
    seen = {(row["arm"], row["dimension"]) for row in rows}
    for dimension in ("rate", "speed", "rate_speed", "lookahead", "jitter", "stagger", "sync"):
        assert ("planar_arm", dimension) in seen, dimension
    for dimension in ("rate", "speed", "rate_speed", "lookahead", "jitter", "stagger"):
        assert ("ur5e", dimension) in seen, dimension

    def p95(layout, rate):
        hits = [r for r in rows if r["arm"] == "planar_arm" and r["dimension"] == "rate" and r["layout"] == layout
                and float(r["value"]) == rate]
        assert len(hits) == 1, (layout, rate, len(hits))
        return hits[0]["metrics"]["tracking_p95_deg"]

    assert p95("ff=trajectory@trajectory+fb=board_gains@board+la=5.5ms", 50) == pytest.approx(0.775, rel=0.03)
    assert p95("ff=trajectory@trajectory+fb=board_gains@board+la=5.5ms+interp=secant_mid_ff", 50) == pytest.approx(
        0.201, rel=0.03
    )
    assert "summary" in ledger and {"planar_arm", "ur5e"} <= set(ledger["summary"])
