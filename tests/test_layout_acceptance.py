"""Acceptance for CONTRACT amendment 3: feed-forward and feedback as pure functions, and the layout.

Written by the dispatcher BEFORE the build and read-only to the builder (CONTRACT §3.7). Every check is observable
behaviour: the ServoCommand a tick sends, a pure function's output against an independent oracle (MuJoCo's
`mj_inverse` through `arm.nominal_dynamics`), or a bench episode's metrics. A builder that cannot pass one of these
reports FAIL; it never edits this file.

Zach, 2026-11-20: "The feedforward models it should accept as pure functions. the feedback model as well. Success to
me looks like: Given a feed forward model, Given a feedback model. what is the best way you basically lay these
things out...?"
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from robot_msgs.control import ServoCommand
from robot_msgs.sensor import JointState
from robot_msgs.trajectory import JointTrajectory

from robot_controllers.controllers.joint_impedance import JointImpedanceConfig, JointImpedanceController
from robot_controllers.controllers.joint_trajectory import JointTrajectoryConfig, JointTrajectoryController
from robot_controllers.controllers.layout import LayoutConfig, LayoutController
from robot_controllers.models import PD, ZERO_FEEDFORWARD, InertiaScaledPD, Layout

JOINTS = ("shoulder", "elbow")
SAMPLE_S = 0.004
K = (40.0, 25.0)
D = (3.2, 1.5)
FIELDS = ("position", "velocity", "feedforward_effort", "stiffness", "damping", "max_effort")


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────────────


def min_jerk(start=(0.1, -0.4), delta=(0.5, -0.3), duration_s=0.8, rest_s=0.2, accelerations=True, effort=True):
    times = np.round(np.arange(0.0, duration_s + rest_s + SAMPLE_S / 2, SAMPLE_S), 12)
    phase = np.clip(times / duration_s, 0.0, 1.0)[:, None]
    start, delta = np.asarray(start), np.asarray(delta)
    positions = start + delta * (10 * phase**3 - 15 * phase**4 + 6 * phase**5)
    velocities = delta / duration_s * (30 * phase**2 - 60 * phase**3 + 30 * phase**4)
    acc = delta / duration_s**2 * (60 * phase - 180 * phase**2 + 120 * phase**3)
    return JointTrajectory(
        t=times,
        joint_names=JOINTS,
        positions=positions,
        velocities=velocities,
        accelerations=acc if accelerations else None,
        effort=(0.7 * acc + np.array([1.0, -0.5])) if effort else None,
    )


def measured(q, qd) -> JointState:
    """Joints out of the controller's order plus an extra one: a controller must select by name."""
    return JointState(name=("elbow", "wrist", "shoulder"), position=[q[1], 9.9, q[0]], velocity=[qd[1], 9.9, qd[0]])


def wobble(k: int) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic measured state that is never on the reference, so feedback terms are non-zero."""
    return (np.array([0.1 + 0.02 * math.sin(0.3 * k), -0.4 + 0.03 * math.cos(0.2 * k)]),
            np.array([0.2 * math.cos(0.1 * k), -0.1 * math.sin(0.25 * k)]))


IRREGULAR_DT = [0.0, 0.01, 0.0093, 0.0121, 0.01, 0.031, 0.002, 0.01, 0.0107, 0.0099] * 12


def assert_same_command(a: ServoCommand, b: ServoCommand, tol: float, where: str) -> None:
    assert tuple(a.name) == tuple(b.name), where
    for name in FIELDS:
        x, y = getattr(a, name), getattr(b, name)
        if x is None or y is None:
            assert x is None and y is None, f"{where}: {name} {x!r} vs {y!r}"
            continue
        np.testing.assert_allclose(np.asarray(x, float), np.asarray(y, float), rtol=0, atol=tol, equal_nan=True,
                                   err_msg=f"{where}: {name}")


def run_pair(reference, candidate, dts, *, replace_at=None, replacement=None, tol=1e-12):
    for k, dt in enumerate(dts):
        if replace_at is not None and k == replace_at:
            reference.replace_input(replacement)
            candidate.replace_input(replacement)
        q, qd = wobble(k)
        state = measured(q, qd)
        expected, _ = reference.tick(state, dt)
        got, defines = candidate.tick(state, dt)
        assert defines == {} or not defines
        assert_same_command(expected, got, tol, f"tick {k}")
        assert candidate.status == reference.status, f"tick {k}"


# ── 1. the composed controller reaches every built controller, tick for tick ───────────────────────────────


def test_default_layout_is_the_simple_jtc_forward_the_trajectory_feedback_on_the_board():
    # Zach: "just forwards the joint trajectory its given, and feedback runs on the driver".
    config = JointTrajectoryConfig(max_effort_nm=(6.0, 4.0))
    run_pair(
        JointTrajectoryController(min_jerk(), config),
        LayoutController(min_jerk(), LayoutConfig(max_effort_nm=(6.0, 4.0))),
        IRREGULAR_DT,
        replace_at=40,
        replacement=min_jerk(start=(0.3, -0.5), delta=(-0.2, 0.2), duration_s=0.4),
    )


@pytest.mark.parametrize("location", ["board", "host", "host_spring"])
def test_pd_feedback_at_each_place_is_the_matching_joint_impedance(location):
    fb_at = {"board": "board", "host": "host", "host_spring": "split"}[location]
    run_pair(
        JointImpedanceController(min_jerk(), JointImpedanceConfig(location=location, stiffness_nm_rad=K,
                                                                  damping_nm_s_rad=D)),
        LayoutController(min_jerk(), LayoutConfig(feedback=PD(K, D), layout=Layout("trajectory", fb_at))),
        IRREGULAR_DT,
        replace_at=55,
        replacement=min_jerk(start=(0.2, -0.6), delta=(0.1, 0.3), duration_s=0.5),
    )


def test_plan_time_feedforward_equals_host_feedforward_on_the_sample_grid():
    calls = {"n": 0}

    def gravity_like(q, qd, qdd):
        calls["n"] += 1
        return np.array([3.0, 1.0]) * np.cos(q) + 0.2 * qd + 0.05 * qdd

    plan = LayoutController(min_jerk(), LayoutConfig(feedforward=gravity_like, layout=Layout("plan", "board")))
    plan_calls = calls["n"]
    host = LayoutController(min_jerk(), LayoutConfig(feedforward=gravity_like, layout=Layout("host_reference", "board")))
    assert plan_calls >= 1  # plan time evaluates at build
    run_pair(host, plan, [0.0] + [SAMPLE_S] * 240, tol=1e-9)
    assert plan.diagnostics()["plan_compute_s"] > 0.0
    assert host.diagnostics()["plan_compute_s"] == 0.0


def test_placements_other_than_trajectory_ignore_the_effort_column():
    with_effort = LayoutController(min_jerk(effort=True), LayoutConfig(feedforward=ZERO_FEEDFORWARD,
                                                                       layout=Layout("host_reference", "board")))
    command, _ = with_effort.tick(measured(*wobble(0)), 0.0)
    assert np.allclose(command.feedforward_effort, 0.0)
    none = LayoutController(min_jerk(effort=True), LayoutConfig(layout=Layout("none", "board")))
    command, _ = none.tick(measured(*wobble(0)), 0.0)
    assert command.feedforward_effort is None or np.allclose(command.feedforward_effort, 0.0)


# ── 2. what each placement evaluates, and on which state ───────────────────────────────────────────────────


def linear_ff(q, qd, qdd):
    return 2.0 * q + 3.0 * qd + 5.0 * qdd


@pytest.mark.parametrize("with_acc", [True, False])
def test_host_measured_feedforward_uses_measured_position_and_velocity_and_reference_acceleration(with_acc):
    traj = min_jerk(accelerations=with_acc)
    ctrl = LayoutController(traj, LayoutConfig(feedforward=linear_ff, layout=Layout("host_measured", "board")))
    acc = traj.accelerations if with_acc else np.gradient(traj.velocities, traj.t, axis=0)
    for k, dt in enumerate([0.0] + [SAMPLE_S] * 30):
        q, qd = wobble(k)
        command, _ = ctrl.tick(measured(q, qd), dt)
        expected = 2.0 * q + 3.0 * qd + 5.0 * acc[k]
        np.testing.assert_allclose(command.feedforward_effort, expected, atol=1e-9, err_msg=f"tick {k}")
        ff, fb_host = ctrl.last_split()
        np.testing.assert_allclose(ff, expected, atol=1e-9)
        np.testing.assert_allclose(fb_host, 0.0, atol=0)


def test_host_reference_feedforward_uses_the_reference_not_the_measurement():
    traj = min_jerk()
    ctrl = LayoutController(traj, LayoutConfig(feedforward=linear_ff, layout=Layout("host_reference", "board")))
    for k, dt in enumerate([0.0] + [SAMPLE_S] * 30):
        command, _ = ctrl.tick(measured(*wobble(k)), dt)
        expected = 2.0 * traj.positions[k] + 3.0 * traj.velocities[k] + 5.0 * traj.accelerations[k]
        np.testing.assert_allclose(command.feedforward_effort, expected, atol=1e-9, err_msg=f"tick {k}")


def test_host_feedback_rides_in_the_feedforward_field_and_is_reported_split():
    def fb(q_ref, qd_ref, q, qd):
        return np.array([10.0, 20.0]) * (q_ref - q) + np.array([1.0, 2.0]) * (qd_ref - qd)

    fb_pd = PD((10.0, 20.0), (1.0, 2.0))
    traj = min_jerk()
    ctrl = LayoutController(traj, LayoutConfig(feedforward=linear_ff, feedback=fb_pd, layout=Layout("host_reference",
                                                                                                      "host")))
    for k, dt in enumerate([0.0] + [SAMPLE_S] * 30):
        q, qd = wobble(k)
        command, _ = ctrl.tick(measured(q, qd), dt)
        ff = 2.0 * traj.positions[k] + 3.0 * traj.velocities[k] + 5.0 * traj.accelerations[k]
        fb_expected = fb(traj.positions[k], traj.velocities[k], q, qd)
        np.testing.assert_allclose(command.feedforward_effort, ff + fb_expected, atol=1e-9)
        assert np.isnan(command.position).all()
        assert command.velocity is None
        assert np.array_equal(command.stiffness, [0.0, 0.0]) and np.array_equal(command.damping, [0.0, 0.0])
        split_ff, split_fb = ctrl.last_split()
        np.testing.assert_allclose(split_ff, ff, atol=1e-9)
        np.testing.assert_allclose(split_fb, fb_expected, atol=1e-9)


def coupled_mass(q):
    q = np.asarray(q, float)
    return np.array([[1.0 + 0.5 * math.cos(q[1]), 0.2 + 0.1 * math.cos(q[1])],
                     [0.2 + 0.1 * math.cos(q[1]), 0.3]])


def test_inertia_scaled_pd_on_the_host_is_the_full_mass_matrix_at_the_measured_state():
    fb = InertiaScaledPD(bandwidth_hz=10.0, damping_ratio=0.9, mass_matrix=coupled_mass)
    wn = 2 * math.pi * 10.0
    q_ref, qd_ref = np.array([0.3, -0.2]), np.array([0.5, 0.1])
    q, qd = np.array([0.25, -0.1]), np.array([0.4, 0.3])
    expected = coupled_mass(q) @ (wn**2 * (q_ref - q) + 2 * 0.9 * wn * (qd_ref - qd))
    np.testing.assert_allclose(fb(q_ref, qd_ref, q, qd), expected, rtol=1e-12)
    kp, kd = fb.gains(q_ref)
    np.testing.assert_allclose(kp, np.diag(coupled_mass(q_ref)) * wn**2, rtol=1e-12)
    np.testing.assert_allclose(kd, np.diag(coupled_mass(q_ref)) * 2 * 0.9 * wn, rtol=1e-12)


def test_board_scheduled_gains_follow_the_reference_every_tick():
    fb = InertiaScaledPD(bandwidth_hz=8.0, damping_ratio=1.0, mass_matrix=coupled_mass)
    traj = min_jerk()
    ctrl = LayoutController(traj, LayoutConfig(feedback=fb, layout=Layout("trajectory", "board_scheduled")))
    seen = []
    for k, dt in enumerate([0.0] + [SAMPLE_S] * 150):
        command, _ = ctrl.tick(measured(*wobble(k)), dt)
        kp, kd = fb.gains(traj.positions[k])
        np.testing.assert_allclose(command.stiffness, kp, rtol=1e-9, err_msg=f"tick {k}")
        np.testing.assert_allclose(command.damping, kd, rtol=1e-9, err_msg=f"tick {k}")
        np.testing.assert_allclose(command.position, traj.positions[k], atol=1e-9)
        seen.append(command.stiffness[0])
    assert max(seen) - min(seen) > 1.0  # the elbow moves, so the shoulder's scheduled stiffness changes


def test_board_feedback_sends_constant_gains_from_the_first_sample():
    fb = InertiaScaledPD(bandwidth_hz=8.0, damping_ratio=1.0, mass_matrix=coupled_mass)
    traj = min_jerk()
    ctrl = LayoutController(traj, LayoutConfig(feedback=fb, layout=Layout("trajectory", "board")))
    kp0, kd0 = fb.gains(traj.positions[0])
    for k, dt in enumerate([0.0] + [SAMPLE_S] * 150):
        command, _ = ctrl.tick(measured(*wobble(k)), dt)
        np.testing.assert_allclose(command.stiffness, kp0, rtol=1e-12)
        np.testing.assert_allclose(command.damping, kd0, rtol=1e-12)


def test_lookahead_samples_the_reference_ahead_of_the_clock():
    traj = min_jerk()
    ahead = LayoutController(traj, LayoutConfig(layout=Layout("trajectory", "board", lookahead_s=2 * SAMPLE_S)))
    for k, dt in enumerate([0.0] + [SAMPLE_S] * 40):
        command, _ = ahead.tick(measured(*wobble(k)), dt)
        np.testing.assert_allclose(command.position, traj.positions[k + 2], atol=1e-9, err_msg=f"tick {k}")


def test_invalid_layouts_are_refused_at_build():
    no_gains = lambda q_ref, qd_ref, q, qd: q_ref - q  # noqa: E731  a host-only feedback
    for config in (
        LayoutConfig(layout=Layout("trajectory", "host")),  # host feedback needs a feedback model
        LayoutConfig(layout=Layout("trajectory", "split")),
        LayoutConfig(layout=Layout("trajectory", "board_scheduled")),
        LayoutConfig(feedback=no_gains, layout=Layout("trajectory", "board")),  # the board can only close gains
        LayoutConfig(feedback=no_gains, layout=Layout("trajectory", "split")),
        LayoutConfig(layout=Layout("plan", "board")),  # plan needs a feed-forward model
        LayoutConfig(layout=Layout("host_measured", "board")),
    ):
        with pytest.raises(ValueError):
            LayoutController(min_jerk(), config)
    with pytest.raises(ValueError):
        Layout("cloud", "board")
    with pytest.raises(ValueError):
        Layout("plan", "gpu")
    with pytest.raises(ValueError):
        Layout("plan", "board", lookahead_s=-0.001)


def test_layout_controller_is_replannable_and_done_holds_the_end_at_rest():
    traj = min_jerk(duration_s=0.2, rest_s=0.0)
    ctrl = LayoutController(traj, LayoutConfig(feedforward=linear_ff, feedback=PD(K, D),
                                               layout=Layout("plan", "board")))
    for k in range(80):
        command, _ = ctrl.tick(measured(*wobble(k)), 0.0 if k == 0 else 0.01)
    assert ctrl.status == "done"
    np.testing.assert_allclose(command.position, traj.positions[-1], atol=1e-12)
    np.testing.assert_allclose(command.velocity, 0.0, atol=0)
    ctrl.replace_input(min_jerk(start=tuple(traj.positions[-1]), delta=(0.1, 0.1), duration_s=0.3))
    command, _ = ctrl.tick(measured(*wobble(0)), 0.01)
    assert ctrl.status == "running"
    np.testing.assert_allclose(command.position, traj.positions[-1], atol=1e-9)


# ── 3. the planar arm's feed-forward is pure and matches MuJoCo ────────────────────────────────────────────


def _random_states(n=25, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-2.5, 2.5, (n, 2)), rng.uniform(-3.0, 3.0, (n, 2)), rng.uniform(-20.0, 20.0, (n, 2)))


def test_planar_arm_rigid_body_feedforward_matches_mj_inverse():
    pytest.importorskip("mujoco")
    from robot_controllers.bench.arm import nominal_dynamics
    from robot_controllers.bench.arm_models import planar_arm_feedforward

    ff = planar_arm_feedforward(("gravity", "inertia", "coriolis", "viscous"))
    q, qd, qdd = _random_states()
    for i in range(len(q)):
        np.testing.assert_allclose(ff(q[i], qd[i], qdd[i]), nominal_dynamics(q[i], qd[i], qdd[i]), atol=1e-9,
                                   rtol=0, err_msg=f"state {i}")
    np.testing.assert_allclose(ff(q, qd, qdd), nominal_dynamics(q, qd, qdd), atol=1e-9, rtol=0)  # batched


def test_planar_arm_mass_matrix_matches_mj_inverse_columns():
    pytest.importorskip("mujoco")
    from robot_controllers.bench.arm import nominal_dynamics
    from robot_controllers.bench.arm_models import planar_arm_mass_matrix

    mass = planar_arm_mass_matrix()
    q, _, _ = _random_states(8)
    zero = np.zeros(2)
    for qi in q:
        base = nominal_dynamics(qi, zero, zero)
        columns = np.stack([nominal_dynamics(qi, zero, np.eye(2)[j]) - base for j in range(2)], axis=1)
        np.testing.assert_allclose(mass(qi), columns, atol=1e-9, rtol=0)


def test_planar_arm_feedforward_terms_add_up_and_are_pure():
    from robot_controllers.bench.arm_models import planar_arm_feedforward

    q, qd, qdd = _random_states(10)
    keep = (q.copy(), qd.copy(), qdd.copy())
    parts = {t: planar_arm_feedforward((t,)) for t in ("gravity", "inertia", "coriolis", "viscous", "coulomb")}
    whole = planar_arm_feedforward(("gravity", "inertia", "coriolis", "viscous", "coulomb"))
    total = sum(f(q, qd, qdd) for f in parts.values())
    np.testing.assert_allclose(whole(q, qd, qdd), total, atol=1e-12)
    np.testing.assert_allclose(whole(q, qd, qdd), whole(q, qd, qdd), atol=0)  # same inputs, same output
    for a, b in zip(keep, (q, qd, qdd)):
        assert np.array_equal(a, b)  # inputs never written
    # gravity alone does not depend on velocity or acceleration; viscous alone does not depend on position
    np.testing.assert_allclose(parts["gravity"](q, qd, qdd), parts["gravity"](q, 0 * qd, 0 * qdd), atol=0)
    np.testing.assert_allclose(parts["viscous"](q, qd, qdd), parts["viscous"](0 * q, qd, qdd), atol=0)
    # a payload the model is told about raises the gravity torque at the shoulder when the arm is horizontal
    heavier = planar_arm_feedforward(("gravity",), payload_kg=0.3)
    flat = np.zeros(2)
    assert abs(heavier(flat, flat, flat)[0]) > abs(parts["gravity"](flat, flat, flat)[0]) + 0.5


# ── 4. the metrics and rungs the amendment names ───────────────────────────────────────────────────────────


def test_amendment_three_metrics_are_in_the_catalogue():
    from robot_controllers.metrics.names import BY_NAME

    expected = {
        "fb_share_pct": ("%", "decomposition", "lower"),
        "fb_rms_nm": ("N·m", "decomposition", "lower"),
        "ff_rms_nm": ("N·m", "decomposition", "none"),
        "plan_compute_ms_per_s": ("ms/s", "cost", "lower"),
        "host_duty_pct": ("%", "cost", "lower"),
        "wire_bytes_per_s": ("bytes/s", "cost", "lower"),
        "torque_rate_rms_nm_s": ("N·m/s", "health", "lower"),
    }
    for name, (unit, family, better) in expected.items():
        info = BY_NAME[name]
        assert (info.unit, info.family, info.better) == (unit, family, better), name


TIMING = {"missed_cycles", "latency_p99_ms"}


def test_model_error_and_host_rate_rungs_carry_r2s_gates():
    from robot_controllers.bench.rungs import ladder
    from robot_controllers.registry import JOINT_TRAJECTORY

    rungs = {r.id: r for r in ladder(JOINT_TRAJECTORY)}
    r2 = rungs["R2.FastFreeSpace"]
    r2_gates = {(g.metric, g.op, g.value) for g in r2.gates if g.metric not in TIMING}
    payload = rungs["R8.FastFreeSpace·payload"]
    assert payload.scene.payload_kg == pytest.approx(0.3)
    for rung, rate in ((payload, r2.timing.rate_hz), (rungs["R9.FastFreeSpace·host50Hz"], 50.0),
                       (rungs["R9.FastFreeSpace·host500Hz"], 500.0)):
        assert {(g.metric, g.op, g.value) for g in rung.gates if g.metric not in TIMING} == r2_gates, rung.id
        assert rung.timing.rate_hz == pytest.approx(rate), rung.id
        assert rung.objective == "tracking_p95_deg" and rung.minimize
    # the ten rungs before the amendment are unchanged in order
    assert [r.id for r in ladder(JOINT_TRAJECTORY)][:10] == [
        "R1.FreeSpace", "R2.FastFreeSpace", "R3.BeltImpact@0.05m/s", "R3.BeltImpact@0.10m/s",
        "R3.BeltImpact@0.20m/s", "R4.SidewaysPush", "R5.RapidReplan", "R6.HostStall",
        "R7.BeltImpact@0.10m/s·host50Hz", "R7.BeltImpact@0.10m/s·host500Hz",
    ]


# ── 5. the sweep: recommend() and real episodes ────────────────────────────────────────────────────────────


def _candidate(name, passed, tracking, duty, plan=0.0, wire=1000.0):
    return {"name": name, "passed": passed, "tracking_p95_deg": tracking, "host_duty_pct": duty,
            "plan_compute_ms_per_s": plan, "wire_bytes_per_s": wire}


def test_recommend_takes_the_cheapest_passing_layout_within_ten_percent_of_the_best():
    from robot_controllers.bench.layouts import recommend

    candidates = [
        _candidate("best_but_costly", True, 1.00, 5.0),
        _candidate("close_and_cheap", True, 1.05, 1.0, plan=0.5),
        _candidate("close_cheap_less_plan", True, 1.08, 1.0, plan=0.2),
        _candidate("cheapest_but_far", True, 1.30, 0.1),
        _candidate("failed_gates", False, 0.50, 0.01),
    ]
    pick = recommend(candidates)
    assert pick["name"] == "close_cheap_less_plan"
    assert isinstance(pick["why"], str) and pick["why"]
    nobody = recommend([_candidate("a", False, 1.0, 1.0), _candidate("b", False, 0.9, 2.0)])
    assert nobody["name"] is None and nobody["why"]


@pytest.fixture(scope="module")
def r1():
    pytest.importorskip("mujoco")
    from robot_controllers.bench.rungs import ladder
    from robot_controllers.registry import JOINT_TRAJECTORY

    return {r.id: r for r in ladder(JOINT_TRAJECTORY)}["R1.FreeSpace"]


def _episode(spec, rung):
    from robot_controllers.bench.loop import run_episode
    from robot_controllers.metrics.episode import episode_metrics

    return episode_metrics(run_episode(spec, rung, seed=0), rung)


def test_the_trajectory_board_layout_spec_runs_exactly_like_the_jtc(r1):
    import dataclasses

    from robot_controllers.bench.layouts import layout_spec
    from robot_controllers.registry import controller

    spec = layout_spec("trajectory", "trajectory", "board_gains", "board")
    assert spec.name == "ff=trajectory@trajectory+fb=board_gains@board"
    assert spec.runs_on == "board" and spec.maturity == "built"
    # WHY include_compute off: the send time otherwise carries each run's measured CPU time, so two runs of the
    # same law latch a few microseconds apart; with it off the episode is deterministic and the laws must agree.
    exact = dataclasses.replace(r1, timing=dataclasses.replace(r1.timing, include_compute=False))
    jtc = _episode(controller("joint_trajectory_controller"), exact)
    mine = _episode(spec, exact)
    for name in ("tracking_p95_deg", "tracking_rms_deg", "tracking_max_deg", "final_error_deg", "saturation_share"):
        assert mine[name] == pytest.approx(jtc[name], rel=1e-9, abs=1e-12), name


def test_a_feedforward_model_cuts_feedback_share_and_tracking_error(r1):
    from robot_controllers.bench.layouts import layout_spec

    bare = _episode(layout_spec("none", "none", "board_gains", "board"), r1)
    modelled = _episode(layout_spec("rigid_body", "plan", "board_gains", "board"), r1)
    for metrics in (bare, modelled):
        assert metrics["fb_share_pct"] is not None and 0.0 <= metrics["fb_share_pct"] <= 100.0
        assert metrics["fb_rms_nm"] is not None and metrics["fb_rms_nm"] >= 0.0
    assert modelled["fb_share_pct"] < 0.5 * bare["fb_share_pct"]
    assert modelled["tracking_p95_deg"] < bare["tracking_p95_deg"]
    assert modelled["plan_compute_ms_per_s"] > 0.0 and bare["plan_compute_ms_per_s"] == 0.0


def test_host_feedback_costs_host_compute_that_board_feedback_does_not(r1):
    from robot_controllers.bench.layouts import layout_spec

    board = _episode(layout_spec("rigid_body", "plan", "pd", "board"), r1)
    host = _episode(layout_spec("rigid_body", "host_measured", "pd", "host"), r1)
    assert host["host_duty_pct"] > board["host_duty_pct"]
    for metrics in (board, host):
        assert metrics["wire_bytes_per_s"] == pytest.approx(metrics["wire_bytes_per_tick"] * metrics["host_rate_hz"])
