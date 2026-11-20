"""The layout controller beyond the acceptance file: CTC on the host and split, the abort hold whatever the
layout,
plan-time cost across replans, a refused input changing nothing, and the registry's default."""

from __future__ import annotations

import math

import numpy as np
import pytest
from robot_msgs.sensor import JointState
from robot_msgs.trajectory import JointTrajectory

from robot_controllers.bench.arm_models import planar_arm_feedforward, planar_arm_mass_matrix
from robot_controllers.controllers.layout import LayoutConfig, LayoutController
from robot_controllers.models import PD, ZERO_FEEDFORWARD, InertiaScaledPD, Layout
from robot_controllers.registry import controller

JOINTS = ("shoulder", "elbow")
SAMPLE_S = 0.004


def trajectory(start=(0.1, -1.2), delta=(0.4, 0.3), duration_s=0.6, rest_s=0.2, velocities=True):
    times = np.round(np.arange(0.0, duration_s + rest_s + SAMPLE_S / 2, SAMPLE_S), 12)
    phase = np.clip(times / duration_s, 0.0, 1.0)[:, None]
    start, delta = np.asarray(start), np.asarray(delta)
    return JointTrajectory(
        t=times,
        joint_names=JOINTS,
        positions=start + delta * (10 * phase**3 - 15 * phase**4 + 6 * phase**5),
        velocities=(delta / duration_s * (30 * phase**2 - 60 * phase**3 + 30 * phase**4))
        if velocities
        else None,
    )


def state(q, qd=(0.0, 0.0)):
    return JointState(name=JOINTS, position=np.asarray(q, float), velocity=np.asarray(qd, float))


def ctc():
    return InertiaScaledPD(6.0, 0.9, planar_arm_mass_matrix())


def test_ctc_on_the_host_sends_the_full_mass_matrix_law_and_reports_it_split():
    traj = trajectory()
    ff = planar_arm_feedforward(("gravity",))
    fb = ctc()
    ctrl = LayoutController(
        traj, LayoutConfig(feedforward=ff, feedback=fb, layout=Layout("host_reference", "host"))
    )
    q, qd = np.array([0.12, -1.18]), np.array([0.05, -0.02])
    for k, dt in enumerate([0.0] + [SAMPLE_S] * 20):
        command, _ = ctrl.tick(state(q, qd), dt)
        q_ref, qd_ref = traj.positions[k], traj.velocities[k]
        expected_fb = planar_arm_mass_matrix()(q) @ (
            (2 * math.pi * 6.0) ** 2 * (q_ref - q) + 2 * 0.9 * 2 * math.pi * 6.0 * (qd_ref - qd)
        )
        expected_ff = ff(q_ref, qd_ref, np.zeros(2))  # gravity ignores the acceleration
        np.testing.assert_allclose(
            command.feedforward_effort, expected_ff + expected_fb, rtol=1e-12, atol=1e-12
        )
        split_ff, split_fb = ctrl.last_split()
        np.testing.assert_allclose(split_ff, expected_ff, rtol=1e-12)
        np.testing.assert_allclose(split_fb, expected_fb, rtol=1e-12, atol=1e-12)


def test_split_keeps_the_build_time_gains_spring_on_the_host_damper_on_the_board():
    traj = trajectory()
    fb = ctc()
    kp0, kd0 = fb.gains(traj.positions[0])
    ctrl = LayoutController(traj, LayoutConfig(feedback=fb, layout=Layout("trajectory", "split")))
    q = np.array([0.0, -1.0])
    for k, dt in enumerate([0.0] + [SAMPLE_S] * 100):
        command, _ = ctrl.tick(state(q), dt)
        np.testing.assert_allclose(command.feedforward_effort, kp0 * (traj.positions[k] - q), rtol=1e-12)
        np.testing.assert_array_equal(command.stiffness, [0.0, 0.0])
        np.testing.assert_allclose(command.damping, kd0, rtol=1e-15)
        np.testing.assert_allclose(command.velocity, traj.velocities[k], atol=1e-12)
        assert np.isnan(command.position).all()
    assert abs(fb.gains(traj.positions[100])[0][0] - kp0[0]) > 0.1  # scheduling would have differed


def test_path_tolerance_holds_like_the_jtc_whatever_the_layout():
    traj = trajectory()
    far = state((0.5, -1.2))  # the shoulder 0.4 rad off the start
    for layout, fb in (
        (Layout("trajectory", "board"), None),
        (Layout("plan", "host"), PD(50.0, 2.0)),
        (Layout("host_measured", "split"), PD(50.0, 2.0)),
    ):
        ff = None if layout.feedforward_at == "trajectory" else ZERO_FEEDFORWARD
        ctrl = LayoutController(
            traj, LayoutConfig(feedforward=ff, feedback=fb, layout=layout, path_tolerance_rad=0.05)
        )
        ctrl.tick(state(traj.positions[0]), 0.0)
        command, _ = ctrl.tick(far, SAMPLE_S)
        assert ctrl.status == "aborted" and "shoulder" in ctrl.message
        np.testing.assert_allclose(command.position, [0.5, -1.2])
        np.testing.assert_array_equal(command.velocity, [0.0, 0.0])
        np.testing.assert_array_equal(command.feedforward_effort, [0.0, 0.0])
        assert command.stiffness is None and command.damping is None  # the board's own gains hold it
        again, _ = ctrl.tick(state((0.9, 0.0)), SAMPLE_S)
        assert again is command  # latched once, never re-read
        assert all(np.array_equal(part, [0.0, 0.0]) for part in ctrl.last_split())


def test_plan_time_cost_adds_up_over_replans_and_never_reaches_the_tick():
    calls = {"rows": 0}

    def counting(q, qd, qdd):
        calls["rows"] += int(np.shape(q)[0]) if np.ndim(q) == 2 else 1
        return 0.5 * np.asarray(q)

    ctrl = LayoutController(trajectory(), LayoutConfig(feedforward=counting, layout=Layout("plan", "board")))
    first = ctrl.diagnostics()["plan_compute_s"]
    assert first > 0.0 and calls["rows"] == len(trajectory())
    for _ in range(30):
        ctrl.tick(state((0.1, -1.2)), SAMPLE_S)
    assert calls["rows"] == len(trajectory())  # ticks only interpolate the column
    ctrl.replace_input(trajectory(start=(0.3, -1.0), duration_s=0.3))
    report = ctrl.diagnostics()
    assert report["plan_compute_s"] > first and report["plan_loads"] == 2 and report["active_trajectory"] == 1
    assert all(isinstance(v, (int, float, str)) for v in report.values())


def test_a_refused_input_changes_nothing():
    traj = trajectory()
    ctrl = LayoutController(
        traj, LayoutConfig(feedback=PD((40.0, 25.0), (3.0, 1.0)), layout=Layout("trajectory", "board"))
    )
    ctrl.tick(state(traj.positions[0]), 0.0)
    before, _ = ctrl.tick(state(traj.positions[0]), SAMPLE_S)
    three = JointTrajectory(t=[0.0, 1.0], joint_names=("a", "b", "c"), positions=np.zeros((2, 3)))
    with pytest.raises(ValueError):
        ctrl.replace_input(three)  # two gains for three joints
    after, _ = ctrl.tick(state(traj.positions[0]), SAMPLE_S)
    np.testing.assert_allclose(after.position, traj.positions[2])  # still the old trajectory, on its clock
    assert ctrl.active_input == 0 and np.array_equal(after.stiffness, before.stiffness)


def test_none_feedforward_with_host_feedback_sends_only_the_feedback():
    traj = trajectory()
    fb = PD((40.0, 25.0), (3.0, 1.0))
    ctrl = LayoutController(traj, LayoutConfig(feedback=fb, layout=Layout("none", "host")))
    q, qd = np.array([0.0, -1.1]), np.array([0.1, 0.0])
    command, _ = ctrl.tick(state(q, qd), 0.0)
    np.testing.assert_allclose(command.feedforward_effort, fb(traj.positions[0], traj.velocities[0], q, qd))
    assert np.array_equal(ctrl.last_split()[0], [0.0, 0.0])


def test_a_trajectory_without_velocities_or_accelerations_still_plans_on_its_sampled_velocity():
    traj = trajectory(velocities=False)
    plan = LayoutController(
        traj, LayoutConfig(feedforward=lambda q, qd, qdd: qd + 0.1 * qdd, layout=Layout("plan", "board"))
    )
    host = LayoutController(
        traj,
        LayoutConfig(feedforward=lambda q, qd, qdd: qd + 0.1 * qdd, layout=Layout("host_reference", "board")),
    )
    for dt in [0.0] + [SAMPLE_S] * 100:
        a, _ = plan.tick(state((0.1, -1.2)), dt)
        b, _ = host.tick(state((0.1, -1.2)), dt)
        np.testing.assert_allclose(a.feedforward_effort, b.feedforward_effort, atol=1e-9)
        np.testing.assert_allclose(a.velocity, b.velocity, atol=1e-9)


def test_the_registry_default_is_the_simple_jtc():
    spec = controller("layout_controller")
    assert spec.maturity == "built" and spec.runs_on == "board" and spec.default_config == {}
    config = spec.make_config()
    assert (
        config.layout == Layout("trajectory", "board")
        and config.feedforward is None
        and config.feedback is None
    )
    for name in (
        "disturbance_observer_feedforward",
        "iterative_learning_feedforward",
        "joint_pid_board_integrator",
        "joint_pid_host_integrator",
        "model_predictive_controller",
        "torque_feedback_controller",
    ):
        planned = controller(name)
        assert (
            planned.maturity == "planned" and planned.target is None and planned.prior_art and planned.summary
        )
        assert planned.runs_on in ("board", "host", "split")
