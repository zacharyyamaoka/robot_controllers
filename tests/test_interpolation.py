"""Amendment 4, §4.1, beyond the acceptance file: what each placement reads at the middle of the hold, the
split damper's secant target, per-joint lookahead with a secant and a mid-hold feed-forward, the at-rest rule
per joint, and the Layout's validation and name."""

from __future__ import annotations

import numpy as np
import pytest
from robot_msgs.sensor import JointState
from robot_msgs.trajectory import JointTrajectory

from robot_controllers.bench.arm_models import planar_arm_feedforward, planar_arm_mass_matrix
from robot_controllers.controllers._sampling import TrajectorySampler
from robot_controllers.controllers.layout import LayoutConfig, LayoutController
from robot_controllers.models import PD, InertiaScaledPD, Layout

JOINTS = ("shoulder", "elbow")
SAMPLE_S = 0.004
PERIOD_S = 0.02


def trajectory(duration_s=0.6, rest_s=0.1):
    times = np.round(np.arange(0.0, duration_s + rest_s + SAMPLE_S / 2, SAMPLE_S), 12)
    phase = np.clip(times / duration_s, 0.0, 1.0)[:, None]
    start, delta = np.array([0.1, -1.2]), np.array([0.4, 0.3])
    acc = delta / duration_s**2 * (60 * phase - 180 * phase**2 + 120 * phase**3)
    return JointTrajectory(
        t=times,
        joint_names=JOINTS,
        positions=start + delta * (10 * phase**3 - 15 * phase**4 + 6 * phase**5),
        velocities=delta / duration_s * (30 * phase**2 - 60 * phase**3 + 30 * phase**4),
        accelerations=acc,
        effort=0.5 * acc + np.array([2.0, -1.0]),
    )


def state(q=(0.1, -1.2), qd=(0.0, 0.0)):
    return JointState(name=JOINTS, position=np.asarray(q, float), velocity=np.asarray(qd, float))


def run(config, ticks=40, q=(0.1, -1.2), qd=(0.0, 0.0)):
    controller = LayoutController(trajectory(), config)
    return [controller.tick(state(q, qd), 0.0 if k == 0 else PERIOD_S)[0] for k in range(ticks)], controller


def sample(time):
    return TrajectorySampler(trajectory()).sample(time)


def accel(time):
    traj = trajectory()
    return np.array([np.interp(time, traj.t, traj.accelerations[:, j]) for j in range(2)])


def test_host_reference_reads_position_velocity_and_acceleration_at_the_middle_of_the_hold():
    ff = planar_arm_feedforward(("gravity", "inertia", "coriolis", "viscous"))
    layout = Layout("host_reference", "board", interpolation="secant_mid_ff")
    commands, _ = run(LayoutConfig(feedforward=ff, layout=layout, control_period_s=PERIOD_S), ticks=30)
    end = trajectory().t[-1]
    for k, command in enumerate(commands):
        mid = k * PERIOD_S + PERIOD_S / 2
        p, v, _ = sample(mid)
        a = accel(mid)
        if mid >= end:
            v, a = np.zeros(2), np.zeros(2)
        np.testing.assert_allclose(command.feedforward_effort, ff(p, v, a), atol=1e-9, err_msg=f"tick {k}")
        np.testing.assert_allclose(command.position, sample(k * PERIOD_S)[0], atol=1e-12)


def test_host_measured_keeps_the_measured_state_and_takes_the_acceleration_at_the_middle():
    ff = planar_arm_feedforward(("gravity", "inertia", "coriolis", "viscous"))
    q, qd = (0.15, -1.1), (0.2, -0.1)
    layout = Layout("host_measured", "board", interpolation="secant_mid_ff")
    commands, _ = run(LayoutConfig(feedforward=ff, layout=layout, control_period_s=PERIOD_S), 20, q, qd)
    for k, command in enumerate(commands):
        mid = k * PERIOD_S + PERIOD_S / 2
        np.testing.assert_allclose(command.feedforward_effort, ff(np.array(q), np.array(qd), accel(mid)),
                                   atol=1e-9, err_msg=f"tick {k}")


def test_plan_reads_its_column_at_the_middle_and_equals_host_reference_on_the_grid():
    ff = planar_arm_feedforward(("gravity", "inertia", "coriolis", "viscous"))
    # T = 8 ms: the middle of every hold is a trajectory sample, where plan == host_reference exactly
    period = 2 * SAMPLE_S
    configs = [
        LayoutConfig(feedforward=ff, layout=Layout(at, "board", interpolation="secant_mid_ff"),
                     control_period_s=period)
        for at in ("plan", "host_reference")
    ]
    plan, reference = (LayoutController(trajectory(), config) for config in configs)
    for k in range(60):
        dt = 0.0 if k == 0 else period
        a, _ = plan.tick(state(), dt)
        b, _ = reference.tick(state(), dt)
        np.testing.assert_allclose(a.feedforward_effort, b.feedforward_effort, atol=1e-9, err_msg=f"tick {k}")


def test_split_sends_the_secant_velocity_to_the_board_damper_and_keeps_the_spring_at_the_tick():
    fb = PD((40.0, 25.0), (3.0, 1.0))
    q = (0.12, -1.18)
    layout = Layout("trajectory", "split", interpolation="secant")
    commands, _ = run(LayoutConfig(feedback=fb, layout=layout, control_period_s=PERIOD_S), 30, q)
    end = trajectory().t[-1]
    for k, command in enumerate(commands):
        s = k * PERIOD_S
        p = sample(s)[0]
        secant = np.zeros(2) if s >= end else (sample(s + PERIOD_S)[0] - p) / PERIOD_S
        np.testing.assert_allclose(command.velocity, secant, atol=1e-12, err_msg=f"tick {k}")
        spring = np.array([40.0, 25.0]) * (p - np.array(q))
        np.testing.assert_allclose(command.feedforward_effort, sample(s)[2] + spring, atol=1e-12)
        np.testing.assert_allclose(command.damping, (3.0, 1.0))


def test_host_feedback_is_evaluated_at_the_tick_on_the_tangent_reference():
    fb = InertiaScaledPD(6.0, 0.9, planar_arm_mass_matrix())
    q, qd = np.array([0.12, -1.18]), np.array([0.05, -0.02])
    tangent, _ = run(LayoutConfig(feedback=fb, layout=Layout("trajectory", "host")), 20, q, qd)
    secant, _ = run(
        LayoutConfig(feedback=fb, layout=Layout("trajectory", "host", interpolation="secant_mid_ff"),
                     control_period_s=PERIOD_S), 20, q, qd)
    for k, (a, b) in enumerate(zip(tangent, secant, strict=True)):
        s = k * PERIOD_S
        p, v, _ = sample(s)
        expected = fb(p, v, q, qd)
        np.testing.assert_allclose(a.feedforward_effort - sample(s)[2], expected, atol=1e-9)
        # only the feed-forward moves to the middle of the hold; the host feedback stays at the tick
        np.testing.assert_allclose(b.feedforward_effort - sample(s + PERIOD_S / 2)[2], expected, atol=1e-9)


def test_per_joint_lookahead_with_a_secant_and_the_at_rest_rule_per_joint():
    lookahead = (0.0, 0.1)  # the elbow runs 100 ms ahead: it reaches the end five ticks before the shoulder
    layout = Layout("trajectory", "board", lookahead_s=lookahead, interpolation="secant_mid_ff")
    commands, _ = run(LayoutConfig(layout=layout, control_period_s=PERIOD_S), ticks=40)
    end = trajectory().t[-1]
    for k, command in enumerate(commands):
        for j, ahead in enumerate(lookahead):
            s = k * PERIOD_S + ahead
            p = sample(s)[0][j]
            secant = 0.0 if s >= end else (sample(s + PERIOD_S)[0][j] - p) / PERIOD_S
            assert command.position[j] == pytest.approx(p, abs=1e-12), (k, j)
            assert command.velocity[j] == pytest.approx(secant, abs=1e-9), (k, j)
            assert command.feedforward_effort[j] == pytest.approx(sample(s + PERIOD_S / 2)[2][j], abs=1e-12)


def test_a_replan_to_a_trajectory_of_another_joint_count_is_refused_with_a_per_joint_lookahead():
    controller = LayoutController(trajectory(), LayoutConfig(layout=Layout(lookahead_s=(0.001, 0.002))))
    before, _ = controller.tick(state(), 0.0)
    three = JointTrajectory(t=[0.0, 1.0], joint_names=("a", "b", "c"), positions=np.zeros((2, 3)))
    with pytest.raises(ValueError):
        controller.replace_input(three)
    after, _ = controller.tick(state(), SAMPLE_S)
    assert after.name == JOINTS


def test_layout_validates_and_names_the_new_fields():
    assert Layout(interpolation="secant").name == "trajectory+board+interp=secant"
    assert Layout(lookahead_s=(0.002, 0.008)).name == "trajectory+board+lookahead=0.002/0.008s"
    assert Layout(lookahead_s=(0.0, 0.0)).name == "trajectory+board"
    assert Layout(lookahead_s=[0.001, 0.002]).lookahead_s == (0.001, 0.002)  # hashable, frozen
    assert Layout(lookahead_s=0.003).lookaheads(3).tolist() == [0.003] * 3
    for bad in ((), (0.001, float("nan")), -0.001):
        with pytest.raises(ValueError):
            Layout(lookahead_s=bad)
    with pytest.raises(ValueError):
        LayoutConfig(control_period_s=0.0)
    config = LayoutConfig(layout=Layout(lookahead_s=(0.001, 0.003)))
    report = LayoutController(trajectory(), config).diagnostics()
    assert report["lookahead_s"] == 0.003 and report["lookahead_s.elbow"] == 0.003
    assert report["interpolation"] == "tangent" and report["control_period_s"] == 0.01
