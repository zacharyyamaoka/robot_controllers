"""The planar_arm plant (bench/arm.py, bench/plant.py): kinematics, dynamics, the belt contact's calibration,
the belt's friction, pushes, the hard stop and the bus timing of read/write."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest
from robot_msgs.control import ServoCommand
from robot_msgs.std import Header

from robot_controllers.bench.arm import (
    ARM,
    HOME_Q,
    build_model,
    fk,
    ik,
    jacobian,
    joint_motion_from_tool,
    nominal_dynamics,
    tool_position,
)
from robot_controllers.bench.plant import PlanarArmPlant
from robot_controllers.bench.types import HostTiming, Push, Scene

G = 9.81
DT = ARM.physics_dt_s


# ── kinematics and dynamics ─────────────────────────────────────────────────────────────────────────────────


def test_home_is_link_one_level_and_link_two_hanging_with_the_tool_at_0_30_0_25():
    assert HOME_Q == pytest.approx((0.0, -math.pi / 2), abs=1e-12)
    assert fk(np.array(HOME_Q)) == pytest.approx((0.30, 0.25), abs=1e-12)


@pytest.mark.parametrize("point", [(0.30, 0.25), (0.42, 0.20), (0.25, 0.32), (0.30, 0.155), (0.10, 0.60)])
def test_ik_is_elbow_up_and_inverts_fk(point):
    q = ik(*point)
    assert q[1] <= 0.0  # elbow-up with this sign convention
    assert fk(q) == pytest.approx(point, abs=1e-12)
    elbow_xz = (ARM.link1_length_m * math.cos(q[0]), ARM.base_z_m + ARM.link1_length_m * math.sin(q[0]))
    # the elbow sits above (or on) the shoulder->tool line
    cross = point[0] * (elbow_xz[1] - ARM.base_z_m) - (point[1] - ARM.base_z_m) * elbow_xz[0]
    assert cross >= -1e-12


def test_ik_refuses_a_point_out_of_reach():
    with pytest.raises(ValueError):
        ik(0.60, 0.50)


@pytest.mark.parametrize("q", [(0.3, -1.0), (-0.4, -2.0), HOME_Q, (1.0, -0.5)])
def test_fk_matches_mujoco_tool_body_so_the_sign_convention_is_the_documented_one(q):
    model, _ = build_model(Scene())
    data = mujoco.MjData(model)
    data.qpos[:2] = q
    mujoco.mj_forward(model, data)
    assert tool_position(np.array(q)) == pytest.approx(data.xpos[model.body("tool").id], abs=1e-12)


def test_a_positive_angle_turns_a_link_from_plus_x_toward_plus_z():
    model, _ = build_model(Scene())
    data = mujoco.MjData(model)
    data.qpos[:2] = (0.1, 0.0)  # both links straight out, raised 0.1 rad
    mujoco.mj_forward(model, data)
    assert data.xpos[model.body("tool").id][2] == pytest.approx(
        ARM.base_z_m + ARM.reach_m * math.sin(0.1), abs=1e-12
    )


def test_joint_motion_from_tool_matches_finite_differences():
    t = np.linspace(0.0, 0.5, 501)
    points = np.stack([0.30 + 0.1 * np.sin(4 * t), 0.25 + 0.05 * np.cos(3 * t)], axis=1)
    velocities = np.stack([0.4 * np.cos(4 * t), -0.15 * np.sin(3 * t)], axis=1)
    accelerations = np.stack([-1.6 * np.sin(4 * t), -0.45 * np.cos(3 * t)], axis=1)
    q, qd, qdd = joint_motion_from_tool(points, velocities, accelerations)
    np.testing.assert_allclose(qd[1:-1], np.gradient(q, t, axis=0)[1:-1], atol=2e-4)
    np.testing.assert_allclose(qdd[2:-2], np.gradient(qd, t, axis=0)[2:-2], atol=2e-3)
    np.testing.assert_allclose(np.einsum("nij,nj->ni", jacobian(q), qd), velocities, atol=1e-12)


def test_nominal_dynamics_is_gravity_at_rest_and_the_rigid_body_inertia_in_motion():
    l1, l2 = ARM.link1_length_m, ARM.link2_length_m
    m1, m2, mt = ARM.link1_mass_kg, ARM.link2_mass_kg, ARM.tool_mass_kg
    for q in ((0.0, -math.pi / 2), (0.3, -1.1), (-0.5, -0.4)):
        tau = nominal_dynamics(np.array(q), np.zeros(2), np.zeros(2))
        outer = (m2 * l2 / 2 + mt * l2) * math.cos(q[0] + q[1])
        expected = (G * ((m1 * l1 / 2 + (m2 + mt) * l1) * math.cos(q[0]) + outer), G * outer)
        assert tau == pytest.approx(expected, abs=1e-9)
    # elbow inertia about its axis: rod (m L^2/3) + tool (point mass + sphere) + armature, at any shoulder
    # angle
    q = np.array([0.2, -1.0])
    tau = nominal_dynamics(q, np.zeros(2), np.array([0.0, 1.0])) - nominal_dynamics(
        q, np.zeros(2), np.zeros(2)
    )
    elbow_inertia = m2 * l2**2 / 3 + mt * l2**2 + 0.4 * mt * ARM.tool_radius_m**2 + ARM.armature_kg_m2[1]
    assert tau[1] == pytest.approx(elbow_inertia, rel=1e-6)
    # viscous damping is in the nominal model, dry friction is not
    moving = nominal_dynamics(q, np.array([0.0, 2.0]), np.zeros(2)) - nominal_dynamics(
        q, np.zeros(2), np.zeros(2)
    )
    coriolis_elbow = 0.0  # qd1 = 0: no velocity product reaches the elbow
    assert moving[1] == pytest.approx(ARM.damping_nm_s_rad[1] * 2.0 + coriolis_elbow, abs=1e-9)


# ── the belt contact's calibration ──────────────────────────────────────────────────────────────────────────


def _press_on_still_belt(press_n: float, stiffness: float, damping: float, seconds: float = 0.6):
    model, calibration = build_model(
        Scene(belt=True, contact_stiffness_n_m=stiffness, contact_damping_n_s_m=damping)
    )
    data = mujoco.MjData(model)
    tool = model.body("tool").id
    data.qpos[:2] = ik(0.30, ARM.tool_touch_z_m + 0.0005)
    forces, penetrations = [], []
    for _ in range(int(seconds / DT)):
        data.qfrc_applied[:2] = nominal_dynamics(
            data.qpos[:2].copy(), np.zeros(2), np.zeros(2)
        )  # gravity off
        data.xfrc_applied[tool, :3] = (0.0, 0.0, -press_n)
        mujoco.mj_step(model, data)
        forces.append(sum(data.efc_force[data.contact[i].efc_address] for i in range(data.ncon)))
        penetrations.append(ARM.tool_touch_z_m - data.xpos[tool][2])
    return np.array(forces), np.array(penetrations), calibration


@pytest.mark.parametrize("stiffness,damping,press", [(1.0e5, 200.0, 30.0), (2.0e4, 50.0, 10.0)])
def test_belt_contact_at_rest_follows_the_calibrated_spring_law(stiffness, damping, press):
    """Spring law within 10 %: MuJoCo's negative solref is acceleration units, so the calibration has to
    convert; a missing conversion is off by 1 / ((1 - d) diag) = 2x here, far outside the band."""
    forces, penetrations, calibration = _press_on_still_belt(press, stiffness, damping)
    force, penetration = forces[-400:].mean(), penetrations[-400:].mean()
    assert penetration > 0
    assert force == pytest.approx(stiffness * penetration, rel=0.10)
    assert force == pytest.approx(
        press, rel=0.10
    )  # what holds the press up is the belt (joint friction aside)
    assert calibration.stiffness_n_m == stiffness and calibration.solref[0] < 0


def test_belt_contact_damping_follows_the_calibration_during_an_impact():
    model, _ = build_model(Scene(belt=True))
    data = mujoco.MjData(model)
    tool = model.body("tool").id
    q = ik(0.30, ARM.tool_touch_z_m + 0.0002)
    data.qpos[:2] = q
    data.qvel[:2] = np.linalg.solve(jacobian(q), np.array([0.0, -0.15]))  # 0.15 m/s down onto the belt
    rows = []
    for _ in range(400):
        q_now, qd_now = data.qpos[:2].copy(), data.qvel[:2].copy()
        data.qfrc_applied[:2] = nominal_dynamics(q_now, np.zeros(2), np.zeros(2))
        mujoco.mj_step(model, data)
        force = sum(data.efc_force[data.contact[i].efc_address] for i in range(data.ncon))
        rows.append((force, ARM.tool_touch_z_m - data.xpos[tool][2], -(jacobian(q_now) @ qd_now)[1]))
    rows = np.array(rows)
    pressing = rows[:, 0] > 0.5
    (fit_stiffness, fit_damping), *_ = np.linalg.lstsq(rows[pressing, 1:], rows[pressing, 0], rcond=None)
    assert fit_stiffness == pytest.approx(1.0e5, rel=0.10)
    assert fit_damping == pytest.approx(200.0, rel=0.10)


# ── the plant ───────────────────────────────────────────────────────────────────────────────────────────────


def _hold_command(plant: PlanarArmPlant, stamp: float, q, **gains) -> ServoCommand:
    q = np.asarray(q, dtype=float)
    return ServoCommand(
        name=plant.joint_names,
        position=q,
        velocity=np.zeros(2),
        feedforward_effort=nominal_dynamics(q, np.zeros(2), np.zeros(2)),
        header=Header(stamp=stamp),
        **gains,
    )


def _drive(plant: PlanarArmPlant, until: float, target, period: float = 0.01, **gains) -> None:
    k = 0
    while k * period < until:
        plant.advance(k * period)
        plant.read()
        plant.write(_hold_command(plant, k * period, target, **gains))
        k += 1
    plant.advance(until)


def test_belt_friction_slides_at_mu_times_the_press_without_chatter():
    """Regression for the MuJoCo trap that moved friction out of the contact: a light press (a few N, as a
    soft impedance makes) on a belt sliding past at 0.25 m/s must stay in contact with a steady force."""
    plant = PlanarArmPlant()
    start = ik(0.30, ARM.tool_touch_z_m + 0.002)
    plant.reset(
        Scene(belt=True, belt_speed_m_s=0.25, start_q=tuple(start)), timing=HostTiming(), duration_s=1.2
    )
    press_target = ik(0.30, ARM.tool_touch_z_m - 0.010)
    _drive(plant, 1.2, press_target, stiffness=np.array([40.0, 25.0]), damping=np.array([2.0, 1.0]))
    rows = plant.samples()
    window = rows["t"] > 0.6
    normal, along = rows["contact_force"][window, 2], rows["contact_force"][window, 0]
    assert rows["in_contact"][window].all()
    assert 1.0 < normal.mean() < 10.0 and normal.std() < 0.05 * normal.mean()
    assert along.mean() == pytest.approx(
        ARM.belt_friction * normal.mean(), rel=0.02
    )  # +x: dragged by the belt
    assert not np.isnan(rows["tool_pos"]).any()


def test_read_serves_the_newest_sample_after_the_reply_delay_and_counts_new_samples():
    plant = PlanarArmPlant()
    plant.reset(Scene(), timing=HostTiming(reply_delay_s=0.0015), duration_s=0.2)
    plant.advance(0.1)
    first = plant.read()
    assert first.header.stamp == pytest.approx(0.0985, abs=1e-12)  # floor((0.1 - 0.0015) / 0.25 ms) ticks
    assert first.seq == 1
    again = plant.read()
    assert again.seq == 1 and again.header.stamp == first.header.stamp  # stale: nothing newer has arrived
    plant.advance(0.1001)  # less than one board tick later
    assert plant.read().seq == 1
    plant.advance(0.11)
    later = plant.read()
    assert later.seq == 2 and later.header.stamp == pytest.approx(0.1085, abs=1e-12)
    quantum = 2 * math.pi / 2**14
    assert np.allclose(np.round(later.position / quantum) * quantum, later.position, atol=1e-12)


def test_write_latches_after_the_command_delay():
    plant = PlanarArmPlant()
    plant.reset(Scene(), timing=HostTiming(command_delay_s=0.0055), duration_s=0.2)
    _drive(plant, 0.05, HOME_Q)
    shifted = np.array(HOME_Q) + np.array([0.01, 0.0])
    plant.write(_hold_command(plant, 0.05, shifted))
    plant.advance(0.07)
    tau = plant.samples()["tau"][:, 0]
    latch_tick = int(round(0.0555 / DT))
    assert abs(tau[latch_tick - 1] - tau[latch_tick - 2]) < 0.5
    assert tau[latch_tick] - tau[latch_tick - 1] > 2.5  # kp 300 * 0.01 rad = 3 N*m more, on that tick


def test_a_push_acts_on_the_tool_in_its_window_and_is_recorded():
    plant = PlanarArmPlant()
    push = Push(start_s=0.1, duration_s=0.05, force_n=(15.0, 0.0, 0.0))
    plant.reset(Scene(pushes=(push,)), duration_s=0.3)
    _drive(plant, 0.3, HOME_Q)
    rows = plant.samples()
    inside = (rows["t"] >= 0.1 - 1e-9) & (rows["t"] < 0.15 - 1e-9)
    assert np.all(rows["external_force"][inside, 0] == 15.0) and np.all(
        rows["external_force"][~inside] == 0.0
    )
    rest_x = rows["tool_pos"][rows["t"] < 0.1, 0].mean()
    assert rows["tool_pos"][inside, 0].max() > rest_x + 0.002  # pushed along +x
    kinds = [(event["kind"], round(event["t"], 6)) for event in plant.events]
    assert ("push_start", 0.1) in kinds and ("push_end", 0.15) in kinds


def test_the_elbow_hard_stop_holds_the_joint_at_its_limit():
    plant = PlanarArmPlant()
    plant.reset(Scene(), duration_s=2.0)
    beyond = np.array([HOME_Q[0], math.radians(-170.0)])
    _drive(plant, 2.0, beyond, max_effort=np.array([12.0, 3.0]))
    elbow = plant.samples()["q"][:, 1]
    stop = ARM.elbow_range_rad[0]
    assert elbow.min() > stop - math.radians(1.0)
    assert elbow[-1] == pytest.approx(stop, abs=math.radians(1.0))


def test_the_plant_records_one_finite_row_per_physics_step():
    plant = PlanarArmPlant()
    plant.reset(Scene(belt=True), duration_s=0.5)
    _drive(plant, 0.5, HOME_Q)
    rows = plant.samples()
    assert len(rows["t"]) == int(round(0.5 / DT))
    np.testing.assert_allclose(np.diff(rows["t"]), DT, atol=1e-12)
    for name, values in rows.items():
        assert values.shape[0] == len(rows["t"]), name
        if values.dtype.kind == "f":
            assert np.isfinite(values).all(), name
    assert plant.meta()["contact_calibration"]["stiffness_n_m"] == 1.0e5
