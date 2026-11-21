"""Amendment 5, §5.8, beyond the acceptance file: BAM's five-bar twin - the vendored model, the drives and
boards against bam_pid_tunning's own table, the loop closure, the closed-chain models against MuJoCo's
constrained dynamics, the model error, the plant, its rungs and pickling."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robot_controllers.bench import fivebar as fb  # noqa: E402
from robot_controllers.bench.arm_tapes import load_tape, resample, tape_record  # noqa: E402
from robot_controllers.bench.types import Push, Scene  # noqa: E402
from robot_controllers.models import ModelError  # noqa: E402

BAM_PID_TUNNING = Path("/home/bam/bam_ws/src/core/mdp/setup/bam_pid_tunning/src")
HOME = np.asarray(fb.FIVEBAR_HOME_Q)


def _bam_pid_tunning():
    if not BAM_PID_TUNNING.exists():
        pytest.skip("bam_ws's bam_pid_tunning is not on this machine")
    sys.path.insert(0, str(BAM_PID_TUNNING))
    try:
        from bam_pid_tunning.plants import bam_fb_v2
    finally:
        sys.path.remove(str(BAM_PID_TUNNING))
    return bam_fb_v2


def _poses(count=12, seed=5):
    """Drive poses about home and along dev000 (where the targets go)."""
    rng = np.random.default_rng(seed)
    _, q, _, _ = resample(load_tape("bam_fb/workspace-dev-000.npz"))
    about_home = HOME + rng.uniform(-np.radians(8.0), np.radians(8.0), size=(count, 6))
    return np.vstack([about_home, q[:: max(1, len(q) // count)]])


# ── the model ─────────────────────────────────────────────────────────────────────────────────────────────


def test_the_vendored_mjcf_is_reduced_mjcfs_output_and_loads_with_six_drives_and_two_passive_hinges():
    sidecar = json.loads(fb.MODEL_PATH.with_suffix(".json").read_text())
    assert hashlib.sha256(fb.MODEL_PATH.read_bytes()).hexdigest() == sidecar["sha256"]
    plant = fb.load_model()
    names = {plant.joint(j).name for j in range(plant.njnt)}
    assert names == set(fb.MJCF_JOINTS) | set(fb.PASSIVE_JOINTS)  # the claw hinge is gone
    assert plant.nq == plant.nv == 8 and plant.neq == 1 and plant.opt.timestep == 0.00025
    np.testing.assert_allclose(plant.eq_solref[0], fb.CLOSURE_SOLREF)
    np.testing.assert_allclose(plant.eq_solimp[0], fb.CLOSURE_SOLIMP)
    truth = fb.FIVEBAR_TRUTH
    for j, name in enumerate(fb.MJCF_JOINTS):
        dof = plant.jnt_dofadr[plant.joint(name).id]
        assert plant.dof_armature[dof] == truth.armature_kg_m2[j]
        assert plant.dof_frictionloss[dof] == truth.coulomb_nm[j]
        assert plant.dof_damping[dof] == truth.viscous_nm_s_rad[j]
        np.testing.assert_allclose(plant.dof_solref[dof], fb.FRICTION_SOLREF)
    nominal = fb.load_model(nominal=True)
    assert np.all(nominal.dof_frictionloss == 0.0)
    assert nominal.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    with pytest.raises(ValueError, match="never the plant"):
        fb.load_model(model_error=ModelError(0.1))


def test_the_drives_and_boards_are_bam_pid_tunnings_drives_in_the_benchs_units():
    source = _bam_pid_tunning()
    board = fb.fivebar_board_config()
    by_role = {d.role.lower(): d for d in source.DRIVES}
    for j, name in enumerate(fb.FIVEBAR_JOINTS):
        drive = by_role[name]
        assert drive.joint == fb.MJCF_JOINTS[j] and drive.can_id == fb.CAN_IDS[j]
        gains = source.as_found_gains(
            drive
        )  # the oracle's own moteus -> N.m/rad conversion (Gains.from_moteus)
        assert board.kp_nm_rad[j] == pytest.approx(gains.kp, rel=1e-12)
        assert board.kd_nm_s_rad[j] == pytest.approx(gains.kd, rel=1e-12)
        assert board.torque_cap_nm[j] == pytest.approx(drive.torque_limit_nm)
        assert fb.FIVEBAR_TRUTH.armature_kg_m2[j] == drive.truth.armature_kg_m2
        assert fb.FIVEBAR_TRUTH.coulomb_nm[j] == drive.truth.coulomb_nm
        assert fb.FIVEBAR_TRUTH.viscous_nm_s_rad[j] == drive.truth.viscous_nm_s_rad
        assert board.encoder_counts_per_rev == int(drive.truth.encoder_counts_per_rev)
        assert board.velocity_filter_hz == drive.truth.velocity_filter_hz
        assert board.encoder_delay_s == drive.truth.measurement_delay_s
    home = np.radians([source.HOME_DEG[name] for name in fb.MJCF_JOINTS])
    np.testing.assert_allclose(fb.FIVEBAR_HOME_Q, home, atol=1e-15)


# ── the loop ──────────────────────────────────────────────────────────────────────────────────────────────


def test_the_loop_closes_on_the_working_assembly_and_is_a_pure_function_of_the_drives():
    chain = fb._Chain(fb.load_model(nominal=True))
    poses = _poses()
    first = [chain.close(q).copy() for q in poses]
    for q, qpos in zip(poses, first, strict=True):
        chain._pose(qpos, com=True)
        assert np.linalg.norm(chain.residual()) < 1e-10
        assert chain._branch() == chain.working_branch
        np.testing.assert_array_equal(qpos[chain.act], q)
    # the same answer in reverse order: no warm start leaks between calls
    again = [chain.close(q).copy() for q in poses[::-1]][::-1]
    np.testing.assert_array_equal(np.asarray(first), np.asarray(again))


def test_the_plant_starts_closed_and_its_soft_loop_stays_within_microns_along_dev000():
    from robot_controllers.bench.layouts import layout_spec
    from robot_controllers.bench.loop import run_episode

    rung = fb.path_rung("fast")
    log = run_episode(
        layout_spec("trajectory", "trajectory", "board_gains", "board"), rung, plant_factory=fb.FivebarPlant
    )
    assert (
        log.meta["loop_gap_max_m"] < 2e-4
    )  # ArmSim's stiffened closure: ~10 um through its move; dev000 at full speed
    assert not any(e["kind"] == "diverged" for e in log.events)
    np.testing.assert_allclose(log.q[0], rung.scene.start_q, atol=1e-3)


# ── the closed-chain models ───────────────────────────────────────────────────────────────────────────────


def test_gravity_is_arm_sims_least_squares_over_tau_and_lambda():
    model = fb.load_model(nominal=True)
    chain = fb._Chain(model)
    data = mujoco.MjData(model)
    gravity = fb.fivebar_feedforward("gravity")
    for q in _poses(8):
        tau = gravity(q, np.zeros(6), np.zeros(6))
        data.qpos[:] = chain.close(q)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        jacobian = chain.jacobian()
        columns = np.hstack(
            [np.eye(model.nv)[:, chain.act], jacobian.T]
        )  # bam_pid_tunning mujoco_arm.py:260-275
        oracle = np.linalg.lstsq(columns, data.qfrc_bias, rcond=None)[0][:6]
        np.testing.assert_allclose(tau, oracle, rtol=0, atol=1e-9)


def test_rigid_body_drives_the_constrained_plant_along_the_commanded_acceleration():
    """The oracle is MuJoCo's own constrained forward dynamics: the true model (soft closure on, friction
    off), at a closure-consistent state, given tau = rigid_body(q, qd, qdd) on the drives, accelerates the
    drives at qdd."""
    plant_model = fb.load_model()
    plant_model.dof_frictionloss[:] = 0.0
    plant_model.jnt_limited[:] = 0
    data = mujoco.MjData(plant_model)
    chain = fb._Chain(plant_model)
    rigid = fb.fivebar_feedforward("rigid_body")
    rng = np.random.default_rng(9)
    for q in _poses(6):
        qd, qdd = rng.uniform(-1.0, 1.0, 6), rng.uniform(-5.0, 5.0, 6)
        qpos, qvel, _, _ = chain.state(q, qd, qdd)
        data.qpos[:], data.qvel[:] = qpos, qvel
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[chain.act] = rigid(q, qd, qdd)
        mujoco.mj_forward(plant_model, data)
        unpowered = mujoco.MjData(plant_model)
        unpowered.qpos[:], unpowered.qvel[:] = qpos, qvel
        mujoco.mj_forward(plant_model, unpowered)
        # the soft closure (solimp 0.9999) leaves a sliver: well under 1 % of how far the arm is from free
        # fall
        scale = np.max(np.abs(unpowered.qacc[chain.act] - qdd))
        assert np.max(np.abs(data.qacc[chain.act] - qdd)) < 0.01 * scale


def test_coulomb_adds_the_drives_friction_and_the_mass_matrix_is_the_inertia_on_the_drives():
    rigid, coulomb = fb.fivebar_feedforward("rigid_body"), fb.fivebar_feedforward("rigid_body_coulomb")
    qd = np.array([0.3, -0.2, 0.004, 1.0, -1.0, 0.0])
    zero = np.zeros(6)
    extra = coulomb(HOME, qd, zero) - rigid(HOME, qd, zero)
    np.testing.assert_allclose(
        extra, np.asarray(fb.FIVEBAR_TRUTH.coulomb_nm) * np.tanh(qd / 0.01), atol=1e-12
    )
    mass = fb.fivebar_mass_matrix()(HOME)
    base = rigid(HOME, zero, zero)
    columns = np.stack([rigid(HOME, zero, np.eye(6)[j]) - base for j in range(6)], axis=1)
    np.testing.assert_allclose(mass, columns, atol=1e-10)
    np.testing.assert_allclose(mass, mass.T, atol=1e-14)
    assert np.all(np.linalg.eigvalsh(mass) > 0.0)
    rows = _poses(3)[:3]
    assert fb.fivebar_mass_matrix()(rows).shape == (3, 6, 6)
    assert rigid(rows, np.zeros_like(rows), np.zeros_like(rows)).shape == (3, 6)


def test_model_error_scales_the_links_and_friction_and_leaves_the_nominal_model_and_the_plant_alone():
    q, qd, qdd = HOME + 0.1, np.full(6, 0.4), np.full(6, 2.0)
    exact = fb.fivebar_feedforward("rigid_body_coulomb")
    assert fb.fivebar_feedforward("rigid_body_coulomb", ModelError()) == exact
    np.testing.assert_array_equal(
        fb.fivebar_feedforward("rigid_body_coulomb", ModelError())(q, qd, qdd), exact(q, qd, qdd)
    )
    # the private copy's links (the bodies that carry a joint, model order) are scaled by link_factors; the
    # welded claw rides with hard_stop_3; the base is never scaled; the armature stays the drive's
    error = ModelError(0.3, seed=4)
    nominal, scaled = fb.load_model(nominal=True), fb.fivebar_feedforward("gravity", error).private().model
    links = [b for b in range(nominal.nbody) if nominal.body_jntnum[b] > 0]
    factor = dict(zip(links, error.link_factors(len(links)), strict=True))
    factor[nominal.body("moving_finger").id] = factor[nominal.body("hard_stop_3").id]
    for body in range(1, nominal.nbody):
        expected = factor.get(body, 1.0)
        assert scaled.body_mass[body] == pytest.approx(nominal.body_mass[body] * expected, rel=1e-12)
        np.testing.assert_allclose(
            scaled.body_inertia[body], nominal.body_inertia[body] * expected, rtol=1e-12
        )
    np.testing.assert_array_equal(scaled.dof_armature, nominal.dof_armature)
    np.testing.assert_allclose(scaled.dof_damping, nominal.dof_damping * error.friction_factor, rtol=1e-12)
    wrong = fb.fivebar_feedforward("rigid_body_coulomb", ModelError(0.3, seed=4))
    assert np.max(np.abs(wrong(q, qd, qdd) - exact(q, qd, qdd))) > 1e-3
    friction_only = fb.fivebar_feedforward("rigid_body_coulomb", ModelError(0.0, friction_scale=0.5))
    gap = friction_only(q, qd, qdd) - exact(q, qd, qdd)
    expected = 0.5 * (np.asarray(fb.FIVEBAR_TRUTH.coulomb_nm) * np.tanh(qd / 0.01))
    assert np.all(np.abs(gap) >= np.abs(expected) - 1e-9)  # + half the viscous on top
    # the plant's model is untouched by any of it
    assert fb._plant_model(0.0, fb.FIVEBAR_TRUTH).body_mass.tolist() == fb.load_model().body_mass.tolist()


def test_the_models_pickle_and_compare_by_value():
    for model in (
        fb.fivebar_feedforward("gravity"),
        fb.fivebar_feedforward("rigid_body", ModelError(0.1, seed=2)),
        fb.fivebar_mass_matrix(),
        fb.fivebar_mass_matrix(ModelError(0.3)),
    ):
        model(HOME) if isinstance(model, fb.FivebarMassMatrix) else model(HOME, np.zeros(6), np.zeros(6))
        again = pickle.loads(pickle.dumps(model))
        assert again == model and hash(again) == hash(model) and repr(again) == repr(model)
    assert repr(fb.fivebar_feedforward("rigid_body")) == "fivebar_feedforward('rigid_body')"


# ── the plant ─────────────────────────────────────────────────────────────────────────────────────────────


def test_the_plant_refuses_a_belt_and_pushes_and_carries_a_payload_at_the_tool():
    plant = fb.FivebarPlant()
    with pytest.raises(ValueError, match="no belt"):
        plant.reset(Scene(belt=True))
    with pytest.raises(ValueError, match="no belt"):
        plant.reset(Scene(pushes=(Push(0.1, 0.1, (1.0, 0.0, 0.0)),)))
    plain, loaded = fb.load_model(), fb.load_model(payload_kg=0.5)
    assert loaded.body("payload").mass[0] == 0.5
    assert loaded.body_subtreemass[0] == pytest.approx(plain.body_subtreemass[0] + 0.5)
    plant.reset(Scene(payload_kg=0.5), duration_s=0.1)
    plant.advance(0.1)
    assert plant.meta()["payload_kg"] == 0.5 and plant.sample_count == 400
    np.testing.assert_allclose(plant.samples()["tool_pos"][0], fb.tool_position(HOME), atol=2e-3)


# ── the targets ───────────────────────────────────────────────────────────────────────────────────────────


def test_the_path_rungs_are_dev000_at_three_speeds_from_the_station_home():
    rungs = fb.fivebar_rungs()
    assert sorted(rungs) == sorted(
        [*(f"bam_fivebar/path@{s}" for s in ("slow", "nominal", "fast")), "bam_fivebar/calibration"]
    )
    peaks = []
    for speed, fraction in fb.SPEEDS.items():
        rung = rungs[f"bam_fivebar/path@{speed}"]
        assert rung.plant == "bam_fivebar" and rung.timing.rate_hz == 100.0
        np.testing.assert_allclose(rung.scene.start_q, fb.STATION_HOME_Q, atol=1e-12)
        trajectory = rung.inputs(np.asarray(rung.scene.start_q))[0][1]
        tape = load_tape(fb.TAPES[fraction])
        assert trajectory.t[-1] == pytest.approx(float(tape.t[-1]))
        assert rung.duration_s == pytest.approx(float(tape.t[-1]) + 0.6)
        np.testing.assert_allclose(
            trajectory.effort,
            fb.fivebar_feedforward("rigid_body")(
                trajectory.positions, trajectory.velocities, trajectory.accelerations
            ),
            atol=1e-12,
        )
        peaks.append(rung.params["peak_joint_speed_deg_s"])
        assert rung.params["tape_sha256"] == tape_record(fb.TAPES[fraction])["sha256"]
    assert peaks[0] < peaks[1] < peaks[2]
    assert pickle.loads(pickle.dumps(rungs["bam_fivebar/path@nominal"])).id == "bam_fivebar/path@nominal"


def test_the_calibration_sweep_moves_every_drive_at_least_5_degrees_on_a_closable_loop():
    rung = fb.calibration_rung()
    trajectory = rung.inputs(np.asarray(rung.scene.start_q))[0][1]
    q = np.asarray(trajectory.positions)
    assert np.all(np.degrees(q.max(0) - q.min(0)) >= 5.0)
    np.testing.assert_allclose(q[0], fb.FIVEBAR_HOME_Q, atol=1e-12)
    np.testing.assert_allclose(q[-1], fb.FIVEBAR_HOME_Q, atol=1e-12)
    chain = fb._Chain(fb.load_model(nominal=True))
    for row in q[::25]:
        chain.close(row)  # raises where the loop cannot close on the working assembly
    gravity = fb.fivebar_feedforward("rigid_body")(
        q[::25], np.asarray(trajectory.velocities)[::25], np.asarray(trajectory.accelerations)[::25]
    )
    assert np.all(np.abs(gravity) < 0.75 * np.asarray(fb.FIVEBAR_TRUTH.torque_cap_nm))


def test_the_arm_registers_itself():
    pytest.importorskip(
        "robot_controllers.bench.arm_planar", reason="the registry's planar module lands with lane A"
    )
    from robot_controllers.bench.arms import arm

    spec = arm("bam_fivebar")
    assert spec is fb.SPEC and spec.joint_names == fb.FIVEBAR_JOINTS and spec.real_station == "bam_fb"
    assert isinstance(spec.plant(), fb.FivebarPlant)
    assert spec.feedforward("gravity", None) == fb.fivebar_feedforward("gravity")
    np.testing.assert_allclose(spec.tool_position(HOME), fb.tool_position(HOME))
    assert math.isclose(spec.board_config().kp_nm_rad[1], 144.0 / (2.0 * math.pi))
