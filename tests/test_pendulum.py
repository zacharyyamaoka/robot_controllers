"""Amendment 5, §5.8, beyond the acceptance file: the pendulum twin - its hinge against the twin of record's
terms, the board against the deployment record, the torque scale, the model error, the plant and its rungs."""

from __future__ import annotations

import ast
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from robot_controllers.bench import pendulum as pd  # noqa: E402
from robot_controllers.bench.arm_tapes import load_tape  # noqa: E402
from robot_controllers.bench.types import Scene  # noqa: E402
from robot_controllers.models import ModelError  # noqa: E402

PENDULUM_SIM = Path("/home/bam/arm_motion_bench/src/arm_motion_bench/baselines/pendulum_sim.py")
PENDULUM_ROBOT = Path("/home/bam/arm_motion_bench/src/arm_motion_bench/robots/pendulum.py")


def _constants(path: Path, *names: str) -> dict[str, float]:
    if not path.exists():
        pytest.skip("arm_motion_bench is not on this machine")
    out = {}
    for node in ast.parse(path.read_text()).body:
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        if getattr(target, "id", None) in names and isinstance(node.value, ast.Constant):
            out[target.id] = node.value.value
    return out


def test_the_hinge_is_the_twin_of_records_gravity_and_inertia():
    model = pd.load_model()
    data = mujoco.MjData(model)
    truth = pd.PENDULUM_TRUTH
    assert model.nv == 1 and model.opt.timestep == truth.physics_dt_s
    assert model.dof_frictionloss[0] == truth.coulomb_nm and model.dof_damping[0] == truth.viscous_nm_s_rad
    for q in np.linspace(-3.0, 3.3, 9):
        data.qpos[0], data.qvel[0] = q, 0.0
        mujoco.mj_forward(model, data)
        # qfrc_bias is the torque that holds the hinge still: + G sin(q - bottom), the sign
        # identify.fit_gravity fits
        assert data.qfrc_bias[0] == pytest.approx(
            truth.gravity_nm * math.sin(q - truth.gravity_bottom_rad), abs=1e-9
        )
        full = np.zeros((1, 1))
        mujoco.mj_fullM(model, data, full)
        assert full[0, 0] == pytest.approx(
            truth.inertia_kg_m2, abs=2e-9
        )  # + the body's 1e-9 kg.m^2 (twin.py keeps it too)
    sidecar = json.loads(pd.MODEL_PATH.with_suffix(".json").read_text())
    assert (
        sidecar["constants"]["gravity_nm"] == truth.gravity_nm
        and sidecar["constants"]["torque_scale"] == truth.torque_scale
    )


def test_the_twins_terms_and_the_board_are_arm_motion_benchs():
    sim = _constants(PENDULUM_SIM, "TWIN_TORQUE_SCALE", "BOARD_KP_NM_PER_REV", "BOARD_KD_NM_PER_REV_S")
    robot = _constants(PENDULUM_ROBOT, "BOTTOM_RAD", "TORQUE_CAP_NM", "TIP_LENGTH_M", "GEAR_RATIO")
    truth, board = pd.PENDULUM_TRUTH, pd.pendulum_board_config()
    assert truth.torque_scale == sim["TWIN_TORQUE_SCALE"]
    assert board.kp_nm_rad[0] == pytest.approx(sim["BOARD_KP_NM_PER_REV"] / (2 * math.pi), rel=1e-15)
    assert board.kd_nm_s_rad[0] == pytest.approx(sim["BOARD_KD_NM_PER_REV_S"] / (2 * math.pi), rel=1e-15)
    assert board.torque_cap_nm[0] == robot["TORQUE_CAP_NM"]
    assert board.encoder_counts_per_rev == 2**14 * int(robot["GEAR_RATIO"])
    assert pd.PENDULUM_HOME_Q == (robot["BOTTOM_RAD"],) and pd.TIP_LENGTH_M == robot["TIP_LENGTH_M"]
    # the twin's own feed-forward (pendulum_sim.py TWIN_FF) is these terms / the torque scale
    text = PENDULUM_SIM.read_text()
    assert (
        "4.42266 / TWIN_TORQUE_SCALE" in text and "0.257822" in text and "0.12011 / TWIN_TORQUE_SCALE" in text
    )
    assert "1.12 / TWIN_TORQUE_SCALE" in text and "0.0873596 / TWIN_TORQUE_SCALE" in text


def test_the_plant_applies_the_torque_scale_and_logs_the_boards_torque():
    plant = pd.PendulumPlant()
    plant.reset(Scene(start_q=(1.2,)), duration_s=0.05)
    plant.advance(0.05)
    tau = plant.samples()["tau"][-1, 0]
    assert tau == pytest.approx(pd.pendulum_feedforward("gravity")(np.array([1.2]), 0, 0)[0], abs=0.05)
    assert plant._data.qfrc_applied[0] == pytest.approx(pd.PENDULUM_TRUTH.torque_scale * tau, rel=1e-12)
    tip = plant.samples()["tool_pos"][-1]
    np.testing.assert_allclose(np.linalg.norm(tip), pd.TIP_LENGTH_M, atol=1e-12)
    with pytest.raises(ValueError, match="no payload"):
        plant.reset(Scene(payload_kg=0.2))
    with pytest.raises(ValueError, match="no belt"):
        plant.reset(Scene(belt=True))


def test_model_error_scales_the_link_and_friction_and_the_mass_matrix_follows():
    error = ModelError(0.3, seed=4)
    link = float(error.link_factors(1)[0])
    q, qd, qdd = np.array([1.0]), np.array([0.7]), np.array([5.0])
    exact, wrong = (
        pd.pendulum_feedforward("rigid_body_coulomb"),
        pd.pendulum_feedforward("rigid_body_coulomb", error),
    )
    m = pd.FF_MODEL
    expected = (
        link * m.gravity_nm * np.sin(q - m.gravity_bottom_rad)
        + link * m.inertia_kg_m2 * qdd
        + error.friction_factor * (m.coulomb_nm * np.tanh(qd / m.band_rad_s) + m.viscous_nm_per_rad_s * qd)
    )
    np.testing.assert_allclose(wrong(q, qd, qdd), expected, atol=1e-12)
    assert np.max(np.abs(wrong(q, qd, qdd) - exact(q, qd, qdd))) > 1e-3
    assert pd.pendulum_mass_matrix(error)(q)[0, 0] == pytest.approx(link * m.inertia_kg_m2)
    assert pd.pendulum_mass_matrix()(np.zeros((4, 1))).shape == (4, 1, 1)
    assert pd.pendulum_feedforward("gravity", ModelError()) == pd.pendulum_feedforward("gravity")
    for model in (wrong, pd.pendulum_mass_matrix(error)):
        again = pickle.loads(pickle.dumps(model))
        assert again == model and repr(again) == repr(model)


def test_the_path_rungs_are_dev007_at_three_speeds_and_the_calibration_sweeps_8_degrees():
    rungs = pd.pendulum_rungs()
    assert sorted(rungs) == sorted(
        [*(f"pendulum/path@{s}" for s in ("slow", "nominal", "fast")), "pendulum/calibration"]
    )
    peaks = []
    for speed, fraction in pd.SPEEDS.items():
        rung = rungs[f"pendulum/path@{speed}"]
        tape = load_tape(pd.TAPES[fraction])
        assert rung.plant == "pendulum" and rung.scene.start_q == (float(tape.positions[0][0]),)
        trajectory = rung.inputs(np.asarray(rung.scene.start_q))[0][1]
        assert trajectory.t[-1] == pytest.approx(float(tape.t[-1]))
        peaks.append(rung.params["peak_joint_speed_deg_s"])
    assert peaks[0] < peaks[1] < peaks[2] and peaks[2] < math.degrees(5.0)  # the joint's measured 5 rad/s
    calibration = rungs["pendulum/calibration"].inputs(np.asarray(pd.PENDULUM_HOME_Q))[0][1]
    q = np.asarray(calibration.positions)[:, 0]
    assert math.degrees(q.max() - q.min()) >= 5.0 and q[0] == q[-1] == pytest.approx(pd.BOTTOM_RAD)


def test_the_arm_registers_itself():
    pytest.importorskip(
        "robot_controllers.bench.arm_planar", reason="the registry's planar module lands with lane A"
    )
    from robot_controllers.bench.arms import arm

    spec = arm("pendulum")
    assert spec is pd.SPEC and spec.joint_names == ("can11",) and spec.real_station == "pendulum"
    assert isinstance(spec.plant(), pd.PendulumPlant)
