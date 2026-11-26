"""Amendment 4, §4.5, beyond the acceptance file: the UR5e plant's record and kinematics, what it refuses, the
payload, the layout models resolved on it, and its rungs' speed variants and pickling."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

pytest.importorskip("mujoco")

from robot_controllers.bench.layouts import feedback_model, feedforward_model, layout_spec  # noqa: E402
from robot_controllers.bench.rungs import speed_variant  # noqa: E402
from robot_controllers.bench.types import Push, Scene  # noqa: E402
from robot_controllers.bench.ur5e import (  # noqa: E402
    UR5E_HOME_Q,
    UR5E_JOINTS,
    UR5E_WAYPOINTS,
    Ur5ePlant,
    load_model,
    tool_position,
    ur5e_board_config,
    ur5e_ctc_bandwidth_hz,
    ur5e_mass_matrix,
    ur5e_rungs,
)

RUNGS = {r.id: r for r in ur5e_rungs()}


def test_the_plant_holds_home_and_records_the_tool_site_on_its_own_kinematics():
    plant = Ur5ePlant()
    plant.reset(Scene(), 0, duration_s=0.2)
    plant.advance(0.2)
    samples = plant.samples()
    assert samples["q"].shape == (800, 6) and samples["tool_pos"].shape == (800, 3)
    np.testing.assert_allclose(samples["q"][-1], UR5E_HOME_Q, atol=2e-3)  # the gravity hold keeps it there
    np.testing.assert_allclose(samples["tool_pos"], tool_position(samples["q"]), atol=1e-12)
    assert plant.read().name == UR5E_JOINTS and plant.meta()["plant"] == "ur5e"


def test_mounted_model_uses_its_gravity_with_the_same_board_loop():
    def upside_down(payload, truth):
        model = load_model(payload_kg=payload, truth=truth)
        model.opt.gravity[2] *= -1
        return model

    plant = Ur5ePlant(model_factory=upside_down)
    plant.reset(Scene(), duration_s=0.2)
    plant.advance(0.2)
    np.testing.assert_allclose(plant.samples()["q"][-1], UR5E_HOME_Q, atol=2e-3)
    np.testing.assert_allclose(plant.samples()["tool_pos"], plant.tool_position(plant.samples()["q"]), atol=1e-12)
    assert plant.meta()["model"] == "caller-supplied UR5e scene"


def test_a_belt_or_a_push_is_refused_and_a_payload_is_a_point_mass_at_the_tool():
    plant = Ur5ePlant()
    for scene in (Scene(belt=True), Scene(pushes=(Push(0.1, 0.1, (1.0, 0.0, 0.0)),))):
        with pytest.raises(ValueError):
            plant.reset(scene, 0)
    bare, loaded = load_model(), load_model(payload_kg=1.5)
    assert loaded.body_mass.sum() - bare.body_mass.sum() == pytest.approx(1.5)
    import mujoco

    data = mujoco.MjData(loaded)
    data.qpos[:] = UR5E_HOME_Q
    mujoco.mj_kinematics(loaded, data)
    np.testing.assert_allclose(data.xpos[loaded.body("payload").id], tool_position(UR5E_HOME_Q), atol=1e-12)


def test_the_layout_models_resolve_on_the_ur5e():
    board = ur5e_board_config()
    pd = feedback_model("pd", "ur5e")
    np.testing.assert_allclose(pd.gains(np.zeros(6))[0], board.kp_nm_rad)
    ctc = feedback_model("ctc", "ur5e")
    assert ctc.bandwidth_hz == pytest.approx(ur5e_ctc_bandwidth_hz()) and ctc.damping_ratio == 0.9
    # the boards are at one bandwidth, so ctc's diagonal gains at home equal the board's kp on every joint
    np.testing.assert_allclose(ctc.gains(np.asarray(UR5E_HOME_Q))[0], board.kp_nm_rad, rtol=1e-9)
    assert repr(feedforward_model("rigid_body", "ur5e")) == "ur5e_feedforward('rigid_body')"
    spec = layout_spec("rigid_body", "host_measured", "ctc", "host", arm="ur5e")
    assert spec.name == "ff=rigid_body@host_measured+fb=ctc@host" and spec.summary.endswith("on the ur5e.")
    with pytest.raises(ValueError):
        layout_spec("rigid_body", "plan", "pd", "board", arm="kuka")
    mass = ur5e_mass_matrix()(np.asarray(UR5E_WAYPOINTS[1]))
    np.testing.assert_allclose(mass, mass.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(mass) > 0)


def test_the_u_rungs_follow_the_waypoints_and_take_speed_variants():
    u2 = RUNGS["U2.FastFreeSpace"]
    trajectory = u2.inputs(np.asarray(UR5E_HOME_Q))[0][1]
    assert trajectory.joint_names == UR5E_JOINTS
    for index, point in enumerate(UR5E_WAYPOINTS):
        arrive = 0.0 if index == 0 else 0.3 + index * 0.45 + (index - 1) * 0.3  # the hold, then each rest
        state = trajectory.joint_state_at(arrive + 0.15, extrapolate="clamp")  # inside the 0.3 s after it
        np.testing.assert_allclose(state.position, point, atol=1e-9)
    faster = speed_variant(u2, 0.35)
    assert faster.id == "U2.FastFreeSpace·move350ms" and faster.plant == "ur5e"
    assert faster.params["peak_joint_speed_deg_s"] > u2.params.get("peak_joint_speed_deg_s", 0.0)
    assert faster.duration_s == pytest.approx(0.3 + 3 * (0.35 + 0.3) + 0.6)
    clone = pickle.loads(pickle.dumps(faster))
    np.testing.assert_array_equal(
        clone.inputs(np.zeros(6))[0][1].positions, faster.inputs(np.zeros(6))[0][1].positions
    )
