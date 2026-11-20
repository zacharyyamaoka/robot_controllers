"""The pure models of amendment 3 beyond the acceptance file: PD / InertiaScaledPD on batches, the layout's
own
validation, and the planar arm's closed form against MuJoCo with a payload and in its parts."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_controllers.bench.arm import ARM, HOME_Q, build_mjcf
from robot_controllers.bench.arm_models import (
    COULOMB_SMOOTHING_RAD_S,
    RIGID_BODY,
    planar_arm_feedforward,
    planar_arm_mass_matrix,
)
from robot_controllers.models import PD, ZERO_FEEDFORWARD, InertiaScaledPD, Layout, has_gains


def _states(n=12, seed=7):
    rng = np.random.default_rng(seed)
    return rng.uniform(-2.5, 2.5, (n, 2)), rng.uniform(-3.0, 3.0, (n, 2)), rng.uniform(-20.0, 20.0, (n, 2))


def test_pd_is_the_constant_gain_law_on_one_state_and_on_a_batch():
    pd = PD((300.0, 200.0), (7.5, 4.0))
    q_ref, qd_ref, q, qd = _states(9)[0], _states(9, 1)[0], _states(9, 2)[0], _states(9, 3)[0]
    batch = pd(q_ref, qd_ref, q, qd)
    assert batch.shape == (9, 2)
    for i in range(9):
        expected = np.array([300.0, 200.0]) * (q_ref[i] - q[i]) + np.array([7.5, 4.0]) * (qd_ref[i] - qd[i])
        np.testing.assert_allclose(pd(q_ref[i], qd_ref[i], q[i], qd[i]), expected, rtol=1e-15)
        np.testing.assert_allclose(batch[i], expected, rtol=1e-15)
    kp, kd = pd.gains(q_ref)
    assert kp.shape == kd.shape == (9, 2) and np.all(kp == [300.0, 200.0]) and np.all(kd == [7.5, 4.0])
    scalar = PD(10.0, 1.0)
    np.testing.assert_array_equal(scalar.gains(np.zeros(3))[0], [10.0, 10.0, 10.0])
    with pytest.raises(ValueError):
        PD((-1.0, 2.0), 1.0)  # a board cannot render a negative gain
    assert PD((1.0, 2.0), 3.0) == PD([1, 2], 3)  # normalised, so equal configs are equal


def test_inertia_scaled_pd_on_a_batch_is_the_per_state_law():
    mass = planar_arm_mass_matrix()
    fb = InertiaScaledPD(6.0, 0.9, mass)
    q_ref, qd_ref, q, qd = (_states(6, s)[0] for s in (1, 2, 3, 4))
    batch = fb(q_ref, qd_ref, q, qd)
    kp, kd = fb.gains(q_ref)
    for i in range(6):
        np.testing.assert_allclose(batch[i], fb(q_ref[i], qd_ref[i], q[i], qd[i]), rtol=1e-14)
        np.testing.assert_allclose(kp[i], fb.gains(q_ref[i])[0], rtol=1e-15)
        np.testing.assert_allclose(kd[i], fb.gains(q_ref[i])[1], rtol=1e-15)
    # at a reference equal to the state the law is the diagonal gains (the board's rendering) plus the
    # coupling
    wn = 2 * math.pi * 6.0
    single_q = np.array([0.2, -1.0])
    error = np.array([0.01, 0.0])
    tau = fb(single_q + error, np.zeros(2), single_q, np.zeros(2))
    np.testing.assert_allclose(tau, mass(single_q)[:, 0] * wn**2 * 0.01, rtol=1e-12)
    assert abs(tau[1]) > 0.0  # the elbow feels the shoulder's error through M12: what the board layout drops
    with pytest.raises(ValueError):
        InertiaScaledPD(0.0, 0.9, mass)


def test_zero_feedforward_and_gains_detection():
    assert np.array_equal(ZERO_FEEDFORWARD(np.ones((4, 2)), None, None), np.zeros((4, 2)))
    assert repr(ZERO_FEEDFORWARD) == "ZERO_FEEDFORWARD"
    assert has_gains(PD(1.0, 1.0)) and has_gains(InertiaScaledPD(1.0, 1.0, planar_arm_mass_matrix()))
    assert not has_gains(lambda q_ref, qd_ref, q, qd: q_ref - q) and not has_gains(None)


def test_layout_names_and_checks_its_models():
    assert Layout().name == "trajectory+board"
    assert Layout("plan", "split", lookahead_s=0.008).name == "plan+split+lookahead=0.008s"
    with pytest.raises(ValueError):
        Layout("plan", "board").check(None, None)
    with pytest.raises(TypeError):
        Layout("trajectory", "host").check(None, 3.0)
    Layout("trajectory", "board").check(None, None)  # the simple JTC: nothing to evaluate
    Layout("host_reference", "host").check(ZERO_FEEDFORWARD, lambda q_ref, qd_ref, q, qd: q_ref - q)


# ── the planar arm's closed form ───────────────────────────────────────────────────────────────────────────


def _mj_inverse_with_payload(payload_kg, q, qd, qdd):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        build_mjcf(belt=False, payload_kg=payload_kg, frictionloss=False, joint_limits=False)
    )
    data = mujoco.MjData(model)
    out = np.empty_like(q)
    for i in range(len(q)):
        data.qpos[:], data.qvel[:], data.qacc[:] = q[i], qd[i], qdd[i]
        mujoco.mj_inverse(model, data)
        out[i] = data.qfrc_inverse
    return out


def test_a_payload_the_model_is_told_about_matches_mujoco_with_that_payload():
    # an independent oracle for the payload term: MuJoCo's own model of the arm carrying 0.3 kg on the tool
    q, qd, qdd = _states(15, 11)
    told = planar_arm_feedforward(RIGID_BODY, payload_kg=0.3)
    np.testing.assert_allclose(told(q, qd, qdd), _mj_inverse_with_payload(0.3, q, qd, qdd), atol=1e-9, rtol=0)
    # and the nominal model is off by exactly what the R8 rung leaves for feedback
    unknown = planar_arm_feedforward(RIGID_BODY)
    assert np.abs(unknown(q, qd, qdd) - _mj_inverse_with_payload(0.3, q, qd, qdd)).max() > 0.5


def test_coulomb_is_the_smoothed_frictionloss_and_nothing_else():
    q, qd, qdd = _states(8, 5)
    coulomb = planar_arm_feedforward(("coulomb",))
    expected = np.array(ARM.frictionloss_nm) * np.tanh(qd / COULOMB_SMOOTHING_RAD_S)
    np.testing.assert_allclose(coulomb(q, qd, qdd), expected, rtol=1e-15)
    np.testing.assert_array_equal(coulomb(q, qd, qdd), coulomb(0 * q, qd, 0 * qdd))


def test_mass_matrix_on_a_batch_is_symmetric_positive_and_per_row():
    mass = planar_arm_mass_matrix()
    q, _, _ = _states(10, 9)
    batch = mass(q)
    assert batch.shape == (10, 2, 2)
    for i in range(10):
        np.testing.assert_array_equal(batch[i], mass(q[i]))
        np.testing.assert_allclose(batch[i], batch[i].T, rtol=0)
        assert np.all(np.linalg.eigvalsh(batch[i]) > 0)
    # the inertia term of the feed-forward is M(q) qdd
    inertia = planar_arm_feedforward(("inertia",))
    _, _, qdd = _states(10, 4)
    np.testing.assert_allclose(inertia(q, 0 * qdd, qdd), np.einsum("nij,nj->ni", batch, qdd), atol=1e-12)
    assert repr(mass) == "planar_arm_mass_matrix()"
    assert repr(planar_arm_feedforward(("gravity",))) == "planar_arm_feedforward(('gravity',), payload_kg=0)"


def test_models_refuse_unknown_terms_and_bad_payloads():
    with pytest.raises(ValueError):
        planar_arm_feedforward(("gravity", "magic"))
    with pytest.raises(ValueError):
        planar_arm_feedforward(())
    with pytest.raises(ValueError):
        planar_arm_feedforward(("gravity", "gravity"))
    with pytest.raises(ValueError):
        planar_arm_feedforward(("gravity",), payload_kg=-0.1)
    assert planar_arm_mass_matrix()(np.asarray(HOME_Q)).shape == (2, 2)
