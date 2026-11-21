"""Acceptance for CONTRACT amendment 5, lane B1: BAM's five-bar arm and the 1-DOF pendulum as twins.

Written by the dispatcher BEFORE the build and read-only to the builders (CONTRACT §5.10). Oracles: bam_pid_tunning's own
`DRIVES` table (imported from bam_ws when it is on this machine), arm_motion_bench's frozen `FF_MODEL` (read from its
source), and physics: a feed-forward that is right leaves little for feedback to carry.

Zach, 2026-11-20: "we want to test in SIM so you can baseline in UR5e to start. and then we want to do on like bam 5bar Sim
and real, and then also the 1 dof pendulum sim and real".
"""

from __future__ import annotations

import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from robot_controllers.models import ModelError

BAM_PID_TUNNING = Path("/home/bam/bam_ws/src/core/mdp/setup/bam_pid_tunning/src")
PENDULUM_FF = Path("/home/bam/arm_motion_bench/src/arm_motion_bench/baselines/pendulum_ff.py")
FIVEBAR_JOINTS = ("j1", "j2", "c1", "w1", "w2", "w3")


def episode(spec, rung, seed=0):
    from robot_controllers.bench.loop import run_episode
    from robot_controllers.metrics.episode import episode_metrics

    log = run_episode(spec, rung, seed=seed)
    return log, episode_metrics(log, rung)


def spec(*layout, lookahead_s=0.0, interpolation="tangent", arm):
    from robot_controllers.bench.layouts import layout_spec

    return layout_spec(*layout, lookahead_s=lookahead_s, interpolation=interpolation, arm=arm)


def hold_rung(arm_name: str, q, seconds: float = 1.5):
    """A rung that holds the arm at `q` (the plant starts there), effort = the arm's nominal gravity."""
    import dataclasses

    from robot_controllers.bench.rungs import HoldScript, rung
    from robot_controllers.bench.types import Scene

    base = rung("R2.FastFreeSpace")
    return dataclasses.replace(
        base, id=f"{arm_name}.Hold", plant=arm_name, inputs=HoldScript(seconds), duration_s=seconds,
        scene=Scene(start_q=tuple(float(v) for v in q)), gates=(), params={},
    )


def test_both_arms_are_registered_after_the_ur5e():
    from robot_controllers.bench.arms import arm, arms

    names = [a.name for a in arms()]
    assert names == ["planar_arm", "ur5e", "bam_fivebar", "pendulum"]
    assert arm("bam_fivebar").joint_names == FIVEBAR_JOINTS
    assert arm("pendulum").joint_names == ("can11",)
    assert len(arm("bam_fivebar").home_q) == 6 and len(arm("pendulum").home_q) == 1


def test_fivebar_boards_are_bam_pid_tunnings_drives():
    if not BAM_PID_TUNNING.exists():
        pytest.skip("bam_ws's bam_pid_tunning is not on this machine")
    sys.path.insert(0, str(BAM_PID_TUNNING))
    try:
        from bam_pid_tunning.plants.bam_fb_v2 import DRIVES
    finally:
        sys.path.remove(str(BAM_PID_TUNNING))
    from robot_controllers.bench.arms import arm

    board = arm("bam_fivebar").board_config()
    by_role = {d.role.lower(): d for d in DRIVES}
    for j, name in enumerate(FIVEBAR_JOINTS):
        drive = by_role[name]
        kp, kd = drive.as_found_moteus[0], drive.as_found_moteus[1]
        assert board.kp_nm_rad[j] == pytest.approx(kp), name
        assert board.kd_nm_s_rad[j] == pytest.approx(kd), name
        assert board.torque_cap_nm[j] == pytest.approx(drive.torque_limit_nm), name
    assert board.encoder_counts_per_rev == int(DRIVES[0].truth.encoder_counts_per_rev)


def _pendulum_ff_model() -> dict:
    tree = ast.parse(PENDULUM_FF.read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "FF_MODEL":
            value = node.value
            return {
                k.value: (v.value if isinstance(v, ast.Constant) else None) for k, v in zip(value.keys, value.values, strict=True)
            }
    raise AssertionError("FF_MODEL not found")


def test_pendulum_feedforward_is_arm_motion_benchs_frozen_ff_model():
    if not PENDULUM_FF.exists():
        pytest.skip("arm_motion_bench is not on this machine")
    from robot_controllers.bench.arms import arm

    m = _pendulum_ff_model()
    full = arm("pendulum").feedforward("rigid_body_coulomb", None)
    rigid = arm("pendulum").feedforward("rigid_body", None)
    gravity = arm("pendulum").feedforward("gravity", None)
    rng = np.random.default_rng(11)
    for _ in range(10):
        q, qd, qdd = rng.uniform(-3, 3, 1), rng.uniform(-4, 4, 1), rng.uniform(-20, 20, 1)
        g = m["ff_gravity_nm"] * np.sin(q - m["ff_gravity_bottom_rad"])
        inertial = m["ff_inertia_kg_m2"] * qdd
        friction = m["ff_coulomb_nm"] * np.tanh(qd / m["ff_band_rad_s"]) + m["ff_viscous_nm_per_rad_s"] * qd
        np.testing.assert_allclose(full(q, qd, qdd), g + inertial + friction, rtol=0, atol=1e-9)
        np.testing.assert_allclose(rigid(q, qd, qdd), g + inertial + m["ff_viscous_nm_per_rad_s"] * qd, rtol=0, atol=1e-9)
        np.testing.assert_allclose(gravity(q, qd, qdd), g, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    ("arm_name", "offset_deg"),
    [("bam_fivebar", (0.0, 10.0, -10.0, 15.0, 0.0, 0.0)), ("pendulum", (40.0,))],
)
def test_gravity_feedforward_carries_a_held_pose_so_feedback_carries_little(arm_name, offset_deg):
    from robot_controllers.bench.arms import arm

    spec_arm = arm(arm_name)
    pose = np.asarray(spec_arm.home_q) + np.radians(offset_deg)
    _, with_ff = episode(spec("gravity", "plan", "board_gains", "board", arm=arm_name), hold_rung(arm_name, pose))
    _, without = episode(spec("none", "none", "board_gains", "board", arm=arm_name), hold_rung(arm_name, pose))
    assert with_ff["diverged"] == 0 and with_ff["tracking_max_deg"] < 1.0
    assert with_ff["fb_share_lf_pct"] < 25.0, with_ff["fb_share_lf_pct"]
    assert without["fb_share_lf_pct"] > 75.0, without["fb_share_lf_pct"]


@pytest.mark.parametrize("arm_name", ["bam_fivebar", "pendulum"])
def test_model_error_changes_the_model_and_never_the_plant(arm_name):
    from robot_controllers.bench.arms import arm

    spec_arm = arm(arm_name)
    exact = spec_arm.feedforward("rigid_body", None)
    nominal = spec_arm.feedforward("rigid_body", ModelError())
    wrong = spec_arm.feedforward("rigid_body", ModelError(0.3, seed=4))
    n = len(spec_arm.joint_names)
    q = np.asarray(spec_arm.home_q) + 0.2
    qd, qdd = np.full(n, 0.5), np.full(n, 3.0)
    np.testing.assert_array_equal(exact(q, qd, qdd), nominal(q, qd, qdd))
    assert np.max(np.abs(wrong(q, qd, qdd) - exact(q, qd, qdd))) > 1e-4
    _, a = episode(spec("none", "none", "board_gains", "board", arm=arm_name), hold_rung(arm_name, spec_arm.home_q, 0.5))
    _, b = episode(spec("none", "none", "board_gains", "board", arm=arm_name), hold_rung(arm_name, spec_arm.home_q, 0.5))
    assert a["tracking_max_deg"] == b["tracking_max_deg"]


@pytest.mark.parametrize("arm_name", ["bam_fivebar", "pendulum"])
def test_each_new_arm_has_its_fixed_targets_and_the_v1_recipe_beats_the_as_built_jtc(arm_name):
    from robot_controllers.bench.arms import arm
    from robot_controllers.bench.rungs import rate_variant

    targets = {t.id: t for t in arm(arm_name).targets()}
    for suffix in ("path@slow", "path@nominal", "path@fast", "calibration", "collision"):
        assert f"{arm_name}/{suffix}" in targets, sorted(targets)
    nominal = rate_variant(targets[f"{arm_name}/path@nominal"].rung(), 50.0)
    _, built = episode(spec("trajectory", "trajectory", "board_gains", "board", arm=arm_name), nominal)
    _, v1 = episode(
        spec("trajectory", "trajectory", "board_gains", "board", lookahead_s=0.0055, interpolation="secant_mid_ff",
             arm=arm_name),
        nominal,
    )
    assert v1["diverged"] == 0 and built["diverged"] == 0
    assert v1["tracking_p95_deg"] < built["tracking_p95_deg"]
    assert math.isfinite(v1["tool_p95_mm"])
