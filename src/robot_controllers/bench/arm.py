"""planar_arm: a two-joint arm in the x-z plane over a conveyor belt (CONTRACT §3), built in code for MuJoCo
3.11.

    from robot_controllers.bench.arm import ARM, fk, ik, nominal_dynamics, build_model
    q = ik(0.30, 0.25)                      # elbow-up, analytic -> HOME_Q = (0, -pi/2)
    x, z = fk(q)
    tau = nominal_dynamics(q, qd, qdd)      # mj_inverse on the nominal model: gravity + inertia feed-forward

Geometry and sign (chosen here, the contract leaves it open): both joints rotate about world **-y** (MJCF axis
`0 -1 0`), so a positive angle turns the link from +x toward +z: counter-clockwise when you look along +y with
x to the right and z up, the usual planar-arm convention. q = (0, 0) points both links along +x. With the
shoulder at (0, 0, 0.50) the tool centre is

    x = L1 cos q1 + L2 cos(q1 + q2),   z = 0.50 + L1 sin q1 + L2 sin(q1 + q2).

"Elbow-up" (the elbow above the shoulder-tool line, as the arm reaches down to the belt) is q2 < 0. Home (0,
-90 deg) puts link 1 level and link 2 hanging straight down, the tool at (0.30, 0.25).

The belt is a box whose top is at z = 0.15, on a slide joint along x whose velocity the plant sets every step
(so it moves at the scene's belt speed whatever the tool does to it). Only the tool sphere touches it, through
one explicit frictionless contact pair whose normal stiffness and damping are calibrated to the scene's N/m
and N*s/m (see `ContactCalibration`); the plant adds the belt's Coulomb friction (mu 0.6) itself, see
`ArmConfig`. The elbow's joint range is the hard stop the S2 rung seeks.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # MuJoCo is the bench extra: imported only where a model is built
    import mujoco

    from robot_controllers.bench.types import Scene

JOINT_NAMES: tuple[str, str] = ("shoulder", "elbow")


@dataclass(frozen=True)
class ArmConfig:
    """The planar arm, SI units. Defaults are CONTRACT §3; change them only as a contract barrier."""

    base_z_m: float = 0.50  # shoulder height above the floor (m)
    link1_length_m: float = 0.30  # shoulder -> elbow (m)
    link2_length_m: float = 0.25  # elbow -> tool centre (m)
    link1_mass_kg: float = 1.2  # uniform rod (kg)
    link2_mass_kg: float = 0.8  # uniform rod (kg)
    tool_radius_m: float = 0.015  # tool sphere radius (m)
    tool_mass_kg: float = 0.3  # tool sphere mass, at the tip (kg)
    armature_kg_m2: tuple[float, float] = (0.020, 0.010)  # gearbox reflected inertia per joint (kg*m^2)
    frictionloss_nm: tuple[float, float] = (0.30, 0.20)  # gearbox dry friction per joint (N*m)
    damping_nm_s_rad: tuple[float, float] = (0.05, 0.05)  # gearbox viscous friction per joint (N*m*s/rad)
    # WHY these stops: -135 deg folds the elbow 45 deg past home, beyond every rung's path (R1 reaches -112.6
    # deg, R5 targets stay within home +-20 deg); -15 deg is past R1's most open pose (-40.6 deg). S2 seeks
    # -135.
    elbow_range_rad: tuple[float, float] = (math.radians(-135.0), math.radians(-15.0))
    # WHY 2 ms: a stiff stop that MuJoCo still integrates (its time constant must stay above 2 physics steps).
    # (time constant s, damping ratio) of the elbow limit
    hard_stop_solref: tuple[float, float] = (0.002, 1.0)
    belt_top_z_m: float = 0.15  # belt surface height (m)
    belt_half_length_m: float = 4.0  # long enough that the tool stays over it for a whole rung (m)
    belt_friction: float = 0.6  # Coulomb friction coefficient tool-belt
    # WHY a small constant impedance: MuJoCo's soft constraint adds an apparent mass d/(1-d) times the
    # contact's mass to the normal direction; at d = 0.02 that is 2 %, so the belt behaves as the
    # spring-damper it is calibrated to. The default (0.9-0.95) would make an impact 10-20x more sluggish than
    # the spring says.
    contact_impedance: float = 0.02
    # WHY friction outside MuJoCo's contact (measured 2026-11-12, MuJoCo 3.11): a tool held against a belt
    # that slides past it keeps MuJoCo's soft friction cone in its "middle zone", whose cost pushes the
    # contact apart: at 2-5 N of press and 0.25 m/s the elliptic and the pyramidal cones both chattered (in
    # contact 9-46 % of the dwell, normal force std 4-9 N on a 2-5 N mean), at any friction regularisation
    # that still slides. So the pair is frictionless (condim 1, the calibrated normal spring-damper only) and
    # the plant adds Coulomb friction mu * F_n * tanh(|v_slip| / friction_slip_speed) against the slip, on the
    # tool, every step.
    friction_slip_speed_m_s: float = 0.02  # below this slip speed friction fades linearly (m/s)
    physics_dt_s: float = 0.00025  # 4 kHz, the board rate (s)

    @property
    def reach_m(self) -> float:
        return self.link1_length_m + self.link2_length_m

    @property
    def tool_touch_z_m(self) -> float:
        """Tool-centre height at which the sphere just touches the belt surface (m)."""
        return self.belt_top_z_m + self.tool_radius_m


ARM = ArmConfig()


# ── kinematics (numpy, no MuJoCo) ───────────────────────────────────────────────────────────────────────────


def fk(q: Any, arm: ArmConfig = ARM) -> tuple[Any, Any]:
    """Tool centre (x, z) in metres for joint angles `q` (rad), shape (2,) or (..., 2)."""
    q = np.asarray(q, dtype=float)
    shoulder, elbow = q[..., 0], q[..., 1]
    x = arm.link1_length_m * np.cos(shoulder) + arm.link2_length_m * np.cos(shoulder + elbow)
    z = arm.base_z_m + arm.link1_length_m * np.sin(shoulder) + arm.link2_length_m * np.sin(shoulder + elbow)
    if q.ndim == 1:
        return float(x), float(z)
    return x, z


def tool_position(q: Any, arm: ArmConfig = ARM) -> np.ndarray:
    """Tool centre as world (x, y, z) in metres, shape (3,) or (N, 3); y is always 0."""
    q = np.asarray(q, dtype=float)
    x, z = fk(q, arm)
    if q.ndim == 1:
        return np.array([x, 0.0, z])
    return np.stack([np.asarray(x), np.zeros_like(x), np.asarray(z)], axis=-1)


def ik(x: float, z: float, arm: ArmConfig = ARM) -> np.ndarray:
    """Joint angles (rad) that put the tool centre at (x, z), elbow-up (q2 <= 0), analytic.

    Raises ValueError outside the annulus the arm can reach.
    """
    l1, l2 = arm.link1_length_m, arm.link2_length_m
    dz = z - arm.base_z_m
    cos_elbow = (x * x + dz * dz - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    if not -1.0 - 1e-12 <= cos_elbow <= 1.0 + 1e-12:
        raise ValueError(f"tool point ({x:.4f}, {z:.4f}) m is out of the arm's reach")
    elbow = -math.acos(min(1.0, max(-1.0, cos_elbow)))  # elbow-up: negative with this sign convention
    shoulder = math.atan2(dz, x) - math.atan2(l2 * math.sin(elbow), l1 + l2 * math.cos(elbow))
    return np.array([shoulder, elbow])


def jacobian(q: Any, arm: ArmConfig = ARM) -> np.ndarray:
    """d(x, z)/dq, shape (2, 2) or (N, 2, 2)."""
    q = np.asarray(q, dtype=float)
    l1, l2 = arm.link1_length_m, arm.link2_length_m
    s1, c1 = np.sin(q[..., 0]), np.cos(q[..., 0])
    s12, c12 = np.sin(q[..., 0] + q[..., 1]), np.cos(q[..., 0] + q[..., 1])
    rows = np.stack(
        [
            np.stack([-l1 * s1 - l2 * s12, -l2 * s12], axis=-1),
            np.stack([l1 * c1 + l2 * c12, l2 * c12], axis=-1),
        ],
        axis=-2,
    )
    return rows


def joint_motion_from_tool(
    points: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray, arm: ArmConfig = ARM
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint (q, qd, qdd) for a tool path given as (N, 2) arrays of (x, z), (xd, zd), (xdd, zdd).

    q by `ik` per sample; qd = J^-1 pd; qdd = J^-1 (pdd - Jdot qd), all analytic, so the trajectory's velocity
    and acceleration are the path's own, not finite differences.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    q = np.array([ik(x, z, arm) for x, z in points])
    jac = jacobian(q, arm)
    qd = np.linalg.solve(jac, np.atleast_2d(velocities)[..., None])[..., 0]
    l1, l2 = arm.link1_length_m, arm.link2_length_m
    s1, c1 = np.sin(q[:, 0]), np.cos(q[:, 0])
    s12, c12 = np.sin(q[:, 0] + q[:, 1]), np.cos(q[:, 0] + q[:, 1])
    w1, w12 = qd[:, 0], qd[:, 0] + qd[:, 1]
    jdot_qd = np.stack([-l1 * c1 * w1**2 - l2 * c12 * w12**2, -l1 * s1 * w1**2 - l2 * s12 * w12**2], axis=-1)
    qdd = np.linalg.solve(jac, (np.atleast_2d(accelerations) - jdot_qd)[..., None])[..., 0]
    return q, qd, qdd


HOME_TOOL_XZ: tuple[float, float] = (0.30, 0.25)
HOME_Q: tuple[float, float] = tuple(float(v) for v in ik(*HOME_TOOL_XZ))  # (0, -pi/2)


# ── the MJCF ────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContactCalibration:
    """How the belt contact's MuJoCo parameters were chosen so the normal force follows F = K pen + B
    pen_rate.

    MuJoCo's direct (negative) solref is in acceleration units. For a contact row with impedance d its force
    is f = (aref - a1) / R, aref = -(B'/d) v - (K'/d) r, R = (1 - d)/d * diag, where (K', B') = -solref and
    `diag` is MuJoCo's approximate inverse mass of the row (efc_diagApprox, from the bodies' invweight0, so
    the same in every configuration). Dropping the small a1/R term (the 2 % apparent mass) gives K = K' / ((1
    - d) diag), so K' = K (1 - d) diag, and the same for B. `diag` is read from a probe contact, not computed
    by hand.
    """

    stiffness_n_m: float  # the scene's target normal stiffness
    damping_n_s_m: float  # the scene's target normal damping
    impedance: float  # solimp d, constant
    diag_approx: float  # MuJoCo's inverse-mass estimate of the normal row (1/kg)
    solref: tuple[float, float]  # what the pair carries: (-K', -B')


def _rod_inertia(mass_kg: float, length_m: float) -> str:
    perpendicular = mass_kg * length_m**2 / 12.0
    return f"{1e-6:.6g} {perpendicular:.9g} {perpendicular:.9g}"


def build_mjcf(
    *,
    belt: bool,
    payload_kg: float = 0.0,
    arm: ArmConfig = ARM,
    frictionloss: bool = True,
    joint_limits: bool = True,
    pair_solref: tuple[float, float] = (-1.0, -1.0),
) -> str:
    """The MJCF text. `frictionloss=False, joint_limits=False, belt=False` is the nominal (feed-forward)
    model."""
    l1, l2 = arm.link1_length_m, arm.link2_length_m
    tool_mass = arm.tool_mass_kg + payload_kg
    tool_inertia = 0.4 * tool_mass * arm.tool_radius_m**2
    friction = arm.frictionloss_nm if frictionloss else (0.0, 0.0)
    if joint_limits:
        low, high = arm.elbow_range_rad
        elbow_limit = (
            f'limited="true" range="{low:.12g} {high:.12g}" '
            f'solreflimit="{arm.hard_stop_solref[0]:g} {arm.hard_stop_solref[1]:g}"'
        )
    else:
        elbow_limit = 'limited="false"'
    belt_body = belt_pair = ""
    if belt:
        half_height = 0.05
        belt_body = f"""
    <body name="belt" pos="0.30 0 {arm.belt_top_z_m - half_height:.12g}">
      <joint name="belt" type="slide" axis="1 0 0" damping="0"/>
      <inertial pos="0 0 0" mass="50" diaginertia="1 1 1"/>
      <geom name="belt" type="box" size="{arm.belt_half_length_m:g} 0.15 {half_height:g}"
            contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>
    </body>"""
        d = arm.contact_impedance
        # condim 1: the normal spring-damper only; the plant adds the belt friction (see ArmConfig)
        belt_pair = f"""
  <contact>
    <pair name="tool_belt" geom1="tool" geom2="belt" condim="1"
          solref="{pair_solref[0]:.12g} {pair_solref[1]:.12g}" solimp="{d:g} {d:g} 0.001 0.5 2"/>
  </contact>"""
    return f"""<mujoco model="planar_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{arm.physics_dt_s:.12g}" gravity="0 0 -9.81" integrator="Euler" cone="elliptic"/>
  <worldbody>
    <body name="link1" pos="0 0 {arm.base_z_m:.12g}">
      <joint name="shoulder" type="hinge" axis="0 -1 0" armature="{arm.armature_kg_m2[0]:g}"
             frictionloss="{friction[0]:g}" damping="{arm.damping_nm_s_rad[0]:g}" limited="false"/>
      <inertial pos="{l1 / 2:.12g} 0 0" mass="{arm.link1_mass_kg:g}"
                diaginertia="{_rod_inertia(arm.link1_mass_kg, l1)}"/>
      <geom type="capsule" fromto="0 0 0 {l1:.12g} 0 0" size="0.012" contype="0" conaffinity="0"
            rgba="0.7 0.7 0.7 1"/>
      <body name="link2" pos="{l1:.12g} 0 0">
        <joint name="elbow" type="hinge" axis="0 -1 0" armature="{arm.armature_kg_m2[1]:g}"
               frictionloss="{friction[1]:g}" damping="{arm.damping_nm_s_rad[1]:g}" {elbow_limit}/>
        <inertial pos="{l2 / 2:.12g} 0 0" mass="{arm.link2_mass_kg:g}"
                  diaginertia="{_rod_inertia(arm.link2_mass_kg, l2)}"/>
        <geom type="capsule" fromto="0 0 0 {l2:.12g} 0 0" size="0.010" contype="0" conaffinity="0"
              rgba="0.6 0.6 0.8 1"/>
        <body name="tool" pos="{l2:.12g} 0 0">
          <inertial pos="0 0 0" mass="{tool_mass:.12g}"
                    diaginertia="{tool_inertia:.9g} {tool_inertia:.9g} {tool_inertia:.9g}"/>
          <geom name="tool" type="sphere" size="{arm.tool_radius_m:g}" contype="0" conaffinity="0"
                rgba="0.9 0.4 0.1 1"/>
        </body>
      </body>
    </body>{belt_body}
  </worldbody>{belt_pair}
</mujoco>
"""


@functools.lru_cache(maxsize=16)
def _calibrated_model(
    belt: bool, payload_kg: float, stiffness_n_m: float, damping_n_s_m: float, arm: ArmConfig
) -> tuple[mujoco.MjModel, ContactCalibration | None]:
    import mujoco

    if not belt:
        return mujoco.MjModel.from_xml_string(build_mjcf(belt=False, payload_kg=payload_kg, arm=arm)), None
    probe = mujoco.MjModel.from_xml_string(build_mjcf(belt=True, payload_kg=payload_kg, arm=arm))
    data = mujoco.MjData(probe)
    # a probe contact: the tool 5 mm into a stationary belt, to read MuJoCo's inverse-mass estimate of the
    # normal row
    data.qpos[:2] = ik(0.30, arm.tool_touch_z_m - 0.005, arm)
    mujoco.mj_forward(probe, data)
    if data.ncon < 1:
        raise RuntimeError("belt calibration: the probe pose made no tool-belt contact")
    # efc_diagA is MuJoCo 3.11's name for the inverse-mass estimate (efc_diagApprox before 3.x)
    diag = float(data.efc_diagA[data.contact[0].efc_address])
    d = arm.contact_impedance
    solref = (-stiffness_n_m * (1.0 - d) * diag, -damping_n_s_m * (1.0 - d) * diag)
    model = mujoco.MjModel.from_xml_string(
        build_mjcf(belt=True, payload_kg=payload_kg, arm=arm, pair_solref=solref)
    )
    return model, ContactCalibration(stiffness_n_m, damping_n_s_m, d, diag, solref)


def build_model(scene: Scene, arm: ArmConfig = ARM) -> tuple[mujoco.MjModel, ContactCalibration | None]:
    """The MuJoCo model for one scene (cached: models are immutable once built, MjData is per episode)."""
    return _calibrated_model(
        bool(scene.belt),
        float(scene.payload_kg),
        float(scene.contact_stiffness_n_m),
        float(scene.contact_damping_n_s_m),
        arm,
    )


# ── nominal dynamics: what the input scripts put in a trajectory's effort ──────────────────────────────────


@functools.lru_cache(maxsize=4)
def _nominal(arm: ArmConfig) -> tuple[mujoco.MjModel, mujoco.MjData]:
    import mujoco

    # WHY no frictionloss and no joint limit: both are constraints, and mj_inverse would put their
    # (unknowable) forces into the feed-forward; the nominal model is rigid body + armature + viscous damping,
    # no payload.
    model = mujoco.MjModel.from_xml_string(
        build_mjcf(belt=False, arm=arm, frictionloss=False, joint_limits=False)
    )
    return model, mujoco.MjData(model)


def nominal_dynamics(q: Any, qd: Any, qdd: Any, arm: ArmConfig = ARM) -> np.ndarray:
    """Joint torques (N*m) that make the nominal arm follow (q, qd, qdd): mj_inverse, shape (2,) or (N, 2).

    Gravity + inertia (incl. armature) + Coriolis + the gearbox's viscous damping; no dry friction, no
    payload, no contact. This is the feed-forward every JointTrajectory rung puts in `effort`.
    """
    import mujoco

    model, data = _nominal(arm)
    single = np.ndim(q) == 1
    q, qd, qdd = (np.atleast_2d(np.asarray(v, dtype=float)) for v in (q, qd, qdd))
    out = np.empty_like(q)
    for row in range(q.shape[0]):
        data.qpos[:] = q[row]
        data.qvel[:] = qd[row]
        data.qacc[:] = qdd[row]
        mujoco.mj_inverse(model, data)
        out[row] = data.qfrc_inverse
    return out[0] if single else out
