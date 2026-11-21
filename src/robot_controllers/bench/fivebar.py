"""BAM's five-bar arm (bam_fb_v2) as a twin: a closed-chain MuJoCo plant behind the moteus board model, its
closed-chain models and its fixed targets (CONTRACT amendment 5, §5.8).

    from robot_controllers.bench.fivebar import FivebarPlant, fivebar_feedforward, fivebar_rungs
    ff = fivebar_feedforward("rigid_body")       # (6,) -> (6,), (N, 6) -> (N, 6): the six drives' torques
    log = run_episode(spec, fivebar_rungs()["bam_fivebar/path@nominal"], plant_factory=FivebarPlant)

Zach, 2026-11-20: "we want to test in SIM so you can baseline in UR5e to start. and then we want to do on like
bam 5bar Sim and real".

- **Model.** `models/bam_fb_v2_bare.xml`: bam_pid_tunning's `reduced_mjcf()` output byte for byte (bundle
  09374d7, every geom removed; provenance in `bam_fb_v2_bare.json`). Six drives j1 j2 c1 w1 w2 w3 = revolute_3
  / 6 / 7 / 11 / 14 / 17 (CAN 11-16); the elbow revolute_8 and the coupler revolute_34 are the loop's passive
  hinges, closed by its one `connect`. On load the claw hinge (revolute_5) is removed: the claw rides welded
  shut, as bam_arm_fb_sim and arm_motion_bench hold it, so the plant has the six drives and nothing that
  flops.
- **Truth** (`FivebarTruth`, bam_pid_tunning `bam_fb_v2.py` DRIVES): per drive armature, Coulomb (MuJoCo
  frictionloss, it sticks: solref 0.002 and 5 noslip passes as ArmSim) and viscous damping; the closure
  stiffened as ArmSim does (solref 0.002, solimp 0.9999 0.99999). 4 kHz physics, the boards' rate.
- **Boards.** The as-found moteus gains, kp 200 / 144 / 200 and kd 4 / 8 / 4 on the five-bar, kp 75 / kd 0.75
  on the wrist, read in moteus units (N·m per output rev) and converted to the bench's N·m/rad (÷ 2π, as
  bam_pid_tunning's `Gains.from_moteus`); caps 12.5 N·m (five-bar) and 2 A × 0.5087 N·m/A = 1.0174 N·m
  (wrist); encoder 163840 counts/rev (2^14 at the rotor through the 10:1), velocity filter 400 Hz, one 4 kHz
  tick of encoder delay (bam_fb_v2.py JMC_LOOP / GIM_LOOP). The integrator (ki) is not modelled. The bus: the
  bench's default 5.5 ms down, 1.5 ms up (bam_fivebar RACK_BOARD) is the twin's nominal.
- **Feed-forward** (`fivebar_feedforward(name, model_error)`, pure): closed-chain inverse dynamics on the six
  drives. For a drive state (q, qd, qdd) the passive hinges are solved onto the loop (`_Chain.close`, below),
  their velocities and accelerations follow from the loop constraint (S, the selection, and the constraint's
  velocity-product term), MuJoCo's `mj_inverse` on a private constraint-free copy gives the tree's generalized
  forces for that closure-consistent motion, and the drives' torques are their projection τ = Sᵀ·qfrc_inverse.
  `gravity` (qd = qdd = 0), `rigid_body` (+ inertia, Coriolis, viscous), `rigid_body_coulomb` (+
  Coulomb·tanh(qd / 0.01)).
- **Mass matrix** (`fivebar_mass_matrix`): the closed chain's inertia on the drives, Sᵀ·M·S.
- **The nominal model is the twin's own links and drives** (armature, viscous, Coulomb), as the UR5e's is:
  `ModelError` is then the whole of what the controller gets wrong. bam_pid_tunning's own nominal uses the
  description's 0.0058 kg·m² armature on J2 and C1 (truth 0.0052 / 0.0050), a smaller error of the same kind.

The loop closure (`_Chain`): in the five-bar's plane every loop hinge is parallel, so the cut point on L2
circles the elbow hinge and the one on C2 circles the coupler hinge; the loop closes where the two circles
meet, two points, one per assembly. Each is a starting guess for a Gauss-Newton solve on the connect residual
over the two passive hinges, and the answer is the one on the working assembly (the export pose's: the sign of
the passive Jacobian's in-plane determinant). WHY analytic first (arm_motion_bench robots/bam_fb.py `Loop`,
160-235): Newton from the export pose lands on the other assembly for about a third of the reachable drive
set, and which third flips with sub-degree changes. WHY it is pure: the guess depends on the drives alone, so
equal inputs give equal outputs, never a warm start's.
"""

from __future__ import annotations

import functools
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_controllers.bench.arms import ArmSpec, register
from robot_controllers.bench.board import BoardConfig
from robot_controllers.bench.plant import PREROLL_S, TIME_TOLERANCE_S, ServoPlant
from robot_controllers.bench.types import HostTiming, Rung, Scene
from robot_controllers.models import ModelError

ARM_NAME = "bam_fivebar"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "bam_fb_v2_bare.xml"
FIVEBAR_JOINTS: tuple[str, ...] = ("j1", "j2", "c1", "w1", "w2", "w3")
MJCF_JOINTS: tuple[str, ...] = (
    "revolute_3",
    "revolute_6",
    "revolute_7",
    "revolute_11",
    "revolute_14",
    "revolute_17",
)
CAN_IDS: tuple[int, ...] = (11, 12, 13, 14, 15, 16)
PASSIVE_JOINTS: tuple[str, str] = (
    "revolute_8",
    "revolute_34",
)  # the elbow (l1 -> l2) and the coupler (c1 -> c2)
CLAW_JOINT = "revolute_5"
TOOL_SITE = "tool"  # reduced_mjcf's site at the claw body's origin
# WHY the vendored model's own home (its `home` keyframe = bam_pid_tunning bam_fb_v2.py:55-57 HOME_DEG, gate
# 1's "above (from park)" stop, proven clear of both self-collision rules), not arm_motion_bench's station
# home (robots/bam_fb.py:59, where its tapes start): the targets start from their own tape's first sample
# whatever home is, and a pose 10 deg out from the station home in j2 / c1 (the acceptance hold) needs 14.5
# N.m of J2 gravity against its 12.5 N.m cap (measured 2026-11-20: the loop's transmission halves there),
# where the same offsets from this home need 6.5 N.m (0.52 of the cap). The claw is welded shut here (0), not
# at the keyframe's 38.5 deg.
FIVEBAR_HOME_Q: tuple[float, ...] = tuple(
    math.radians(degrees) for degrees in (0.0, 18.85, 40.91, 27.61, 90.0, 0.0)
)
# arm_motion_bench's station home (robots/bam_fb.py:59): every bam_fb tape, dev000 included, starts and ends
# here
STATION_HOME_Q: tuple[float, ...] = (
    0.02352648644982546,
    0.7011585996259065,
    0.9565127140169948,
    0.4141484027007757,
    1.547269840341145,
    9.01198379919866e-13,
)
MOTEUS_RAD_PER_REV = (
    2.0 * math.pi
)  # moteus kp is N·m per output revolution: kp_rad = kp_moteus / 2π (Gains.from_moteus)
COULOMB_SMOOTHING_RAD_S = (
    0.01  # rigid_body_coulomb = ... + Coulomb · tanh(qd / 0.01 rad/s), as the UR5e and the planar arm
)
FEEDFORWARD_NAMES: tuple[str, ...] = ("gravity", "rigid_body", "rigid_body_coulomb")
# WHY these (bam_pid_tunning plants/mujoco_arm.py:57-65, measured there on this five-bar 2026-10-25): a 2 ms
# friction constraint and noslip keep a stuck drive stuck as the rack's do; the default closure regularisation
# read the crank 18 % lighter than the rigid linkage, (0.9999, 0.99999) within 0.3 %, closed to ~10 um through
# a 0.25 rad move.
FRICTION_SOLREF = (0.002, 1.0)
NOSLIP_ITERATIONS = 5
CLOSURE_SOLREF = (0.002, 1.0)
CLOSURE_SOLIMP = (0.9999, 0.99999, 0.001, 0.5, 2.0)
CLOSURE_TOLERANCE_M = 1e-12
CLOSURE_ITERATIONS = 30
VELOCITY_PRODUCT_STEP_RAD = 1e-6  # the central difference of the loop Jacobian along qd: a 1 urad step


@dataclass(frozen=True)
class FivebarTruth:
    """The twin's hidden per-drive truth, j1 j2 c1 w1 w2 w3 (bam_pid_tunning plants/bam_fb_v2.py
    DRIVES:97-139), and the boards as found (`as_found_moteus`, moteus units)."""

    # J1: the description (no pan twin on this mount) and rack_truths can11's friction; J2 / C1:
    # rack_truths.TWIN_SEEDED_PLL can12 / can13 (round 10); W1-W3: bam_ur_wrist_coverage refit_rig it5 (CAN
    # 14-16)
    armature_kg_m2: tuple[float, ...] = (0.0058, 0.0052156, 0.004959, 0.001, 0.001347, 0.001111)
    coulomb_nm: tuple[float, ...] = (0.70, 0.593149, 0.621496, 0.0413, 0.065646, 0.061131)
    viscous_nm_s_rad: tuple[float, ...] = (0.08, 0.14375, 0.140625, 0.0, 0.000448, 0.0)
    # the five-bar's JMC 60ASM400 + AD60 10:1 at 12.5 N.m; the wrist's GIM4305-10 at 2 A x 0.5087 N.m/A
    # (bam_fb_v2.py:73)
    torque_cap_nm: tuple[float, ...] = (12.5, 12.5, 12.5, 1.0174, 1.0174, 1.0174)
    # the boards' live servo.pid_position (N.m/rev, N.m.s/rev; the rack read 2026-11-10/13): five-bar kp 200 /
    # 144 / 200, kd 4 / 8 / 4; the wrist kp 75 / kd 0.75 since the rack owner's never-limp runners
    # (bam_fb_v2.py docstring)
    kp_moteus: tuple[float, ...] = (200.0, 144.0, 200.0, 75.0, 75.0, 75.0)
    kd_moteus: tuple[float, ...] = (4.0, 8.0, 4.0, 0.75, 0.75, 0.75)
    encoder_counts_per_rev: int = 163840  # JMC_LOOP / GIM_LOOP (bam_fb_v2.py:69-72)
    velocity_filter_hz: float = 400.0
    encoder_delay_s: float = 0.00025  # one 4 kHz tick (bam_fb_v2.py:66-68 says why not the host delay)
    gear_ratio: float = 10.0  # every drive; the board's frame is the output's, so nothing here reads it
    physics_dt_s: float = 0.00025  # 4 kHz, the boards' rate


FIVEBAR_TRUTH = FivebarTruth()


# ── the MuJoCo model ───────────────────────────────────────────────────────────────────────────────────────


def _link_factors(model: Any, model_error: ModelError | None) -> dict[str, float]:
    """Each body's mass/inertia factor: a body that carries a joint is a link (model order, `link_factors`); a
    body without one rides with its nearest jointed ancestor (the welded claw with hard_stop_3); the base is
    never scaled."""
    import mujoco

    links = [b for b in range(model.nbody) if model.body_jntnum[b] > 0]
    factors = (
        np.ones(len(links))
        if model_error is None
        else np.asarray(model_error.link_factors(len(links)), dtype=float)
    )
    by_link = dict(zip(links, factors.tolist(), strict=True))
    out: dict[str, float] = {}
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor > 0 and ancestor not in by_link:
            ancestor = int(model.body_parentid[ancestor])
        out[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)] = by_link.get(ancestor, 1.0)
    return out


def load_model(
    *,
    payload_kg: float = 0.0,
    nominal: bool = False,
    model_error: ModelError | None = None,
    truth: FivebarTruth = FIVEBAR_TRUTH,
) -> Any:
    """A fresh MjModel of the vendored five-bar: the claw welded shut, the closure stiffened, the drives'
    truth.

    `nominal`: the controller's private copy, constraints disabled (mj_inverse must not fold the loop's own
    force or a friction constraint into its answer: kin.py:190-201), no frictionloss, `model_error` applied to
    its links (mass and inertia) and its damping. The plant (`nominal=False`) carries frictionloss with noslip
    and an optional point payload at the tool site; it never takes a model error.
    """
    import mujoco

    if model_error is not None and not nominal:
        raise ValueError("ModelError changes what the controller is told, never the plant")
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
    spec.delete(spec.joint(CLAW_JOINT))
    for key in list(spec.keys):
        spec.delete(key)  # the 9-joint home keyframe no longer fits
    spec.option.timestep = truth.physics_dt_s
    friction = 1.0 if model_error is None else float(model_error.friction_factor)
    for name, armature, coulomb, viscous in zip(
        MJCF_JOINTS, truth.armature_kg_m2, truth.coulomb_nm, truth.viscous_nm_s_rad, strict=True
    ):
        joint = spec.joint(name)
        # WHY on the spec, not the compiled model: MuJoCo bakes a lone body's dof mass at compile time
        # (mujoco_arm.py:158-159)
        joint.armature = armature
        # WHY a 3-vector: MuJoCo 3.11's joint damping is a polynomial in qd; only the linear term is set
        joint.damping = [viscous * friction, 0.0, 0.0]
        joint.frictionloss = 0.0 if nominal else coulomb
        if not nominal:
            joint.solref_friction = FRICTION_SOLREF
    if model_error is not None:
        for name in PASSIVE_JOINTS:
            spec.joint(name).damping = [float(spec.joint(name).damping[0]) * friction, 0.0, 0.0]
    for equality in spec.equalities:
        equality.solref = CLOSURE_SOLREF
        equality.solimp = CLOSURE_SOLIMP
    if not nominal:
        spec.option.noslip_iterations = NOSLIP_ITERATIONS
    payload = float(payload_kg)
    if payload and nominal:
        raise ValueError("the nominal model carries no payload: the controller does not know it")
    if payload:
        site = spec.site(TOOL_SITE)
        # WHY a child body at the tool site (as the UR5e's payload): a point mass where it is held, the
        # vendored link's own inertials untouched (the diaginertia only satisfies MuJoCo's body-inertia check)
        body = site.parent.add_body(name="payload", pos=list(site.pos))
        body.mass = payload
        body.inertia = [1e-9, 1e-9, 1e-9]
        body.explicitinertial = True
    if model_error is not None and not model_error.is_nominal:
        factors = _link_factors(spec.compile(), model_error)
        for body in spec.bodies:
            factor = factors.get(body.name, 1.0)
            if factor != 1.0:
                body.mass = body.mass * factor
                body.inertia = [value * factor for value in body.inertia]
    model = spec.compile()
    if nominal:
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    return model


@functools.lru_cache(maxsize=8)
def _plant_model(payload_kg: float, truth: FivebarTruth) -> Any:
    # models are immutable once built; MjData is per episode
    return load_model(payload_kg=payload_kg, truth=truth)


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class _Chain:
    """The closed chain's kinematics on one model and a scratch MjData: the loop closed from the drives, the
    selection S (nv x 6) and the loop constraint's velocity-product term. Every call poses its own data from
    its arguments."""

    def __init__(self, model: Any) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        if model.nq != model.nv:
            raise ValueError("the five-bar's chain expects hinges only (qpos and qvel indexed alike)")
        ids = [model.joint(name).id for name in MJCF_JOINTS]
        self.act = np.array([int(model.jnt_dofadr[j]) for j in ids])
        self.elbow, self.coupler = (model.joint(name).id for name in PASSIVE_JOINTS)
        self.passive = np.array([int(model.jnt_dofadr[self.elbow]), int(model.jnt_dofadr[self.coupler])])
        connects = [e for e in range(model.neq) if model.eq_type[e] == mujoco.mjtEq.mjEQ_CONNECT]
        if len(connects) != 1:
            raise ValueError(f"the five-bar has one loop closure, found {len(connects)} connects")
        eq = connects[0]
        bodies = (int(model.eq_obj1id[eq]), int(model.eq_obj2id[eq]))
        anchors = (model.eq_data[eq, 0:3].copy(), model.eq_data[eq, 3:6].copy())
        self.b1, self.b2 = bodies
        self.a1, self.a2 = anchors
        elbow_body, coupler_body = int(model.jnt_bodyid[self.elbow]), int(model.jnt_bodyid[self.coupler])
        if set(bodies) != {elbow_body, coupler_body}:
            raise ValueError("the loop's cut is not between the elbow's and the coupler's links")
        side = {bodies[0]: anchors[0], bodies[1]: anchors[1]}
        self.l2_body, self.l2_anchor = elbow_body, side[elbow_body]
        self.c2_body, self.c2_anchor = coupler_body, side[coupler_body]
        self._jac1 = np.zeros((3, model.nv))
        self._jac2 = np.zeros((3, model.nv))
        self.qpos0 = model.qpos0.copy()
        self._pose(self.qpos0, com=True)
        self.working_branch = self._branch()

    # ── posing ─────────────────────────────────────────────────────────────────────────────────────────────

    def _pose(self, qpos: np.ndarray, *, com: bool) -> None:
        self.data.qpos[:] = qpos
        self._mujoco.mj_kinematics(self.model, self.data)
        if com:
            self._mujoco.mj_comPos(self.model, self.data)

    def _point(self, body: int, anchor: np.ndarray) -> np.ndarray:
        data = self.data
        return data.xpos[body] + data.xmat[body].reshape(3, 3) @ anchor

    def residual(self) -> np.ndarray:
        """The cut's gap (m, world) at the posed data: the point on body1 minus the point on body2."""
        return self._point(self.b1, self.a1) - self._point(self.b2, self.a2)

    def jacobian(self) -> np.ndarray:
        """d residual / d qvel (3 x nv) at the posed data (needs mj_comPos)."""
        mujoco = self._mujoco
        mujoco.mj_jac(self.model, self.data, self._jac1, None, self._point(self.b1, self.a1), self.b1)
        mujoco.mj_jac(self.model, self.data, self._jac2, None, self._point(self.b2, self.a2), self.b2)
        return self._jac1 - self._jac2

    def _plane(self) -> np.ndarray:
        normal = self.data.xaxis[self.elbow].copy()
        normal /= np.linalg.norm(normal)
        first = np.cross(normal, [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0])
        first /= np.linalg.norm(first)
        return np.vstack([first, np.cross(normal, first)])

    def _branch(self) -> float:
        """The assembly the posed loop is on: the sign of the passive Jacobian's in-plane determinant."""
        return float(np.sign(np.linalg.det(self._plane() @ self.jacobian()[:, self.passive])))

    # ── the loop ───────────────────────────────────────────────────────────────────────────────────────────

    def close(self, drives: Any) -> np.ndarray:
        """Full qpos with the drives at `drives` (rad, j1 j2 c1 w1 w2 w3) and the passive hinges on the loop,
        on the working assembly; the data is left posed there (kinematics and com). ValueError where the loop
        cannot close."""
        start = self.qpos0.copy()
        start[self.act] = np.asarray(drives, dtype=float)
        self._pose(start, com=False)
        data = self.data
        basis = self._plane()
        normal = data.xaxis[self.elbow] / np.linalg.norm(data.xaxis[self.elbow])
        elbow, coupler = basis @ data.xanchor[self.elbow], basis @ data.xanchor[self.coupler]
        on_l2 = basis @ self._point(self.l2_body, self.l2_anchor)
        on_c2 = basis @ self._point(self.c2_body, self.c2_anchor)
        coupler_sign = float(np.sign(data.xaxis[self.coupler] @ normal))
        reach_l2, reach_c2 = float(np.linalg.norm(on_l2 - elbow)), float(np.linalg.norm(on_c2 - coupler))
        span = float(np.linalg.norm(coupler - elbow))
        if span < 1e-9 or span > reach_l2 + reach_c2 or span < abs(reach_l2 - reach_c2):
            raise ValueError(
                "bam_fivebar: the loop does not close at "
                f"{np.round(np.degrees(start[self.act]), 3).tolist()} deg"
            )
        along = (span**2 + reach_l2**2 - reach_c2**2) / (2.0 * span)
        height = math.sqrt(max(reach_l2**2 - along**2, 0.0))
        direction = (coupler - elbow) / span
        across = np.array([-direction[1], direction[0]])

        def angle(vector: np.ndarray) -> float:
            return math.atan2(float(vector[1]), float(vector[0]))

        for side in (1.0, -1.0):
            meet = elbow + along * direction + side * height * across
            guess = start.copy()
            guess[self.passive[0]] += _wrap(angle(meet - elbow) - angle(on_l2 - elbow))
            guess[self.passive[1]] += coupler_sign * _wrap(angle(meet - coupler) - angle(on_c2 - coupler))
            qpos, gap = self._newton(guess)
            if gap <= 1e-9 and self._branch() == self.working_branch:
                return qpos
        raise ValueError(
            f"bam_fivebar: near a fold at {np.round(np.degrees(start[self.act]), 3).tolist()} deg"
        )

    def _newton(self, qpos: np.ndarray) -> tuple[np.ndarray, float]:
        """Gauss-Newton on the cut's gap over the passive hinges (kin.py:91-114); leaves the data posed."""
        qpos = qpos.copy()
        gap = math.inf
        for _ in range(CLOSURE_ITERATIONS):
            self._pose(qpos, com=True)
            residual = self.residual()
            gap = float(np.linalg.norm(residual))
            if gap < CLOSURE_TOLERANCE_M:
                break
            step = np.linalg.lstsq(self.jacobian()[:, self.passive], -residual, rcond=None)[0]
            qpos[self.passive] += step
        self._pose(qpos, com=True)
        return qpos, gap

    def selection(self) -> np.ndarray:
        """S (nv x 6) at the posed data: joint velocities of the closed chain per unit drive velocity
        (kin.py:116-130; the passive rows from the loop constraint, Jc_p qd_p = -Jc_a qd_a)."""
        jacobian = self.jacobian()
        out = np.zeros((self.model.nv, len(self.act)))
        out[self.act, np.arange(len(self.act))] = 1.0
        out[self.passive, :] = np.linalg.lstsq(jacobian[:, self.passive], -jacobian[:, self.act], rcond=None)[
            0
        ]
        return out

    def velocity_product(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """(d Jc / dt) qvel (3,): the loop's acceleration at zero qacc, by a central difference of Jc along
        qvel. Poses the data away from qpos: the caller re-poses."""
        speed = float(np.linalg.norm(qvel))
        if speed == 0.0:
            return np.zeros(3)
        step = VELOCITY_PRODUCT_STEP_RAD / speed
        self._pose(qpos + step * qvel, com=True)
        ahead = self.jacobian()
        self._pose(qpos - step * qvel, com=True)
        behind = self.jacobian()
        return (ahead - behind) @ qvel / (2.0 * step)

    def state(self, q: Any, qd: Any, qdd: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(qpos, qvel, qacc, S) of the whole tree for one drive state, closure-consistent to second order."""
        qpos = self.close(q)
        selection = self.selection()
        qvel = selection @ np.asarray(qd, dtype=float)
        jacobian = self.jacobian()
        bias = self.velocity_product(qpos, qvel)
        qacc = np.zeros(self.model.nv)
        qacc[self.act] = np.asarray(qdd, dtype=float)
        rhs = -(jacobian[:, self.act] @ qacc[self.act] + bias)
        qacc[self.passive] = np.linalg.lstsq(jacobian[:, self.passive], rhs, rcond=None)[0]
        self._pose(qpos, com=True)
        return qpos, qvel, qacc, selection


# ── the models (pure functions of their inputs) ────────────────────────────────────────────────────────────


class _Private:
    """A private nominal model, its chain and the Coulomb vector the controller is told: what makes each
    five-bar model function pure from the outside. Built on first use; never pickled."""

    __slots__ = ("model", "chain", "coulomb")

    def __init__(self, model_error: ModelError | None) -> None:
        self.model = load_model(nominal=True, model_error=model_error)
        self.chain = _Chain(self.model)
        friction = 1.0 if model_error is None else float(model_error.friction_factor)
        self.coulomb = np.asarray(FIVEBAR_TRUTH.coulomb_nm, dtype=float) * friction


def _nominal(model_error: ModelError | None) -> ModelError | None:
    return None if model_error is None or model_error.is_nominal else model_error


def _rows(q: Any, qd: Any, qdd: Any) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    single = np.ndim(q) == 1
    q, qd, qdd = (np.atleast_2d(np.asarray(v, dtype=np.float64)) for v in (q, qd, qdd))
    return single, q, qd, qdd


class _Lazy:
    __slots__ = ("model_error", "_private")

    def __init__(self, model_error: ModelError | None) -> None:
        self.model_error = _nominal(model_error)
        self._private: _Private | None = None

    def private(self) -> _Private:
        if self._private is None:
            self._private = _Private(self.model_error)
        return self._private

    def __getstate__(self) -> dict[str, Any]:
        return {"model_error": self.model_error}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.model_error = state["model_error"]
        self._private = None


class FivebarFeedforward(_Lazy):
    """`ff(q, qd, qdd) -> tau` on the six drives: (6,) -> (6,), (N, 6) -> (N, 6). Pure; closed-chain
    `mj_inverse` on a private constraint-free nominal model projected through S (`gravity` with qd and qdd
    zero)."""

    __slots__ = ("name",)

    def __init__(self, name: str, model_error: ModelError | None = None) -> None:
        if name not in FEEDFORWARD_NAMES:
            raise ValueError(f"unknown five-bar feed-forward {name!r}; known: {FEEDFORWARD_NAMES}")
        super().__init__(model_error)
        self.name = name

    def __call__(self, q: Any, qd: Any, qdd: Any) -> np.ndarray:
        import mujoco

        private = self.private()
        model, chain = private.model, private.chain
        data = chain.data
        single, q, qd, qdd = _rows(q, qd, qdd)
        out = np.empty((q.shape[0], len(MJCF_JOINTS)))
        gravity_only = self.name == "gravity"
        zero = np.zeros(len(MJCF_JOINTS))
        for row in range(q.shape[0]):
            if gravity_only:
                qpos, qvel, qacc, selection = chain.state(q[row], zero, zero)
            else:
                qpos, qvel, qacc, selection = chain.state(q[row], qd[row], qdd[row])
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            data.qacc[:] = qacc
            mujoco.mj_inverse(model, data)
            out[row] = selection.T @ data.qfrc_inverse
        if self.name == "rigid_body_coulomb":
            out = out + private.coulomb * np.tanh(qd / COULOMB_SMOOTHING_RAD_S)
        return out[0] if single else out

    def __getstate__(self) -> dict[str, Any]:
        return {"model_error": self.model_error, "name": self.name}

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        self.name = state["name"]

    def __repr__(self) -> str:
        error = "" if self.model_error is None else f", {self.model_error!r}"
        return f"fivebar_feedforward({self.name!r}{error})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FivebarFeedforward) and (other.name, other.model_error) == (
            self.name,
            self.model_error,
        )

    def __hash__(self) -> int:
        return hash(("fivebar_feedforward", self.name, self.model_error))


class FivebarMassMatrix(_Lazy):
    """`mass(q) -> M(q)`: the closed chain's inertia on the six drives, Sᵀ·M·S, armature included: (6,) -> (6,
    6), (N, 6) -> (N, 6, 6). Pure."""

    __slots__ = ()

    def __call__(self, q: Any) -> np.ndarray:
        import mujoco

        private = self.private()
        model, chain = private.model, private.chain
        q = np.asarray(q, dtype=np.float64)
        rows = np.atleast_2d(q)
        out = np.zeros((rows.shape[0], len(MJCF_JOINTS), len(MJCF_JOINTS)))
        full = np.zeros((model.nv, model.nv))
        for row in range(rows.shape[0]):
            chain.close(rows[row])
            selection = chain.selection()
            # the position stage only (as the UR5e's): close() left kinematics and com; composite inertias,
            # then M
            mujoco.mj_crb(model, chain.data)
            mujoco.mj_fullM(model, chain.data, full)
            out[row] = selection.T @ full @ selection
        return out[0] if q.ndim == 1 else out

    def __repr__(self) -> str:
        return (
            "fivebar_mass_matrix()"
            if self.model_error is None
            else f"fivebar_mass_matrix({self.model_error!r})"
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FivebarMassMatrix) and other.model_error == self.model_error

    def __hash__(self) -> int:
        return hash(("fivebar_mass_matrix", self.model_error))


def fivebar_feedforward(terms: str, model_error: ModelError | None = None) -> FivebarFeedforward:
    """The five-bar feed-forward named `terms` (`gravity`, `rigid_body`, `rigid_body_coulomb`) on a model
    `model_error` away from the twin; None and `ModelError()` are the nominal model exactly."""
    return FivebarFeedforward(terms, model_error)


def fivebar_mass_matrix(model_error: ModelError | None = None) -> FivebarMassMatrix:
    """The closed chain's inertia on the six drives, M_act(q) = Sᵀ M(q) S, on a model `model_error` away from
    the twin."""
    return FivebarMassMatrix(model_error)


@functools.lru_cache(maxsize=1)
def _kinematics() -> _Chain:
    return _Chain(load_model(nominal=True))


def tool_position(q: Any) -> np.ndarray:
    """The tool site's world position (m) from the six drives, the loop closed (encoder FK): (6,) -> (3,), (N,
    6) -> (N, 3)."""
    chain = _kinematics()
    site = chain.model.site(TOOL_SITE).id
    q = np.asarray(q, dtype=np.float64)
    rows = np.atleast_2d(q)
    out = np.empty((rows.shape[0], 3))
    for row in range(rows.shape[0]):
        chain.close(rows[row])
        out[row] = chain.data.site_xpos[site]
    return out[0] if q.ndim == 1 else out


def closed_qpos(q: Any) -> np.ndarray:
    """The full model qpos (the drives and the two passive hinges) for six drive angles, loop closed."""
    return _kinematics().close(q).copy()


# ── the boards ─────────────────────────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def fivebar_board_config() -> BoardConfig:
    """The five-bar's boards as found: kp, kd from moteus units (per output rev) to N·m/rad, the caps, encoder
    163840 counts/rev, a 400 Hz velocity filter and one tick of encoder delay, at 4 kHz."""
    truth = FIVEBAR_TRUTH
    return BoardConfig(
        kp_nm_rad=tuple(kp / MOTEUS_RAD_PER_REV for kp in truth.kp_moteus),
        kd_nm_s_rad=tuple(kd / MOTEUS_RAD_PER_REV for kd in truth.kd_moteus),
        torque_cap_nm=truth.torque_cap_nm,
        rate_hz=1.0 / truth.physics_dt_s,
        encoder_delay_s=truth.encoder_delay_s,
        velocity_filter_hz=truth.velocity_filter_hz,
        encoder_counts_per_rev=truth.encoder_counts_per_rev,
    )


# ── the plant ──────────────────────────────────────────────────────────────────────────────────────────────


class FivebarPlant(ServoPlant):
    """The `bam_fivebar` plant: the closed-chain bam_fb_v2 in MuJoCo behind one moteus board model per drive
    (the interface of `PlanarArmPlant`; no belt, no pushes). It commands and logs the six drives only;
    `meta()` carries the loop's largest gap over the episode."""

    name = ARM_NAME
    joint_names: tuple[str, ...] = FIVEBAR_JOINTS

    def __init__(
        self, *, board_config: BoardConfig | None = None, truth: FivebarTruth = FIVEBAR_TRUTH
    ) -> None:
        self.truth = truth
        super().__init__(board_config or fivebar_board_config(), truth.physics_dt_s)
        self._gravity = FivebarFeedforward("gravity")

    @property
    def home_q(self) -> tuple[float, ...]:
        return FIVEBAR_HOME_Q

    def tool_position(self, q: Any) -> np.ndarray:
        """The tool site (x, y, z) for drive angles `q`, (6,) -> (3,), (N, 6) -> (N, 3)."""
        return tool_position(q)

    def reset(
        self,
        scene: Scene,
        seed: int = 0,
        *,
        timing: HostTiming | None = None,
        duration_s: float = 5.0,
    ) -> None:
        """A new episode, as `PlanarArmPlant.reset`: the arm at rest at `scene.start_q` (or home) with the
        loop closed, the boards holding it with the nominal gravity feed-forward through a 20 ms unrecorded
        pre-roll."""
        import mujoco

        if scene.belt or scene.pushes:
            raise ValueError("the five-bar plant has no belt and takes no pushes")
        self._mujoco = mujoco
        self.scene = scene
        self.seed = int(seed)
        self.timing = timing or HostTiming()
        self._reset_bus(seed, self.timing)
        self.contact_calibration = None
        self._model = model = _plant_model(float(scene.payload_kg), self.truth)
        self._data = data = mujoco.MjData(model)
        self._chain = chain = _Chain(model)
        self._act = chain.act
        start_q = np.asarray(scene.start_q if scene.start_q is not None else self.home_q, dtype=float)
        if start_q.shape != (len(self.joint_names),):
            raise ValueError(f"start_q must be {len(self.joint_names)} drive angles, got {start_q}")
        self.start_q = tuple(float(v) for v in start_q)
        data.qpos[:] = chain.close(start_q)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        self._tool_site = model.site(TOOL_SITE).id
        self._warnings_seen = self._warning_count()
        self.events: list[dict[str, Any]] = []
        self._loop_gap_max = 0.0
        preroll_ticks = int(round(PREROLL_S / self._dt))
        self._first_k = -preroll_ticks
        self._k = self._first_k
        hold = self._gravity(start_q, np.zeros_like(start_q), np.zeros_like(start_q))
        self.boards.reset(self._first_k * self._dt, start_q, feedforward=hold)
        self._allocate(preroll_ticks + int(math.ceil(duration_s / self._dt)) + 8)
        self._last_sample_k: int | None = None
        self._seq = 0
        self._writes = 0
        self._now = self._first_k * self._dt
        self.advance(0.0)

    def advance(self, until: float) -> None:
        """Run physics and boards over every tick with k * dt < `until` (s)."""
        mujoco, model, data, boards = self._mujoco, self._model, self._data, self.boards
        dt, site, act = self._dt, self._tool_site, self._act
        qpos, qvel, qfrc, site_xpos, efc_pos = (
            data.qpos,
            data.qvel,
            data.qfrc_applied,
            data.site_xpos,
            data.efc_pos,
        )
        act_list = act.tolist()
        first_k = self._first_k
        k = self._k
        gap = self._loop_gap_max
        while k * dt < until - TIME_TOLERANCE_S:
            t = k * dt
            if k - first_k >= self._capacity:
                self._grow()
            q_true = [qpos[i] for i in act_list]
            qd_true = [qvel[i] for i in act_list]
            tau = boards.step(t, q_true)
            for index, value in zip(act_list, tau, strict=True):
                qfrc[index] = value
            mujoco.mj_step(model, data)
            # the equality rows come first in efc: the connect's 3 position errors at the step's state
            if data.ne:
                gap = max(gap, float(np.linalg.norm(efc_pos[: data.ne])))
            row = k - first_k
            self._sample_q[row] = boards.measured_position
            self._sample_qd[row] = boards.measured_velocity
            self._sample_tau[row] = tau
            for joint in boards.watchdog_entries:
                self.events.append({"t": t, "kind": "watchdog", "joint": self.joint_names[joint]})
            if k >= 0:
                self._t[k] = t
                self._q[k] = q_true
                self._qd[k] = qd_true
                self._tau[k] = tau
                self._at_cap[k] = boards.at_cap
                self._mode[k] = boards.mode
                self._tool_pos[k] = site_xpos[site]
                self._tau_ff[k] = boards.tau_ff
                self._tau_fb_board[k] = boards.tau_fb
                self._latched_tick[k] = boards.latched_tag
            k += 1
        self._k = k
        self._loop_gap_max = gap
        self._now = max(self._now, float(until))
        self._check_divergence()

    def meta(self) -> dict[str, Any]:
        """What this plant was, for EpisodeLog.meta."""
        return {
            "plant": self.name,
            "physics_dt_s": self._dt,
            "preroll_s": -self._first_k * self._dt,
            "start_q": list(self.start_q),
            "board": asdict(self.board_config),
            "truth": asdict(self.truth),
            "model": MODEL_PATH.name,
            "tool_site": TOOL_SITE,
            "payload_kg": float(self.scene.payload_kg),
            "loop_gap_max_m": float(self._loop_gap_max),
        }


# ── the targets ────────────────────────────────────────────────────────────────────────────────────────────

# WHY 0.5 nominal: the speed the real showcase ran dev000 at (arm_motion_bench
# scripts/bam_fb_real_showcase.py: the rung with the lowest moving RMS and stop error that ends at READY F),
# so a real log of the same tape compares one to one.
SPEEDS: dict[str, float] = {"slow": 0.25, "nominal": 0.5, "fast": 1.0}
TAPES: dict[float, str] = {
    1.0: "bam_fb/workspace-dev-000.npz",
    0.5: "bam_fb/workspace-dev-000__s0.5.npz",
    0.25: "bam_fb/workspace-dev-000__s0.25.npz",
}
CALIBRATION_AMPLITUDE_RAD = math.radians(8.0)  # every drive sweeps +-8 deg (>= 5 deg peak to peak, §5.4)
CALIBRATION_DURATION_S = 4.0
CALIBRATION_FREQUENCIES_HZ = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75)  # distinct, so no two drives move in step
PAD_S = 0.6  # each rung runs 0.6 s past its script (as R1 / R2)


@functools.lru_cache(maxsize=8)
def path_trajectory(speed: float) -> Any:
    """dev000 at speed fraction `speed` (0.25, 0.5 or 1.0) on the 4 ms grid, effort = nominal `rigid_body`."""
    from robot_controllers.bench.arm_tapes import tape_trajectory

    return tape_trajectory(TAPES[float(speed)], FivebarFeedforward("rigid_body"))


@functools.lru_cache(maxsize=1)
def calibration_trajectory() -> Any:
    """Every drive swept +-8 deg about home at once (distinct frequencies), at rest at both ends; effort =
    rigid_body."""
    from robot_msgs.trajectory import JointTrajectory

    from robot_controllers.bench.arm_tapes import sine_sweep

    t, q, qd, qdd = sine_sweep(
        FIVEBAR_HOME_Q,
        amplitude_rad=CALIBRATION_AMPLITUDE_RAD,
        duration_s=CALIBRATION_DURATION_S,
        frequencies_hz=CALIBRATION_FREQUENCIES_HZ,
    )
    return JointTrajectory(
        t=t,
        joint_names=FIVEBAR_JOINTS,
        positions=q,
        velocities=qd,
        accelerations=qdd,
        effort=FivebarFeedforward("rigid_body")(q, qd, qdd),
    )


def _rung(
    rung_id: str, title: str, description: str, script: Any, start_q: tuple[float, ...], params: dict
) -> Rung:
    from robot_controllers.bench.rungs import rung
    from robot_controllers.registry import JOINT_TRAJECTORY

    return Rung(
        id=rung_id,
        title=title,
        box=JOINT_TRAJECTORY,
        family="free_space",
        description=description,
        duration_s=round(script.duration_s + PAD_S, 6),
        inputs=script,
        scene=Scene(start_q=start_q),
        timing=HostTiming(),
        gates=rung("R2.FastFreeSpace").gates,
        objective="tracking_p95_deg",
        params=params,
        plant=ARM_NAME,
    )


def _speed_params(trajectory: Any) -> dict[str, float]:
    q = np.asarray(trajectory.positions)
    tool = tool_position(q)
    t = np.asarray(trajectory.t)
    return {
        "peak_joint_speed_deg_s": math.degrees(float(np.max(np.abs(np.asarray(trajectory.velocities))))),
        "peak_tool_speed_m_s": float(np.max(np.linalg.norm(np.gradient(tool, t, axis=0), axis=1))),
    }


def path_rung(speed: str) -> Rung:
    """`bam_fivebar/path@<speed>`: arm_motion_bench's BAM FB workspace v2 dev000 (the ISO 9283 line and
    rectangle) at slow 0.25, nominal 0.5 or fast 1.0, from its own start (home), host 100 Hz, R2's gates."""
    from robot_controllers.bench.arm_tapes import TrajectoryScript, tape_record

    fraction = SPEEDS[speed]
    trajectory = path_trajectory(fraction)
    record = tape_record(TAPES[fraction])
    script = TrajectoryScript(functools.partial(path_trajectory, fraction), float(trajectory.t[-1]))
    return _rung(
        f"{ARM_NAME}/path@{speed}",
        f"BAM five-bar dev000 path at {fraction:g}",
        "arm_motion_bench's BAM FB workspace v2 dev000 tape (ISO 9283 line + rectangle, bam_ruckig) at speed "
        f"{fraction:g}, resampled onto the 4 ms grid. Tests tracking on the closed-chain arm on the tape "
        "the real arm runs.",
        script,
        tuple(float(v) for v in trajectory.positions[0]),
        {
            "speed_fraction": fraction,
            "tape": record["file"],
            "tape_sha256": record["sha256"],
            **_speed_params(trajectory),
        },
    )


def calibration_rung() -> Rung:
    """`bam_fivebar/calibration`: every drive swept +-8 deg about home at once for 4 s: each lag shows."""
    trajectory = calibration_trajectory()
    from robot_controllers.bench.arm_tapes import TrajectoryScript

    return _rung(
        f"{ARM_NAME}/calibration",
        "BAM five-bar calibration sweep",
        "Every drive sweeps +-8 deg about home at once (sin^2-windowed sines, 0.5-1.75 Hz), at rest at both "
        "ends. Tests nothing by itself: it is the run a bus calibration reads each joint's delay from.",
        TrajectoryScript(calibration_trajectory, CALIBRATION_DURATION_S),
        FIVEBAR_HOME_Q,
        {"amplitude_deg": math.degrees(CALIBRATION_AMPLITUDE_RAD), **_speed_params(trajectory)},
    )


def fivebar_rungs() -> dict[str, Rung]:
    """The five-bar's path rungs at three speeds and its calibration sweep, by id."""
    rungs = [path_rung(speed) for speed in SPEEDS] + [calibration_rung()]
    return {r.id: r for r in rungs}


# ── the registry ───────────────────────────────────────────────────────────────────────────────────────────


def _plant(board_config: BoardConfig | None = None) -> FivebarPlant:
    return FivebarPlant(board_config=board_config)


def _feedforward(name: str, model_error: ModelError | None = None) -> FivebarFeedforward:
    return FivebarFeedforward(name, model_error)


def _targets() -> tuple[Any, ...]:
    raise NotImplementedError("bam_fivebar's Target records land with lane A's bench.targets (CONTRACT §5.4)")


SPEC = register(
    ArmSpec(
        name=ARM_NAME,
        title="BAM five-bar (6 drives)",
        joint_names=FIVEBAR_JOINTS,
        home_q=FIVEBAR_HOME_Q,
        plant=_plant,
        board_config=fivebar_board_config,
        feedforward=_feedforward,
        mass_matrix=fivebar_mass_matrix,
        tool_position=tool_position,
        targets=_targets,
        real_station="bam_fb",
    )
)
