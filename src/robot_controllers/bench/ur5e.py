"""The UR5e plant: MuJoCo Menagerie's UR5e behind the same moteus board model, its models and its two rungs
(CONTRACT amendment 4, §4.5).

    from robot_controllers.bench.ur5e import Ur5ePlant, ur5e_rungs, ur5e_feedforward, ur5e_mass_matrix
    rungs = {r.id: r for r in ur5e_rungs()}             # U1.FreeSpace, U2.FastFreeSpace (not in ladder())
    log = run_episode(layout_spec(..., arm="ur5e"), rungs["U2.FastFreeSpace"])   # rung.plant == "ur5e"

Zach, 2026-11-20: "Can you please run examples on the UR robot arm. and also try trajectories at different
speeds." The bench's planar arm is a toy; the UR5e is a real six-joint arm with real inertials, so a
conclusion that holds on both is not an artefact of two links.

- **Model.** `models/ur5e_bare.xml`, vendored byte for byte from bam_ws `bam_pid_tunning/tests/fixtures`: the
  Menagerie UR5e's kinematics and inertials (BSD-3, ROS-Industrial; `models/LICENSE`), no meshes, actuators
  or contacts. Its integrator (implicitfast) and 4 kHz step are kept as vendored.
- **Truth** (bam_pid_tunning `arm_truths.ur5e_arm`): armature (0.5 x3, 0.08 x3) kg·m², frictionloss (4.0 x3,
  0.8 x3) N·m, damping (2.0 x3, 0.3 x3) N·m·s/rad, set on the loaded model.
- **Home and tool.** Home (0, -1.2, 1.4, -1.6, -1.57, 0) rad; the tool is the flange site `tool`, 0.1 m out
  along wrist_3 (Menagerie's attachment site). `Scene.payload_kg` adds a point mass there; a belt or a push
  is refused.
- **Boards.** The moteus model with the UR5e datasheet's caps (150 x3, 28 x3) N·m and gains at the twin
  shoulder's bandwidth: kp_j = M_jj(home)·ωn², kd_j = 2ζωn·M_jj(home), ωn and ζ from the planar shoulder's
  configured kp, kd and M11(home) (39.7 rad/s and 0.496), unrounded.
- **Feed-forward** (`ur5e_feedforward(name)`, pure): `gravity`, `rigid_body` (+ inertia, Coriolis, viscous:
  the nominal model the U rungs fill effort with), `rigid_body_coulomb` (+ frictionloss·tanh(qd / 0.01)); each
  is `mj_inverse` on its own private copy of the nominal model (no payload, no dry friction).
- **Feedback.** `ur5e_mass_matrix()` is `mj_fullM`; `ctc` (layouts) is InertiaScaledPD ζ 0.9 at the bandwidth
  whose shoulder_lift kp at home equals the board's (`ur5e_ctc_bandwidth_hz`).
- **Model error** (amendment 5, §5.3): `ur5e_feedforward(name, model_error)` and `ur5e_mass_matrix(model_error)`
  scale, on their private copy, the mass and inertia of every body that carries a joint (model order) by
  `model_error.link_factors(6)` and `dof_damping` and the dry friction by `friction_factor`; the armature stays.
- **Registry** (§5.1): the module registers `UR5E_ARM` with `bench.arms` at import; its fixed targets are
  `bench.targets.ur5e_targets()`.
"""

from __future__ import annotations

import dataclasses
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

MODEL_PATH = Path(__file__).resolve().parent / "models" / "ur5e_bare.xml"
UR5E_JOINTS: tuple[str, ...] = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
UR5E_HOME_Q: tuple[float, ...] = (0.0, -1.2, 1.4, -1.6, -1.57, 0.0)
TOOL_SITE = "tool"
SAMPLE_PERIOD_S = 0.004  # every trajectory is sampled every 4 ms, as the planar rungs
COULOMB_SMOOTHING_RAD_S = 0.01  # rigid_body_coulomb = ... + frictionloss · tanh(qd / 0.01 rad/s)
FEEDFORWARD_NAMES: tuple[str, ...] = ("gravity", "rigid_body", "rigid_body_coulomb")
# The U rungs' joint path (§4.5): W0 = home -> W1 -> W2 -> W0, a 0.3 s hold first and a 0.3 s rest after each
UR5E_WAYPOINTS: tuple[tuple[float, ...], ...] = (
    UR5E_HOME_Q,
    (0.9, -0.8, 1.0, -1.8, -1.57, 0.6),
    (-0.6, -1.5, 1.8, -1.9, -1.2, -0.5),
    UR5E_HOME_Q,
)
HOLD_S = 0.3
REST_S = 0.3


@dataclass(frozen=True)
class Ur5eTruth:
    """The simulated UR5e's joint truths (bam_pid_tunning arm_truths.ur5e_arm) and its boards' torque caps."""

    armature_kg_m2: tuple[float, ...] = (0.5, 0.5, 0.5, 0.08, 0.08, 0.08)
    frictionloss_nm: tuple[float, ...] = (4.0, 4.0, 4.0, 0.8, 0.8, 0.8)
    damping_nm_s_rad: tuple[float, ...] = (2.0, 2.0, 2.0, 0.3, 0.3, 0.3)
    torque_cap_nm: tuple[float, ...] = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)  # UR5e datasheet
    physics_dt_s: float = 0.00025  # 4 kHz, the vendored model's timestep and the board rate


UR5E_TRUTH = Ur5eTruth()


# ── the MuJoCo model ───────────────────────────────────────────────────────────────────────────────────────


def _xml(payload_kg: float) -> str:
    text = MODEL_PATH.read_text()
    if payload_kg == 0.0:
        return text
    # WHY a child body at the tool site: a point mass where the payload is held, leaving the vendored link's
    # own inertials untouched (the diaginertia only keeps MuJoCo's body-inertia check satisfied)
    anchor = '<site name="tool" pos="0 0.1 0"'
    if text.count(anchor) != 1:
        raise RuntimeError(f"{MODEL_PATH.name}: the tool site is not where the payload goes")
    payload = (
        f'<body name="payload" pos="0 0.1 0"><inertial mass="{payload_kg:.12g}" pos="0 0 0" '
        'diaginertia="1e-9 1e-9 1e-9"/></body>\n                  '
    )
    return text.replace(anchor, payload + anchor)


def load_model(*, payload_kg: float = 0.0, frictionloss: bool = True, truth: Ur5eTruth = UR5E_TRUTH) -> Any:
    """A fresh MjModel of the vendored UR5e with the truth's armature, damping and (optionally)
    frictionloss."""
    import mujoco

    payload = float(payload_kg)
    if payload == 0.0:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    else:
        model = mujoco.MjModel.from_xml_string(_xml(payload))
    model.dof_armature[:] = truth.armature_kg_m2
    model.dof_damping[:] = truth.damping_nm_s_rad
    model.dof_frictionloss[:] = truth.frictionloss_nm if frictionloss else 0.0
    return model


@functools.lru_cache(maxsize=8)
def _plant_model(payload_kg: float, truth: Ur5eTruth) -> Any:
    # models are immutable once built; MjData is per episode
    return load_model(payload_kg=payload_kg, frictionloss=True, truth=truth)


def _told(model_error: ModelError | None) -> ModelError | None:
    """The model error a model carries: None for the nominal model (so `ModelError()` is exactly today's)."""
    if model_error is None:
        return None
    if not isinstance(model_error, ModelError):
        raise TypeError(f"model_error must be a ModelError, got {model_error!r}")
    return None if model_error.is_nominal else model_error


def jointed_bodies(model: Any) -> list[int]:
    """The bodies that carry a joint, in model order: the links a `ModelError` scales (§5.3)."""
    return [body for body in range(model.nbody) if model.body_jntnum[body] > 0]


class _Private:
    """A private nominal model and its data: what makes each UR5e model function pure from the outside."""

    __slots__ = ("model", "data")

    def __init__(self, model_error: ModelError | None = None) -> None:
        import mujoco

        # WHY no frictionloss: dry friction is a constraint, and mj_inverse would put its (unknowable)
        # force into the feed-forward; the nominal model is rigid body + armature + viscous damping, no
        # payload (as the planar arm's nominal_dynamics)
        self.model = load_model(frictionloss=False)
        if model_error is not None:
            # WHY the armature stays: it is the drive's reflected inertia, not the link's (§5.3)
            bodies = jointed_bodies(self.model)
            for body, factor in zip(bodies, model_error.link_factors(len(bodies)), strict=True):
                self.model.body_mass[body] *= factor
                self.model.body_inertia[body] *= factor
            self.model.dof_damping[:] *= model_error.friction_factor
        self.data = mujoco.MjData(self.model)


def _rows(q: Any, qd: Any, qdd: Any) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    single = np.ndim(q) == 1
    q, qd, qdd = (np.atleast_2d(np.asarray(v, dtype=np.float64)) for v in (q, qd, qdd))
    return single, q, qd, qdd


class Ur5eFeedforward:
    """`ff(q, qd, qdd) -> tau` on the UR5e: (6,) -> (6,), (N, 6) -> (N, 6). Pure; `mj_inverse` on a private
    copy of the nominal model (`gravity` with qd and qdd zero), with `model_error`'s links and friction."""

    __slots__ = ("name", "model_error", "_private", "_friction")

    def __init__(self, name: str, model_error: ModelError | None = None) -> None:
        if name not in FEEDFORWARD_NAMES:
            raise ValueError(f"unknown UR5e feed-forward {name!r}; known: {FEEDFORWARD_NAMES}")
        self.name = name
        self.model_error = _told(model_error)
        self._private: _Private | None = None
        friction = np.asarray(UR5E_TRUTH.frictionloss_nm, dtype=np.float64)
        self._friction = friction if self.model_error is None else friction * self.model_error.friction_factor

    def __call__(self, q: Any, qd: Any, qdd: Any) -> np.ndarray:
        import mujoco

        if self._private is None:
            self._private = _Private(self.model_error)
        model, data = self._private.model, self._private.data
        single, q, qd, qdd = _rows(q, qd, qdd)
        out = np.empty((q.shape[0], model.nv))
        gravity_only = self.name == "gravity"
        for row in range(q.shape[0]):
            data.qpos[:] = q[row]
            if gravity_only:
                data.qvel[:] = 0.0
                data.qacc[:] = 0.0
            else:
                data.qvel[:] = qd[row]
                data.qacc[:] = qdd[row]
            mujoco.mj_inverse(model, data)
            out[row] = data.qfrc_inverse
        if self.name == "rigid_body_coulomb":
            out = out + self._friction * np.tanh(qd / COULOMB_SMOOTHING_RAD_S)
        return out[0] if single else out

    def __repr__(self) -> str:
        error = "" if self.model_error is None else f", model_error={self.model_error!r}"
        return f"ur5e_feedforward({self.name!r}{error})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ur5eFeedforward) and (other.name, other.model_error) == (
            self.name,
            self.model_error,
        )

    def __hash__(self) -> int:
        return hash(("ur5e_feedforward", self.name, self.model_error))


class Ur5eMassMatrix:
    """`mass(q) -> M(q)`: the UR5e's joint-space inertia, armature included (`mj_fullM`), (6,) -> (6, 6),
    (N, 6) -> (N, 6, 6). Pure; with `model_error`'s links."""

    __slots__ = ("model_error", "_private")

    def __init__(self, model_error: ModelError | None = None) -> None:
        self.model_error = _told(model_error)
        self._private: _Private | None = None

    def __call__(self, q: Any) -> np.ndarray:
        import mujoco

        if self._private is None:
            self._private = _Private(self.model_error)
        model, data = self._private.model, self._private.data
        q = np.asarray(q, dtype=np.float64)
        rows = np.atleast_2d(q)
        out = np.zeros((rows.shape[0], model.nv, model.nv))
        for row in range(rows.shape[0]):
            data.qpos[:] = rows[row]
            # the position stage only: kinematics, composite inertias, M (what mj_forward computes M with)
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            mujoco.mj_crb(model, data)
            mujoco.mj_fullM(model, data, out[row])
        return out[0] if q.ndim == 1 else out

    def __repr__(self) -> str:
        return "ur5e_mass_matrix()" if self.model_error is None else f"ur5e_mass_matrix({self.model_error!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ur5eMassMatrix) and other.model_error == self.model_error

    def __hash__(self) -> int:
        return hash(("ur5e_mass_matrix", self.model_error))


def ur5e_feedforward(terms: str, model_error: ModelError | None = None) -> Ur5eFeedforward:
    """The UR5e feed-forward named `terms`: `gravity`, `rigid_body` or `rigid_body_coulomb` (§4.5), with
    `model_error`'s links and friction (§5.3; None = the nominal model)."""
    return Ur5eFeedforward(terms, model_error)


def ur5e_mass_matrix(model_error: ModelError | None = None) -> Ur5eMassMatrix:
    """The UR5e's joint-space inertia M(q), armature included, no payload (with `model_error`'s links)."""
    return Ur5eMassMatrix(model_error)


class _Kinematics:
    """Tool-site forward kinematics on a private model."""

    def __init__(self) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = load_model(frictionloss=False)
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site(TOOL_SITE).id

    def __call__(self, q: Any) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        rows = np.atleast_2d(q)
        out = np.empty((rows.shape[0], 3))
        for row in range(rows.shape[0]):
            self.data.qpos[:] = rows[row]
            self._mujoco.mj_kinematics(self.model, self.data)
            out[row] = self.data.site_xpos[self.site]
        return out[0] if q.ndim == 1 else out


@functools.lru_cache(maxsize=1)
def _kinematics() -> _Kinematics:
    return _Kinematics()


def tool_position(q: Any) -> np.ndarray:
    """The tool site's world position (m) for UR5e joint angles: (6,) -> (3,), (N, 6) -> (N, 3)."""
    return _kinematics()(q)


# ── the boards ─────────────────────────────────────────────────────────────────────────────────────────────


def _full_mass(model: Any, q: Any) -> np.ndarray:
    import mujoco

    data = mujoco.MjData(model)
    data.qpos[: len(q)] = q
    mujoco.mj_forward(model, data)
    full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, full)
    return full


def planar_shoulder_bandwidth() -> tuple[float, float]:
    """(ωn rad/s, ζ) of the planar twin's shoulder at home: ωn = √(kp / M11), ζ = kd / (2 √(kp · M11)), from
    its configured board gains and MuJoCo's M11(home) (armature included)."""
    from robot_controllers.bench.arm import ARM, HOME_Q, build_model

    planar = BoardConfig()
    kp, kd = float(planar.kp_nm_rad[0]), float(planar.kd_nm_s_rad[0])
    model, _ = build_model(Scene(), ARM)
    m11 = float(_full_mass(model, HOME_Q)[0, 0])
    return math.sqrt(kp / m11), kd / (2.0 * math.sqrt(kp * m11))


@functools.lru_cache(maxsize=1)
def ur5e_board_config() -> BoardConfig:
    """The UR5e's boards: the datasheet caps, encoder 2^14 and every other field at BoardConfig's default, and
    gains at the planar shoulder's bandwidth: kp_j = M_jj(home)·ωn², kd_j = 2ζωn·M_jj(home), unrounded.

    WHY the planar shoulder's bandwidth: the two arms' boards are then equally stiff relative to their own
    inertia, so a layout that wins on one is not winning on a stiffer loop.
    """
    wn, zeta = planar_shoulder_bandwidth()
    diagonal = np.diag(_full_mass(_plant_model(0.0, UR5E_TRUTH), UR5E_HOME_Q))
    return BoardConfig(
        kp_nm_rad=tuple(float(v) for v in diagonal * wn**2),
        kd_nm_s_rad=tuple(float(v) for v in 2.0 * zeta * wn * diagonal),
        torque_cap_nm=UR5E_TRUTH.torque_cap_nm,
        rate_hz=1.0 / UR5E_TRUTH.physics_dt_s,
        encoder_counts_per_rev=2**14,
    )


def ur5e_ctc_bandwidth_hz() -> float:
    """The CTC bandwidth (Hz) whose shoulder_lift kp at home, M_22(home)·ωn², equals the board's (as §3.6
    on the planar shoulder). With the boards above that is the planar shoulder's ωn / 2π."""
    lift = UR5E_JOINTS.index("shoulder_lift_joint")
    inertia = float(ur5e_mass_matrix()(UR5E_HOME_Q)[lift, lift])
    return math.sqrt(float(ur5e_board_config().kp_nm_rad[lift]) / inertia) / (2.0 * math.pi)


# ── the plant ──────────────────────────────────────────────────────────────────────────────────────────────


class Ur5ePlant(ServoPlant):
    """The `ur5e` plant: the vendored UR5e in MuJoCo behind one moteus board model per joint (the interface of
    `PlanarArmPlant`; no belt, no pushes)."""

    name = "ur5e"
    joint_names: tuple[str, ...] = UR5E_JOINTS

    def __init__(self, *, board_config: BoardConfig | None = None, truth: Ur5eTruth = UR5E_TRUTH) -> None:
        self.truth = truth
        super().__init__(board_config or ur5e_board_config(), truth.physics_dt_s)
        self._gravity = Ur5eFeedforward("gravity")

    @property
    def home_q(self) -> tuple[float, ...]:
        return UR5E_HOME_Q

    def tool_position(self, q: Any) -> np.ndarray:
        """The tool site (x, y, z) for joint angles `q`, (6,) -> (3,), (N, 6) -> (N, 3)."""
        return tool_position(q)

    def reset(
        self,
        scene: Scene,
        seed: int = 0,
        *,
        timing: HostTiming | None = None,
        duration_s: float = 5.0,
    ) -> None:
        """A new episode, as `PlanarArmPlant.reset`: the arm at rest at `scene.start_q` (or home), the boards
        holding it with the nominal gravity feed-forward through a 20 ms unrecorded pre-roll."""
        import mujoco

        if scene.belt or scene.pushes:
            raise ValueError("the UR5e plant has no belt and takes no pushes; use the planar arm's rungs")
        self._mujoco = mujoco
        self.scene = scene
        self.seed = int(seed)
        self.timing = timing or HostTiming()
        self._reset_bus(seed, self.timing)
        self.contact_calibration = None
        self._model = model = _plant_model(float(scene.payload_kg), self.truth)
        self._data = data = mujoco.MjData(model)
        start_q = np.asarray(scene.start_q if scene.start_q is not None else self.home_q, dtype=float)
        if start_q.shape != (len(self.joint_names),):
            raise ValueError(f"start_q must be {len(self.joint_names)} joint angles, got {start_q}")
        self.start_q = tuple(float(v) for v in start_q)
        data.qpos[:] = start_q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        self._tool_site = model.site(TOOL_SITE).id
        self._warnings_seen = self._warning_count()
        self.events: list[dict[str, Any]] = []
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
        dt, site = self._dt, self._tool_site
        qpos, qvel, qfrc, site_xpos = data.qpos, data.qvel, data.qfrc_applied, data.site_xpos
        first_k = self._first_k
        k = self._k
        while k * dt < until - TIME_TOLERANCE_S:
            t = k * dt
            if k - first_k >= self._capacity:
                self._grow()
            q_true = qpos.tolist()
            qd_true = qvel.tolist()
            tau = boards.step(t, q_true)
            qfrc[:] = tau
            mujoco.mj_step(model, data)
            # after mj_step the kinematics are those of the state at t (integration came last)
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
        }


# ── the path and the rungs ─────────────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=32)
def _path_trajectory(
    move_s: float, waypoints: tuple[tuple[float, ...], ...], hold_s: float, rest_s: float
) -> Any:
    """Minimum-jerk joint moves through `waypoints` (a hold first, a rest after each), every 4 ms, the end
    included; velocity and acceleration analytic, effort = the nominal `rigid_body` model."""
    from robot_msgs.trajectory import JointTrajectory

    from robot_controllers.bench.rungs import min_jerk

    points = np.asarray(waypoints, dtype=float)
    total = hold_s + (len(points) - 1) * (move_s + rest_s)
    count = int(round(total / SAMPLE_PERIOD_S))
    t = np.arange(count + 1) * SAMPLE_PERIOD_S
    t[-1] = total
    q = np.tile(points[0], (count + 1, 1))
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    for index in range(len(points) - 1):
        start = hold_s + index * (move_s + rest_s)
        delta = points[index + 1] - points[index]
        after = t >= start - 1e-12
        local = np.clip(t[after] - start, 0.0, move_s)
        s, ds, dds = min_jerk(local / move_s)
        moving = t[after] < start + move_s - 1e-12
        q[after] = points[index] + np.outer(s, delta)
        qd[after] = np.outer(np.where(moving, ds, 0.0) / move_s, delta)
        qdd[after] = np.outer(np.where(moving, dds, 0.0) / move_s**2, delta)
    return JointTrajectory(
        t=t,
        joint_names=UR5E_JOINTS,
        positions=q,
        velocities=qd,
        accelerations=qdd,
        effort=Ur5eFeedforward("rigid_body")(q, qd, qdd),
    )


@dataclass(frozen=True)
class Ur5ePathScript:
    """One JointTrajectory at t = 0: W0 -> W1 -> W2 -> W0 at `move_s` per move (§4.5). Frozen, so a rung
    pickles into a worker process; `respeed` is what `rungs.speed_variant` re-times it with."""

    move_s: float
    waypoints: tuple[tuple[float, ...], ...] = UR5E_WAYPOINTS
    hold_s: float = HOLD_S
    rest_s: float = REST_S

    def __call__(self, start_q: np.ndarray) -> list[tuple[float, Any]]:
        return [(0.0, _path_trajectory(float(self.move_s), self.waypoints, self.hold_s, self.rest_s))]

    @property
    def duration_s(self) -> float:
        return self.hold_s + (len(self.waypoints) - 1) * (self.move_s + self.rest_s)

    def respeed(self, move_s: float) -> Ur5ePathScript:
        return dataclasses.replace(self, move_s=float(move_s))


def ur5e_rungs() -> tuple[Rung, ...]:
    """`U1.FreeSpace` (1.2 s per move, R1's gates) and `U2.FastFreeSpace` (0.45 s, R2's gates) on the UR5e:
    `plant = "ur5e"`, not in `ladder()`."""
    from robot_controllers.bench.rungs import rung
    from robot_controllers.registry import JOINT_TRAJECTORY

    path_text = " -> ".join("W0 W1 W2 W0".split())
    waypoint_text = "; ".join("(" + ", ".join(f"{v:g}" for v in point) + ")" for point in UR5E_WAYPOINTS[:3])
    rack = HostTiming()
    out = []
    for order, (rung_id, title, base, move_s, what) in enumerate(
        (
            ("U1.FreeSpace", "UR5e free-space motion", "R1.FreeSpace", 1.2,
             "Tests plain tracking on a six-joint arm with the feed-forward the trajectory carries."),
            ("U2.FastFreeSpace", "UR5e fast free-space motion", "R2.FastFreeSpace", 0.45,
             "Tests tracking when inertia dominates, with the pan joint near its torque cap."),
        ),
        start=1,
    ):
        script = Ur5ePathScript(move_s)
        out.append(
            Rung(
                id=rung_id,
                title=title,
                box=JOINT_TRAJECTORY,
                family="free_space",
                description=f"Three minimum-jerk joint moves {path_text} on the UR5e, {move_s:g} s each "
                f"with 0.3 s rests. {what}",
                duration_s=round(script.duration_s + 0.6, 6),
                inputs=script,
                timing=rack,
                gates=rung(base).gates,
                objective="tracking_p95_deg",
                params={"move_s": move_s, "rest_s": REST_S, "waypoints_rad": waypoint_text},
                order=order,
                plant="ur5e",
            )
        )
    return tuple(out)


# ── the registry (amendment 5, §5.1) ───────────────────────────────────────────────────────────────────────


def ur5e_plant(board_config: BoardConfig | None = None) -> Ur5ePlant:
    """A fresh UR5e plant (its own boards unless `board_config` says otherwise)."""
    return Ur5ePlant(board_config=board_config)


def ur5e_targets() -> tuple[Any, ...]:
    """The UR5e's fixed targets (§5.4): U2's path at three speeds, a calibration sweep and a collision."""
    from robot_controllers.bench.targets import ur5e_targets as targets

    return targets()


UR5E_ARM = register(
    ArmSpec(
        name="ur5e",
        title="UR5e (6 joints)",
        joint_names=UR5E_JOINTS,
        home_q=UR5E_HOME_Q,
        plant=ur5e_plant,
        board_config=ur5e_board_config,
        feedforward=ur5e_feedforward,
        mass_matrix=ur5e_mass_matrix,
        tool_position=tool_position,
        targets=ur5e_targets,
    )
)
