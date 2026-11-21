"""The 1-DOF pendulum (the One Dof Test Rack's CAN 11 "Pendulum Arm") as a twin: one MuJoCo hinge behind the
moteus board model, arm_motion_bench's identified feed-forward and its fixed targets (CONTRACT amendment 5,
§5.8).

    from robot_controllers.bench.pendulum import PendulumPlant, pendulum_feedforward, pendulum_rungs
    ff = pendulum_feedforward("rigid_body_coulomb")   # (1,) -> (1,), (N, 1) -> (N, 1): FF_MODEL

Zach, 2026-11-20: "and then also the 1 dof pendulum sim and real".

- **Frame.** q is the BOARD frame (rad), what moteus reports for CAN 11 through its 10:1, as
  arm_motion_bench's pendulum station, its twin and its tapes use it. Home is the gravity bottom, board +0.263
  (robots/pendulum.py:47).
- **Model.** `models/pendulum_can11_bare.xml`: bam_rack_twin's `bare_hinge_xml` with an identified gravity
  (one body on one hinge; half the axis inertia in a point mass that gives G·sin(q − bottom), half in the
  armature), filled with the twin of record can11-r18a's terms (arm_motion_bench
  baselines/pendulum_sim.py:84-89): G 4.42266 N·m, bottom 0.257822 rad, axis inertia 0.12011 kg·m², Coulomb
  1.12 N·m (frictionloss, it sticks), viscous 0.0873596 N·m·s/rad. The plant applies 1.04 × the board's torque
  (the twin's torque scale: the motor delivers more than moteus thinks). Not modelled: LuGre pre-sliding,
  Stribeck, the rotor ripple (provenance in `pendulum_can11_bare.json`).
- **Board.** CAN 11's flashed position loop, kp 400 N·m/rev and kd 8 N·m·s/rev (pendulum_sim.py:245-246, the
  deployment record), in the bench's N·m/rad (÷ 2π); cap 12.6 N·m (robots/pendulum.py:92); encoder 2^14 counts
  at the rotor through the 10:1 = 163840 per output rev. WHY not the ×2 the rig's runner sends on every
  command (kp_scale_cmd 2, pendulum_sim.py PD_FIELDS): `board_gains` means the board's configured gains; a
  layout that scales them says so (`gain_scale`, §5.3). The integrator (ki 16000, ilimit 0.5) is not modelled.
- **Feed-forward** (`pendulum_feedforward(name, model_error)`, pure, closed form): arm_motion_bench's frozen
  `pendulum_ff.FF_MODEL` (pendulum_ff.py:548), identified on the real rig, in board torque:
  `rigid_body_coulomb` = 4.113478·sin(q − 0.252969) + 0.10572·qdd + 0.612227·tanh(qd / 0.05) + 0.081449·qd;
  `rigid_body` without the Coulomb term; `gravity` the G term alone. So "the model exactly" (ModelError()) is
  the rig's real model, 3-12 % off the twin's truth (G 4.25 and J 0.115 in board torque), as the real arm's
  is.
- **ModelError**: the one link's G and J × its link factor (the identified J is the axis total; the rotor's
  ~2 % rides along: nothing on the rig separates it), Coulomb and viscous × the friction factor.
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

ARM_NAME = "pendulum"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "pendulum_can11_bare.xml"
PENDULUM_JOINTS: tuple[str, ...] = ("can11",)
BOTTOM_RAD = (
    0.263  # board frame: arm_motion_bench robots/pendulum.py:47 (the probe fits +0.261 .. +0.263), the home
)
TIP_LENGTH_M = 0.339  # robots/pendulum.py:44 (CAD max radius): the tool
PENDULUM_HOME_Q: tuple[float, ...] = (BOTTOM_RAD,)
MOTEUS_RAD_PER_REV = 2.0 * math.pi
FEEDFORWARD_NAMES: tuple[str, ...] = ("gravity", "rigid_body", "rigid_body_coulomb")


@dataclass(frozen=True)
class PendulumTruth:
    """The twin's hidden truth (can11-r18a via pendulum_sim.py:84-89; the MJCF carries the mechanical terms)
    and the board as flashed."""

    gravity_nm: float = 4.42266
    gravity_bottom_rad: float = 0.257822
    inertia_kg_m2: float = 0.12011
    coulomb_nm: float = 1.12
    viscous_nm_s_rad: float = 0.0873596
    torque_scale: float = 1.04  # plant N·m per board N·m (pendulum_sim.py:86)
    torque_cap_nm: float = 12.6
    kp_moteus: float = 400.0  # N·m per output rev (pendulum_sim.py:245)
    kd_moteus: float = 8.0  # N·m·s per output rev (pendulum_sim.py:246)
    encoder_counts_per_rev: int = 2**14 * 10  # a 14-bit rotor encoder through the 10:1
    physics_dt_s: float = 0.00025  # 4 kHz, the boards' rate (the twin steps at 5 kHz)


PENDULUM_TRUTH = PendulumTruth()


@dataclass(frozen=True)
class PendulumModel:
    """The identified feed-forward, board torque: arm_motion_bench baselines/pendulum_ff.py:548 FF_MODEL,
    frozen there from the 2026-11-19 real sessions (`identify`), copied value for value."""

    gravity_nm: float = 4.113478
    gravity_bottom_rad: float = 0.252969
    inertia_kg_m2: float = 0.10572
    coulomb_nm: float = 0.612227
    band_rad_s: float = 0.05
    viscous_nm_per_rad_s: float = 0.081449

    def scaled(self, model_error: ModelError | None) -> PendulumModel:
        """This model `model_error` away: G, J × the link factor; Coulomb, viscous × the friction factor."""
        if model_error is None or model_error.is_nominal:
            return self
        link = float(model_error.link_factors(1)[0])
        friction = float(model_error.friction_factor)
        return PendulumModel(
            gravity_nm=self.gravity_nm * link,
            gravity_bottom_rad=self.gravity_bottom_rad,
            inertia_kg_m2=self.inertia_kg_m2 * link,
            coulomb_nm=self.coulomb_nm * friction,
            band_rad_s=self.band_rad_s,
            viscous_nm_per_rad_s=self.viscous_nm_per_rad_s * friction,
        )


FF_MODEL = PendulumModel()


def _nominal(model_error: ModelError | None) -> ModelError | None:
    return None if model_error is None or model_error.is_nominal else model_error


class PendulumFeedforward:
    """`ff(q, qd, qdd) -> tau` (board N·m): (1,) -> (1,), (N, 1) -> (N, 1). Pure, closed form."""

    __slots__ = ("name", "model_error", "_model")

    def __init__(self, name: str, model_error: ModelError | None = None) -> None:
        if name not in FEEDFORWARD_NAMES:
            raise ValueError(f"unknown pendulum feed-forward {name!r}; known: {FEEDFORWARD_NAMES}")
        self.name = name
        self.model_error = _nominal(model_error)
        self._model = FF_MODEL.scaled(self.model_error)

    def __call__(self, q: Any, qd: Any, qdd: Any) -> np.ndarray:
        model = self._model
        q = np.asarray(q, dtype=np.float64)
        tau = model.gravity_nm * np.sin(q - model.gravity_bottom_rad)
        if self.name == "gravity":
            return tau
        qd = np.asarray(qd, dtype=np.float64)
        tau = tau + model.inertia_kg_m2 * np.asarray(qdd, dtype=np.float64) + model.viscous_nm_per_rad_s * qd
        if self.name == "rigid_body_coulomb":
            tau = tau + model.coulomb_nm * np.tanh(qd / model.band_rad_s)
        return tau

    def __getstate__(self) -> dict[str, Any]:
        return {"name": self.name, "model_error": self.model_error}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["name"], state["model_error"])

    def __repr__(self) -> str:
        error = "" if self.model_error is None else f", {self.model_error!r}"
        return f"pendulum_feedforward({self.name!r}{error})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PendulumFeedforward) and (other.name, other.model_error) == (
            self.name,
            self.model_error,
        )

    def __hash__(self) -> int:
        return hash(("pendulum_feedforward", self.name, self.model_error))


class PendulumMassMatrix:
    """`mass(q) -> [[J]]`: the identified axis inertia (1,) -> (1, 1), (N, 1) -> (N, 1, 1). Pure."""

    __slots__ = ("model_error", "_inertia")

    def __init__(self, model_error: ModelError | None = None) -> None:
        self.model_error = _nominal(model_error)
        self._inertia = FF_MODEL.scaled(self.model_error).inertia_kg_m2

    def __call__(self, q: Any) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if q.ndim == 1:
            return np.full((1, 1), self._inertia)
        return np.full((q.shape[0], 1, 1), self._inertia)

    def __getstate__(self) -> dict[str, Any]:
        return {"model_error": self.model_error}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__init__(state["model_error"])

    def __repr__(self) -> str:
        return (
            "pendulum_mass_matrix()"
            if self.model_error is None
            else f"pendulum_mass_matrix({self.model_error!r})"
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PendulumMassMatrix) and other.model_error == self.model_error

    def __hash__(self) -> int:
        return hash(("pendulum_mass_matrix", self.model_error))


def pendulum_feedforward(terms: str, model_error: ModelError | None = None) -> PendulumFeedforward:
    """The pendulum feed-forward named `terms` (`gravity`, `rigid_body`, `rigid_body_coulomb`): FF_MODEL
    `model_error` away; None and `ModelError()` are FF_MODEL exactly."""
    return PendulumFeedforward(terms, model_error)


def pendulum_mass_matrix(model_error: ModelError | None = None) -> PendulumMassMatrix:
    return PendulumMassMatrix(model_error)


def tool_position(q: Any) -> np.ndarray:
    """The tip, 0.339 m out, in the pivot frame (z up, x where the tip goes as q rises from the bottom): (L
    sin θ, 0, −L cos θ), θ = q − 0.263 (robots/pendulum.py's station kinematics). (1,) -> (3,), (N, 1) -> (N,
    3)."""
    q = np.asarray(q, dtype=np.float64)
    theta = np.atleast_2d(q)[:, 0] - BOTTOM_RAD
    out = np.stack(
        [TIP_LENGTH_M * np.sin(theta), np.zeros_like(theta), -TIP_LENGTH_M * np.cos(theta)], axis=1
    )
    return out[0] if q.ndim == 1 else out


# ── the model and the board ────────────────────────────────────────────────────────────────────────────────


def load_model() -> Any:
    import mujoco

    return mujoco.MjModel.from_xml_path(str(MODEL_PATH))


@functools.lru_cache(maxsize=1)
def _plant_model() -> Any:
    return load_model()


@functools.lru_cache(maxsize=1)
def pendulum_board_config() -> BoardConfig:
    """CAN 11's board: kp 400 N·m/rev and kd 8 N·m·s/rev as N·m/rad, the 12.6 N·m cap, 163840 counts per
    output rev, the bench's encoder delay and velocity filter, 4 kHz."""
    truth = PENDULUM_TRUTH
    return BoardConfig(
        kp_nm_rad=(truth.kp_moteus / MOTEUS_RAD_PER_REV,),
        kd_nm_s_rad=(truth.kd_moteus / MOTEUS_RAD_PER_REV,),
        torque_cap_nm=(truth.torque_cap_nm,),
        rate_hz=1.0 / truth.physics_dt_s,
        encoder_counts_per_rev=truth.encoder_counts_per_rev,
    )


class PendulumPlant(ServoPlant):
    """The `pendulum` plant: the bare hinge in MuJoCo behind CAN 11's board model (the interface of
    `PlanarArmPlant`; no belt, no pushes, no payload). The hinge feels the board's torque × 1.04."""

    name = ARM_NAME
    joint_names: tuple[str, ...] = PENDULUM_JOINTS

    def __init__(
        self, *, board_config: BoardConfig | None = None, truth: PendulumTruth = PENDULUM_TRUTH
    ) -> None:
        self.truth = truth
        super().__init__(board_config or pendulum_board_config(), truth.physics_dt_s)
        self._gravity = PendulumFeedforward("gravity")

    @property
    def home_q(self) -> tuple[float, ...]:
        return PENDULUM_HOME_Q

    def tool_position(self, q: Any) -> np.ndarray:
        return tool_position(q)

    def reset(
        self,
        scene: Scene,
        seed: int = 0,
        *,
        timing: HostTiming | None = None,
        duration_s: float = 5.0,
    ) -> None:
        """A new episode, as `PlanarArmPlant.reset`: the hinge at rest at `scene.start_q` (or the bottom), the
        board holding it with the identified gravity feed-forward through a 20 ms unrecorded pre-roll."""
        import mujoco

        if scene.belt or scene.pushes:
            raise ValueError("the pendulum plant has no belt and takes no pushes")
        if scene.payload_kg:
            raise ValueError("the pendulum plant carries no payload (its arm is the rig's own)")
        self._mujoco = mujoco
        self.scene = scene
        self.seed = int(seed)
        self.timing = timing or HostTiming()
        self._reset_bus(seed, self.timing)
        self.contact_calibration = None
        self._model = model = _plant_model()
        self._data = data = mujoco.MjData(model)
        start_q = np.asarray(scene.start_q if scene.start_q is not None else self.home_q, dtype=float)
        if start_q.shape != (1,):
            raise ValueError(f"start_q must be 1 joint angle, got {start_q}")
        self.start_q = tuple(float(v) for v in start_q)
        data.qpos[0] = start_q[0]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        self._warnings_seen = self._warning_count()
        self.events: list[dict[str, Any]] = []
        preroll_ticks = int(round(PREROLL_S / self._dt))
        self._first_k = -preroll_ticks
        self._k = self._first_k
        hold = self._gravity(start_q, np.zeros(1), np.zeros(1))
        self.boards.reset(self._first_k * self._dt, start_q, feedforward=hold)
        self._allocate(preroll_ticks + int(math.ceil(duration_s / self._dt)) + 8)
        self._last_sample_k: int | None = None
        self._seq = 0
        self._writes = 0
        self._now = self._first_k * self._dt
        self.advance(0.0)

    def advance(self, until: float) -> None:
        """Run physics and the board over every tick with k * dt < `until` (s)."""
        mujoco, model, data, boards = self._mujoco, self._model, self._data, self.boards
        dt, scale = self._dt, float(self.truth.torque_scale)
        qpos, qvel, qfrc = data.qpos, data.qvel, data.qfrc_applied
        first_k = self._first_k
        k = self._k
        while k * dt < until - TIME_TOLERANCE_S:
            t = k * dt
            if k - first_k >= self._capacity:
                self._grow()
            q_true = float(qpos[0])
            qd_true = float(qvel[0])
            tau = boards.step(t, (q_true,))
            qfrc[0] = scale * tau[0]
            mujoco.mj_step(model, data)
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
                theta = q_true - BOTTOM_RAD
                self._tool_pos[k] = (TIP_LENGTH_M * math.sin(theta), 0.0, -TIP_LENGTH_M * math.cos(theta))
                self._tau_ff[k] = boards.tau_ff
                self._tau_fb_board[k] = boards.tau_fb
                self._latched_tick[k] = boards.latched_tag
            k += 1
        self._k = k
        self._now = max(self._now, float(until))
        self._check_divergence()

    def meta(self) -> dict[str, Any]:
        return {
            "plant": self.name,
            "physics_dt_s": self._dt,
            "preroll_s": -self._first_k * self._dt,
            "start_q": list(self.start_q),
            "board": asdict(self.board_config),
            "truth": asdict(self.truth),
            "model": MODEL_PATH.name,
            "tool": f"the tip, {TIP_LENGTH_M} m out",
        }


# ── the targets ────────────────────────────────────────────────────────────────────────────────────────────

# WHY dev-007 at 0.5 nominal: the real showcase (arm_motion_bench scripts/pendulum_showcase.py SEGMENTS)
# played the v2 dev tour from the bottom and ran tasks 005-007 at 0.5, 007 being its 138 deg (63 cm) swing, so
# a real log of it compares one to one; fast (1.0) reaches 4.95 rad/s, the joint's 5 rad/s.
SPEEDS: dict[str, float] = {"slow": 0.25, "nominal": 0.5, "fast": 1.0}
TAPES: dict[float, str] = {
    1.0: "pendulum/workspace-dev-007.npz",
    0.5: "pendulum/workspace-dev-007__s0.5.npz",
    0.25: "pendulum/workspace-dev-007__s0.25.npz",
}
CALIBRATION_AMPLITUDE_RAD = math.radians(8.0)
CALIBRATION_DURATION_S = 4.0
CALIBRATION_FREQUENCIES_HZ = (1.0,)
PAD_S = 0.6


@functools.lru_cache(maxsize=8)
def path_trajectory(speed: float) -> Any:
    """dev-007 at speed fraction `speed` on the 4 ms grid, effort = the nominal `rigid_body` model (FF_MODEL,
    no Coulomb)."""
    from robot_controllers.bench.arm_tapes import tape_trajectory

    return tape_trajectory(TAPES[float(speed)], PendulumFeedforward("rigid_body"))


@functools.lru_cache(maxsize=1)
def calibration_trajectory() -> Any:
    from robot_msgs.trajectory import JointTrajectory

    from robot_controllers.bench.arm_tapes import sine_sweep

    t, q, qd, qdd = sine_sweep(
        PENDULUM_HOME_Q,
        amplitude_rad=CALIBRATION_AMPLITUDE_RAD,
        duration_s=CALIBRATION_DURATION_S,
        frequencies_hz=CALIBRATION_FREQUENCIES_HZ,
    )
    return JointTrajectory(
        t=t,
        joint_names=PENDULUM_JOINTS,
        positions=q,
        velocities=qd,
        accelerations=qdd,
        effort=PendulumFeedforward("rigid_body")(q, qd, qdd),
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
    t = np.asarray(trajectory.t)
    tool = tool_position(np.asarray(trajectory.positions))
    return {
        "peak_joint_speed_deg_s": math.degrees(float(np.max(np.abs(np.asarray(trajectory.velocities))))),
        "peak_tool_speed_m_s": float(np.max(np.linalg.norm(np.gradient(tool, t, axis=0), axis=1))),
    }


def path_rung(speed: str) -> Rung:
    """`pendulum/path@<speed>`: arm_motion_bench's pendulum workspace v2 dev-007 (a 138 deg swing) at slow
    0.25, nominal 0.5 or fast 1.0, from its own start, host 100 Hz, R2's gates."""
    from robot_controllers.bench.arm_tapes import TrajectoryScript, tape_record

    fraction = SPEEDS[speed]
    trajectory = path_trajectory(fraction)
    record = tape_record(TAPES[fraction])
    return _rung(
        f"{ARM_NAME}/path@{speed}",
        f"Pendulum dev-007 swing at {fraction:g}",
        "arm_motion_bench's pendulum workspace v2 dev-007 tape (a 138 deg swing, bam_ruckig) at speed "
        f"{fraction:g}, resampled onto the 4 ms grid. Tests tracking through a gravity load that changes "
        "sign, on the tape the rig runs.",
        TrajectoryScript(functools.partial(path_trajectory, fraction), float(trajectory.t[-1])),
        (float(trajectory.positions[0][0]),),
        {
            "speed_fraction": fraction,
            "tape": record["file"],
            "tape_sha256": record["sha256"],
            **_speed_params(trajectory),
        },
    )


def calibration_rung() -> Rung:
    from robot_controllers.bench.arm_tapes import TrajectoryScript

    trajectory = calibration_trajectory()
    return _rung(
        f"{ARM_NAME}/calibration",
        "Pendulum calibration sweep",
        "The joint sweeps +-8 deg about the bottom (a sin^2-windowed 1 Hz sine), at rest at both ends. "
        "The run a bus calibration reads the joint's delay from.",
        TrajectoryScript(calibration_trajectory, CALIBRATION_DURATION_S),
        PENDULUM_HOME_Q,
        {"amplitude_deg": math.degrees(CALIBRATION_AMPLITUDE_RAD), **_speed_params(trajectory)},
    )


def pendulum_rungs() -> dict[str, Rung]:
    """The pendulum's path rungs at three speeds and its calibration sweep, by id (`plant = "pendulum"`)."""
    rungs = [path_rung(speed) for speed in SPEEDS] + [calibration_rung()]
    return {r.id: r for r in rungs}


# ── the registry ───────────────────────────────────────────────────────────────────────────────────────────


def _plant(board_config: BoardConfig | None = None) -> PendulumPlant:
    return PendulumPlant(board_config=board_config)


def _feedforward(name: str, model_error: ModelError | None = None) -> PendulumFeedforward:
    return PendulumFeedforward(name, model_error)


def _targets() -> tuple[Any, ...]:
    raise NotImplementedError(
        "the pendulum's Target records land with lane A's bench.targets (CONTRACT §5.4)"
    )


SPEC = register(
    ArmSpec(
        name=ARM_NAME,
        title="1-DOF pendulum (CAN 11)",
        joint_names=PENDULUM_JOINTS,
        home_q=PENDULUM_HOME_Q,
        plant=_plant,
        board_config=pendulum_board_config,
        feedforward=_feedforward,
        mass_matrix=pendulum_mass_matrix,
        tool_position=tool_position,
        targets=_targets,
        real_station="pendulum",
    )
)
