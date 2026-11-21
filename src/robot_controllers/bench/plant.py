"""PlanarArmPlant: the planar arm in MuJoCo behind its servo boards, as a host sees it over a bus (CONTRACT
§3).

    plant = PlanarArmPlant()
    plant.reset(rung.scene, seed, timing=rung.timing)
    plant.advance(wake_t)                    # physics + boards up to the host's wake
    state = plant.read()                     # ServoState: the newest board sample that has arrived
    plant.write(command)                     # latches on the boards at command.header.stamp + command delay

Every physics step (4 kHz) the boards tick on the joints' true positions (their encoders delay and quantise
them), their torques drive MuJoCo, the scene's pushes and the belt's friction act on the tool, and one plant
row is recorded: t, q, qd, tau, at_cap, board_mode, tool_pos, contact_force, in_contact, external_force
(EpisodeLog's plant arrays). Before t = 0 the plant runs a short unrecorded pre-roll with the boards holding
the start pose (gravity feed-forward), so the first host read has a board sample from before it and the
encoders are settled.

Time is simulated seconds since the episode start; the board samples at k * dt, the host gets a sample
`reply_delay_s` after it was taken, and a command latches `command_delay_s` after it was sent (HostTiming).

The bus (amendment 4, §4.2) is `ServoPlant`'s, shared with every plant behind the same boards (the UR5e's
too): each joint's part of a command latches at sent + its own delay, plus one seeded uniform draw per
command (in whole board ticks) when the timing has `command_delay_jitter_s`, first in first out per joint (a
command that would latch before the one sent before it latches on the same tick). With the defaults it is the
bus as before.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import numpy as np
from robot_msgs.control import ServoCommand, ServoState
from robot_msgs.std import Header

from robot_controllers.bench.arm import (
    ARM,
    HOME_Q,
    JOINT_NAMES,
    ArmConfig,
    build_model,
    nominal_dynamics,
    tool_position,
)
from robot_controllers.bench.board import BoardConfig, ServoBoards
from robot_controllers.bench.types import HostTiming, Scene

PREROLL_S = 0.02  # unrecorded hold before t = 0 (s): longer than the reply delay and the encoder's settling
TIME_TOLERANCE_S = 1e-9
# WHY a fixed second entropy word: the bus jitter draws come from the episode seed but must never share a
# stream with the host loop's wake jitter (np.random.default_rng(seed)), so adding bus jitter shifts no wake.
BUS_JITTER_STREAM = 0xB05_2026
# WHY the planar elbow's S2 stop (arm.ArmConfig.hard_stop_solref): a 2 ms time constant is a stiff stop MuJoCo still
# integrates at 4 kHz, and the collision targets are then the same kind of wall the S2 rung seeks
STOP_SOLREF: tuple[float, float] = (0.002, 1.0)
STOP_FAR_RAD = 1.0e3  # the open side of a one-sided stop on a joint that had no range


class ServoPlant:
    """What every plant behind the moteus board model shares: the bus (`read`, `write`), the per-step record
    (`samples`) and MuJoCo's divergence warnings. A subclass builds the model, steps it (`advance`) and knows
    its own kinematics (`tool_position`)."""

    name = "servo_plant"
    joint_names: tuple[str, ...] = ()

    def __init__(self, board_config: BoardConfig, physics_dt_s: float) -> None:
        self.board_config = board_config
        if abs(self.board_config.dt_s - physics_dt_s) > 1e-12:
            raise ValueError(
                "the boards tick once per physics step: BoardConfig.rate_hz must be 1 / physics_dt_s"
            )
        self.boards = ServoBoards(self.joint_names, self.board_config)
        self._dt = physics_dt_s
        self._model = None
        self._data = None

    @property
    def now(self) -> float:
        """The time the plant has been advanced to (s)."""
        return self._now

    def _reset_bus(self, seed: int, timing: HostTiming) -> None:
        self._delays = timing.command_delays(len(self.joint_names))
        self._delay_steps = timing.delay_steps(len(self.joint_names))
        self._delay_jitter = float(timing.command_delay_jitter_s)
        self._bus_rng = np.random.default_rng([int(seed), BUS_JITTER_STREAM])
        self._last_latch = [-math.inf] * len(self.joint_names)

    def tool_position(self, q: Any) -> np.ndarray:
        raise NotImplementedError

    # ── hard stops (amendment 5, Scene.stops) ──────────────────────────────────────────────────────────────

    def _stopped_model(self, model: Any, scene: Any, joint_ids: Any, start_q: Any) -> Any:
        """`model`, or a private copy of it with the scene's stops as joint-range limits (`STOP_SOLREF`), each
        intersected with the joint's own range. `joint_ids[j]` is plant joint j's MuJoCo joint id. A plant's reset
        calls this before it makes its MjData, and its advance calls `_record_stops(k, t)` after each `mj_step`."""
        # (plant joint, MuJoCo joint id, side, angle, middle of the stopped range, qpos address)
        self._stops: list[tuple[int, int, str, float, float, int]] = []
        if not scene.stops:
            return model
        import copy

        import mujoco

        stopped = copy.deepcopy(model)  # WHY a copy: the plants cache their models across episodes
        for joint, angle, side in scene.stops:
            if not 0 <= joint < len(self.joint_names):
                raise ValueError(f"stop on joint {joint}: the {self.name} has {len(self.joint_names)} joints")
            start = float(start_q[joint])
            if (side == "upper" and start > angle) or (side == "lower" and start < angle):
                raise ValueError(
                    f"the {self.name} starts at {start:.4f} rad on {self.joint_names[joint]}, past its "
                    f"{side} stop at {angle:.4f} rad"
                )
            jid = int(joint_ids[joint])
            low, high = (
                (float(stopped.jnt_range[jid][0]), float(stopped.jnt_range[jid][1]))
                if stopped.jnt_limited[jid]
                else (-STOP_FAR_RAD, STOP_FAR_RAD)
            )
            low, high = (low, min(high, angle)) if side == "upper" else (max(low, angle), high)
            stopped.jnt_limited[jid] = 1
            stopped.jnt_range[jid] = (low, high)
            stopped.jnt_solref[jid] = STOP_SOLREF
            # WHY the range's middle: a joint-limit row only says which joint; the stop's row is the one at the
            # stop's end of the range, the joint's other (original or far) limit is not this stop
            self._stops.append((joint, jid, side, angle, 0.5 * (low + high), int(stopped.jnt_qposadr[jid])))
        self._stop_touched = [False] * len(self._stops)
        self._limit_row = int(mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT)
        return stopped

    def _record_stops(self, k: int, t: float) -> None:
        """The stops' constraint torque on their joints at step k (after its mj_step), and each first touch."""
        data = self._data
        torque = [0.0] * len(self.joint_names)
        count = int(data.nefc)
        kinds = data.efc_type[:count]
        ids = data.efc_id[:count]
        forces = data.efc_force[:count]
        for index, (joint, jid, side, angle, middle, qpos_address) in enumerate(self._stops):
            position = float(data.qpos[qpos_address])
            if (position > middle) != (side == "upper"):
                continue  # at the other end of the range
            for row in range(count):
                if int(kinds[row]) != self._limit_row or int(ids[row]) != jid:
                    continue
                # a limit row pushes the joint back inside its range: + at the lower end, - at the upper
                force = float(forces[row])
                torque[joint] += -force if side == "upper" else force
                if force > 0.0 and not self._stop_touched[index]:
                    self._stop_touched[index] = True
                    self.events.append(
                        {"t": t, "kind": "stop_touch", "joint": self.joint_names[joint], "angle_rad": angle,
                         "side": side}
                    )
        if k >= 0:
            self._tau_stop[k] = torque

    # ── the bus ────────────────────────────────────────────────────────────────────────────────────────────

    def read(self) -> ServoState:
        """The newest board sample taken at or before now - reply_delay, stamped with its sample time.

        `seq` counts distinct samples served: it advances by one per reply that carries a new sample, and
        repeats when the host reads again before a newer sample has arrived (a stale reply).
        """
        target = self._now - self.timing.reply_delay_s
        k = min(int(math.floor(target / self._dt + TIME_TOLERANCE_S)), self._k - 1)
        k = max(k, self._first_k)
        if k != self._last_sample_k:
            self._seq += 1
            self._last_sample_k = k
        row = k - self._first_k
        joints = len(self.joint_names)
        return ServoState(
            name=self.joint_names,
            position=self._sample_q[row].copy(),
            velocity=self._sample_qd[row].copy(),
            effort=self._sample_tau[row].copy(),
            fault=np.zeros(joints, dtype=np.int64),
            seq=self._seq,
            header=Header(stamp=k * self._dt, frame_id=self.name),
        )

    def write(
        self, command: ServoCommand, defines: Mapping[str, float] | None = None
    ) -> float | tuple[float, ...]:
        """Send `command` (its `header.stamp` = when it went on the bus) and any position `defines` (joint ->
        rad, a homing step's set_output_exact); returns the time they latch on the boards (one per joint when
        the joints' latches differ).

        Each write is numbered 0, 1, ... in order; `latched_tick` in the samples is that number for the
        command each board ran (-1 = the pre-roll hold), so a host-tick row (one per write) maps onto the
        physics steps."""
        sent = float(command.header.stamp)
        delays = self._delays
        for at, step_delays in self._delay_steps:  # amendment 5: the bus as it is at the send time
            if sent < at:
                break
            delays = step_delays
        jitter = 0.0
        if self._delay_jitter > 0.0:
            draw = float(self._bus_rng.uniform(-self._delay_jitter, self._delay_jitter))
            # WHY whole board ticks: the send time carries the tick's measured CPU time, which differs run to
            # run by microseconds. A continuous draw puts latches anywhere in a 0.25 ms board tick, so that
            # noise moved 8-20 of R2's ~540 latches to the next tick in every repeat run (measured 2026-11-20)
            # and the same seed was not the same episode. Rounded to the tick a board can resolve anyway (and
            # kept inside +-jitter), a seed fixes every latch; a jitter under one tick has no effect.
            bound = math.floor(self._delay_jitter / self._dt + 1e-9)
            jitter = min(max(round(draw / self._dt), -bound), bound) * self._dt
        latch = []
        for j, delay in enumerate(delays):
            # WHY max with the previous latch: a CAN bus delivers in order, so a command that would arrive
            # before the one sent ahead of it waits for it and latches on the same board tick
            at = max(sent + (delay + jitter if jitter else delay), self._last_latch[j])
            latch.append(at)
        self._last_latch = latch
        uniform = all(at == latch[0] for at in latch)
        self.boards.schedule(command, latch[0] if uniform else latch, defines, tag=self._writes)
        self._writes += 1
        return latch[0] if uniform else tuple(latch)

    def _allocate(self, rows: int) -> None:
        joints = len(self.joint_names)
        self._capacity = rows
        # board samples (pre-roll included), indexed k - first_k: what read() serves
        self._sample_q = np.zeros((rows, joints))
        self._sample_qd = np.zeros((rows, joints))
        self._sample_tau = np.zeros((rows, joints))
        # plant rows (t >= 0), indexed k
        self._t = np.zeros(rows)
        self._q = np.zeros((rows, joints))
        self._qd = np.zeros((rows, joints))
        self._tau = np.zeros((rows, joints))
        self._at_cap = np.zeros((rows, joints), dtype=bool)
        self._mode = np.zeros((rows, joints), dtype=np.int64)
        self._tool_pos = np.zeros((rows, 3))
        self._contact_force = np.zeros((rows, 3))
        self._in_contact = np.zeros(rows, dtype=bool)
        self._external_force = np.zeros((rows, 3))
        # the torque split (amendment 3): the feed-forward the board added, its PD term, the command it ran
        self._tau_ff = np.zeros((rows, joints))
        self._tau_fb_board = np.zeros((rows, joints))
        self._latched_tick = np.full((rows, joints), -1, dtype=np.int64)
        self._tau_stop = np.zeros((rows, joints))  # amendment 5: the hard stops' torque (Scene.stops)

    def _grow(self) -> None:
        names = (
            "_sample_q",
            "_sample_qd",
            "_sample_tau",
            "_t",
            "_q",
            "_qd",
            "_tau",
            "_at_cap",
            "_mode",
            "_tool_pos",
            "_contact_force",
            "_in_contact",
            "_external_force",
            "_tau_ff",
            "_tau_fb_board",
            "_latched_tick",
            "_tau_stop",
        )
        old = {name: getattr(self, name) for name in names}
        self._allocate(self._capacity * 2)
        for name, array in old.items():
            getattr(self, name)[: len(array)] = array

    def _warning_count(self) -> int:
        warnings = self._data.warning
        return int(
            sum(
                warnings[int(kind)].number
                for kind in (
                    self._mujoco.mjtWarning.mjWARN_BADQACC,
                    self._mujoco.mjtWarning.mjWARN_BADQVEL,
                    self._mujoco.mjtWarning.mjWARN_BADQPOS,
                )
            )
        )

    def _check_divergence(self) -> None:
        warnings = self._warning_count()
        if warnings > self._warnings_seen:
            # MuJoCo resets the state on a non-finite or exploding acceleration; the episode is no longer
            # physical
            self._warnings_seen = warnings
            self.events.append(
                {"t": self._now, "kind": "diverged", "why": "MuJoCo reset the state (bad qacc/qvel/qpos)"}
            )

    # ── the record ─────────────────────────────────────────────────────────────────────────────────────────

    @property
    def sample_count(self) -> int:
        return max(self._k, 0)

    def samples(self) -> dict[str, np.ndarray]:
        """The plant rows recorded so far (t >= 0), copies, keyed as EpisodeLog's plant fields (`tau_stop` only
        when the scene has stops)."""
        rows = self.sample_count
        if getattr(getattr(self, "scene", None), "stops", ()) and not getattr(self, "_stops", None):
            # WHY loud: a plant that silently ignored a stop would score a collision target as free motion
            raise NotImplementedError(f"the {self.name} plant does not honour Scene.stops (see _stopped_model)")
        extra = {"tau_stop": self._tau_stop[:rows].copy()} if getattr(self, "_stops", None) else {}
        return {**extra,
            "t": self._t[:rows].copy(),
            "q": self._q[:rows].copy(),
            "qd": self._qd[:rows].copy(),
            "tau": self._tau[:rows].copy(),
            "at_cap": self._at_cap[:rows].copy(),
            "board_mode": self._mode[:rows].copy(),
            "tool_pos": self._tool_pos[:rows].copy(),
            "contact_force": self._contact_force[:rows].copy(),
            "in_contact": self._in_contact[:rows].copy(),
            "external_force": self._external_force[:rows].copy(),
            "tau_ff": self._tau_ff[:rows].copy(),
            "tau_fb_board": self._tau_fb_board[:rows].copy(),
            "latched_tick": self._latched_tick[:rows].copy(),
        }


class PlanarArmPlant(ServoPlant):
    """The `planar_arm` plant: MuJoCo arm + belt + one moteus board model per joint."""

    name = "planar_arm"
    joint_names: tuple[str, ...] = JOINT_NAMES

    def __init__(self, *, board_config: BoardConfig | None = None, arm: ArmConfig = ARM) -> None:
        self.arm = arm
        super().__init__(board_config or BoardConfig(), arm.physics_dt_s)

    @property
    def home_q(self) -> tuple[float, ...]:
        return HOME_Q

    def tool_position(self, q: Any) -> np.ndarray:
        """The tool centre (x, y, z) for joint angles `q`, (2,) -> (3,), (N, 2) -> (N, 3): the arm's fk."""
        return tool_position(q, self.arm)

    # ── episode setup ──────────────────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        scene: Scene,
        seed: int = 0,
        *,
        timing: HostTiming | None = None,
        duration_s: float = 5.0,
    ) -> None:
        """A new episode: build (or reuse) the scene's model, put the arm at `scene.start_q` (or home) at
        rest, the boards holding it, and run the pre-roll. `seed` is kept for the record: nothing in this
        plant is random (no sensor noise); the loop's jitter has its own seed."""
        import mujoco

        self._mujoco = mujoco
        self.scene = scene
        self.seed = int(seed)
        self.timing = timing or HostTiming()
        self._reset_bus(seed, self.timing)
        model, self.contact_calibration = build_model(scene, self.arm)
        start_q = np.asarray(scene.start_q if scene.start_q is not None else self.home_q, dtype=float)
        if start_q.shape != (2,):
            raise ValueError(f"start_q must be 2 joint angles, got {start_q}")
        joint_ids = [model.joint(name).id for name in self.joint_names]
        self._model = model = self._stopped_model(model, scene, joint_ids, start_q)
        self._data = data = mujoco.MjData(model)
        self.start_q = tuple(float(v) for v in start_q)
        data.qpos[:2] = start_q
        data.qvel[:] = 0.0
        self._belt_dof = model.joint("belt").dofadr[0] if scene.belt else None
        self._belt_speed = float(scene.belt_speed_m_s) if scene.belt else 0.0
        if self._belt_dof is not None:
            data.qvel[self._belt_dof] = self._belt_speed
        mujoco.mj_forward(model, data)
        self._tool_body = model.body("tool").id
        self._tool_geom = model.geom("tool").id
        self._pushes = tuple(
            (float(p.start_s), float(p.start_s + p.duration_s), tuple(float(f) for f in p.force_n))
            for p in scene.pushes
        )
        self._push_active = [False] * len(self._pushes)
        self._push_done = [False] * len(self._pushes)
        self._normal_force = 0.0  # last step's belt normal force on the tool (N): what the friction uses
        self._touched = False
        self._warnings_seen = self._warning_count()
        self.events: list[dict[str, Any]] = []

        preroll_ticks = int(round(PREROLL_S / self._dt))
        self._first_k = -preroll_ticks
        self._k = self._first_k
        hold_feedforward = nominal_dynamics(start_q, np.zeros(2), np.zeros(2), self.arm)
        self.boards.reset(self._first_k * self._dt, start_q, feedforward=hold_feedforward)
        self._allocate(preroll_ticks + int(math.ceil(duration_s / self._dt)) + 8)
        self._last_sample_k: int | None = None
        self._seq = 0
        self._writes = 0  # commands written this episode: the tag the boards report while each one runs
        self._now = self._first_k * self._dt
        self.advance(0.0)

    # ── physics ────────────────────────────────────────────────────────────────────────────────────────────

    def advance(self, until: float) -> None:
        """Run physics and boards over every tick with k * dt < `until` (s)."""
        mujoco, model, data, boards = self._mujoco, self._model, self._data, self.boards
        dt, arm = self._dt, self.arm
        l1, l2, radius = arm.link1_length_m, arm.link2_length_m, arm.tool_radius_m
        mu, slip_speed = arm.belt_friction, arm.friction_slip_speed_m_s
        tool_body, tool_geom = self._tool_body, self._tool_geom
        belt_dof, belt_speed = self._belt_dof, self._belt_speed
        qpos, qvel, qfrc, xfrc, xpos = data.qpos, data.qvel, data.qfrc_applied, data.xfrc_applied, data.xpos
        first_k = self._first_k
        k = self._k
        while k * dt < until - TIME_TOLERANCE_S:
            t = k * dt
            if k - first_k >= self._capacity:
                self._grow()
            q_shoulder, q_elbow = float(qpos[0]), float(qpos[1])
            qd_shoulder, qd_elbow = float(qvel[0]), float(qvel[1])
            tau = boards.step(t, (q_shoulder, q_elbow))
            qfrc[0], qfrc[1] = tau[0], tau[1]
            # the scene's pushes on the tool
            push_x = push_y = push_z = 0.0
            for index, (start, end, force) in enumerate(self._pushes):
                if start - TIME_TOLERANCE_S <= t < end - TIME_TOLERANCE_S:
                    push_x, push_y, push_z = push_x + force[0], push_y + force[1], push_z + force[2]
                    if not self._push_active[index]:
                        self._push_active[index] = True
                        self.events.append({"t": t, "kind": "push_start", "force_n": list(force)})
                elif self._push_active[index] and not self._push_done[index]:
                    self._push_done[index] = True
                    self.events.append({"t": t, "kind": "push_end"})
            # the belt's Coulomb friction on the tool, from last step's normal force (one 0.25 ms tick behind)
            friction_x = 0.0
            if belt_dof is not None:
                qvel[belt_dof] = belt_speed
                if self._normal_force > 0.0:
                    # contact-point velocity along x: the tool centre's plus omega x r (r = R down the -z
                    # axis)
                    sum_angle, sum_rate = q_shoulder + q_elbow, qd_shoulder + qd_elbow
                    tool_vx = -l1 * math.sin(q_shoulder) * qd_shoulder - l2 * math.sin(sum_angle) * sum_rate
                    slip = tool_vx + radius * sum_rate - belt_speed
                    friction_x = -mu * self._normal_force * math.tanh(slip / slip_speed)
            xfrc[tool_body, 0] = push_x + friction_x
            xfrc[tool_body, 1] = push_y
            xfrc[tool_body, 2] = push_z
            xfrc[tool_body, 4] = (
                -radius * friction_x
            )  # the friction acts at the contact point, R below the centre
            mujoco.mj_step(model, data)
            # after mj_step, kinematics and contact forces are those of the state at t (integration came last)
            if self._stops:
                self._record_stops(k, t)
            normal = 0.0
            normal_z = 0.0
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                force = float(data.efc_force[contact.efc_address])
                if force <= 0.0:
                    continue
                direction = -1.0 if contact.geom[0] == tool_geom else 1.0  # +normal acts on geom2
                normal += force
                normal_z += direction * force * float(contact.frame[2])
            self._normal_force = normal
            row = k - first_k
            self._sample_q[row] = boards.measured_position
            self._sample_qd[row] = boards.measured_velocity
            self._sample_tau[row] = tau
            for joint in boards.watchdog_entries:
                self.events.append({"t": t, "kind": "watchdog", "joint": self.joint_names[joint]})
            if k >= 0:
                self._t[k] = t
                self._q[k] = (q_shoulder, q_elbow)
                self._qd[k] = (qd_shoulder, qd_elbow)
                self._tau[k] = tau
                self._at_cap[k] = boards.at_cap
                self._mode[k] = boards.mode
                self._tool_pos[k] = xpos[tool_body]
                self._contact_force[k] = (friction_x, 0.0, normal_z)
                self._external_force[k] = (push_x, push_y, push_z)
                self._tau_ff[k] = boards.tau_ff
                self._tau_fb_board[k] = boards.tau_fb
                self._latched_tick[k] = boards.latched_tag
                touching = normal > 1e-6
                self._in_contact[k] = touching
                if touching and not self._touched:
                    self._touched = True
                    self.events.append({"t": t, "kind": "touch"})
            k += 1
        self._k = k
        self._now = max(self._now, float(until))
        self._check_divergence()

    def meta(self) -> dict[str, Any]:
        """What this plant was, for EpisodeLog.meta."""
        calibration = self.contact_calibration
        return {
            "plant": self.name,
            "physics_dt_s": self._dt,
            "preroll_s": -self._first_k * self._dt,
            "start_q": list(self.start_q),
            "board": asdict(self.board_config),
            "arm": asdict(self.arm),
            "contact_calibration": None if calibration is None else asdict(calibration),
        }
