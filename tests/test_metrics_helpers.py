"""Hand-built EpisodeLogs and Rungs with known answers, for the metrics tests (no bench, no MuJoCo).

`make_log` builds a two-joint log: host ticks at `rate_hz` from 0 to `duration_s`, plant samples at
`plant_rate_hz`. By default the plant follows the reference exactly (zero tracking error), every state is
`latency_s` old when the command goes out, nothing touches, nothing saturates. Tests then overwrite the arrays
whose answer they know (EpisodeLog is a plain mutable dataclass).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from robot_msgs.trajectory import JointTrajectory

from robot_controllers.bench.types import EpisodeLog, Gate, HostTiming, Push, Rung, Scene
from robot_controllers.registry import JOINT_COMMAND_SEQUENCE, JOINT_TRAJECTORY

JOINTS = ("shoulder", "elbow")
LINK_1_M, LINK_2_M = 0.30, 0.25
BASE_Z_M = 0.50


def planar_fk(q: np.ndarray) -> np.ndarray:
    """(N, 2) joint angles -> (N, 3) tool position, a stand-in planar arm in the x-z plane."""
    q = np.atleast_2d(q)
    x = LINK_1_M * np.cos(q[:, 0]) + LINK_2_M * np.cos(q[:, 0] + q[:, 1])
    z = BASE_Z_M - LINK_1_M * np.sin(q[:, 0]) - LINK_2_M * np.sin(q[:, 0] + q[:, 1])
    return np.column_stack([x, np.zeros_like(x), z])


def smooth_reference(t: np.ndarray) -> np.ndarray:
    """A gentle two-joint motion (rad) that ends at 0.6 s and holds."""
    s = np.clip(np.asarray(t, dtype=float) / 0.6, 0.0, 1.0)
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5  # min-jerk
    return np.column_stack([0.2 + 0.3 * blend, 0.5 - 0.2 * blend])


def make_log(
    *,
    duration_s: float = 1.0,
    rate_hz: float = 100.0,
    plant_rate_hz: float = 4000.0,
    latency_s: float = 0.003,
    reference: Callable[[np.ndarray], np.ndarray] = smooth_reference,
    sent_t: np.ndarray | None = None,
    rung: str = "R1.FreeSpace",
    controller: str = "test_controller",
) -> EpisodeLog:
    period = 1.0 / rate_hz
    if sent_t is None:
        sent_t = np.arange(0.0, duration_s - 1e-9, period)
    sent_t = np.asarray(sent_t, dtype=float)
    ticks = sent_t.size
    t = np.arange(0.0, duration_s + 1e-9, 1.0 / plant_rate_hz)
    q = reference(t)
    qd = np.gradient(q, t, axis=0)
    ref_position = reference(sent_t)
    nan = np.full((ticks, len(JOINTS)), np.nan)
    return EpisodeLog(
        controller=controller,
        rung=rung,
        seed=0,
        joint_names=JOINTS,
        nominal_period_s=period,
        tick_t=sent_t.copy(),
        sent_t=sent_t.copy(),
        state_stamp=sent_t - latency_s,
        state_seq=np.arange(ticks),
        compute_s=np.full(ticks, 20e-6),
        cmd_position=ref_position.copy(),
        cmd_velocity=np.gradient(ref_position, sent_t, axis=0) if ticks > 1 else nan.copy(),
        cmd_feedforward=np.zeros((ticks, len(JOINTS))),
        cmd_stiffness=nan.copy(),
        cmd_damping=nan.copy(),
        cmd_max_effort=nan.copy(),
        ref_position=ref_position,
        ref_velocity=None,
        ref_tool_pos=planar_fk(ref_position),
        status=np.array(["running"] * ticks, dtype=object),
        input_index=np.zeros(ticks, dtype=int),
        t=t,
        q=q,
        qd=qd,
        tau=np.zeros_like(q),
        at_cap=np.zeros(q.shape, dtype=bool),
        board_mode=np.zeros(q.shape, dtype=int),
        tool_pos=planar_fk(q),
        contact_force=np.zeros((t.size, 3)),
        in_contact=np.zeros(t.size, dtype=bool),
        external_force=np.zeros((t.size, 3)),
        events=[],
        meta={},
    )


def trajectory_script(duration_s: float = 0.6) -> Callable[[np.ndarray], list]:
    """An input script whose one JointTrajectory lasts `duration_s` (the schedule the slip is measured on)."""

    def script(start_q: np.ndarray) -> list:
        times = np.linspace(0.0, duration_s, 7)
        positions = smooth_reference(times)
        return [(0.0, JointTrajectory(t=times, joint_names=JOINTS, positions=positions))]

    return script


def make_rung(
    rung_id: str = "R1.FreeSpace",
    *,
    family: str = "free_space",
    gates: tuple[Gate, ...] = (),
    objective: str = "tracking_p95_deg",
    minimize: bool = True,
    scene: Scene | None = None,
    order: int = 1,
    duration_s: float = 1.0,
    inputs: Callable[[np.ndarray], list] | None = None,
    sequence_box: bool = False,
) -> Rung:
    return Rung(
        id=rung_id,
        title=rung_id,
        box=JOINT_COMMAND_SEQUENCE if sequence_box else JOINT_TRAJECTORY,
        family=family,
        description="A hand-built rung for metric tests.",
        duration_s=duration_s,
        inputs=inputs or trajectory_script(),
        scene=scene or Scene(),
        timing=HostTiming(),
        gates=gates,
        objective=objective,
        minimize=minimize,
        order=order,
    )


# ── a belt impact and a push with known answers ─────────────────────────────────────────────────────────────

PLANT_DT = 0.00025
TOUCH_S = 0.5
DWELL = (0.6, 1.4)


def belt_trace(duration_s: float = 1.6):
    """Touch at 0.5 s: a 40 N spike 5-7 ms after it, a steady 10 N, a 50 N knock at 1.45 s (after the dwell).

    Returns (t, contact_force, in_contact, tool_pos); the tool comes down at 0.05 m/s onto the belt top (0.15
    m).
    """
    t = np.arange(0.0, duration_s, PLANT_DT)
    force = np.zeros((t.size, 3))
    in_contact = (t >= TOUCH_S) & (t < 1.5)
    force[in_contact, 2] = 10.0
    force[(t >= TOUCH_S + 0.005) & (t < TOUCH_S + 0.007), 2] = 40.0
    force[(t >= 1.45) & (t < 1.452), 2] = 50.0
    tool = np.zeros((t.size, 3))
    tool[:, 0] = 0.30
    tool[:, 2] = np.where(t < TOUCH_S, 0.15 + 0.05 * (TOUCH_S - t), 0.15)
    return t, force, in_contact, tool


def _replace_plant(log: EpisodeLog, t: np.ndarray, tool: np.ndarray) -> None:
    n = t.size
    log.t, log.tool_pos = t, tool
    log.q, log.qd, log.tau = np.zeros((n, 2)), np.zeros((n, 2)), np.zeros((n, 2))
    log.at_cap, log.board_mode = np.zeros((n, 2), dtype=bool), np.zeros((n, 2), dtype=int)
    log.contact_force, log.in_contact = np.zeros((n, 3)), np.zeros(n, dtype=bool)
    log.external_force = np.zeros((n, 3))


def belt_log() -> EpisodeLog:
    log = make_log(duration_s=1.6, rung="R3.BeltImpact@0.10m/s")
    t, force, in_contact, tool = belt_trace()
    _replace_plant(log, t, tool)
    log.contact_force, log.in_contact = force, in_contact
    log.meta.update({"dwell": DWELL, "press_target_n": 10.0})
    return log


def belt_rung(**overrides) -> Rung:
    return make_rung(
        "R3.BeltImpact@0.10m/s",
        family="contact",
        scene=Scene(belt=True),
        objective="contact_max_n",
        **overrides,
    )


def push_path(duration_s: float = 1.4):
    """+x push in [0.5, 0.9]: 3 mm out, then -2 mm overshoot, a 20 ms dip inside 1 mm, out again, back at
    1.05."""
    t = np.arange(0.0, duration_s, PLANT_DT)
    displacement_mm = np.zeros(t.size)
    displacement_mm[(t >= 0.5) & (t < 0.9)] = 3.0
    displacement_mm[(t >= 0.9) & (t < 1.0)] = -2.0
    displacement_mm[(t >= 1.0) & (t < 1.02)] = 0.5
    displacement_mm[(t >= 1.02) & (t < 1.05)] = 1.5
    displacement_mm[t >= 1.05] = 0.2
    tool = np.tile([0.30, 0.0, 0.25], (t.size, 1))
    tool[:, 0] += displacement_mm / 1000.0
    return t, tool


def push_log() -> EpisodeLog:
    log = make_log(duration_s=1.4, rung="R4.SidewaysPush")
    _replace_plant(log, *push_path())
    return log


def push_rung(**overrides) -> Rung:
    scene = Scene(pushes=(Push(0.5, 0.4, (15.0, 0.0, 0.0)),))
    return make_rung(
        "R4.SidewaysPush", family="disturbance", scene=scene, objective="push_recovery_s", **overrides
    )
