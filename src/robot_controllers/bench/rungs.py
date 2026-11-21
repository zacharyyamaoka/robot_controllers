"""The rung ladder: the ruler every controller of a box is measured on (CONTRACT §3, frozen with its gates).

    from robot_controllers.bench.rungs import ladder
    for rung in ladder(JOINT_TRAJECTORY): ...          # R1.FreeSpace ... R9.FastFreeSpace·host500Hz
    for rung in ladder(JOINT_COMMAND_SEQUENCE): ...    # S1.MoveTo, S2.SeekHardStop, S3.HostStall

JointTrajectory rungs: the tool path is drawn in (x, z) (minimum-jerk moves, quintic blends, constant-speed
segments), turned into joints by the analytic elbow-up `ik` with analytic velocity and acceleration, sampled
every 4 ms, and `effort` = `nominal_dynamics(q, qd, qdd)` (gravity + inertia feed-forward).
JointCommandSequence rungs build robot_msgs `JointCommandSequence`s. Timing gates (`missed_cycles == 0`,
`latency_p99_ms <= 1000 / rate`) sit on every rung except the two stall rungs, computed from that rung's host
rate.

Gate numbers are the ruler: never move them after seeing results; a controller that fails is a finding.

Input scripts are small frozen dataclasses (not closures) so a rung pickles into a worker process.

Amendment 4 (§4.4) adds `R10.BeltSync@0.25m/s` (the tool keeping pace with a point riding the belt) and four
variant functions, which are not ladder rungs: `rate_variant` (another host rate, its timing gates with it),
`speed_variant` (the same free-space path at another move time), `bus_variant` (another command delay, per
joint, with jitter) and `sync_variant` (R10 at another belt speed).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from robot_msgs.control import (
    Elapsed,
    JointCommand,
    JointCommandSequence,
    JointCommandStep,
    Reached,
    Stall,
)
from robot_msgs.trajectory import JointTrajectory

from robot_controllers.bench.arm import ARM, HOME_Q, JOINT_NAMES, ik, joint_motion_from_tool, nominal_dynamics
from robot_controllers.bench.board import BoardConfig
from robot_controllers.bench.types import Gate, HostTiming, Push, Rung, Scene
from robot_controllers.core import Box
from robot_controllers.registry import JOINT_COMMAND_SEQUENCE, JOINT_TRAJECTORY

SAMPLE_PERIOD_S = 0.004  # every trajectory is sampled every 4 ms (CONTRACT §3)
TOOL_PATH = ((0.30, 0.25), (0.42, 0.20), (0.25, 0.32), (0.30, 0.25))  # R1/R2/R6: home -> out -> up-in -> home
REST_S = 0.3  # R1/R2/R6 rests between moves
BELT_X_M = 0.30  # where the belt rungs touch down
APPROACH_ABOVE_M = 0.040  # the belt rungs start with the tool 40 mm above the surface
CRUISE_M = 0.020  # ... cover the last 20 mm at constant touch speed
AIM_BELOW_M = 0.010  # ... aiming 10 mm below the surface
LIFT_M = 0.040  # ... and lift 40 mm after the dwell
DWELL_S = 0.8
LIFT_S = 0.4
BELT_SPEED_M_S = 0.25
PRESS_TARGET_N = 10.0
REPLAN_PERIOD_S = 0.15
REPLANS = 8
REPLAN_MOVE_S = 0.6
REPLAN_RANGE_RAD = math.radians(20.0)
REPLAN_TARGET_SEED = (
    20261112  # WHY fixed in the rung: the targets are part of the ruler, the same for everyone
)
STALL_S = 0.5
PAYLOAD_KG = 0.3  # R8: on the tool, known to no feed-forward model (amendment 3)
# R10 (amendment 4, §4.4): a virtual item rides the belt along +x, 50 mm above its top (the tool centre's z)
SYNC_POINTS_X_M = (0.11, 0.16, 0.36, 0.40)  # hold, cruise start, cruise end, rest
SYNC_Z_M = 0.20
SYNC_HOLD_S = 0.3
SYNC_BELT_SPEED_M_S = 0.25
SYNC_TOLERANCE_MM = 5.0
TIMING_METRICS = ("missed_cycles", "latency_p99_ms")  # the two gates timing_gates() adds


# ── paths ───────────────────────────────────────────────────────────────────────────────────────────────────


def min_jerk(tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """s, ds/dtau, d2s/dtau2 of the minimum-jerk blend 10 tau^3 - 15 tau^4 + 6 tau^5 on tau in [0, 1]."""
    tau = np.clip(tau, 0.0, 1.0)
    return (
        10 * tau**3 - 15 * tau**4 + 6 * tau**5,
        30 * tau**2 - 60 * tau**3 + 30 * tau**4,
        60 * tau - 180 * tau**2 + 120 * tau**3,
    )


def quintic(
    start: float,
    start_velocity: float,
    start_acceleration: float,
    end: float,
    end_velocity: float,
    end_acceleration: float,
    duration: float,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Position, velocity and acceleration of the quintic polynomial meeting both ends' (p, v, a) in
    `duration`."""
    big_t = duration
    c0, c1, c2 = start, start_velocity, start_acceleration / 2.0
    c3 = (
        20 * (end - start)
        - (8 * end_velocity + 12 * start_velocity) * big_t
        - (3 * start_acceleration - end_acceleration) * big_t**2
    ) / (2 * big_t**3)
    c4 = (
        30 * (start - end)
        + (14 * end_velocity + 16 * start_velocity) * big_t
        + (3 * start_acceleration - 2 * end_acceleration) * big_t**2
    ) / (2 * big_t**4)
    c5 = (
        12 * (end - start)
        - 6 * (end_velocity + start_velocity) * big_t
        - (start_acceleration - end_acceleration) * big_t**2
    ) / (2 * big_t**5)
    t = np.clip(np.asarray(t, dtype=float), 0.0, big_t)
    position = c0 + c1 * t + c2 * t**2 + c3 * t**3 + c4 * t**4 + c5 * t**5
    velocity = c1 + 2 * c2 * t + 3 * c3 * t**2 + 4 * c4 * t**3 + 5 * c5 * t**4
    acceleration = 2 * c2 + 6 * c3 * t + 12 * c4 * t**2 + 20 * c5 * t**3
    return position, velocity, acceleration


# A tool-path segment: ("hold", point, duration) | ("min_jerk", start, end, duration)
# | ("blend", start, end, start_speed, end_speed, duration): a quintic along the line, speeds along it, at
# rest in acc
# | ("cruise", start, end, duration): constant speed along the line.
Segment = tuple


def _segment_samples(segment: Segment, local_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kind = segment[0]
    count = len(local_t)
    if kind == "hold":
        point = np.asarray(segment[1], dtype=float)
        return np.tile(point, (count, 1)), np.zeros((count, 2)), np.zeros((count, 2))
    start, end = np.asarray(segment[1], dtype=float), np.asarray(segment[2], dtype=float)
    delta = end - start
    length = float(np.linalg.norm(delta))
    direction = delta / length if length > 0 else np.zeros(2)
    if kind == "min_jerk":
        duration = segment[3]
        s, ds, dds = min_jerk(local_t / duration)
        return start + np.outer(s, delta), np.outer(ds / duration, delta), np.outer(dds / duration**2, delta)
    if kind == "blend":
        start_speed, end_speed, duration = segment[3], segment[4], segment[5]
        u, du, ddu = quintic(0.0, start_speed, 0.0, length, end_speed, 0.0, duration, local_t)
        return start + np.outer(u, direction), np.outer(du, direction), np.outer(ddu, direction)
    if kind == "cruise":
        duration = segment[3]
        speed = length / duration
        u = speed * np.clip(local_t, 0.0, duration)
        return (
            start + np.outer(u, direction),
            np.outer(np.full(count, speed), direction),
            np.zeros((count, 2)),
        )
    raise ValueError(f"unknown tool-path segment {kind!r}")


def _duration(segment: Segment) -> float:
    return float(segment[-1])


def sample_tool_path(segments: tuple[Segment, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(t, points, velocities, accelerations) of a piecewise tool path, every 4 ms, end included."""
    total = sum(_duration(segment) for segment in segments)
    count = int(round(total / SAMPLE_PERIOD_S))
    t = np.arange(count + 1) * SAMPLE_PERIOD_S
    t[-1] = total
    boundaries = np.cumsum([0.0] + [_duration(segment) for segment in segments])
    which = np.clip(np.searchsorted(boundaries, t + 1e-12, side="right") - 1, 0, len(segments) - 1)
    points, velocities, accelerations = (
        np.zeros((count + 1, 2)),
        np.zeros((count + 1, 2)),
        np.zeros((count + 1, 2)),
    )
    for index, segment in enumerate(segments):
        mask = which == index
        if mask.any():
            p, v, a = _segment_samples(segment, t[mask] - boundaries[index])
            points[mask], velocities[mask], accelerations[mask] = p, v, a
    return t, points, velocities, accelerations


@functools.lru_cache(maxsize=64)
def _tool_path_trajectory(segments: tuple[Segment, ...]) -> JointTrajectory:
    t, points, velocities, accelerations = sample_tool_path(segments)
    q, qd, qdd = joint_motion_from_tool(points, velocities, accelerations)
    return JointTrajectory(
        t=t,
        joint_names=JOINT_NAMES,
        positions=q,
        velocities=qd,
        accelerations=qdd,
        effort=nominal_dynamics(q, qd, qdd),
    )


def _joint_quintic_trajectory(
    start: np.ndarray,
    start_velocity: np.ndarray,
    start_acceleration: np.ndarray,
    target: np.ndarray,
    duration: float,
) -> JointTrajectory:
    count = int(round(duration / SAMPLE_PERIOD_S))
    t = np.arange(count + 1) * SAMPLE_PERIOD_S
    t[-1] = duration
    q, qd, qdd = (np.zeros((count + 1, len(start))) for _ in range(3))
    for j in range(len(start)):
        q[:, j], qd[:, j], qdd[:, j] = quintic(
            start[j], start_velocity[j], start_acceleration[j], target[j], 0.0, 0.0, duration, t
        )
    return JointTrajectory(
        t=t,
        joint_names=JOINT_NAMES,
        positions=q,
        velocities=qd,
        accelerations=qdd,
        effort=nominal_dynamics(q, qd, qdd),
    )


# ── input scripts (start joints -> [(t, message), ...]) ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolPathScript:
    """One JointTrajectory at t = 0 that follows a fixed tool path (the rung's scene starts the arm on it)."""

    segments: tuple[Segment, ...]

    def __call__(self, start_q: np.ndarray) -> list[tuple[float, Any]]:
        return [(0.0, _tool_path_trajectory(self.segments))]

    @property
    def duration_s(self) -> float:
        return sum(_duration(segment) for segment in self.segments)


@dataclass(frozen=True)
class HoldScript:
    """One JointTrajectory at t = 0 holding the start joints for `duration_s`, effort = gravity."""

    duration_s: float

    def __call__(self, start_q: np.ndarray) -> list[tuple[float, Any]]:
        start = np.asarray(start_q, dtype=float)
        zero = np.zeros_like(start)
        return [(0.0, _joint_quintic_trajectory(start, zero, zero, start, self.duration_s))]


@dataclass(frozen=True)
class ReplanScript:
    """A trajectory at t = 0 from the start (at rest) to targets[0], then one every `period_s` from the
    previous trajectory's state at that instant (position, velocity and acceleration) to the next target, at
    rest.

    WHY the acceleration too (the contract names position and velocity): a jump in reference acceleration is a
    jump in the feed-forward torque the trajectory carries; matching it keeps the replan itself smooth, so
    what the rung measures is the controller's handover, not a step the script put in.
    """

    targets: tuple[tuple[float, ...], ...]
    period_s: float = REPLAN_PERIOD_S
    move_s: float = REPLAN_MOVE_S

    def __call__(self, start_q: np.ndarray) -> list[tuple[float, Any]]:
        return list(_replan_chain(tuple(float(v) for v in start_q), self.targets, self.period_s, self.move_s))


@functools.lru_cache(maxsize=16)
def _replan_chain(
    start_q: tuple[float, ...], targets: tuple[tuple[float, ...], ...], period_s: float, move_s: float
) -> tuple[tuple[float, JointTrajectory], ...]:
    position = np.asarray(start_q, dtype=float)
    velocity = np.zeros_like(position)
    acceleration = np.zeros_like(position)
    chain = []
    for index, target in enumerate(targets):
        trajectory = _joint_quintic_trajectory(position, velocity, acceleration, np.asarray(target), move_s)
        chain.append((index * period_s, trajectory))
        # the state of this trajectory `period_s` in: where the next one starts (the quintic's exact values)
        for j in range(len(position)):
            p, v, a = quintic(
                position[j], velocity[j], acceleration[j], target[j], 0.0, 0.0, move_s, np.array([period_s])
            )
            position[j], velocity[j], acceleration[j] = p[0], v[0], a[0]
    return tuple(chain)


@dataclass(frozen=True)
class SequenceScript:
    """One JointCommandSequence at t = 0."""

    sequence: JointCommandSequence

    def __call__(self, start_q: np.ndarray) -> list[tuple[float, Any]]:
        return [(0.0, self.sequence)]


# ── the gates ───────────────────────────────────────────────────────────────────────────────────────────────


def timing_gates(rate_hz: float) -> tuple[Gate, ...]:
    period_ms = 1000.0 / rate_hz
    return (
        Gate("missed_cycles", "==", 0, "the host kept its rate: no period went by without a command"),
        Gate(
            "latency_p99_ms",
            "<=",
            period_ms,
            f"a command goes out within one host period ({period_ms:g} ms) of the sample it came from",
        ),
    )


_DIVERGED = Gate("diverged", "==", 0, "nothing went non-finite or out of bounds")
_FAULTS = Gate("faults", "==", 0, "no board fault and no controller abort")


def _belt_gates() -> tuple[Gate, ...]:
    return (
        Gate("contact_peak_n", "<=", 60, "the impact transient stays under 60 N"),
        Gate("dwell_bounces", "==", 0, "once pressing, the tool never leaves the belt"),
        Gate(
            "dwell_force_error_n", "<=", 10, f"presses with {PRESS_TARGET_N:g} N, within 10 N, over the dwell"
        ),
        Gate("dwell_contact_fraction", ">=", 95, "in contact for at least 95 % of the dwell"),
        _DIVERGED,
    )


# ── the rungs ───────────────────────────────────────────────────────────────────────────────────────────────


def free_space_segments(move_s: float) -> tuple[Segment, ...]:
    """R1/R2/R6: rest, then three minimum-jerk moves through TOOL_PATH with a rest after each."""
    segments: list[Segment] = [("hold", TOOL_PATH[0], REST_S)]
    for start, end in zip(TOOL_PATH[:-1], TOOL_PATH[1:], strict=True):
        segments += [("min_jerk", start, end, move_s), ("hold", end, REST_S)]
    return tuple(segments)


def belt_segments(touch_speed_m_s: float) -> tuple[tuple[Segment, ...], tuple[float, float]]:
    """R3/R7: (segments, dwell window). From 40 mm above the belt at x = 0.30: speed up over 20 mm to the
    touch speed, cruise the last 20 mm at it, slow to rest 10 mm below the surface (a quintic over 10 mm),
    dwell 0.8 s, lift 40 mm (minimum-jerk, 0.4 s), rest. Heights are the tool centre's (the sphere's bottom
    touches at belt top + radius)."""
    touch_z = ARM.tool_touch_z_m
    start = (BELT_X_M, touch_z + APPROACH_ABOVE_M)
    cruise = (BELT_X_M, touch_z + CRUISE_M)
    surface = (BELT_X_M, touch_z)
    bottom = (BELT_X_M, touch_z - AIM_BELOW_M)
    lifted = (BELT_X_M, touch_z - AIM_BELOW_M + LIFT_M)
    speed = touch_speed_m_s
    # WHY these blend times: 2 * distance / speed is the time a constant acceleration would take, so the
    # quintic speeds up (and slows down) monotonically without overshooting the touch speed or the bottom
    speed_up_s = 2.0 * (APPROACH_ABOVE_M - CRUISE_M) / speed
    cruise_s = CRUISE_M / speed
    slow_down_s = 2.0 * AIM_BELOW_M / speed
    first_rest_s = 0.2
    segments: tuple[Segment, ...] = (
        ("hold", start, first_rest_s),
        ("blend", start, cruise, 0.0, speed, speed_up_s),
        ("cruise", cruise, surface, cruise_s),
        ("blend", surface, bottom, speed, 0.0, slow_down_s),
        ("hold", bottom, DWELL_S),
        ("min_jerk", bottom, lifted, LIFT_S),
        ("hold", lifted, 0.3),
    )
    dwell_start = first_rest_s + speed_up_s + cruise_s + slow_down_s
    return segments, (dwell_start, dwell_start + DWELL_S)


def _belt_rung(
    order: int,
    rung_id: str,
    title: str,
    touch_speed: float,
    timing: HostTiming,
    family: str,
    description: str,
) -> Rung:
    segments, dwell = belt_segments(touch_speed)
    script = ToolPathScript(segments)
    start_q = tuple(float(v) for v in ik(*segments[0][1]))
    scene = Scene(
        belt=True,
        belt_speed_m_s=BELT_SPEED_M_S,
        start_q=start_q,
        extras={"dwell": dwell, "press_target_n": PRESS_TARGET_N, "touch_x_m": BELT_X_M},
    )
    return Rung(
        id=rung_id,
        title=title,
        box=JOINT_TRAJECTORY,
        family=family,
        description=description,
        duration_s=round(script.duration_s + 0.2, 6),
        inputs=script,
        scene=scene,
        timing=timing,
        gates=(*_belt_gates(), *timing_gates(timing.rate_hz)),
        objective="contact_max_n",
        minimize=True,
        params={
            "touch_speed_m_s": touch_speed,
            "belt_speed_m_s": BELT_SPEED_M_S,
            "dwell_s": DWELL_S,
            "press_target_n": PRESS_TARGET_N,
            "aim_below_surface_mm": AIM_BELOW_M * 1000,
            "approach_mm": APPROACH_ABOVE_M * 1000,
            "cruise_mm": CRUISE_M * 1000,
            "host_rate_hz": timing.rate_hz,
        },
        order=order,
    )


def replan_targets(seed: int = REPLAN_TARGET_SEED) -> tuple[tuple[float, float], ...]:
    """R5's nine targets (the first trajectory's and eight replans'), uniform within +-20 deg of home,
    seeded."""
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-REPLAN_RANGE_RAD, REPLAN_RANGE_RAD, size=(REPLANS + 1, 2))
    return tuple((float(HOME_Q[0] + a), float(HOME_Q[1] + b)) for a, b in offsets)


def _joint_trajectory_ladder() -> tuple[Rung, ...]:
    box = JOINT_TRAJECTORY
    rack = HostTiming()
    r1_script = ToolPathScript(free_space_segments(0.8))
    r2_script = ToolPathScript(free_space_segments(0.3))
    tool_path_text = " -> ".join(f"({x:.2f}, {z:.2f})" for x, z in TOOL_PATH)
    free_space_params = {"tool_path_m": tool_path_text, "rest_s": REST_S}
    rungs = [
        Rung(
            id="R1.FreeSpace",
            title="Free-space motion",
            box=box,
            family="free_space",
            description="Three minimum-jerk tool moves through free space, 0.8 s each with 0.3 s rests. "
            "Tests plain tracking with the feed-forward the trajectory carries.",
            duration_s=round(r1_script.duration_s + 0.6, 6),
            inputs=r1_script,
            timing=rack,
            gates=(
                Gate("tracking_p95_deg", "<=", 1.0, "tracks a slow free-space path within a degree"),
                Gate(
                    "saturation_share",
                    "<=",
                    2,
                    "the motion fits the boards' torque (at the cap < 2 % of the time)",
                ),
                _DIVERGED,
                _FAULTS,
                *timing_gates(rack.rate_hz),
            ),
            objective="tracking_p95_deg",
            params={**free_space_params, "move_s": 0.8},
            order=1,
        ),
        Rung(
            id="R2.FastFreeSpace",
            title="Fast free-space motion",
            box=box,
            family="free_space",
            description="The same tool path at 0.3 s per move, near the boards' torque limits. "
            "Tests tracking when inertia, not gravity, dominates.",
            duration_s=round(r2_script.duration_s + 0.6, 6),
            inputs=r2_script,
            timing=rack,
            gates=(
                Gate("tracking_p95_deg", "<=", 3.0, "tracks a fast free-space path within 3 degrees"),
                Gate("saturation_share", "<=", 5, "the fast motion stays mostly inside the boards' torque"),
                _DIVERGED,
                *timing_gates(rack.rate_hz),
            ),
            objective="tracking_p95_deg",
            params={**free_space_params, "move_s": 0.3},
            order=2,
        ),
    ]
    for order, speed in ((3, 0.05), (4, 0.10), (5, 0.20)):
        rungs.append(
            _belt_rung(
                order,
                f"R3.BeltImpact@{speed:.2f}m/s",
                f"Belt impact at {speed:.2f} m/s",
                speed,
                rack,
                "contact",
                f"The tool comes down onto a belt moving at {BELT_SPEED_M_S} m/s, the last 20 mm at "
                f"{speed:.2f} m/s, aiming 10 mm below the surface, dwells 0.8 s and lifts. "
                "Tests the impact peak and a steady press.",
            )
        )
    push = Push(start_s=0.5, duration_s=0.4, force_n=(15.0, 0.0, 0.0))
    hold_script = HoldScript(
        1.8
    )  # WHY 0.2 s short of the run: the schedule then ends inside it, so its slip is measurable
    rungs.append(
        Rung(
            id="R4.SidewaysPush",
            title="Sideways push",
            box=box,
            family="disturbance",
            description="The arm holds home while a 15 N push along +x acts on the tool for 0.4 s. "
            "Tests stiffness under a disturbance and how fast and cleanly it recovers.",
            duration_s=2.0,
            inputs=hold_script,
            scene=Scene(pushes=(push,), extras={"push_force_n": 15.0}),
            timing=rack,
            gates=(
                Gate("push_recovery_s", "<=", 0.5, "back within 1 mm of the pre-push pose within 0.5 s"),
                Gate("push_overshoot_mm", "<=", 5, "springs back past the pose by no more than 5 mm"),
                _DIVERGED,
                *timing_gates(rack.rate_hz),
            ),
            objective="push_recovery_s",
            params={"push_n": 15.0, "push_start_s": 0.5, "push_duration_s": 0.4},
            order=6,
        )
    )
    targets = replan_targets()
    replan_end = REPLANS * REPLAN_PERIOD_S + REPLAN_MOVE_S
    rungs.append(
        Rung(
            id="R5.RapidReplan",
            title="Rapid replanning",
            box=box,
            family="replanning",
            description="A new 0.6 s trajectory every 0.15 s, eight times, each from the previous one's "
            "state at that instant to rest at a seeded target within 20 degrees of home. "
            "Tests taking a new plan mid-motion without a jump.",
            duration_s=round(replan_end + 0.7, 6),
            inputs=ReplanScript(targets),
            scene=Scene(extras={"targets_rad": targets, "replan_period_s": REPLAN_PERIOD_S}),
            timing=rack,
            gates=(
                Gate(
                    "replans_accepted",
                    ">=",
                    REPLANS,
                    "takes every new trajectory (a refusal fails, it is not skipped)",
                ),
                Gate(
                    "command_jump_max_deg",
                    "<=",
                    2,
                    "a replan never jumps the commanded position by more than 2 degrees",
                    applies="if_measured",
),
                Gate("replan_settle_s", "<=", 0.5, "settles within 0.5 s after the last plan ends"),
                _DIVERGED,
                *timing_gates(rack.rate_hz),
            ),
            objective="tracking_p95_deg",
            params={
                "replan_period_s": REPLAN_PERIOD_S,
                "replans": REPLANS,
                "move_s": REPLAN_MOVE_S,
                "target_range_deg": math.degrees(REPLAN_RANGE_RAD),
                "target_seed": REPLAN_TARGET_SEED,
            },
            order=7,
        )
    )
    stall_timing = HostTiming(jitter_s=0.0003, stalls=((0.8, STALL_S),))
    rungs.append(
        Rung(
            id="R6.HostStall",
            title="Host stall",
            box=box,
            family="timing",
            description="R1's path with 0.3 ms of wake jitter and one 0.5 s host stall at t = 0.8 s "
            "(the F09 hang); the "
            "boards' watchdog brakes the arm. Tests that the controller comes back on schedule.",
            duration_s=round(r1_script.duration_s + 1.2, 6),
            inputs=r1_script,
            timing=stall_timing,
            gates=(
                _DIVERGED,
                Gate(
                    "schedule_slip_ms",
                    "<=",
                    20,
                    "ends on the input's schedule after the stall (a slipping clock fails)",
                ),
                Gate("settle_time_s", "<=", 1.0, "settles within 1 s after the reference ends"),
            ),
            objective="tracking_p95_deg",
            params={
                **free_space_params,
                "move_s": 0.8,
                "jitter_ms": 0.3,
                "stall_start_s": 0.8,
                "stall_s": STALL_S,
            },
            order=8,
        )
    )
    for order, rate in ((9, 50.0), (10, 500.0)):
        rungs.append(
            _belt_rung(
                order,
                f"R7.BeltImpact@0.10m/s·host{rate:.0f}Hz",
                f"Belt impact, host at {rate:.0f} Hz",
                0.10,
                HostTiming(rate_hz=rate),
                "timing",
                f"Rung R3 at 0.10 m/s with the host loop at {rate:.0f} Hz instead of 100 Hz. "
                "Tests how the host rate moves the impact and the press.",
            )
        )
    rungs.extend(_amendment_three_rungs(rungs[1]))
    rungs.append(sync_variant(SYNC_BELT_SPEED_M_S))
    return tuple(rungs)


def _amendment_three_rungs(fast: Rung) -> list[Rung]:
    """R8 and R9 (CONTRACT amendment 3, §3.5): R2's path with model error, and at the other host rates.

    WHY R2's gates, timing gates at each rung's own rate (as R7 does it): every number in the layout sweep is
    then on a frozen rung, and the only thing that differs from R2 is the one condition the rung names.
    """
    r2_gates = tuple(g for g in fast.gates if g.metric not in ("missed_cycles", "latency_p99_ms"))
    rungs = [
        Rung(
            id="R8.FastFreeSpace·payload",
            title="Fast free-space motion with an unmodelled payload",
            box=fast.box,
            family="disturbance",
            description=f"R2's fast tool path with a {PAYLOAD_KG:g} kg payload on the tool that no "
            "feed-forward model knows (the trajectory's effort column and every model are the nominal "
            "arm's). "
            "Tests how much of a model error feedback closes, and where it is closed.",
            duration_s=fast.duration_s,
            inputs=fast.inputs,
            scene=Scene(payload_kg=PAYLOAD_KG),
            timing=fast.timing,
            gates=(*r2_gates, *timing_gates(fast.timing.rate_hz)),
            objective="tracking_p95_deg",
            params={**fast.params, "payload_kg": PAYLOAD_KG},
            order=11,
        )
    ]
    for order, rate in ((12, 50.0), (13, 500.0)):
        timing = HostTiming(rate_hz=rate)
        rungs.append(
            Rung(
                id=f"R9.FastFreeSpace·host{rate:.0f}Hz",
                title=f"Fast free-space motion, host at {rate:.0f} Hz",
                box=fast.box,
                family="timing",
                description=f"Rung R2 with the host loop at {rate:.0f} Hz instead of 100 Hz. "
                "Tests how the host rate moves tracking for a law closed on the host vs on the board.",
                duration_s=fast.duration_s,
                inputs=fast.inputs,
                timing=timing,
                gates=(*r2_gates, *timing_gates(rate)),
                objective="tracking_p95_deg",
                params={**fast.params, "host_rate_hz": rate},
                order=order,
            )
        )
    return rungs


# ── amendment 4 (§4.4): the sync rung and the variants ───────────────────────────────────────────────────


def sync_segments(belt_speed_m_s: float) -> tuple[tuple[Segment, ...], tuple[float, float]]:
    """R10's tool path and its sync window (the cruise): hold at (0.11, 0.20) for 0.3 s, a quintic blend to
    (0.16, 0.20) from rest to the belt speed v, cruise to (0.36, 0.20) at v, blend to (0.40, 0.20) from v to
    rest, hold 0.3 s. Each blend takes 2 · its distance / v (as the belt rungs' touch-down: the speed rises
    monotonically to v without overshooting it)."""
    speed = float(belt_speed_m_s)
    if not (math.isfinite(speed) and speed > 0.0):
        raise ValueError(f"belt_speed_m_s must be finite and > 0, got {belt_speed_m_s!r}")
    hold, start, end, rest = ((x, SYNC_Z_M) for x in SYNC_POINTS_X_M)
    blend_in_s = 2.0 * (start[0] - hold[0]) / speed
    cruise_s = (end[0] - start[0]) / speed
    blend_out_s = 2.0 * (rest[0] - end[0]) / speed
    segments: tuple[Segment, ...] = (
        ("hold", hold, SYNC_HOLD_S),
        ("blend", hold, start, 0.0, speed, blend_in_s),
        ("cruise", start, end, cruise_s),
        ("blend", end, rest, speed, 0.0, blend_out_s),
        ("hold", rest, SYNC_HOLD_S),
    )
    window_start = SYNC_HOLD_S + blend_in_s
    return segments, (window_start, window_start + cruise_s)


def sync_variant(belt_speed_m_s: float) -> Rung:
    """`R10.BeltSync@<v>m/s` (order 14, family sync): the tool keeps pace with a point riding the belt.

    WHY the belt off: the rung measures timing against a moving target, and contact would add its own error;
    the "item" is the reference's own tool point over the cruise, 50 mm above the belt top.
    """
    speed = float(belt_speed_m_s)
    segments, window = sync_segments(speed)
    script = ToolPathScript(segments)
    rack = HostTiming()
    return Rung(
        id=f"R10.BeltSync@{speed:.2f}m/s",
        title=f"Belt sync at {speed:.2f} m/s",
        box=JOINT_TRAJECTORY,
        family="sync",
        description=f"A virtual item rides the belt along +x at {speed:.2f} m/s, 50 mm above its top; the "
        "tool blends up to its speed, keeps pace with it for 200 mm and slows to rest (the belt itself is "
        "off). Tests how closely the tool keeps time with a moving target.",
        duration_s=round(script.duration_s + 0.2, 6),
        inputs=script,
        scene=Scene(start_q=tuple(float(v) for v in ik(*segments[0][1]))),
        timing=rack,
        gates=(
            Gate(
                "sync_error_p95_mm",
                "<=",
                SYNC_TOLERANCE_MM,
                f"the tool stays within {SYNC_TOLERANCE_MM:g} mm of a point riding the belt",
            ),
            Gate("saturation_share", "<=", 5, "the sync motion stays mostly inside the boards' torque"),
            _DIVERGED,
            *timing_gates(rack.rate_hz),
        ),
        objective="sync_error_p95_mm",
        minimize=True,
        params={
            "belt_speed_m_s": speed,
            "sync_start_s": window[0],
            "sync_end_s": window[1],
            "item_z_m": SYNC_Z_M,
            "cruise_mm": 1000.0 * (SYNC_POINTS_X_M[2] - SYNC_POINTS_X_M[1]),
            "host_rate_hz": rack.rate_hz,
        },
        order=14,
    )


def rate_variant(rung: Rung, rate_hz: float) -> Rung:
    """`rung` with the host at `rate_hz` (every other timing field kept): id `<id>·host<rate>Hz`, its two
    timing gates replaced by `timing_gates(rate_hz)`, `params["host_rate_hz"] = rate_hz`. Not a ladder
    rung."""
    rate = float(rate_hz)
    if not (math.isfinite(rate) and rate > 0.0):
        raise ValueError(f"rate_hz must be finite and > 0, got {rate_hz!r}")
    return dataclasses.replace(
        rung,
        id=f"{rung.id}·host{rate:g}Hz",
        title=f"{rung.title}, host at {rate:g} Hz",
        timing=dataclasses.replace(rung.timing, rate_hz=rate),
        gates=(*(g for g in rung.gates if g.metric not in TIMING_METRICS), *timing_gates(rate)),
        params={**rung.params, "host_rate_hz": rate},
    )


def _respeed(script: Any, move_s: float, rung: Rung) -> Any:
    """The rung's input script with every move taking `move_s`: a script that can re-time itself (the UR5e
    path's `respeed`) does; a free-space tool path (R1/R2's) is rebuilt at the new move time."""
    if callable(getattr(script, "respeed", None)):
        return script.respeed(move_s)
    base_move = rung.params.get("move_s")
    if isinstance(script, ToolPathScript) and script.segments == free_space_segments(float(base_move)):
        return ToolPathScript(free_space_segments(move_s))
    raise ValueError(
        f"speed_variant re-times free-space paths (R1, R2, U1, U2); {rung.id}'s script is not one"
    )


def _home(plant: str) -> np.ndarray:
    """The home joints of `plant`, any arm in the registry (§5.1)."""
    if plant == "planar_arm":
        return np.asarray(HOME_Q, dtype=float)
    from robot_controllers.bench.arms import arm

    return np.asarray(arm(plant).home_q, dtype=float)


def _tool_positions(plant: str, q: np.ndarray) -> np.ndarray:
    """(N, 3) tool positions of joint samples on `plant`'s own kinematics (the registry's, §5.1)."""
    if plant == "planar_arm":
        from robot_controllers.bench.arm import tool_position

        return tool_position(q)
    from robot_controllers.bench.arms import arm

    return np.asarray(arm(plant).tool_position(q))


def speed_variant(rung: Rung, move_s: float) -> Rung:
    """The same free-space path at `move_s` per move (R1, R2, U1, U2): id `<id>·move<ms>ms`, duration =
    script + 0.6 s (as R1/R2), the base rung's gates, and params `move_s`, `peak_joint_speed_deg_s` and
    `peak_tool_speed_m_s` measured on the script's own reference. Not a ladder rung."""
    if "move_s" not in rung.params:
        raise ValueError(f"{rung.id} has no params['move_s'] to vary")
    move = float(move_s)
    if not (math.isfinite(move) and move > 0.0):
        raise ValueError(f"move_s must be finite and > 0, got {move_s!r}")
    script = _respeed(rung.inputs, move, rung)
    start = np.asarray(rung.scene.start_q, dtype=float) if rung.scene.start_q else _home(rung.plant)
    trajectory = script(start)[0][1]
    times = np.asarray(trajectory.t, dtype=float)
    joint_speed = float(np.max(np.abs(np.asarray(trajectory.velocities))))
    tool = _tool_positions(rung.plant, np.asarray(trajectory.positions))
    tool_speed = float(np.max(np.linalg.norm(np.gradient(tool, times, axis=0), axis=1)))
    return dataclasses.replace(
        rung,
        id=f"{rung.id}·move{move * 1000:g}ms",
        title=f"{rung.title}, {move:g} s per move",
        duration_s=round(script.duration_s + 0.6, 6),
        inputs=script,
        params={
            **rung.params,
            "move_s": move,
            "peak_joint_speed_deg_s": math.degrees(joint_speed),
            "peak_tool_speed_m_s": tool_speed,
        },
    )


def _ms_text(seconds: float | tuple[float, ...]) -> str:
    if np.ndim(seconds) == 0:
        return f"{float(seconds) * 1e3:g}"
    return "/".join(f"{float(v) * 1e3:g}" for v in seconds)


def bus_variant(
    rung: Rung, command_delay_s: float | tuple[float, ...], jitter_s: float = 0.0
) -> Rung:
    """`rung` on another bus: `command_delay_s` (one value, or one per plant joint) and
    `command_delay_jitter_s = jitter_s`. id `<id>·bus<ms>ms[±<ms>ms]`, a tuple written `<ms>/<ms>/...`. Not a
    ladder rung."""
    if np.ndim(command_delay_s) == 0:
        delay: float | tuple[float, ...] = float(command_delay_s)
    else:
        delay = tuple(float(v) for v in command_delay_s)
    jitter = float(jitter_s)
    delay_ms = _ms_text(delay)
    suffix = f"·bus{delay_ms}ms" + (f"±{jitter * 1e3:g}ms" if jitter else "")
    timing = dataclasses.replace(rung.timing, command_delay_s=delay, command_delay_jitter_s=jitter)
    return dataclasses.replace(
        rung,
        id=f"{rung.id}{suffix}",
        title=f"{rung.title}, bus {delay_ms} ms" + (f" ± {jitter * 1e3:g} ms" if jitter else ""),
        timing=timing,
        params={**rung.params, "command_delay_ms": delay_ms, "command_delay_jitter_ms": jitter * 1e3},
    )


def _move_to_sequence(target: np.ndarray, name: str) -> JointCommandSequence:
    caps = BoardConfig().torque_cap_nm
    gravity = nominal_dynamics(target, np.zeros(2), np.zeros(2))
    move = JointCommandStep(
        name="move",
        commands=tuple(
            JointCommand.to(
                joint, float(target[j]), speed=1.5, max_effort=caps[j], feedforward_effort=float(gravity[j])
            )
            for j, joint in enumerate(JOINT_NAMES)
        ),
        # Reached's own speed (0.01 rad/s) and dwell (0.05 s) defaults: measured to fire through the board's
        # count-derivative velocity (0.83 s here, vs 0.81 s at a looser 0.05 rad/s)
        done_when=(Reached(joints=JOINT_NAMES, tolerance=math.radians(1.0)),),
        timeout=3.0,
    )
    hold = JointCommandStep(
        name="hold",
        commands=tuple(
            JointCommand.hold(joint, max_effort=caps[j], feedforward_effort=float(gravity[j]))
            for j, joint in enumerate(JOINT_NAMES)
        ),
        done_when=(Elapsed(duration=0.5),),
        timeout=2.0,
    )
    return JointCommandSequence(joint_names=JOINT_NAMES, steps=(move, hold), name=name)


def _seek_sequence() -> JointCommandSequence:
    caps = BoardConfig().torque_cap_nm
    gravity = nominal_dynamics(np.asarray(HOME_Q), np.zeros(2), np.zeros(2))
    stop = ARM.elbow_range_rad[0]
    seek = JointCommandStep(
        name="seek_elbow_stop",
        commands=(
            JointCommand.hold("shoulder", max_effort=caps[0], feedforward_effort=float(gravity[0])),
            # WHY 1.2 rad: past the stop (45 deg = 0.79 rad from home), so the setpoint keeps leading once it
            # is hit
            JointCommand.move_by("elbow", -1.2, speed=0.5, max_effort=3.0),
        ),
        done_when=(Stall(joint="elbow"),),
        timeout=4.0,
        define_position={"elbow": stop},
    )
    return JointCommandSequence(joint_names=JOINT_NAMES, steps=(seek,), name="S2.SeekHardStop")


def _sequence_ladder() -> tuple[Rung, ...]:
    box = JOINT_COMMAND_SEQUENCE
    rack = HostTiming()
    target = ik(0.42, 0.20)
    move_to = SequenceScript(_move_to_sequence(target, "S1.MoveTo"))
    seek = SequenceScript(_seek_sequence())
    move_params = {
        "target_deg": f"({math.degrees(target[0]):.2f}, {math.degrees(target[1]):.2f})",
        "speed_rad_s": 1.5,
        "reached_tolerance_deg": 1.0,
        "hold_s": 0.5,
    }
    return (
        Rung(
            id="S1.MoveTo",
            title="Move to a pose",
            box=box,
            family="sequence",
            description="Moves both joints to a pose at 1.5 rad/s with the boards' full torque until Reached "
            "(1 degree), then holds 0.5 s. Tests that a step finishes on its done-when.",
            duration_s=2.5,
            inputs=move_to,
            timing=rack,
            gates=(
                Gate("steps_done", ">=", len(move_to.sequence), "every step finished on its done-when"),
                Gate("reached_error_deg", "<=", 1.0, "within a degree of the pose when Reached fired"),
                _FAULTS,
                *timing_gates(rack.rate_hz),
            ),
            objective="sequence_time_s",
            params=move_params,
            order=1,
        ),
        Rung(
            id="S2.SeekHardStop",
            title="Seek a hard stop",
            box=box,
            family="sequence",
            description="Holds the shoulder and drives the elbow toward its hard stop at 0.5 rad/s "
            "under 3 N·m "
            "until Stall, then defines that position. Tests a homing step that ends on a contact.",
            duration_s=3.0,
            inputs=seek,
            timing=rack,
            gates=(
                Gate(
                    "steps_done", ">=", len(seek.sequence), "the stall was detected and the position defined"
                ),
                _FAULTS,
                *timing_gates(rack.rate_hz),
            ),
            objective="sequence_time_s",
            params={
                "speed_rad_s": 0.5,
                "max_effort_nm": 3.0,
                "hard_stop_deg": math.degrees(ARM.elbow_range_rad[0]),
            },
            order=2,
        ),
        Rung(
            id="S3.HostStall",
            title="Host stall mid-move",
            box=box,
            family="sequence",
            description="S1 with a 0.5 s host stall at t = 0.3 s, mid-move; the boards' watchdog brakes. "
            "Tests that the sequence still finishes without running away.",
            duration_s=3.0,
            inputs=move_to,
            timing=HostTiming(stalls=((0.3, STALL_S),)),
            gates=(
                Gate("steps_done", ">=", len(move_to.sequence), "every step finished despite the stall"),
                _DIVERGED,
            ),
            objective="sequence_time_s",
            params={**move_params, "stall_start_s": 0.3, "stall_s": STALL_S},
            order=3,
        ),
    )


@functools.lru_cache(maxsize=8)
def ladder(box: Box) -> tuple[Rung, ...]:
    """Every rung of `box`, simplest first. A box with no rungs yet (PoseTrajectory, ServoCommand) gives
    ()."""
    if box == JOINT_TRAJECTORY:
        return _joint_trajectory_ladder()
    if box == JOINT_COMMAND_SEQUENCE:
        return _sequence_ladder()
    return ()


def rung(rung_id: str) -> Rung:
    """One rung by id, from any box."""
    for box in (JOINT_TRAJECTORY, JOINT_COMMAND_SEQUENCE):
        for candidate in ladder(box):
            if candidate.id == rung_id:
                return candidate
    raise KeyError(f"no rung {rung_id!r}")
