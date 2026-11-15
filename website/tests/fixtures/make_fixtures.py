"""Write the site's fixture results: synthetic, plausible results-1 documents for the two boxes with a ladder.

    env -u VIRTUAL_ENV uv run --frozen --project /home/bam/robot_controllers --extra test python /home/bam/robot_controllers/website/tests/fixtures/make_fixtures.py

WHY a fixture: the site lane builds while the bench, metrics and integration lanes are still writing the real
ledger. These documents follow CONTRACT §5 (schema, rung records, RunResult rows, the trace keys and their rates)
so every page and plot path is exercised, and they say they are synthetic (`generated.fixture = true`, which the
site prints as a banner). Numbers are invented to be plausible for the planar arm, not measured.

The traces are built first and the headline metrics are computed from them, so a plot and its number agree.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from robot_controllers.bench.types import Gate, judge
from robot_controllers.metrics.names import METRICS
from robot_controllers.registry import JOINT_COMMAND_SEQUENCE, JOINT_TRAJECTORY, in_box

OUT = Path(__file__).resolve().parent / "results"
SCHEMA = "robot-controllers/results-1"
JOINTS = ("shoulder", "elbow")
TRACE_HZ = 200.0
CONTACT_HZ = 1000.0

# ── the planar arm (CONTRACT §3), only what the fixture needs ───────────────────────────────────────────────
LINK1_M, LINK2_M, BASE_Z_M = 0.30, 0.25, 0.50
BELT_TOP_Z_M, TOOL_RADIUS_M = 0.15, 0.015


def forward_kinematics(shoulder_rad: np.ndarray, elbow_rad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tool (x, z) in metres. q = 0 points the link along +x; positive rotates toward +z (fixture convention)."""
    x = LINK1_M * np.cos(shoulder_rad) + LINK2_M * np.cos(shoulder_rad + elbow_rad)
    z = BASE_Z_M + LINK1_M * np.sin(shoulder_rad) + LINK2_M * np.sin(shoulder_rad + elbow_rad)
    return x, z


def inverse_kinematics(x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Elbow-up analytic IK for the fixture convention."""
    dz = z - BASE_Z_M
    cos_elbow = np.clip((x**2 + dz**2 - LINK1_M**2 - LINK2_M**2) / (2 * LINK1_M * LINK2_M), -1.0, 1.0)
    elbow = np.arccos(cos_elbow)  # positive elbow keeps the elbow above the tool line for a downward reach
    shoulder = np.arctan2(dz, x) - np.arctan2(LINK2_M * np.sin(elbow), LINK1_M + LINK2_M * np.cos(elbow))
    return shoulder, elbow


def min_jerk(fraction: np.ndarray) -> np.ndarray:
    s = np.clip(fraction, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def tool_path(times: np.ndarray, waypoints: list[tuple[float, float]], move_s: float, rest_s: float,
              start_rest_s: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """Min-jerk straight-line moves through tool waypoints, resting between them."""
    x = np.full_like(times, waypoints[0][0])
    z = np.full_like(times, waypoints[0][1])
    start = start_rest_s
    for (x0, z0), (x1, z1) in zip(waypoints[:-1], waypoints[1:], strict=True):
        inside = times >= start
        blend = min_jerk((times - start) / move_s)
        x = np.where(inside, x0 + (x1 - x0) * blend, x)
        z = np.where(inside, z0 + (z1 - z0) * blend, z)
        start += move_s + rest_s
    return x, z


# ── the ladder (CONTRACT §3, transcribed for the fixture; the bench lane's rungs.py is the real one) ───────

TIMING_WHY_MISSED = "the host kept its period: no cycle was skipped"
TIMING_WHY_LATENCY = "sensor-to-command latency stays under one host period"


def timing_gates(rate_hz: float) -> list[Gate]:
    return [Gate("missed_cycles", "==", 0, TIMING_WHY_MISSED),
            Gate("latency_p99_ms", "<=", round(1000.0 / rate_hz, 4), TIMING_WHY_LATENCY)]


def belt_gates() -> list[Gate]:
    return [
        Gate("contact_peak_n", "<=", 60, "the impact transient stays under 60 N"),
        Gate("dwell_bounces", "==", 0, "the tool stays on the belt during the dwell"),
        Gate("dwell_force_error_n", "<=", 10, "the dwell force is within 10 N of the 10 N target press"),
        Gate("dwell_contact_fraction", ">=", 95, "in contact for at least 95 % of the dwell"),
        Gate("diverged", "==", 0, "nothing went non-finite or out of bounds"),
    ]


def timing(rate_hz: float = 100.0, *, jitter_s: float = 0.0, stalls: tuple = ()) -> dict:
    return {"rate_hz": rate_hz, "command_delay_s": 0.0055, "reply_delay_s": 0.0015, "jitter_s": jitter_s,
            "stalls": [list(s) for s in stalls], "include_compute": True}


def scene(*, belt: bool = False, pushes: tuple = (), start_q=None) -> dict:
    return {"belt": belt, "belt_speed_m_s": 0.25, "contact_stiffness_n_m": 100000.0, "contact_damping_n_s_m": 200.0,
            "pushes": [{"start_s": p[0], "duration_s": p[1], "force_n": list(p[2])} for p in pushes],
            "payload_kg": 0.0, "start_q": start_q, "extras": {}}


def belt_rung(order: int, rung_id: str, title: str, touch_speed: float, rate_hz: float = 100.0) -> dict:
    return dict(
        id=rung_id, title=title, family="contact", order=order,
        description=(f"The tool comes straight down onto a belt moving 0.25 m/s at {touch_speed:g} m/s, aiming 10 mm "
                     "below the surface, dwells 0.8 s and lifts. It tests the impact transient and how steadily the "
                     "controller presses on a moving surface."),
        duration_s=2.2, params={"touch_speed_m_s": touch_speed, "dwell_s": 0.8, "press_target_n": 10.0,
                                "approach_mm": 40.0, "aim_below_mm": 10.0},
        timing=timing(rate_hz), scene=scene(belt=True), gates=belt_gates() + timing_gates(rate_hz),
        objective="contact_max_n", minimize=True)


JT_RUNGS = [
    dict(id="R1.FreeSpace", title="Free-space motion", family="free_space", order=1,
         description="Three min-jerk moves between tool points, 0.8 s each with 0.3 s rests. It tests plain "
                     "tracking with gravity and inertia feed-forward.",
         duration_s=3.5, params={"move_s": 0.8, "rest_s": 0.3}, timing=timing(), scene=scene(),
         gates=[Gate("tracking_p95_deg", "<=", 1.0, "tracks a slow path within a degree"),
                Gate("saturation_share", "<=", 2, "the torque cap is hit at most 2 % of the time"),
                Gate("diverged", "==", 0, "nothing went non-finite or out of bounds"),
                Gate("faults", "==", 0, "no board fault and no controller abort")] + timing_gates(100),
         objective="tracking_p95_deg", minimize=True),
    dict(id="R2.FastFreeSpace", title="Fast free-space motion", family="free_space", order=2,
         description="The same path with 0.3 s moves. It tests tracking near the torque cap.",
         duration_s=2.0, params={"move_s": 0.3, "rest_s": 0.3}, timing=timing(), scene=scene(),
         gates=[Gate("tracking_p95_deg", "<=", 3.0, "tracks a fast path within three degrees"),
                Gate("saturation_share", "<=", 5, "the torque cap is hit at most 5 % of the time"),
                Gate("diverged", "==", 0, "nothing went non-finite or out of bounds")] + timing_gates(100),
         objective="tracking_p95_deg", minimize=True),
    belt_rung(3, "R3.BeltImpact@0.05m/s", "Belt impact at 0.05 m/s", 0.05),
    belt_rung(4, "R3.BeltImpact@0.10m/s", "Belt impact at 0.10 m/s", 0.10),
    belt_rung(5, "R3.BeltImpact@0.20m/s", "Belt impact at 0.20 m/s", 0.20),
    dict(id="R4.SidewaysPush", title="Sideways push", family="disturbance", order=6,
         description="The arm holds at home and is pushed +15 N along x for 0.4 s. It tests how fast and how cleanly "
                     "it recovers.",
         duration_s=2.0, params={"push_n": 15.0, "push_s": 0.4}, timing=timing(),
         scene=scene(pushes=((0.5, 0.4, (15.0, 0.0, 0.0)),)),
         gates=[Gate("push_recovery_s", "<=", 0.5, "back within 1 mm half a second after release"),
                Gate("push_overshoot_mm", "<=", 5, "overshoots the hold by at most 5 mm"),
                Gate("diverged", "==", 0, "nothing went non-finite or out of bounds")] + timing_gates(100),
         objective="push_recovery_s", minimize=True),
    dict(id="R5.RapidReplan", title="Rapid replanning", family="replanning", order=7,
         description="A new trajectory every 0.15 s, eight times, each from the previous one's state to a seeded "
                     "target within 20 deg of home. It tests replacing the input while moving.",
         duration_s=2.2, params={"replan_period_s": 0.15, "replans": 8, "target_range_deg": 20.0},
         timing=timing(), scene=scene(),
         gates=[Gate("replans_accepted", ">=", 8, "takes every new trajectory"),
                Gate("command_jump_max_deg", "<=", 2, "no command step larger than 2 deg at a replan"),
                Gate("replan_settle_s", "<=", 0.5, "settles within half a second of the last trajectory"),
                Gate("diverged", "==", 0, "nothing went non-finite or out of bounds")] + timing_gates(100),
         objective="tracking_p95_deg", minimize=True),
    dict(id="R6.HostStall", title="Host stall", family="timing", order=8,
         description="R1's path with 0.3 ms jitter and one 0.5 s host stall at t = 0.8 s, so the board watchdog "
                     "brakes. It tests staying on schedule and recovering after a hang.",
         duration_s=3.5, params={"stall_s": 0.5, "stall_at_s": 0.8}, timing=timing(jitter_s=0.0003, stalls=((0.8, 0.5),)),
         scene=scene(),
         gates=[Gate("diverged", "==", 0, "nothing went non-finite or out of bounds"),
                Gate("schedule_slip_ms", "<=", 20, "the controller's clock ends within 20 ms of the schedule"),
                Gate("settle_time_s", "<=", 1.0, "settles within a second after the reference ends")],
         objective="tracking_p95_deg", minimize=True),
    belt_rung(9, "R7.BeltImpact@0.10m/s·host50Hz", "Belt impact, host at 50 Hz", 0.10, rate_hz=50.0),
    belt_rung(10, "R7.BeltImpact@0.10m/s·host500Hz", "Belt impact, host at 500 Hz", 0.10, rate_hz=500.0),
]

JCS_RUNGS = [
    dict(id="S1.MoveTo", title="Move to a pose", family="sequence", order=1,
         description="Both joints move to a pose at 1.5 rad/s until Reached (1 deg), then hold 0.5 s. It tests a "
                     "plain move-and-hold sequence.",
         duration_s=2.5, params={"speed_rad_s": 1.5, "tolerance_rad": 0.01745}, timing=timing(), scene=scene(),
         gates=[Gate("steps_done", ">=", 2, "every step finished on its done-when"),
                Gate("reached_error_deg", "<=", 1.0, "ends within a degree of the target"),
                Gate("faults", "==", 0, "no board fault and no controller abort")],
         objective="sequence_time_s", minimize=True),
    dict(id="S2.SeekHardStop", title="Seek a hard stop", family="sequence", order=2,
         description="The elbow moves toward its joint limit at 0.5 rad/s under 3 N·m until Stall, then defines "
                     "that position. It tests homing against a hard stop.",
         duration_s=3.0, params={"speed_rad_s": 0.5, "max_effort_nm": 3.0}, timing=timing(), scene=scene(),
         gates=[Gate("steps_done", ">=", 1, "every step finished on its done-when"),
                Gate("faults", "==", 0, "no board fault and no controller abort")],
         objective="sequence_time_s", minimize=True),
    dict(id="S3.HostStall", title="Host stall mid-move", family="sequence", order=3,
         description="S1 with a 0.5 s host stall at t = 0.3 s. It tests that the sequence survives the watchdog "
                     "brake and still finishes.",
         duration_s=3.0, params={"stall_s": 0.5, "stall_at_s": 0.3}, timing=timing(stalls=((0.3, 0.5),)),
         scene=scene(),
         gates=[Gate("steps_done", ">=", 2, "every step finished on its done-when"),
                Gate("diverged", "==", 0, "nothing went non-finite or out of bounds")],
         objective="sequence_time_s", minimize=True),
]

# ── controller profiles: the invented headline numbers each run is built to hit ─────────────────────────────

PROFILES = {
    # compute p50/p99 in µs; fields = non-NaN ServoCommand fields per joint
    "joint_trajectory_controller": dict(compute=(8.1, 11.2), fields=3, lag=0.010, soft=False, time_driven=True,
                                        replannable=True),
    # the same law compiled (CONTRACT amendment 2): identical tracking, less compute
    "joint_trajectory_controller_cpp": dict(compute=(6.3, 8.9), fields=3, lag=0.010, soft=False, time_driven=True,
                                            replannable=True),
    "tape_feeder": dict(compute=(5.2, 7.4), fields=3, lag=0.010, soft=False, time_driven=False, replannable=False),
    "joint_impedance_board": dict(compute=(11.6, 15.9), fields=5, lag=0.022, soft=True, time_driven=True,
                                  replannable=True),
    "joint_impedance_host": dict(compute=(12.4, 17.8), fields=3, lag=0.026, soft=True, time_driven=True,
                                 replannable=True),
    "joint_impedance_host_spring": dict(compute=(12.2, 16.7), fields=4, lag=0.024, soft=True, time_driven=True,
                                        replannable=True),
}

# tracking p95 (deg) per rung, and the other numbers the gates read (missing = a healthy default)
HEADLINE = {
    "joint_trajectory_controller": {
        "R1.FreeSpace": dict(track=0.38), "R2.FastFreeSpace": dict(track=1.62, saturation=1.8),
        "R3.BeltImpact@0.05m/s": dict(track=0.62, peak=48.2, dwell=31.0),
        "R3.BeltImpact@0.10m/s": dict(track=0.71, peak=71.5, dwell=31.6),
        "R3.BeltImpact@0.20m/s": dict(track=0.83, peak=118.0, dwell=32.2),
        "R4.SidewaysPush": dict(track=0.41, recovery=0.14, overshoot=0.6, deflection=2.1),
        "R5.RapidReplan": dict(track=0.74, replans=8, jump=0.9, settle=0.31),
        "R6.HostStall": dict(track=6.8, slip=0.0, settle=0.42),
        "R7.BeltImpact@0.10m/s·host50Hz": dict(track=0.92, peak=73.8, dwell=31.9),
        "R7.BeltImpact@0.10m/s·host500Hz": dict(track=0.55, peak=70.2, dwell=31.4),
    },
    "tape_feeder": {
        "R1.FreeSpace": dict(track=0.39), "R2.FastFreeSpace": dict(track=1.60, saturation=1.7),
        "R3.BeltImpact@0.05m/s": dict(track=0.63, peak=49.0, dwell=31.4),
        "R3.BeltImpact@0.10m/s": dict(track=0.72, peak=72.3, dwell=31.9),
        "R3.BeltImpact@0.20m/s": dict(track=0.84, peak=119.0, dwell=32.5),
        "R4.SidewaysPush": dict(track=0.41, recovery=0.14, overshoot=0.6, deflection=2.1),
        "R5.RapidReplan": dict(track=9.8, replans=0, jump=0.4, settle=None),
        "R6.HostStall": dict(track=11.9, slip=500.0, settle=0.38),
        "R7.BeltImpact@0.10m/s·host50Hz": dict(track=0.95, peak=74.1, dwell=32.0),
        "R7.BeltImpact@0.10m/s·host500Hz": dict(track=0.56, peak=70.9, dwell=31.5),
    },
    "joint_impedance_board": {
        "R1.FreeSpace": dict(track=0.86), "R2.FastFreeSpace": dict(track=3.38, saturation=0.6),
        "R3.BeltImpact@0.05m/s": dict(track=1.4, peak=18.6, dwell=9.1),
        "R3.BeltImpact@0.10m/s": dict(track=1.5, peak=27.9, dwell=9.0),
        "R3.BeltImpact@0.20m/s": dict(track=1.7, peak=52.3, dwell=8.7),
        "R4.SidewaysPush": dict(track=5.2, recovery=0.33, overshoot=3.1, deflection=27.0),
        "R5.RapidReplan": dict(track=1.21, replans=8, jump=0.9, settle=0.42),
        "R6.HostStall": dict(track=7.4, slip=0.0, settle=0.61),
        "R7.BeltImpact@0.10m/s·host50Hz": dict(track=1.9, peak=29.4, dwell=8.9),
        "R7.BeltImpact@0.10m/s·host500Hz": dict(track=1.3, peak=27.5, dwell=9.1),
    },
    "joint_impedance_host": {
        "R1.FreeSpace": dict(track=0.97, saturation=0.4), "R2.FastFreeSpace": dict(track=4.1, saturation=2.4),
        "R3.BeltImpact@0.05m/s": dict(track=1.6, peak=21.3, dwell=8.2, fraction=99.2),
        "R3.BeltImpact@0.10m/s": dict(track=1.8, peak=34.6, dwell=7.6, fraction=97.8),
        "R3.BeltImpact@0.20m/s": dict(track=2.3, peak=41.0, dwell=5.4, bounces=3, fraction=88.5),
        "R4.SidewaysPush": dict(track=6.1, recovery=0.71, overshoot=6.4, deflection=31.0),
        "R5.RapidReplan": dict(track=1.35, replans=8, jump=1.1, settle=0.47),
        "R6.HostStall": dict(track=9.6, slip=0.0, settle=1.34),
        "R7.BeltImpact@0.10m/s·host50Hz": dict(track=3.1, peak=46.0, dwell=6.1, bounces=4, fraction=81.0),
        "R7.BeltImpact@0.10m/s·host500Hz": dict(track=1.1, peak=28.8, dwell=8.9),
    },
    "joint_impedance_host_spring": {
        "R1.FreeSpace": dict(track=0.91, saturation=0.2), "R2.FastFreeSpace": dict(track=3.52, saturation=1.1),
        "R3.BeltImpact@0.05m/s": dict(track=1.5, peak=19.4, dwell=8.8),
        "R3.BeltImpact@0.10m/s": dict(track=1.6, peak=29.1, dwell=8.7),
        "R3.BeltImpact@0.20m/s": dict(track=1.9, peak=48.7, dwell=8.6),
        "R4.SidewaysPush": dict(track=5.6, recovery=0.38, overshoot=3.8, deflection=29.0),
        "R5.RapidReplan": dict(track=1.28, replans=8, jump=1.0, settle=0.44),
        "R6.HostStall": dict(track=7.9, slip=0.0, settle=0.72),
        "R7.BeltImpact@0.10m/s·host50Hz": dict(track=2.2, peak=31.8, dwell=8.5),
        "R7.BeltImpact@0.10m/s·host500Hz": dict(track=1.4, peak=28.2, dwell=8.8),
    },
}

HEADLINE["joint_trajectory_controller_cpp"] = HEADLINE["joint_trajectory_controller"]  # one law, two implementations

JCS_HEADLINE = {
    "S1.MoveTo": dict(steps=(2, 2), reached=0.42, time_s=1.38),
    "S2.SeekHardStop": dict(steps=(1, 1), reached=None, time_s=2.1),
    "S3.HostStall": dict(steps=(2, 2), reached=0.47, time_s=1.91),
}


# ── helpers ─────────────────────────────────────────────────────────────────────────────────────────────────


def sig4(value: float | None) -> float | None:
    """Round to 4 significant digits (CONTRACT §5); None and non-finite stay None."""
    if value is None or not math.isfinite(value):
        return None
    if value == 0:
        return 0.0
    return float(f"{value:.4g}")


def series(values: np.ndarray) -> list[float | None]:
    return [sig4(float(v)) for v in values]


def compute_trace(rng: np.random.Generator, profile: dict, duration_s: float, rate_hz: float,
                  stalls: list) -> tuple[np.ndarray, np.ndarray]:
    tick_t = np.arange(0.0, duration_s, 1.0 / rate_hz)
    for start, length in stalls:
        tick_t = tick_t[(tick_t < start) | (tick_t >= start + length)]
    p50, p99 = profile["compute"]
    # lognormal with the requested median and 99th percentile
    sigma = math.log(p99 / p50) / 2.326
    compute = p50 * np.exp(rng.normal(0.0, sigma, len(tick_t)))
    return tick_t, compute


def contact_time_base(duration_s: float, touch_s: float | None) -> np.ndarray:
    """200 Hz everywhere, 1 kHz inside [touch - 20 ms, touch + 80 ms] (CONTRACT §5)."""
    coarse = np.arange(0.0, duration_s, 1.0 / TRACE_HZ)
    if touch_s is None:
        return coarse
    fine = np.arange(touch_s - 0.020, touch_s + 0.080, 1.0 / CONTACT_HZ)
    outside = coarse[(coarse < touch_s - 0.020) | (coarse >= touch_s + 0.080)]
    return np.sort(np.concatenate([outside, fine]))


def p95(values: np.ndarray) -> float:
    return float(np.percentile(np.abs(values), 95))


# ── one run ─────────────────────────────────────────────────────────────────────────────────────────────────


def jt_run(controller_name: str, rung: dict, seed: int) -> dict:
    # WHY not hash(): str hashes are salted per process, and the fixture must come out byte-identical every run
    rng = np.random.default_rng([seed, sum(map(ord, controller_name)), sum(map(ord, rung["id"]))])
    profile, target = PROFILES[controller_name], HEADLINE[controller_name][rung["id"]]
    duration = rung["duration_s"]
    times = np.arange(0.0, duration, 1.0 / TRACE_HZ)
    home = (0.30, 0.25)
    events: list[dict] = []
    touch_s = None
    lift_s = None

    if rung["family"] in ("free_space", "timing"):
        move = rung["params"]["move_s"] if "move_s" in rung["params"] else 0.8
        rest = rung["params"].get("rest_s", 0.3)
        ref_x, ref_z = tool_path(times, [home, (0.42, 0.20), (0.25, 0.32), home], move, rest)
    elif rung["family"] == "contact":
        speed = rung["params"]["touch_speed_m_s"]
        start_z = BELT_TOP_Z_M + TOOL_RADIUS_M + 0.040
        aim_z = BELT_TOP_Z_M + TOOL_RADIUS_M - 0.010
        approach_s = 0.25 + 0.03 / max(speed, 1e-3) * 0.5
        constant_s = 0.020 / speed
        down_start = 0.3
        ref_z = np.full_like(times, start_z)
        # accelerate over the first 30 mm, then constant speed for the last 20 mm
        accel_end = down_start + approach_s
        in_accel = (times >= down_start) & (times < accel_end)
        fraction = (times - down_start) / approach_s
        ref_z = np.where(in_accel, start_z - 0.030 * fraction**2, ref_z)
        in_const = (times >= accel_end) & (times < accel_end + constant_s)
        ref_z = np.where(in_const, start_z - 0.030 - speed * (times - accel_end), ref_z)
        dwell_start = accel_end + constant_s
        lift_s = dwell_start + rung["params"]["dwell_s"]
        ref_z = np.where((times >= dwell_start) & (times < lift_s), aim_z, ref_z)
        ref_z = np.where(times >= lift_s, aim_z + (start_z - aim_z) * min_jerk((times - lift_s) / 0.4), ref_z)
        ref_x = np.full_like(times, 0.30)
        surface_z = BELT_TOP_Z_M + TOOL_RADIUS_M
        lag_s = profile["lag"] * 0.4
        touch_s = float(accel_end + (start_z - 0.030 - surface_z) / speed + lag_s)
        events.append({"t": round(touch_s, 4), "kind": "touch"})
    else:  # disturbance and replanning hold at home or wander near it
        ref_x, ref_z = np.full_like(times, home[0]), np.full_like(times, home[1])

    ref_q = np.stack(inverse_kinematics(ref_x, ref_z), axis=1)

    if rung["family"] == "replanning":
        home_q = ref_q[0].copy()
        targets = home_q + rng.uniform(-np.radians(20), np.radians(20), size=(8, 2))
        current = home_q.copy()
        for index, target_q in enumerate(targets):
            start = 0.3 + 0.15 * index
            blend = min_jerk((times - start) / 0.6)[:, None]
            ref_q = np.where((times >= start)[:, None], current + (target_q - current) * blend, ref_q)
            current = current + (target_q - current) * float(min_jerk(np.array([0.15 / 0.6]))[0])
            events.append({"t": round(start, 4), "kind": "replan" if profile["replannable"] else "refused"})

    # the tracking error: lag behind the reference velocity, a little noise, the rung's disturbance
    velocity = np.gradient(ref_q, times, axis=0)
    error = -profile["lag"] * velocity + rng.normal(0.0, 0.0004, ref_q.shape)
    if rung["family"] == "disturbance":
        push = ((times >= 0.5) & (times < 0.9)).astype(float)
        recover = np.where(times >= 0.9, np.exp(-(times - 0.9) / max(target.get("recovery", 0.3) / 3, 1e-3)), 0.0)
        shape = push * (1 - np.exp(-(times - 0.5) / 0.04)) + recover * (1 - np.exp(-0.4 / 0.04))
        error += np.outer(shape, [0.6, -1.0]) * 0.01
        events += [{"t": 0.5, "kind": "push_start"}, {"t": 0.9, "kind": "push_end"}]
    if rung["family"] == "replanning" and not profile["replannable"]:
        error += ref_q - ref_q[0]  # it keeps feeding its first trajectory: every replan is an error
    if rung["family"] == "timing":
        stall_start, stall_len = rung["timing"]["stalls"][0]
        events += [{"t": stall_start, "kind": "stall_start"}, {"t": stall_start + stall_len, "kind": "stall_end"},
                   {"t": round(stall_start + 0.25, 4), "kind": "watchdog"}]
        frozen = (times >= stall_start + 0.25) & (times < stall_start + stall_len + 0.15)
        index_at = np.searchsorted(times, stall_start + 0.25)
        held = ref_q[index_at]
        error = np.where(frozen[:, None], held - ref_q, error)
        if not profile["time_driven"]:  # it resumes where it stopped: the rest of the path slips
            slip_samples = int(stall_len * TRACE_HZ)
            late = np.roll(ref_q, slip_samples, axis=0)
            late[:slip_samples] = ref_q[0]
            after = times >= stall_start + stall_len + 0.15
            error = np.where(after[:, None], late - ref_q, error)

    worst = max(p95(error[:, 0]), p95(error[:, 1]))
    error *= np.radians(target["track"]) / max(worst, 1e-12)
    q = ref_q + error
    q_deg, ref_deg, err_deg = np.degrees(q), np.degrees(ref_q), np.degrees(error)
    tool_x, tool_z = forward_kinematics(q[:, 0], q[:, 1])
    ref_tool_x, ref_tool_z = forward_kinematics(ref_q[:, 0], ref_q[:, 1])
    tool_error_mm = np.hypot(tool_x - ref_tool_x, tool_z - ref_tool_z) * 1000

    # motor torque: gravity on the links plus an inertial term and noise
    gravity = np.stack([
        9.81 * ((1.2 * 0.15 + 0.8 * 0.30 + 0.3 * 0.30) * np.cos(q[:, 0]) + (0.8 * 0.125 + 0.3 * 0.25) * np.cos(q.sum(1))),
        9.81 * (0.8 * 0.125 + 0.3 * 0.25) * np.cos(q.sum(1)),
    ], axis=1)
    accel = np.gradient(velocity, times, axis=0)
    tau = gravity + accel * np.array([0.12, 0.03]) + rng.normal(0.0, 0.02, q.shape)

    trace: dict[str, list] = {"t": series(times)}
    for index, joint in enumerate(JOINTS):
        trace[f"q_deg.{joint}"] = series(q_deg[:, index])
        trace[f"ref_deg.{joint}"] = series(ref_deg[:, index])
        trace[f"err_deg.{joint}"] = series(err_deg[:, index])
        trace[f"tau_nm.{joint}"] = series(tau[:, index])
    trace["tool_x_mm"] = series(tool_x * 1000)
    trace["tool_z_mm"] = series(tool_z * 1000)

    metrics: dict[str, float | int | None] = {m.name: None for m in METRICS}
    worst_joint = int(np.argmax([p95(err_deg[:, 0]), p95(err_deg[:, 1])]))
    metrics.update(
        tracking_rms_deg=float(np.sqrt(np.mean(err_deg[:, worst_joint] ** 2))),
        tracking_p95_deg=p95(err_deg[:, worst_joint]),
        tracking_max_deg=float(np.max(np.abs(err_deg))),
        tool_rms_mm=float(np.sqrt(np.mean(tool_error_mm**2))),
        tool_p95_mm=p95(tool_error_mm),
        final_error_deg=float(np.max(np.abs(err_deg[-1]))),
        settle_time_s=target.get("settle", 0.18 if not profile["soft"] else 0.27),
        saturation_share=target.get("saturation", 0.0),
        faults=0, diverged=0,
    )

    if rung["family"] == "contact":
        contact_t = contact_time_base(duration, touch_s)
        force = np.zeros_like(contact_t)
        since = contact_t - touch_s
        on = (since >= 0) & (contact_t < lift_s + 0.05)
        ring_hz = 38.0 if not profile["soft"] else 22.0
        transient = (target["peak"] - target["dwell"]) * np.exp(-np.clip(since - 0.006, 0, None) / 0.012)
        rise = np.clip(since / 0.006, 0, 1)
        force = np.where(on, rise * (target["dwell"] + transient * np.cos(2 * np.pi * ring_hz * np.clip(since - 0.006, 0, None))), 0.0)
        force = np.where(on, force + rng.normal(0.0, 0.25, len(force)), 0.0)
        force = np.clip(force, 0.0, None)
        # place the peak exactly (the transient's first maximum)
        window = (since >= 0) & (since <= 0.030)
        force[window] *= target["peak"] / max(force[window].max(), 1e-9)
        dwell_window = (contact_t >= touch_s + 0.10) & (contact_t < lift_s)
        bounces = int(target.get("bounces", 0))
        if bounces:
            starts = np.linspace(touch_s + 0.2, lift_s - 0.15, bounces)
            for start in starts:
                force[(contact_t >= start) & (contact_t < start + 0.035)] = 0.0
        fraction = target.get("fraction", 100.0)
        trace["contact_t"] = series(contact_t)
        trace["contact_fn_n"] = series(force)
        dwell_force = force[dwell_window]
        metrics.update(
            touch_speed_m_s=rung["params"]["touch_speed_m_s"] * 0.97,
            contact_peak_n=float(force[window].max()),
            contact_max_n=float(force.max()),
            dwell_force_mean_n=float(target["dwell"]),
            dwell_force_std_n=float(np.std(dwell_force)),
            dwell_force_error_n=abs(float(target["dwell"]) - rung["params"]["press_target_n"]),
            dwell_bounces=bounces,
            dwell_contact_fraction=fraction,
            dwell_slip_mm=0.4 if not profile["soft"] else 1.6,
        )
    if rung["family"] == "disturbance":
        metrics.update(push_deflection_mm=target["deflection"], push_recovery_s=target["recovery"],
                       push_overshoot_mm=target["overshoot"])
    if rung["family"] == "replanning":
        metrics.update(replans_accepted=target["replans"], command_jump_max_deg=target["jump"],
                       replan_settle_s=target["settle"],
                       torque_rate_max_nm_s=float(np.max(np.abs(np.gradient(tau, times, axis=0)))))
    else:
        metrics.update(command_jump_max_deg=float(np.max(np.abs(np.diff(ref_deg, axis=0)))) * 2,
                       torque_rate_max_nm_s=float(np.max(np.abs(np.gradient(tau, times, axis=0)))))

    rate = rung["timing"]["rate_hz"]
    stalls = rung["timing"]["stalls"]
    tick_t, compute = compute_trace(rng, profile, duration, rate, stalls)
    trace["tick_t"] = series(tick_t)
    trace["compute_us"] = series(compute)
    stall_ms = max((s[1] * 1000 for s in stalls), default=0.0)
    metrics.update(
        period_jitter_p50_ms=0.004 + (0.21 if rung["timing"]["jitter_s"] else 0.0),
        period_jitter_p99_ms=0.012 + (0.62 if rung["timing"]["jitter_s"] else 0.0),
        period_jitter_max_ms=(stall_ms or 0.02) + (0.0 if not stalls else 1000 / rate),
        missed_cycles=int(round(stall_ms / (1000 / rate))) if stalls else 0,
        worst_stall_ms=(stall_ms + 1000 / rate) if stalls else 1000 / rate + 0.02,
        latency_p50_ms=1.62 + compute.mean() / 1000,
        latency_p99_ms=1.75 + np.percentile(compute, 99) / 1000,
        latency_max_ms=1.76 + compute.max() / 1000,
        stale_replies=0,
        schedule_slip_ms=target.get("slip", 0.0 if profile["time_driven"] else 0.6),
        watchdog_trips=1 if stalls else 0,
        compute_us_p50=float(np.percentile(compute, 50)),
        compute_us_p99=float(np.percentile(compute, 99)),
        compute_us_max=float(compute.max()),
        command_fields=profile["fields"],
        wire_bytes_per_tick=len(JOINTS) * (2 + 4 * profile["fields"]),
        host_rate_hz=rate,
    )
    return finish(controller_name, rung, seed, metrics, trace, events)


def jcs_run(controller_name: str, rung: dict, seed: int) -> dict:
    rng = np.random.default_rng([seed, sum(map(ord, rung["id"]))])
    target = JCS_HEADLINE[rung["id"]]
    duration = rung["duration_s"]
    times = np.arange(0.0, duration, 1.0 / TRACE_HZ)
    home_q = np.stack(inverse_kinematics(np.array([0.30]), np.array([0.25])), axis=1)[0]
    events: list[dict] = []
    q = np.tile(home_q, (len(times), 1))
    tau_extra = np.zeros_like(q)
    if rung["id"] in ("S1.MoveTo", "S3.HostStall"):
        goal = home_q + np.radians([25.0, -30.0])
        ramp_s = np.abs(goal - home_q).max() / 1.5
        progress = np.clip(times / ramp_s, 0, 1)[:, None]
        if rung["id"] == "S3.HostStall":
            stall_start, stall_len = rung["timing"]["stalls"][0]
            shifted = np.where(times < stall_start, times, np.where(times < stall_start + stall_len, stall_start,
                                                                     times - stall_len))
            progress = np.clip(shifted / ramp_s, 0, 1)[:, None]
            events += [{"t": stall_start, "kind": "stall_start"}, {"t": stall_start + stall_len, "kind": "stall_end"},
                       {"t": round(stall_start + 0.25, 4), "kind": "watchdog"}]
        q = home_q + (goal - home_q) * min_jerk(progress) + rng.normal(0, 0.0003, q.shape)
        done_t = target["time_s"] - 0.5
        events += [{"t": round(done_t, 4), "kind": "step_done", "step": 0, "outcome": "Reached"},
                   {"t": target["time_s"], "kind": "step_done", "step": 1, "outcome": "Elapsed"}]
    else:  # seek the elbow's hard stop
        limit = home_q[1] + np.radians(55.0)
        q[:, 1] = np.minimum(home_q[1] + 0.5 * times, limit)
        stalled = q[:, 1] >= limit
        tau_extra[:, 1] = np.where(stalled, 3.0 * np.clip((times - times[np.argmax(stalled)]) / 0.15, 0, 1), 0.0)
        events += [{"t": target["time_s"], "kind": "step_done", "step": 0, "outcome": "Stall elbow"}]
    tau = np.stack([9.81 * 0.44 * np.cos(q[:, 0]), 9.81 * 0.175 * np.cos(q.sum(1))], axis=1) + tau_extra
    tool_x, tool_z = forward_kinematics(q[:, 0], q[:, 1])
    trace: dict[str, list] = {"t": series(times)}
    for index, joint in enumerate(JOINTS):
        trace[f"q_deg.{joint}"] = series(np.degrees(q[:, index]))
        trace[f"tau_nm.{joint}"] = series(tau[:, index])
    trace["tool_x_mm"] = series(tool_x * 1000)
    trace["tool_z_mm"] = series(tool_z * 1000)
    profile = dict(compute=(6.3, 9.1))
    tick_t, compute = compute_trace(rng, profile, duration, 100.0, rung["timing"]["stalls"])
    trace["tick_t"] = series(tick_t)
    trace["compute_us"] = series(compute)
    stalls = rung["timing"]["stalls"]
    metrics: dict[str, float | int | None] = {m.name: None for m in METRICS}
    metrics.update(
        steps_done=target["steps"][0], steps_total=target["steps"][1], sequence_time_s=target["time_s"],
        reached_error_deg=target["reached"], faults=0, diverged=0, saturation_share=0.0,
        period_jitter_p50_ms=0.004, period_jitter_p99_ms=0.011, period_jitter_max_ms=500.0 if stalls else 0.02,
        missed_cycles=50 if stalls else 0, worst_stall_ms=510.0 if stalls else 10.02,
        latency_p50_ms=1.63, latency_p99_ms=1.76, latency_max_ms=1.77, stale_replies=0,
        schedule_slip_ms=None, watchdog_trips=1 if stalls else 0,
        compute_us_p50=float(np.percentile(compute, 50)), compute_us_p99=float(np.percentile(compute, 99)),
        compute_us_max=float(compute.max()), command_fields=4, wire_bytes_per_tick=2 * (2 + 4 * 4),
        host_rate_hz=100.0,
        torque_rate_max_nm_s=float(np.max(np.abs(np.gradient(tau, times, axis=0)))),
    )
    return finish(controller_name, rung, seed, metrics, trace, events)


def finish(controller_name: str, rung: dict, seed: int, metrics: dict, trace: dict, events: list) -> dict:
    metrics = {k: (sig4(float(v)) if isinstance(v, float) else v) for k, v in metrics.items()}
    results = [judge(gate, metrics.get(gate.metric)) for gate in rung["gates"]]
    gates = [{"metric": r.gate.metric, "op": r.gate.op, "value": r.gate.value, "why": r.gate.why,
              "measured": r.value, "passed": r.passed, "note": r.note} for r in results]
    if rung["id"] == "R5.RapidReplan" and not PROFILES.get(controller_name, {}).get("replannable", True):
        for gate in gates:
            if gate["metric"] == "replans_accepted":
                gate["note"] = "unsupported: refuses replanning"
    passed = all(g["passed"] for g in gates)
    return {"controller": controller_name, "rung": rung["id"], "seed": seed, "metrics": metrics, "gates": gates,
            "passed": passed, "objective": metrics.get(rung["objective"]), "trace": trace, "events": events,
            "wandb_url": None, "wall_s": 0.0}


# ── the document ────────────────────────────────────────────────────────────────────────────────────────────


def rung_record(rung: dict) -> dict:
    return {**{k: v for k, v in rung.items() if k != "gates"},
            "gates": [{"metric": g.metric, "op": g.op, "value": g.value, "why": g.why} for g in rung["gates"]]}


def summarize(box, rungs: list[dict], results: list[dict]) -> dict:
    """CONTRACT §4's summary: the metrics lane's own `summarize` when it imports, so the fixture carries the real
    shape; otherwise this guess (names-only Pareto lists), which the site also reads."""
    try:
        from robot_controllers.metrics.summary import summarize as real_summarize

        return real_summarize(box, [rung_record(r) for r in rungs], results)
    except Exception as error:  # noqa: BLE001  the metrics lane may be mid-edit; the guess keeps the fixture whole
        print(f"metrics.summary not usable ({type(error).__name__}: {error}); writing the fixture's own summary")
    by = {(r["controller"], r["rung"]): r for r in results}
    names = sorted({r["controller"] for r in results}, key=[r["controller"] for r in results].index)
    per_controller = {}
    for name in names:
        runs = [by[(name, rung["id"])] for rung in rungs if (name, rung["id"]) in by]
        compute = sorted(r["metrics"]["compute_us_p99"] for r in runs)
        per_controller[name] = {
            "gates_passed": sum(g["passed"] for r in runs for g in r["gates"]),
            "gates_total": sum(len(r["gates"]) for r in runs),
            "rungs_passed": sum(r["passed"] for r in runs), "rungs_total": len(runs),
            "objective": {}, "compute_us_p99": sig4(float(np.median(compute))),
        }
    per_rung = {}
    for rung in rungs:
        ranked = sorted((by[(n, rung["id"])] for n in names if (n, rung["id"]) in by),
                        key=lambda r: (r["objective"] is None, (r["objective"] or 0) * (1 if rung["minimize"] else -1)))
        for rank, run in enumerate(ranked, 1):
            per_controller[run["controller"]]["objective"][rung["id"]] = {"value": run["objective"], "rank": rank}
        passing = [r for r in ranked if r["passed"] and r["objective"] is not None]
        per_rung[rung["id"]] = {"objective": rung["objective"], "minimize": rung["minimize"],
                                "best": passing[0]["controller"] if passing else None}
    pareto = {}
    anchor = next((r for r in rungs if r["id"].startswith("R1")), None)
    for rung in rungs:
        if rung["family"] != "contact" or anchor is None:
            continue
        points = {n: (by[(n, anchor["id"])]["metrics"]["tracking_p95_deg"], by[(n, rung["id"])]["metrics"]["contact_max_n"],
                      per_controller[n]["compute_us_p99"]) for n in names}
        pareto[rung["id"]] = [n for n, p in points.items()
                              if not any(all(o[i] <= p[i] for i in range(3)) and o != p for o in points.values())]
    best = sorted(names, key=lambda n: (-per_controller[n]["rungs_passed"], -per_controller[n]["gates_passed"],
                                        per_controller[n]["compute_us_p99"]))[0]
    stats = per_controller[best]
    why = (f"passes {stats['rungs_passed']}/{stats['rungs_total']} rungs and {stats['gates_passed']}/"
           f"{stats['gates_total']} gates at {stats['compute_us_p99']:g} µs p99 per tick")
    if len(names) == 1:
        why += " (the only built controller in this box)"
    return {"per_controller": per_controller, "per_rung": per_rung, "pareto": pareto,
            "recommendation": {"controller": best, "why": why}}


def controller_records(box, configs: dict) -> list[dict]:
    return [{"name": s.name, "title": s.title, "runs_on": s.runs_on, "summary": s.summary, "variant_of": s.variant_of,
             "prior_art": list(s.prior_art), "config": configs.get(s.name, {})} for s in in_box(box, built_only=True)]


IMPEDANCE = {"stiffness_nm_rad": [40.0, 25.0], "damping_ratio": 0.8, "inertia_kg_m2": [0.10, 0.03],
             "damping_nm_s_rad": None, "max_effort_nm": None}
CONFIGS = {
    "joint_trajectory_controller": {"path_tolerance_rad": None},
    "joint_trajectory_controller_cpp": {"path_tolerance_rad": None},
    "tape_feeder": {"control_period_s": 0.01},
    "joint_impedance_board": {"location": "board", **IMPEDANCE},
    "joint_impedance_host": {"location": "host", **IMPEDANCE},
    "joint_impedance_host_spring": {"location": "host_spring", **IMPEDANCE},
}


def document(box, rungs: list[dict], results: list[dict], configs: dict) -> dict:
    return {
        "schema": SCHEMA,
        "box": {"slug": box.slug, "name": box.name, "input": box.input_name, "output": box.output_name,
                "wandb_project": box.wandb_project},
        "generated": {"commit": "fixture", "dirty": False, "robot_msgs": "0.2.0 @ fixture", "plant": "planar_arm",
                      "host": "fixture", "python": "3.12.3", "mujoco": "3.11.0", "date": "2026-11-12",
                      "fixture": True},
        "rungs": [rung_record(r) for r in rungs],
        "controllers": controller_records(box, configs),
        "results": results,
        "summary": summarize(box, rungs, results),
        "wandb": {"project_url": None, "reports": {}},
    }


def main() -> None:
    jt_names = [s.name for s in in_box(JOINT_TRAJECTORY, built_only=True)]
    jt_results = [jt_run(name, rung, 0) for name in jt_names for rung in JT_RUNGS]
    jcs_name = in_box(JOINT_COMMAND_SEQUENCE, built_only=True)[0].name
    jcs_results = [jcs_run(jcs_name, rung, 0) for rung in JCS_RUNGS]
    for box, rungs, results in ((JOINT_TRAJECTORY, JT_RUNGS, jt_results), (JOINT_COMMAND_SEQUENCE, JCS_RUNGS, jcs_results)):
        path = OUT / box.slug / "results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # WHY compact separators: the fixture is committed; indent would triple its size for no reader.
        path.write_text(json.dumps(document(box, rungs, results, CONFIGS), separators=(",", ":"), allow_nan=False) + "\n")
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
