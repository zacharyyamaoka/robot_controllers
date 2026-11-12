"""The rung ladder (bench/rungs.py): the exact ids, gates and objectives of CONTRACT §3, what each input
script draws, and every rung run end-to-end into a well-formed EpisodeLog with a test-only controller."""

from __future__ import annotations

import math
import pickle
import time

import numpy as np
import pytest
from robot_msgs.control import JointCommandSequence, ServoCommand
from robot_msgs.trajectory import JointTrajectory

from robot_controllers import controller
from robot_controllers.bench.arm import ARM, HOME_Q, fk, nominal_dynamics
from robot_controllers.bench.loop import run_episode
from robot_controllers.bench.rungs import ladder, quintic, rung
from robot_controllers.bench.types import EpisodeLog
from robot_controllers.metrics.names import BY_NAME
from robot_controllers.registry import JOINT_COMMAND_SEQUENCE, JOINT_TRAJECTORY, POSE_TRAJECTORY

T100 = (("missed_cycles", "==", 0), ("latency_p99_ms", "<=", 10.0))
BELT = (
    ("contact_peak_n", "<=", 60),
    ("dwell_bounces", "==", 0),
    ("dwell_force_error_n", "<=", 10),
    ("dwell_contact_fraction", ">=", 95),
    ("diverged", "==", 0),
)
# CONTRACT §3, transcribed: id -> (gates, objective)
EXPECTED = {
    "R1.FreeSpace": (
        (
            ("tracking_p95_deg", "<=", 1.0),
            ("saturation_share", "<=", 2),
            ("diverged", "==", 0),
            ("faults", "==", 0),
            *T100,
        ),
        "tracking_p95_deg",
    ),
    "R2.FastFreeSpace": (
        (("tracking_p95_deg", "<=", 3.0), ("saturation_share", "<=", 5), ("diverged", "==", 0), *T100),
        "tracking_p95_deg",
    ),
    "R3.BeltImpact@0.05m/s": ((*BELT, *T100), "contact_max_n"),
    "R3.BeltImpact@0.10m/s": ((*BELT, *T100), "contact_max_n"),
    "R3.BeltImpact@0.20m/s": ((*BELT, *T100), "contact_max_n"),
    "R4.SidewaysPush": (
        (("push_recovery_s", "<=", 0.5), ("push_overshoot_mm", "<=", 5), ("diverged", "==", 0), *T100),
        "push_recovery_s",
    ),
    "R5.RapidReplan": (
        (
            ("replans_accepted", ">=", 8),
            ("command_jump_max_deg", "<=", 2),
            ("replan_settle_s", "<=", 0.5),
            ("diverged", "==", 0),
            *T100,
        ),
        "tracking_p95_deg",
    ),
    "R6.HostStall": (
        (("diverged", "==", 0), ("schedule_slip_ms", "<=", 20), ("settle_time_s", "<=", 1.0)),
        "tracking_p95_deg",
    ),
    "R7.BeltImpact@0.10m/s·host50Hz": (
        (*BELT, ("missed_cycles", "==", 0), ("latency_p99_ms", "<=", 20.0)),
        "contact_max_n",
    ),
    "R7.BeltImpact@0.10m/s·host500Hz": (
        (*BELT, ("missed_cycles", "==", 0), ("latency_p99_ms", "<=", 2.0)),
        "contact_max_n",
    ),
    "S1.MoveTo": (
        (("steps_done", ">=", 2), ("reached_error_deg", "<=", 1.0), ("faults", "==", 0), *T100),
        "sequence_time_s",
    ),
    "S2.SeekHardStop": ((("steps_done", ">=", 1), ("faults", "==", 0), *T100), "sequence_time_s"),
    "S3.HostStall": ((("steps_done", ">=", 2), ("diverged", "==", 0)), "sequence_time_s"),
}
ALL_RUNGS = (*ladder(JOINT_TRAJECTORY), *ladder(JOINT_COMMAND_SEQUENCE))
WALL: dict[str, tuple[float, float]] = {}


class Streamer:
    """Test-only time-driven trajectory streamer (own clock += dt, linear sample, hold the end),
    replannable."""

    def __init__(self, trajectory):
        self.replace_input(trajectory)

    def replace_input(self, trajectory):
        self.trajectory, self.clock, self.status = trajectory, None, "running"

    def tick(self, state, dt):
        self.clock = 0.0 if self.clock is None else self.clock + dt
        trajectory = self.trajectory
        at = min(self.clock, float(trajectory.t[-1]))
        if self.clock >= trajectory.t[-1]:
            self.status = "done"

        def sample(values):
            return np.array([np.interp(at, trajectory.t, values[:, j]) for j in range(values.shape[1])])

        velocity = (
            sample(trajectory.velocities) if self.status == "running" else np.zeros(trajectory.n_joints)
        )
        return ServoCommand(
            name=trajectory.joint_names,
            position=sample(trajectory.positions),
            velocity=velocity,
            feedforward_effort=sample(trajectory.effort),
        ), {}


class Spec:
    def __init__(self, name, factory):
        self.name, self.factory = name, factory

    def build(self, message):
        return self.factory(message)


# ── the ruler ───────────────────────────────────────────────────────────────────────────────────────────────


def test_the_ladders_have_the_contract_ids_in_order():
    assert [r.id for r in ladder(JOINT_TRAJECTORY)] == list(EXPECTED)[:10]
    assert [r.id for r in ladder(JOINT_COMMAND_SEQUENCE)] == list(EXPECTED)[10:]
    assert [r.order for r in ladder(JOINT_TRAJECTORY)] == list(range(1, 11))
    assert [r.order for r in ladder(JOINT_COMMAND_SEQUENCE)] == [1, 2, 3]
    assert ladder(POSE_TRAJECTORY) == ()


@pytest.mark.parametrize("item", ALL_RUNGS, ids=lambda r: r.id)
def test_every_rung_has_exactly_the_contract_gates_and_objective(item):
    gates, objective = EXPECTED[item.id]
    assert [(g.metric, g.op, pytest.approx(g.value)) for g in item.gates] == [
        (metric, op, pytest.approx(value)) for metric, op, value in gates
    ]
    assert item.objective == objective and item.minimize
    for gate in item.gates:
        assert gate.metric in BY_NAME and gate.why
    assert item.box in (JOINT_TRAJECTORY, JOINT_COMMAND_SEQUENCE) and item.description.count(".") >= 2


def test_rungs_pickle_for_worker_processes():
    for item in ALL_RUNGS:
        clone = pickle.loads(pickle.dumps(item))
        assert clone.id == item.id
        start = np.asarray(item.scene.start_q or HOME_Q)
        first, again = item.inputs(start)[0][1], clone.inputs(start)[0][1]
        if isinstance(first, JointTrajectory):
            np.testing.assert_array_equal(first.positions, again.positions)


# ── what the scripts draw ───────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rung_id,move_s", [("R1.FreeSpace", 0.8), ("R2.FastFreeSpace", 0.3)])
def test_free_space_path_visits_the_contract_points_with_rests(rung_id, move_s):
    trajectory = rung(rung_id).inputs(np.asarray(HOME_Q))[0][1]
    assert np.diff(trajectory.t) == pytest.approx(np.full(len(trajectory) - 1, 0.004))
    points = ((0.30, 0.25), (0.42, 0.20), (0.25, 0.32), (0.30, 0.25))
    for index, point in enumerate(points):
        arrive = index * (move_s + 0.3)  # P0 rests 0.3 s first, then each move ends in a 0.3 s rest
        for at in (arrive, arrive + 0.3):
            state = trajectory.joint_state_at(at, extrapolate="clamp")
            assert fk(state.position) == pytest.approx(point, abs=1e-9), (index, at)
            assert np.abs(state.velocity).max() < 1e-9
    # the feed-forward is the nominal inverse dynamics of the trajectory itself (not gravity alone)
    expected = nominal_dynamics(trajectory.positions, trajectory.velocities, trajectory.accelerations)
    np.testing.assert_allclose(trajectory.effort, expected, atol=1e-12)
    moving = np.abs(trajectory.accelerations).max(axis=1) > 1.0
    gravity_only = nominal_dynamics(
        trajectory.positions, 0 * trajectory.velocities, 0 * trajectory.accelerations
    )
    assert np.abs(trajectory.effort[moving] - gravity_only[moving]).max() > 0.05


@pytest.mark.parametrize(
    "speed,rung_id",
    [
        (0.05, "R3.BeltImpact@0.05m/s"),
        (0.10, "R3.BeltImpact@0.10m/s"),
        (0.20, "R3.BeltImpact@0.20m/s"),
        (0.10, "R7.BeltImpact@0.10m/s·host50Hz"),
    ],
)
def test_belt_approach_cruises_at_the_touch_speed_and_aims_10_mm_below(speed, rung_id):
    item = rung(rung_id)
    start = np.asarray(item.scene.start_q)
    assert fk(start) == pytest.approx((0.30, ARM.tool_touch_z_m + 0.040), abs=1e-12)
    trajectory = item.inputs(start)[0][1]
    x, z = fk(trajectory.positions)
    tool_velocity_z = np.gradient(z, trajectory.t)
    dwell_start, dwell_end = item.scene.extras["dwell"]
    descending = trajectory.t <= dwell_start
    assert np.all(tool_velocity_z[descending][1:-1] <= 1e-9)  # never goes back up on the way down
    assert -tool_velocity_z[descending].min() <= speed * (1 + 1e-3)
    crossing = np.flatnonzero(z <= ARM.tool_touch_z_m + 1e-12)[0]
    cruise_band = (z < ARM.tool_touch_z_m + 0.019) & (z > ARM.tool_touch_z_m + 0.001) & descending
    analytic_vz = _tool_velocities(trajectory)[:, 1]
    assert analytic_vz[cruise_band] == pytest.approx(np.full(cruise_band.sum(), -speed), abs=1e-9)
    assert abs(analytic_vz[crossing] + speed) < 0.05 * speed  # touches the surface at the touch speed
    dwell = (trajectory.t >= dwell_start) & (trajectory.t <= dwell_end)
    assert z[dwell] == pytest.approx(np.full(dwell.sum(), ARM.tool_touch_z_m - 0.010), abs=1e-12)
    assert np.allclose(x, 0.30, atol=1e-12)
    assert z[-1] == pytest.approx(ARM.tool_touch_z_m - 0.010 + 0.040, abs=1e-12)
    assert (
        item.scene.belt and item.scene.belt_speed_m_s == 0.25 and item.scene.extras["press_target_n"] == 10.0
    )
    assert item.duration_s > trajectory.t[-1]


def _tool_velocities(trajectory):
    from robot_controllers.bench.arm import jacobian

    return np.einsum("nij,nj->ni", jacobian(trajectory.positions), trajectory.velocities)


def test_push_rung_holds_home_and_pushes_15_n_along_x_for_0_4_s_at_0_5_s():
    item = rung("R4.SidewaysPush")
    push = item.scene.pushes[0]
    assert (push.start_s, push.duration_s, push.force_n) == (0.5, 0.4, (15.0, 0.0, 0.0))
    trajectory = item.inputs(np.asarray(HOME_Q))[0][1]
    np.testing.assert_allclose(trajectory.positions, np.tile(HOME_Q, (len(trajectory), 1)), atol=1e-15)


def test_replan_chain_starts_each_plan_from_the_previous_plans_state():
    item = rung("R5.RapidReplan")
    script = item.inputs(np.asarray(HOME_Q))
    assert [t for t, _ in script] == pytest.approx([0.15 * k for k in range(9)])
    targets = np.asarray(item.scene.extras["targets_rad"])
    assert targets.shape == (9, 2) and np.all(np.abs(targets - np.asarray(HOME_Q)) <= math.radians(20.0))
    assert len({tuple(t) for t in targets.round(9)}) == 9  # seeded, not repeated
    first = script[0][1]
    np.testing.assert_allclose(first.positions[0], HOME_Q, atol=1e-15)
    for (start_t, previous), (next_t, following) in zip(script[:-1], script[1:], strict=True):
        # exact: the previous plan is the quintic its own end conditions define, evaluated at the handover
        for j in range(2):
            p, v, a = quintic(
                previous.positions[0, j],
                previous.velocities[0, j],
                previous.accelerations[0, j],
                previous.positions[-1, j],
                0.0,
                0.0,
                0.6,
                np.array([next_t - start_t]),
            )
            assert following.positions[0, j] == pytest.approx(p[0], abs=1e-12)
            assert following.velocities[0, j] == pytest.approx(v[0], abs=1e-12)
        # and what a controller sampling the 4 ms samples sees there agrees to interpolation error
        handover = previous.joint_state_at(next_t - start_t)
        np.testing.assert_allclose(following.positions[0], handover.position, atol=1e-4)
    for index, (_, trajectory) in enumerate(script):
        assert trajectory.t[-1] == pytest.approx(0.6)
        np.testing.assert_allclose(trajectory.positions[-1], targets[index], atol=1e-12)
        np.testing.assert_allclose(trajectory.velocities[-1], 0.0, atol=1e-12)
    assert item.params["replans"] == 8


def test_sequence_rungs_build_the_contract_sequences():
    move_to = rung("S1.MoveTo").inputs(np.asarray(HOME_Q))[0][1]
    assert isinstance(move_to, JointCommandSequence) and len(move_to) == 2
    move, hold = move_to.steps
    assert [c.mode for c in move.commands] == ["position", "position"]
    assert [c.speed for c in move.commands] == [1.5, 1.5]
    assert [c.max_effort for c in move.commands] == [12.0, 8.0]  # = the boards' caps
    reached = move.done_when[0]
    assert type(reached).__name__ == "Reached" and reached.tolerance == pytest.approx(math.radians(1.0))
    assert [c.mode for c in hold.commands] == ["hold", "hold"]
    assert hold.done_when[0].duration == pytest.approx(0.5)

    seek = rung("S2.SeekHardStop").inputs(np.asarray(HOME_Q))[0][1]
    (step,) = seek.steps
    elbow = step.command("elbow")
    assert elbow.mode == "move_by" and elbow.distance < -math.radians(45) and elbow.speed == 0.5
    assert elbow.max_effort == 3.0 and step.command("shoulder").mode == "hold"
    assert (
        type(step.done_when[0]).__name__ == "Stall"
        and step.define_position["elbow"] == ARM.elbow_range_rad[0]
    )

    assert rung("S3.HostStall").timing.stalls == ((0.3, 0.5),)
    assert (
        rung("R6.HostStall").timing.stalls == ((0.8, 0.5),) and rung("R6.HostStall").timing.jitter_s == 0.0003
    )
    assert rung("R7.BeltImpact@0.10m/s·host50Hz").timing.rate_hz == 50.0
    assert rung("R7.BeltImpact@0.10m/s·host500Hz").timing.rate_hz == 500.0


# ── every rung end-to-end ───────────────────────────────────────────────────────────────────────────────────


def _spec_for(item):
    if item.box == JOINT_COMMAND_SEQUENCE:
        return controller("joint_command_sequence_controller")  # robot_msgs' own sequence controller
    return Spec("test_streamer", Streamer)


@pytest.mark.parametrize("item", ALL_RUNGS, ids=lambda r: r.id)
def test_every_rung_runs_end_to_end_into_a_well_formed_episode_log(item):
    started = time.perf_counter()
    log = run_episode(_spec_for(item), item, seed=0)
    WALL[item.id] = (time.perf_counter() - started, item.duration_s)
    assert isinstance(log, EpisodeLog) and log.rung == item.id and log.joint_names == ("shoulder", "elbow")
    ticks, rows = len(log.tick_t), len(log.t)
    assert rows == int(round(item.duration_s / ARM.physics_dt_s))
    assert ticks > 0.5 * item.duration_s * item.timing.rate_hz
    for name in ("tick_t", "sent_t", "state_stamp", "state_seq", "compute_s", "status", "input_index"):
        assert getattr(log, name).shape == (ticks,), name
    for name in (
        "cmd_position",
        "cmd_velocity",
        "cmd_feedforward",
        "cmd_stiffness",
        "cmd_damping",
        "cmd_max_effort",
    ):
        assert getattr(log, name).shape == (ticks, 2), name
    for name, width in (
        ("q", 2),
        ("qd", 2),
        ("tau", 2),
        ("at_cap", 2),
        ("board_mode", 2),
        ("tool_pos", 3),
        ("contact_force", 3),
        ("external_force", 3),
    ):
        values = getattr(log, name)
        assert values.shape == (rows, width), name
        if values.dtype.kind == "f":
            assert np.isfinite(values).all(), f"{name} has NaN/inf"
    assert log.in_contact.shape == (rows,)
    assert np.all(np.diff(log.sent_t) > 0) and np.all(log.sent_t >= log.state_stamp)
    assert np.isfinite(log.compute_s).all() and np.all(log.compute_s > 0)
    assert [float(e["t"]) for e in log.events] == sorted(float(e["t"]) for e in log.events)
    if item.box == JOINT_TRAJECTORY:
        assert log.ref_position.shape == (ticks, 2) and np.isfinite(log.ref_position).all()
        assert log.ref_tool_pos.shape == (ticks, 3)
        assert log.meta["schedule_end_s"] <= item.duration_s
    else:
        assert log.ref_position is None and log.meta["steps_total"] == len(
            item.inputs(np.asarray(HOME_Q))[0][1]
        )
    if item.scene.belt:
        assert log.meta["dwell"] and log.meta["press_target_n"] == 10.0 and log.event_times("touch")
    if item.scene.pushes:
        assert log.meta["push"] == [0.5, 0.9, "+x"]
    assert log.meta["plant"] == "planar_arm" and log.meta["board"]["kp_nm_rad"] == (300.0, 200.0)


def test_report_wall_time_per_rung():
    """Prints wall seconds per simulated second for the rungs run above (pytest -s shows it); asserts nothing
    about speed, which depends on the machine's load."""
    for rung_id, (wall, simulated) in WALL.items():
        print(
            f"{rung_id:34s} {simulated:5.2f} s simulated {wall:6.2f} s wall {wall / simulated:5.3f} wall/sim"
        )
    assert all(wall > 0 for wall, _ in WALL.values())
