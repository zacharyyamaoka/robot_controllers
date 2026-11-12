"""The built controllers (lane controllers): JTC, tape feeder, joint impedance on board / host / host_spring.

Each test checks observable behaviour: the ServoCommand a tick sends, the controller's status, or what a toy
plant does under those commands. The numeric checks use hand-computed literals, so a wrong law (a flipped
sign, a gain on the wrong joint, a missing term) fails them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np
import pytest
from robot_msgs.control import (
    JointCommand,
    JointCommandSequence,
    JointCommandStep,
    Reached,
    ServoCommand,
)
from robot_msgs.sensor import JointState
from robot_msgs.trajectory import JointTrajectory

from robot_controllers import Controller, Replannable, SelfReport, controller, in_box
from robot_controllers.controllers._sampling import TrajectorySampler
from robot_controllers.controllers.joint_impedance import JointImpedanceConfig, JointImpedanceController
from robot_controllers.controllers.joint_trajectory import (
    JointTrajectoryConfig,
    JointTrajectoryController,
    NotReplannable,
    TapeFeeder,
    TapeFeederConfig,
)
from robot_controllers.registry import JOINT_TRAJECTORY

JOINTS = ("shoulder", "elbow")
PERIOD_S = 0.01  # the rack host's 100 Hz
DESIGN_INERTIA = np.array([0.10, 0.03])  # JointImpedanceConfig's default design inertia (kg·m²)


# ── a toy plant: one double integrator per joint under a servo board's PD ──────────────────────────────────


@dataclass
class ToyBoardPlant:
    """Per joint a double integrator (inertia I; no gravity, friction or delay) under a servo board's PD.

    The board applies a ServoCommand like moteus position mode: tau = K_eff (p - q) + D_eff (v - qd) + ff,
    where K_eff / D_eff are the command's stiffness / damping, or the board's own gains when those are None;
    a NaN position drops the position term; a missing velocity is 0 and a missing feed-forward is 0; the
    torque is clipped at max_effort when it is given. It reports its joints in `reported_order` plus an extra
    `gripper`, so a controller must select its joints by name.
    """

    inertia: np.ndarray = field(default_factory=lambda: DESIGN_INERTIA.copy())
    board_kp: np.ndarray = field(default_factory=lambda: np.array([300.0, 200.0]))
    board_kd: np.ndarray = field(default_factory=lambda: np.array([7.5, 4.0]))
    q: np.ndarray = field(default_factory=lambda: np.zeros(2))
    qd: np.ndarray = field(default_factory=lambda: np.zeros(2))
    reported_order: tuple[str, ...] = ("elbow", "gripper", "shoulder")
    time: float = 0.0

    def state(self) -> JointState:
        by_name = {"gripper": (0.0, 0.0)} | {name: (self.q[i], self.qd[i]) for i, name in enumerate(JOINTS)}
        return JointState(
            name=self.reported_order,
            position=[by_name[name][0] for name in self.reported_order],
            velocity=[by_name[name][1] for name in self.reported_order],
        )

    def apply(self, command: ServoCommand, duration_s: float, step_s: float = 0.25e-3) -> None:
        order = [command.name.index(name) for name in JOINTS]
        position = command.position[order]
        velocity = np.zeros(2) if command.velocity is None else command.velocity[order]
        feedforward = np.zeros(2) if command.feedforward_effort is None else command.feedforward_effort[order]
        stiffness = self.board_kp if command.stiffness is None else command.stiffness[order]
        damping = self.board_kd if command.damping is None else command.damping[order]
        cap = None if command.max_effort is None else command.max_effort[order]
        has_position = ~np.isnan(position)
        for _ in range(round(duration_s / step_s)):
            position_term = np.where(has_position, stiffness * (np.nan_to_num(position) - self.q), 0.0)
            torque = position_term + damping * (velocity - self.qd) + feedforward
            if cap is not None:
                torque = np.clip(torque, -cap, cap)
            self.qd = self.qd + torque / self.inertia * step_s  # semi-implicit Euler
            self.q = self.q + self.qd * step_s
            self.time += step_s


def run_on_plant(
    ctrl, plant: ToyBoardPlant, *, period_s: float, duration_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Tick `ctrl` every `period_s` on the plant; return the tick times and the plant's q at each tick."""
    times, positions = [], []
    for _ in range(round(duration_s / period_s)):
        times.append(plant.time)
        positions.append(plant.q.copy())
        command, defines = ctrl.tick(plant.state(), period_s)
        assert defines == {}
        plant.apply(command, period_s)
    return np.array(times), np.array(positions)


# ── trajectories ────────────────────────────────────────────────────────────────────────────────────────────


def min_jerk_trajectory(
    start=(0.0, 0.0), delta=(0.4, -0.3), duration_s=1.0, sample_s=0.004, inertia=DESIGN_INERTIA, rest_s=0.0
) -> JointTrajectory:
    """A rest-to-rest minimum-jerk move per joint; effort = I qdd, the exact feed-forward of the toy plant."""
    times = np.round(np.arange(0.0, duration_s + rest_s + sample_s / 2, sample_s), 12)
    phase = np.clip(times / duration_s, 0.0, 1.0)[:, None]
    start, delta = np.asarray(start), np.asarray(delta)
    positions = start + delta * (10 * phase**3 - 15 * phase**4 + 6 * phase**5)
    velocities = delta / duration_s * (30 * phase**2 - 60 * phase**3 + 30 * phase**4)
    accelerations = delta / duration_s**2 * (60 * phase - 180 * phase**2 + 120 * phase**3)
    return JointTrajectory(
        t=times,
        joint_names=JOINTS,
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        effort=accelerations * inertia,
    )


def ramp_trajectory(sample_s=PERIOD_S, duration_s=1.0, slope=(1.0, -0.5)) -> JointTrajectory:
    """Positions = slope x t on a grid of `sample_s`: sample k is k x sample_s x slope, easy to read back."""
    times = np.round(np.arange(0.0, duration_s + sample_s / 2, sample_s), 12)
    slope = np.asarray(slope)
    return JointTrajectory(
        t=times,
        joint_names=JOINTS,
        positions=times[:, None] * slope,
        velocities=np.tile(slope, (len(times), 1)),
        effort=np.tile([0.7, -0.2], (len(times), 1)),
    )


def state_at(shoulder=(0.0, 0.0), elbow=(0.0, 0.0)) -> JointState:
    """A measured state, joints deliberately out of the controller's order and with an extra joint."""
    return JointState(
        name=("elbow", "wrist", "shoulder"),
        position=[elbow[0], 9.9, shoulder[0]],
        velocity=[elbow[1], 9.9, shoulder[1]],
    )


# ── the shared sampler is robot_msgs' law, bit for bit ──────────────────────────────────────────────────────


def test_sampler_matches_joint_state_at_bit_for_bit():
    rng = np.random.default_rng(7)
    times = np.cumsum(rng.uniform(0.001, 0.02, 60))
    traj = JointTrajectory(
        t=times,
        joint_names=JOINTS,
        positions=rng.normal(size=(60, 2)),
        velocities=rng.normal(size=(60, 2)),
        effort=rng.normal(size=(60, 2)),
    )
    sampler = TrajectorySampler(traj)
    probes = np.concatenate([times, rng.uniform(times[0] - 0.1, times[-1] + 0.1, 400), [times[0], times[-1]]])
    for time in probes:
        oracle = traj.joint_state_at(float(time), extrapolate="clamp")
        position, velocity, effort = sampler.sample(float(time))
        assert np.array_equal(position, oracle.position), time
        assert np.array_equal(velocity, oracle.velocity), time
        assert np.array_equal(effort, oracle.effort), time


def test_sampler_without_velocities_uses_the_segment_slope_and_zero_effort():
    traj = JointTrajectory(
        t=[0.0, 1.0, 3.0], joint_names=JOINTS, positions=[[0.0, 0.0], [1.0, -2.0], [1.0, 2.0]]
    )
    sampler = TrajectorySampler(traj)
    position, velocity, effort = sampler.sample(2.0)
    assert position == pytest.approx([1.0, 0.0])
    assert velocity == pytest.approx([0.0, 2.0])  # (2 - -2) / (3 - 1)
    assert np.array_equal(effort, [0.0, 0.0])
    assert np.array_equal(sampler.sample(5.0)[1], [0.0, 0.0])  # held at the end: at rest


# ── JTC vs tape feeder: time-driven vs index-driven ─────────────────────────────────────────────────────────


def test_jtc_tracks_a_min_jerk_move_on_a_board_pd_plant():
    traj = min_jerk_trajectory(rest_s=0.3)
    plant = ToyBoardPlant()
    times, measured = run_on_plant(JointTrajectoryController(traj), plant, period_s=PERIOD_S, duration_s=1.3)
    reference = np.array([traj.joint_state_at(t, extrapolate="clamp").position for t in times])
    error_deg = np.degrees(np.abs(reference - measured)).max()
    # A 10 ms zero-order hold lags by about qd x period / 2: 0.75 rad/s peak -> ~0.2 deg.
    assert error_deg < 0.5, error_deg
    assert measured[-1] == pytest.approx([0.4, -0.3], abs=math.radians(0.05))


def test_jtc_is_time_driven_a_late_tick_jumps_ahead():
    traj = ramp_trajectory()
    jtc = JointTrajectoryController(traj)
    state = state_at()
    for _ in range(5):  # ticks at t = 0, 0.01, ..., 0.04
        command, _ = jtc.tick(state, PERIOD_S)
    assert command.position == pytest.approx(traj.positions[4])
    command, _ = jtc.tick(state, 3 * PERIOD_S)  # three periods late
    assert command.position == pytest.approx(traj.positions[7], abs=1e-12)  # 4 + 3 samples: on schedule
    assert jtc.clock_s == pytest.approx(0.07)


def test_tape_feeder_advances_one_sample_per_tick_and_reports_the_slip():
    traj = ramp_trajectory()
    tape = TapeFeeder(traj, TapeFeederConfig(control_period_s=PERIOD_S))
    state = state_at()
    for _ in range(5):
        command, _ = tape.tick(state, PERIOD_S)
    assert command.position == pytest.approx(traj.positions[4])
    assert tape.diagnostics()["schedule_slip_s"] == pytest.approx(0.0, abs=1e-12)
    command, _ = tape.tick(state, 3 * PERIOD_S)
    assert command.position == pytest.approx(traj.positions[5], abs=1e-12)  # one sample, whatever dt says
    assert tape.diagnostics()["schedule_slip_s"] == pytest.approx(2 * PERIOD_S)
    command, _ = tape.tick(state, 0.0)  # even a zero-length period advances the tape
    assert command.position == pytest.approx(traj.positions[6], abs=1e-12)
    assert tape.diagnostics()["mode"] == "index_driven"


def test_first_tick_samples_t0_whatever_dt_the_loop_passes():
    traj = ramp_trajectory()
    for ctrl in (JointTrajectoryController(traj), TapeFeeder(traj), JointImpedanceController(traj)):
        command, _ = ctrl.tick(state_at(), 0.5)
        assert command.name == JOINTS
        assert command.position == pytest.approx(traj.positions[0])  # the board impedance sends q_ref


def test_jtc_command_fields_leave_the_gains_to_the_board():
    jtc = JointTrajectoryController(ramp_trajectory(), JointTrajectoryConfig(max_effort_nm=(12.0, 8.0)))
    jtc.tick(state_at(), PERIOD_S)
    command, _ = jtc.tick(state_at(), PERIOD_S)
    assert command.stiffness is None and command.damping is None
    assert command.velocity == pytest.approx([1.0, -0.5])
    assert command.feedforward_effort == pytest.approx([0.7, -0.2])  # the trajectory's effort
    assert command.max_effort == pytest.approx([12.0, 8.0])
    no_effort = JointTrajectory(t=[0.0, 1.0], joint_names=JOINTS, positions=[[0.0, 0.0], [1.0, 1.0]])
    command, _ = JointTrajectoryController(no_effort).tick(state_at(), PERIOD_S)
    assert np.array_equal(command.feedforward_effort, [0.0, 0.0]) and command.max_effort is None


# ── replanning ──────────────────────────────────────────────────────────────────────────────────────────────


def test_replace_input_splices_the_new_trajectory_at_now():
    first = ramp_trajectory()
    jtc = JointTrajectoryController(first)
    state = state_at()
    for _ in range(31):  # through t = 0.30
        jtc.tick(state, PERIOD_S)
    here = first.joint_state_at(0.30)
    # The new trajectory starts where the old one is at that instant and heads back to zero.
    second = JointTrajectory(
        t=[0.0, 0.5],
        joint_names=JOINTS,
        positions=[here.position, [0.0, 0.0]],
        velocities=[here.velocity, [0.0, 0.0]],
    )
    jtc.replace_input(second)
    assert isinstance(jtc, Replannable)
    command, _ = jtc.tick(state, PERIOD_S)  # the tick of the same wake: t = 0 of the new trajectory
    assert command.position == pytest.approx(here.position)  # no jump at the splice
    command, _ = jtc.tick(state, PERIOD_S)
    assert command.position == pytest.approx(second.joint_state_at(PERIOD_S).position)
    assert jtc.diagnostics()["active_trajectory"] == 1 and jtc.status == "running"


def test_replace_input_restarts_a_finished_jtc_and_leaves_it_alone_when_refused():
    config = JointTrajectoryConfig(max_effort_nm=(12.0, 8.0))
    jtc = JointTrajectoryController(ramp_trajectory(duration_s=0.05), config)
    for _ in range(10):
        jtc.tick(state_at(), PERIOD_S)
    assert jtc.status == "done"
    with pytest.raises(ValueError):  # 3 joints vs a 2-joint max effort: a refused input changes nothing
        jtc.replace_input(JointTrajectory(t=[0.0], joint_names=("a", "b", "c"), positions=[[0.0, 0.0, 0.0]]))
    assert jtc.status == "done" and jtc.active_input == 0
    jtc.replace_input(ramp_trajectory())
    assert jtc.status == "running" and jtc.active_input == 1


def test_tape_feeder_refuses_a_new_trajectory_while_running():
    traj = ramp_trajectory(duration_s=0.05)
    tape = TapeFeeder(traj)
    command, _ = tape.tick(state_at(), PERIOD_S)
    with pytest.raises(NotReplannable) as refusal:
        tape.replace_input(ramp_trajectory(slope=(-1.0, 1.0)))
    assert isinstance(refusal.value, RuntimeError)
    command, _ = tape.tick(state_at(), PERIOD_S)
    assert command.position == pytest.approx(traj.positions[1])  # still the old tape
    for _ in range(10):
        tape.tick(state_at(), PERIOD_S)
    assert tape.status == "done"
    tape.replace_input(ramp_trajectory(slope=(-1.0, 1.0)))  # v1 takes a new trajectory once holding
    assert tape.status == "running" and tape.active_input == 1


# ── the end and the guard ───────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [s.name for s in in_box(JOINT_TRAJECTORY, built_only=True)])
def test_done_after_the_end_and_keeps_commanding_the_end_at_rest(name):
    traj = ramp_trajectory(duration_s=0.1)
    ctrl = controller(name).build(traj)
    statuses = []
    for _ in range(25):
        command, _ = ctrl.tick(state_at(), PERIOD_S)
        statuses.append(ctrl.status)
    assert statuses[9] == "running" and statuses[10:] == ["done"] * 15  # t = 0.10 is the end
    assert set(statuses) <= {"running", "done"}
    if name.startswith("joint_impedance_host"):
        assert np.isnan(command.position).all()
    else:
        assert command.position == pytest.approx(traj.positions[-1])
    if command.velocity is not None:
        assert np.array_equal(command.velocity, [0.0, 0.0])


def test_path_tolerance_aborts_and_holds_where_the_arm_was():
    traj = ramp_trajectory()
    guarded = JointTrajectoryController(traj, JointTrajectoryConfig(path_tolerance_rad=0.05))
    unguarded = JointTrajectoryController(traj)
    guarded.tick(state_at(), PERIOD_S)
    far = state_at(shoulder=(0.3, 0.0), elbow=(-0.01, 0.0))  # shoulder 0.3 rad off the path
    command, _ = guarded.tick(far, PERIOD_S)
    assert guarded.status == "aborted" and "shoulder" in guarded.message and "deg" in guarded.message
    assert command.position == pytest.approx([0.3, -0.01]) and np.array_equal(command.velocity, [0.0, 0.0])
    moved = state_at(shoulder=(0.5, 0.0), elbow=(0.2, 0.0))
    command, _ = guarded.tick(moved, PERIOD_S)
    assert command.position == pytest.approx([0.3, -0.01])  # latched, not re-read
    for _ in range(3):
        unguarded.tick(far, PERIOD_S)
    assert unguarded.status == "running"  # default: no guard


# ── joint impedance: board, host, host_spring ──────────────────────────────────────────────────────────────


def impedance_reference() -> JointTrajectory:
    """At t = 0.5: q_ref = (0.2, -0.1), qd_ref = (0.2, 0.2), effort_ref = (1.0, -0.5)."""
    return JointTrajectory(
        t=[0.0, 1.0],
        joint_names=JOINTS,
        positions=[[0.1, -0.2], [0.3, 0.0]],
        velocities=[[0.2, 0.2], [0.2, 0.2]],
        effort=[[1.0, -0.5], [1.0, -0.5]],
    )


def tick_at_half_second(config: JointImpedanceConfig, state: JointState) -> ServoCommand:
    ctrl = JointImpedanceController(impedance_reference(), config)
    ctrl.tick(state, PERIOD_S)  # t = 0
    command, defines = ctrl.tick(state, 0.5)  # t = 0.5
    assert defines == {}
    return command


ELBOW_DAMPING = 2 * 0.8 * math.sqrt(25.0 * 0.03)  # 1.3856... N·m·s/rad


def test_damping_is_two_zeta_sqrt_k_i_unless_given():
    config = JointImpedanceConfig()
    # shoulder: 2 * 0.8 * sqrt(40 * 0.1) = 3.2
    assert config.damping(JOINTS) == pytest.approx([3.2, ELBOW_DAMPING], rel=1e-15)
    explicit = JointImpedanceConfig(damping_nm_s_rad=(5.0, 1.0))
    assert explicit.damping(JOINTS) == pytest.approx([5.0, 1.0])
    assert JointImpedanceConfig(stiffness_nm_rad=100.0, inertia_kg_m2=0.01, damping_ratio=1.0).damping(
        JOINTS
    ) == pytest.approx([2.0, 2.0])
    with pytest.raises(ValueError):
        JointImpedanceConfig(location="cloud")
    with pytest.raises(ValueError):
        JointImpedanceController(
            JointTrajectory(t=[0.0], joint_names=("a", "b", "c"), positions=[[0.0, 0.0, 0.0]]), config
        )  # a 2-joint stiffness on a 3-joint trajectory


def test_host_impedance_feedforward_is_exactly_the_spring_plus_damper_from_the_measured_state():
    state = state_at(shoulder=(0.05, 0.5), elbow=(-0.3, -0.1))
    command = tick_at_half_second(JointImpedanceConfig(location="host"), state)
    # shoulder: 40 (0.2 - 0.05) + 3.2 (0.2 - 0.5) + 1.0 = 6.0 - 0.96 + 1.0
    # elbow:    25 (-0.1 - -0.3) + D (0.2 - -0.1) - 0.5 = 5.0 + 0.3 D - 0.5
    expected = [6.04, 4.5 + 0.3 * ELBOW_DAMPING]
    assert command.feedforward_effort == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert np.isnan(command.position).all()
    assert np.array_equal(command.stiffness, [0.0, 0.0]) and np.array_equal(command.damping, [0.0, 0.0])
    assert command.velocity is None and command.name == JOINTS


def test_host_spring_sends_board_damping_around_the_reference_velocity():
    state = state_at(shoulder=(0.05, 0.5), elbow=(-0.3, -0.1))
    command = tick_at_half_second(JointImpedanceConfig(location="host_spring"), state)
    assert command.feedforward_effort == pytest.approx([6.0 + 1.0, 5.0 - 0.5], rel=1e-12)  # K e + effort_ref
    assert np.isnan(command.position).all()
    assert np.array_equal(command.stiffness, [0.0, 0.0])
    assert command.damping == pytest.approx([3.2, ELBOW_DAMPING], rel=1e-15)
    assert command.velocity == pytest.approx([0.2, 0.2])


def test_board_impedance_sends_k_and_d_with_the_reference():
    state = state_at(shoulder=(0.05, 0.5), elbow=(-0.3, -0.1))
    command = tick_at_half_second(JointImpedanceConfig(location="board", max_effort_nm=6.0), state)
    assert command.position == pytest.approx([0.2, -0.1])
    assert command.velocity == pytest.approx([0.2, 0.2])
    assert command.feedforward_effort == pytest.approx([1.0, -0.5])
    assert command.stiffness == pytest.approx([40.0, 25.0])
    assert command.damping == pytest.approx([3.2, ELBOW_DAMPING], rel=1e-15)
    assert command.max_effort == pytest.approx([6.0, 6.0])


def test_host_impedance_needs_measured_velocity():
    ctrl = JointImpedanceController(impedance_reference(), JointImpedanceConfig(location="host"))
    with pytest.raises(ValueError, match="velocities"):
        ctrl.tick(JointState(name=JOINTS, position=[0.0, 0.0]), PERIOD_S)


def test_the_three_impedance_locations_render_the_same_law_at_a_fast_host_rate():
    # At 2 kHz with no bus delay, closing K and D on the host or on the board must give the same motion; a
    # wrong sign or a gain on the wrong joint in any rendering would split them apart (or diverge).
    traj = min_jerk_trajectory(duration_s=0.6, rest_s=0.4)
    runs = {}
    for location in ("board", "host", "host_spring"):
        ctrl = JointImpedanceController(traj, JointImpedanceConfig(location=location))
        _, runs[location] = run_on_plant(ctrl, ToyBoardPlant(), period_s=0.5e-3, duration_s=1.0)
    for location in ("host", "host_spring"):
        split_deg = np.degrees(np.abs(runs[location] - runs["board"])).max()
        assert split_deg < 0.05, (location, split_deg)
    assert runs["board"][-1] == pytest.approx([0.4, -0.3], abs=math.radians(0.5))


# ── the registry builds every built controller ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [s.name for s in in_box(JOINT_TRAJECTORY, built_only=True)])
def test_every_built_trajectory_controller_builds_through_the_registry(name):
    traj = min_jerk_trajectory()
    ctrl = controller(name).build(traj)
    assert isinstance(ctrl, Controller) and isinstance(ctrl, SelfReport) and isinstance(ctrl, Replannable)
    command, defines = ctrl.tick(state_at(), PERIOD_S)
    assert isinstance(command, ServoCommand) and command.name == JOINTS and defines == {}
    assert command.header.stamp == 0.0 and command.state_stamp is None  # the loop stamps it, never the ctrl
    report = ctrl.diagnostics()
    assert report["status"] == "running" and report["active_trajectory"] == 0
    assert all(isinstance(v, (int, float, str)) for v in report.values())
    config = controller(name).make_config()
    for config_field in fields(config):  # the site prints the parameters table from this metadata
        assert config_field.metadata["doc"] and "unit" in config_field.metadata


def test_registry_configs_put_each_impedance_variant_where_it_belongs():
    expected = {
        "joint_impedance_board": "board",
        "joint_impedance_host": "host",
        "joint_impedance_host_spring": "host_spring",
    }
    for name, location in expected.items():
        ctrl = controller(name).build(min_jerk_trajectory())
        assert ctrl.location == location
        assert ctrl.diagnostics()["damper_on"] == ("host" if location == "host" else "board")


def test_joint_command_sequence_controller_builds_through_the_registry_and_runs_to_done():
    sequence = JointCommandSequence(
        joint_names=JOINTS,
        steps=(
            JointCommandStep(
                name="move to",
                commands=(
                    JointCommand.to("shoulder", 0.2, speed=1.5, max_effort=12.0),
                    JointCommand.to("elbow", -0.1, speed=1.5, max_effort=8.0),
                ),
                done_when=(Reached(joints=JOINTS, tolerance=math.radians(1.0)),),
                timeout=5.0,
            ),
        ),
    )
    ctrl = controller("joint_command_sequence_controller").build(sequence)
    assert isinstance(ctrl, Controller)
    plant = ToyBoardPlant()
    for _ in range(300):
        command, _ = ctrl.tick(plant.state(), PERIOD_S)
        plant.apply(command, PERIOD_S)
        if ctrl.status != "running":
            break
    assert ctrl.status == "done", ctrl.message
    assert plant.q == pytest.approx([0.2, -0.1], abs=math.radians(1.0))
