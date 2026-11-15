"""The C++ joint trajectory controller against the Python one: the same black box, two implementations
(CONTRACT amendment 2). Every tick of every case must give the same ServoCommand to 1e-12 (it is bit-exact
today), the same status and the same diagnostics, on randomized trajectories, late and irregular dt, the hold
at the end, replace_input mid-run and the path-tolerance abort.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from robot_msgs.control import ServoCommand
from robot_msgs.sensor import JointState
from robot_msgs.trajectory import JointTrajectory

from robot_controllers import Controller, Replannable, SelfReport, controller
from robot_controllers.controllers.joint_trajectory import JointTrajectoryConfig, JointTrajectoryController
from robot_controllers.controllers.joint_trajectory_cpp import JointTrajectoryControllerCpp, load_library

# The contract tolerance; PARITY_TOLERANCE=0 checks the bit-exactness the module docstring claims (it passes).
TOLERANCE = float(os.environ.get("PARITY_TOLERANCE", "1e-12"))
FIELDS = ("position", "velocity", "feedforward_effort", "max_effort", "stiffness", "damping")


def random_trajectory(rng: np.random.Generator, *, joints: int, samples: int, start: float = 0.0,
                      velocities: bool = True, effort: bool = True) -> JointTrajectory:
    """Irregular, strictly increasing sample times and random numbers in every field."""
    steps = rng.uniform(0.001, 0.05, size=samples - 1)
    times = start + np.concatenate([[0.0], np.cumsum(steps)])
    names = tuple(f"joint_{index}" for index in range(joints))
    return JointTrajectory(
        t=times,
        joint_names=names,
        positions=rng.normal(0.0, 1.0, size=(samples, joints)),
        velocities=rng.normal(0.0, 2.0, size=(samples, joints)) if velocities else None,
        effort=rng.normal(0.0, 3.0, size=(samples, joints)) if effort else None,
    )


def state_for(names: tuple[str, ...], positions: np.ndarray, *, extra: bool = True) -> JointState:
    """A measured state in a shuffled order with an extra joint, so both select by name."""
    order = list(reversed(names))
    values = {name: float(positions[i]) for i, name in enumerate(names)}
    if extra:
        order.insert(1, "gripper")
        values["gripper"] = 0.123
    return JointState(name=tuple(order), position=[values[name] for name in order])


def assert_same_command(python: ServoCommand, cpp: ServoCommand, where: str) -> None:
    assert python.name == cpp.name, where
    for name in FIELDS:
        expected, actual = getattr(python, name), getattr(cpp, name)
        if expected is None or actual is None:
            assert expected is None and actual is None, f"{where}: {name} {expected!r} vs {actual!r}"
            continue
        assert expected.shape == actual.shape, f"{where}: {name}"
        both_nan = np.isnan(expected) & np.isnan(actual)
        difference = np.where(both_nan, 0.0, np.abs(expected - actual))
        assert np.all(difference <= TOLERANCE), f"{where}: {name} differs by {difference.max():.3g}"
    assert python.header.stamp == cpp.header.stamp == 0.0 and cpp.state_stamp is None  # the loop stamps it


def assert_same_report(python, cpp, where: str) -> None:
    expected, actual = python.diagnostics(), cpp.diagnostics()
    for key in ("mode", "status", "active_trajectory", "message"):
        assert expected[key] == actual[key], f"{where}: {key} {expected[key]!r} vs {actual[key]!r}"
    for key in ("clock_s", "trajectory_end_s"):
        assert expected[key] == actual[key], f"{where}: {key}"
    assert python.status == cpp.status, where


def run_pair(trajectory: JointTrajectory, dts, *, config=None, measured=None, replace_at=None):
    """Tick both implementations through `dts`; returns the number of ticks compared."""
    python = JointTrajectoryController(trajectory, config)
    cpp = JointTrajectoryControllerCpp(trajectory, config)
    active = trajectory
    for index, dt in enumerate(dts):
        if replace_at and index in replace_at:
            active = replace_at[index]
            python.replace_input(active)
            cpp.replace_input(active)
        positions = measured(index, active) if measured else np.zeros(active.n_joints)
        state = state_for(active.joint_names, positions)
        command_python, defines_python = python.tick(state, dt)
        command_cpp, defines_cpp = cpp.tick(state, dt)
        assert defines_python == defines_cpp == {}
        assert_same_command(command_python, command_cpp, f"tick {index} dt {dt}")
        assert_same_report(python, cpp, f"tick {index}")
    return len(dts)


def test_the_cpp_controller_is_registered_as_a_second_implementation_of_the_jtc():
    spec = controller("joint_trajectory_controller_cpp")
    reference = controller("joint_trajectory_controller")
    assert spec.implementation == "cpp" and reference.implementation == "python"
    assert spec.variant_of == reference.name and spec.box == reference.box
    assert spec.runs_on == reference.runs_on
    assert spec.config_target == reference.config_target
    built = spec.build(random_trajectory(np.random.default_rng(0), joints=2, samples=5))
    assert isinstance(built, Controller) and isinstance(built, Replannable) and isinstance(built, SelfReport)
    report = built.diagnostics()
    assert report["implementation"] == "cpp" and isinstance(report["law_cpu_ns"], int)
    assert load_library().rc_jtc_abi_version() == 1


@pytest.mark.parametrize("seed", range(12))
def test_randomized_trajectories_with_irregular_dt_match_tick_for_tick(seed):
    rng = np.random.default_rng(seed)
    joints = int(rng.integers(1, 8))
    samples = int(rng.integers(1, 60))
    trajectory = random_trajectory(
        rng, joints=joints, samples=samples, start=float(rng.choice([0.0, 0.0, rng.uniform(0.0, 0.2)])),
        velocities=bool(rng.integers(0, 2)), effort=bool(rng.integers(0, 2)),
    )
    # late ticks (3-10 periods), early ticks, zero dt and ordinary jitter, past the end into the hold
    period = 0.01
    dts = np.concatenate([
        rng.uniform(0.0, 2.0 * period, size=60),
        rng.choice([0.0, period, 3 * period, 10 * period], size=40),
    ])
    rng.shuffle(dts)
    # a per-joint cap, a scalar cap, or none
    max_effort = tuple(rng.uniform(0.5, 5.0, size=joints)) if seed % 3 == 0 else (2.0 if seed % 2 else None)
    config = JointTrajectoryConfig(max_effort_nm=max_effort)
    assert run_pair(trajectory, list(dts), config=config) == 100


def test_hold_at_the_end_is_the_last_sample_at_rest_and_done():
    trajectory = random_trajectory(np.random.default_rng(1), joints=3, samples=11)
    python = JointTrajectoryController(trajectory)
    cpp = JointTrajectoryControllerCpp(trajectory)
    state = state_for(trajectory.joint_names, np.zeros(3))
    for index in range(200):
        command_python, _ = python.tick(state, 0.01)
        command_cpp, _ = cpp.tick(state, 0.01)
        assert_same_command(command_python, command_cpp, f"tick {index}")
    assert cpp.status == "done" and "holding its last sample" in cpp.message
    assert np.array_equal(command_cpp.position, trajectory.positions[-1])  # bit-exact end
    assert np.array_equal(command_cpp.velocity, np.zeros(3))


def test_a_single_sample_trajectory_and_one_starting_late_hold_their_samples():
    rng = np.random.default_rng(2)
    run_pair(random_trajectory(rng, joints=2, samples=1, start=0.3), [0.01] * 50)
    run_pair(random_trajectory(rng, joints=2, samples=6, start=0.25), [0.02] * 50)


def test_replace_input_mid_run_restarts_the_clock_in_any_status():
    rng = np.random.default_rng(3)
    first = random_trajectory(rng, joints=3, samples=20)
    replacements = {
        7: random_trajectory(rng, joints=3, samples=15),       # while running
        40: random_trajectory(rng, joints=3, samples=9),       # after the first one is done
        41: random_trajectory(rng, joints=3, samples=4, velocities=False, effort=False),  # the very next tick
        60: random_trajectory(rng, joints=2, samples=12),      # another joint set
    }
    dts = list(rng.uniform(0.0, 0.03, size=120))
    run_pair(first, dts, replace_at=replacements)


def test_path_tolerance_aborts_on_the_same_tick_and_latches_the_same_hold():
    rng = np.random.default_rng(4)
    trajectory = random_trajectory(rng, joints=3, samples=30)
    config = JointTrajectoryConfig(path_tolerance_rad=(0.4, 0.5, 0.6))

    def measured(index, active):
        reference = active.positions[min(index // 3, len(active) - 1)]
        drift = 0.02 * index * np.array([1.0, -0.5, 0.25])
        values = reference + drift
        if index == 45:
            values[1] = math.nan  # a NaN measurement counts as off the path
        return values

    python = JointTrajectoryController(trajectory, config)
    cpp = JointTrajectoryControllerCpp(trajectory, config)
    statuses = []
    for index in range(80):
        state = state_for(trajectory.joint_names, measured(index, trajectory))
        command_python, _ = python.tick(state, 0.01)
        command_cpp, _ = cpp.tick(state, 0.01)
        assert_same_command(command_python, command_cpp, f"tick {index}")
        assert_same_report(python, cpp, f"tick {index}")
        statuses.append(cpp.status)
    assert "aborted" in statuses and "deg off the path" in cpp.message
    assert command_cpp is cpp.tick(state_for(trajectory.joint_names, np.ones(3)), 0.01)[0]  # latched object
    # a new trajectory is taken after an abort, by both
    run_pair(trajectory, [0.01] * 5, config=config,
             replace_at={2: random_trajectory(rng, joints=3, samples=8)})


def test_nan_measurement_aborts_both_with_a_nan_hold():
    trajectory = random_trajectory(np.random.default_rng(5), joints=2, samples=10)
    config = JointTrajectoryConfig(path_tolerance_rad=10.0)
    python = JointTrajectoryController(trajectory, config)
    cpp = JointTrajectoryControllerCpp(trajectory, config)
    state = state_for(trajectory.joint_names, np.array([0.0, math.nan]))
    command_python, _ = python.tick(state, 0.01)
    command_cpp, _ = cpp.tick(state, 0.01)
    assert python.status == cpp.status == "aborted"
    assert_same_command(command_python, command_cpp, "nan")
    assert python.message == cpp.message


@pytest.mark.parametrize("dt", [math.nan, -0.001, math.inf])
def test_an_invalid_dt_raises_the_same_error_and_changes_nothing(dt):
    trajectory = random_trajectory(np.random.default_rng(6), joints=2, samples=10)
    python = JointTrajectoryController(trajectory)
    cpp = JointTrajectoryControllerCpp(trajectory)
    state = state_for(trajectory.joint_names, np.zeros(2))
    python.tick(state, 0.01)
    cpp.tick(state, 0.01)
    with pytest.raises(ValueError) as python_error:
        python.tick(state, dt)
    with pytest.raises(ValueError) as cpp_error:
        cpp.tick(state, dt)
    assert str(python_error.value) == str(cpp_error.value)
    assert python.clock_s == cpp.clock_s == 0.0
    assert_same_command(python.tick(state, 0.02)[0], cpp.tick(state, 0.02)[0], "after the refusal")


def test_a_refused_input_changes_nothing_in_either():
    trajectory = random_trajectory(np.random.default_rng(7), joints=2, samples=10)
    config = JointTrajectoryConfig(max_effort_nm=(1.0, 2.0))
    python = JointTrajectoryController(trajectory, config)
    cpp = JointTrajectoryControllerCpp(trajectory, config)
    three = JointTrajectory(t=[0.0], joint_names=("a", "b", "c"), positions=[[0.0, 0.0, 0.0]])
    for implementation in (python, cpp):
        with pytest.raises(ValueError):
            implementation.replace_input(three)
        assert implementation.active_input == 0 and implementation.status == "running"
    run_pair(trajectory, [0.01] * 20, config=config)


def test_a_missing_joint_raises_keyerror_in_both():
    trajectory = random_trajectory(np.random.default_rng(8), joints=2, samples=10)
    short = JointState(name=("joint_0",), position=[0.0])
    for implementation in (JointTrajectoryController(trajectory), JointTrajectoryControllerCpp(trajectory)):
        with pytest.raises(KeyError):
            implementation.tick(short, 0.01)
