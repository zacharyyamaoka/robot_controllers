"""The host loop (bench/loop.py): wake schedule, stamping, latency, stalls and the watchdog, script delivery
(replans and refusals), the reference on the intended schedule, and sequence step events.

The controllers here are test-only (a time-driven streamer, and robot_msgs' JointCommandSequenceController):
the built controllers are another lane's and are not imported.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from robot_msgs.control import ServoCommand
from robot_msgs.control.sequence_controller import JointCommandSequenceController

from robot_controllers.bench.arm import ARM, HOME_Q, tool_position
from robot_controllers.bench.loop import run_episode
from robot_controllers.bench.rungs import HoldScript, ladder, rung
from robot_controllers.bench.types import HostTiming, Rung
from robot_controllers.registry import JOINT_COMMAND_SEQUENCE, JOINT_TRAJECTORY

BOARD_DT = ARM.physics_dt_s


class Streamer:
    """Test-only time-driven trajectory streamer: own clock += dt, linear sample, hold the end;
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


class NoReplan(Streamer):
    replace_input = None  # type: ignore[assignment]

    def __init__(self, trajectory):
        Streamer.replace_input(self, trajectory)


class Refuser(Streamer):
    def __init__(self, trajectory):
        super().__init__(trajectory)
        self.refusing = True

    def replace_input(self, trajectory):
        if getattr(self, "refusing", False):
            raise RuntimeError("not while running")
        super().replace_input(trajectory)


class Spec:
    """Duck-typed ControllerSpec: run_episode only needs `.name` and `.build(message)`."""

    def __init__(self, name, factory):
        self.name, self.factory = name, factory

    def build(self, message):
        return self.factory(message)


def _hold_rung(timing: HostTiming, duration: float = 1.2) -> Rung:
    return Rung(
        id="T.Hold",
        title="Hold",
        box=JOINT_TRAJECTORY,
        family="timing",
        description="test",
        duration_s=duration,
        inputs=HoldScript(duration - 0.2),
        timing=timing,
    )


def test_latency_is_the_reply_delay_plus_compute_within_one_board_tick():
    timing = HostTiming(reply_delay_s=0.0015, command_delay_s=0.0055)
    log = run_episode(Spec("streamer", Streamer), _hold_rung(timing), seed=0)
    latency = log.sent_t - log.state_stamp
    assert np.all(log.sent_t == log.tick_t + log.compute_s)
    without_compute = latency - log.compute_s
    assert np.all(without_compute >= timing.reply_delay_s - 1e-9)
    assert np.all(without_compute < timing.reply_delay_s + BOARD_DT + 1e-9)
    assert np.all(np.diff(log.state_seq) == 1)  # one fresh sample per tick at 100 Hz
    assert len(log.tick_t) == 120
    assert len(log.meta["compute_cpu_s"]) == 120 and min(log.meta["compute_cpu_s"]) >= 0.0
    np.testing.assert_allclose(log.tick_t, np.arange(120) * 0.01, atol=1e-12)


def test_without_include_compute_the_command_is_sent_at_the_wake():
    log = run_episode(Spec("streamer", Streamer), _hold_rung(HostTiming(include_compute=False)), seed=0)
    assert np.array_equal(log.sent_t, log.tick_t)
    assert np.all(log.compute_s > 0)


def test_a_stall_window_skips_wakes_and_trips_every_board_watchdog():
    timing = HostTiming(stalls=((0.3, 0.5),))
    log = run_episode(Spec("streamer", Streamer), _hold_rung(timing, duration=1.5), seed=0)
    assert not np.any((log.tick_t >= 0.3) & (log.tick_t < 0.8))
    gaps = np.diff(log.sent_t)
    assert gaps.max() == pytest.approx(0.51, abs=0.002)  # last wake 0.29, next 0.80
    last_before = log.sent_t[log.tick_t < 0.3][-1]
    watchdogs = [event for event in log.events if event["kind"] == "watchdog"]
    assert sorted(event["joint"] for event in watchdogs) == ["elbow", "shoulder"]
    expected = last_before + timing.command_delay_s + 0.25
    for event in watchdogs:
        assert expected - 1e-9 <= event["t"] < expected + BOARD_DT + 1e-9
    braking = (log.t > expected + 0.001) & (log.t < 0.8)
    assert np.all(log.board_mode[braking] == 11)
    assert np.all(log.board_mode[log.t > 0.8 + timing.command_delay_s + 0.001] == 0)
    kinds = [event["kind"] for event in log.events]
    assert "stall_start" in kinds and "stall_end" in kinds


def test_jitter_is_seeded_and_clipped_to_half_a_period():
    # large, so the clip at +-5 ms is exercised; WHY include_compute off: a wake never precedes the previous
    # command's send, and that send carries measured (wall-clock) compute, which no seed reproduces
    timing = HostTiming(jitter_s=0.004, include_compute=False)
    first = run_episode(Spec("streamer", Streamer), _hold_rung(timing), seed=3)
    same = run_episode(Spec("streamer", Streamer), _hold_rung(timing), seed=3)
    other = run_episode(Spec("streamer", Streamer), _hold_rung(timing), seed=4)
    np.testing.assert_array_equal(first.tick_t, same.tick_t)
    assert not np.array_equal(first.tick_t, other.tick_t)
    offsets = first.tick_t - np.round(first.tick_t / 0.01) * 0.01
    assert np.all(np.abs(offsets) <= 0.005 + 1e-12) and np.std(offsets) > 0.001
    assert np.all(np.diff(first.tick_t) >= 0)


def test_replans_are_delivered_before_the_tick_at_their_scheduled_time():
    r5 = rung("R5.RapidReplan")
    log = run_episode(Spec("streamer", Streamer), r5, seed=0)
    replans = [event for event in log.events if event["kind"] == "replan"]
    assert len(replans) == 8
    assert [event["scheduled_t"] for event in replans] == pytest.approx([0.15 * k for k in range(1, 9)])
    for event in replans:
        assert event["t"] == pytest.approx(
            event["scheduled_t"], abs=1e-9
        )  # 0.15 s is a whole number of periods
        tick = int(np.argmin(np.abs(log.tick_t - event["t"])))
        assert log.input_index[tick] == event["index"]  # the tick at the replan already uses the new input
    assert log.meta["targets_rad"] and len(log.meta["targets_rad"]) == 9


def test_a_controller_without_replace_input_or_that_raises_is_recorded_as_refused():
    r5 = rung("R5.RapidReplan")
    for factory, why in ((NoReplan, "no replace_input"), (Refuser, "RuntimeError: not while running")):
        log = run_episode(Spec(factory.__name__, factory), r5, seed=0)
        refused = [event for event in log.events if event["kind"] == "refused"]
        assert len(refused) == 8 and all(why in event["why"] for event in refused)
        assert not any(event["kind"] == "replan" for event in log.events)
        assert log.input_index.max() == 8  # the rung's intended input moves on; the reference follows it


def test_the_reference_is_the_active_input_on_its_intended_schedule():
    r5 = rung("R5.RapidReplan")
    log = run_episode(Spec("streamer", Streamer), r5, seed=0)
    script = r5.inputs(np.asarray(HOME_Q))
    for tick in range(0, len(log.sent_t), 7):
        start_t, trajectory = script[log.input_index[tick]]
        expected = trajectory.joint_state_at(log.sent_t[tick] - start_t, extrapolate="clamp")
        np.testing.assert_allclose(log.ref_position[tick], expected.position, atol=1e-12)
    np.testing.assert_allclose(log.ref_tool_pos, tool_position(log.ref_position), atol=1e-15)
    assert log.meta["schedule_end_s"] == pytest.approx(8 * 0.15 + 0.6)


def test_sequence_steps_become_step_done_events_with_their_targets():
    s1 = rung("S1.MoveTo")
    log = run_episode(Spec("seq", JointCommandSequenceController), s1, seed=0)
    steps = [event for event in log.events if event["kind"] == "step_done"]
    assert [event["outcome"] for event in steps] == ["Reached", "Elapsed"]
    sequence = s1.inputs(np.asarray(HOME_Q))[0][1]
    targets = {c.joint: c.position for c in sequence.steps[0].commands}
    assert steps[0]["target"] == pytest.approx(targets)
    assert log.meta["steps_total"] == 2 and log.ref_position is None and log.ref_tool_pos is None
    reached_tick = int(np.argmin(np.abs(log.tick_t - steps[0]["t"])))
    measured = np.array([np.interp(steps[0]["t"], log.t, log.q[:, j]) for j in range(2)])
    assert np.max(np.abs(measured - np.array(list(targets.values())))) < math.radians(1.5)
    assert log.status[reached_tick] == "running" and log.status[-1] == "done"


def test_a_homing_define_reaches_the_board_and_is_recorded():
    s2 = rung("S2.SeekHardStop")
    log = run_episode(Spec("seq", JointCommandSequenceController), s2, seed=0)
    defines = [event for event in log.events if event["kind"] == "define"]
    assert len(defines) == 1 and defines[0]["joint"] == "elbow"
    assert defines[0]["value"] == pytest.approx(ARM.elbow_range_rad[0])
    steps = [event for event in log.events if event["kind"] == "step_done"]
    assert steps and steps[0]["outcome"] == "Stall elbow"
    # the true elbow is at the stop when it stalled (the define is not a fiction)
    assert np.interp(steps[0]["t"], log.t, log.q[:, 1]) == pytest.approx(
        ARM.elbow_range_rad[0], abs=math.radians(1.0)
    )


def test_episode_log_shapes_and_meta_for_a_belt_rung():
    belt = ladder(JOINT_TRAJECTORY)[3]  # R3 at 0.10 m/s
    log = run_episode(Spec("streamer", Streamer), belt, seed=0)
    ticks, joints = len(log.tick_t), len(log.joint_names)
    for name in (
        "cmd_position",
        "cmd_velocity",
        "cmd_feedforward",
        "cmd_stiffness",
        "cmd_damping",
        "cmd_max_effort",
        "ref_position",
        "ref_velocity",
    ):
        assert getattr(log, name).shape == (ticks, joints), name
    assert log.ref_tool_pos.shape == (ticks, 3)
    assert np.isnan(log.cmd_stiffness).all()  # the streamer leaves the board gains alone
    assert log.meta["dwell"][1] - log.meta["dwell"][0] == pytest.approx(0.8)
    assert log.meta["press_target_n"] == 10.0
    assert log.meta["contact_calibration"]["stiffness_n_m"] == 1.0e5
    touches = log.event_times("touch")
    assert len(touches) == 1 and log.in_contact[log.t >= log.meta["dwell"][0]][:100].all()
    assert JOINT_COMMAND_SEQUENCE != belt.box
