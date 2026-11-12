"""The servo-board model (bench/board.py): the moteus position-mode law, encoder, watchdog and frame defines.

The numeric checks use hand-computed literals, and one test replays a closed loop against bam_pid_tunning's
own moteus model (an independent oracle, written for the rack tuner) when that package is on disk.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from robot_msgs.control import ServoCommand
from robot_msgs.std import Header

from robot_controllers.bench.board import OK_MODE, WATCHDOG_MODE, BoardConfig, ServoBoards

JOINTS = ("shoulder", "elbow")
DT = 1.0 / 4000.0
BAM_PID_TUNNING_SRC = Path("/home/bam/bam_ws/src/core/mdp/setup/bam_pid_tunning/src")


def _command(
    position, *, velocity=None, feedforward=None, stiffness=None, damping=None, max_effort=None, names=JOINTS
) -> ServoCommand:
    def arr(values):
        return None if values is None else np.asarray(values, dtype=float)

    return ServoCommand(
        name=names,
        position=arr(position),
        velocity=arr(velocity),
        feedforward_effort=arr(feedforward),
        stiffness=arr(stiffness),
        damping=arr(damping),
        max_effort=arr(max_effort),
        header=Header(),
    )


def _ideal_boards(**overrides) -> ServoBoards:
    """No encoder delay and no quantisation: the law alone."""
    config = BoardConfig(encoder_delay_s=0.0, encoder_counts_per_rev=0, **overrides)
    return ServoBoards(JOINTS, config)


def test_position_mode_law_with_absolute_gains_nan_position_and_feedforward():
    boards = _ideal_boards()
    boards.reset(0.0, [0.1, -1.0])
    boards.schedule(
        _command(
            [0.2, math.nan],
            velocity=[0.5, 0.3],
            feedforward=[1.0, -0.5],
            stiffness=[50.0, 0.0],
            damping=[2.0, 3.0],
        ),
        latch_t=0.001,
    )
    tau = boards.step(0.001, [0.1, -1.0])
    # shoulder: the setpoint walks one tick at its velocity before the law: 50 (0.2 + 0.5 dt - 0.1) + 2 (0.5 -
    # 0) + 1
    assert tau[0] == pytest.approx(50.0 * (0.2 + 0.5 * DT - 0.1) + 2.0 * 0.5 + 1.0, abs=1e-12)
    # elbow: NaN position = no position term at all, whatever the stiffness
    assert tau[1] == pytest.approx(3.0 * 0.3 - 0.5, abs=1e-12)
    assert boards.mode == [OK_MODE, OK_MODE]


def test_none_gains_use_the_configured_ones_and_missing_fields_count_zero():
    boards = _ideal_boards()
    boards.reset(0.0, [0.0, 0.0])
    boards.schedule(_command([0.01, -0.02]), latch_t=0.0)
    tau = boards.step(0.0, [0.0, 0.0])
    assert tau == pytest.approx([300.0 * 0.01, 200.0 * -0.02], abs=1e-12)


def test_torque_is_clipped_to_the_smaller_of_cap_and_max_effort():
    boards = _ideal_boards()
    boards.reset(0.0, [0.0, 0.0])
    boards.schedule(_command([1.0, -1.0], max_effort=[3.0, 50.0]), latch_t=0.0)
    tau = boards.step(0.0, [0.0, 0.0])
    assert tau == [3.0, -8.0]  # 300 N*m demanded -> max_effort 3; -200 -> the elbow board's cap 8
    assert boards.at_cap == [True, True]
    boards.schedule(_command([0.001, 0.0], max_effort=[3.0, 8.0]), latch_t=DT)
    boards.step(DT, [0.0, 0.0])
    assert boards.at_cap == [False, False]


def test_a_command_latches_on_the_first_tick_at_or_after_its_latch_time():
    boards = _ideal_boards()
    boards.reset(0.0, [0.0, 0.0])
    boards.schedule(_command([0.01, 0.0]), latch_t=0.0055)
    taus = [boards.step(k * DT, [0.0, 0.0])[0] for k in range(30)]
    first = next(k for k, value in enumerate(taus) if value != 0.0)
    assert first == 22  # 0.0055 s / 0.25 ms


def test_encoder_delays_by_half_a_millisecond_and_quantises_to_14_bits():
    boards = ServoBoards(JOINTS, BoardConfig())
    boards.reset(0.0, [0.0, 0.0])
    quantum = 2.0 * math.pi / 2**14
    seen = []
    for k in range(8):
        q_true = 0.0 if k < 3 else 0.0101  # the joint jumps on tick 3
        boards.step(k * DT, [q_true, 0.0])
        seen.append(boards.measured_position[0])
    assert seen[:5] == [0.0] * 5 and seen[5] != 0.0  # two ticks (0.5 ms) late
    assert seen[5] == pytest.approx(round(0.0101 / quantum) * quantum, abs=1e-15)
    assert abs(seen[5] - 0.0101) <= quantum / 2


def test_velocity_is_the_count_derivative_through_a_200_hz_first_order_filter():
    boards = ServoBoards(JOINTS, BoardConfig(encoder_delay_s=0.0, encoder_counts_per_rev=0))
    boards.reset(0.0, [0.0, 0.0])
    speed = 1.0
    velocities = []
    for k in range(9):
        boards.step(k * DT, [speed * k * DT, 0.0])
        velocities.append(boards.measured_velocity[0])
    alpha = 1.0 - math.exp(-2.0 * math.pi * 200.0 * DT)
    expected = [speed * (1.0 - (1.0 - alpha) ** k) for k in range(9)]
    assert velocities == pytest.approx(expected, abs=1e-12)


def test_watchdog_enters_mode_11_after_0_25_s_without_a_command_and_brakes_on_kd():
    boards = ServoBoards(JOINTS, BoardConfig(encoder_delay_s=0.0, encoder_counts_per_rev=0))
    boards.reset(0.0, [0.0, 0.0])
    entries = []
    speed = 0.4
    for k in range(1005):
        boards.step(k * DT, [speed * k * DT, 0.0])
        entries += [(k, j) for j in boards.watchdog_entries]
    assert entries == [(1000, 0), (1000, 1)]  # t = 0.25 s, once per board
    assert boards.mode == [WATCHDOG_MODE, WATCHDOG_MODE]
    assert boards.tau[0] == pytest.approx(
        -7.5 * boards.measured_velocity[0], abs=1e-12
    )  # brake on configured kd
    assert boards.measured_velocity[0] == pytest.approx(speed, rel=1e-6)
    boards.schedule(_command([0.4, 0.0]), latch_t=1005 * DT)
    boards.step(1005 * DT, [speed * 1005 * DT, 0.0])
    assert boards.mode == [OK_MODE, OK_MODE] and boards.watchdog_entries == []


def test_a_joint_the_command_does_not_name_keeps_its_watchdog_counting():
    boards = ServoBoards(JOINTS, BoardConfig())
    boards.reset(0.0, [0.0, 0.0])
    for k in range(1100):
        if k % 40 == 0:
            boards.schedule(_command([0.0], names=("shoulder",)), latch_t=k * DT)
        boards.step(k * DT, [0.0, 0.0])
    assert boards.mode == [OK_MODE, WATCHDOG_MODE]


def test_a_define_shifts_the_reported_position_without_a_velocity_spike():
    boards = ServoBoards(JOINTS, BoardConfig())
    boards.reset(0.0, [0.0, -2.0])
    for k in range(20):
        boards.step(k * DT, [0.0, -2.0])
    boards.schedule(_command([0.0, 1.0]), latch_t=20 * DT, defines={"elbow": 1.0})
    tau = boards.step(20 * DT, [0.0, -2.0])
    assert boards.measured_position[1] == pytest.approx(1.0, abs=1e-12)
    assert boards.measured_velocity[1] == pytest.approx(0.0, abs=1e-12)
    assert abs(tau[1]) < 1e-9  # the new setpoint (1.0, in the new frame) is where the joint now reads


def test_step_response_of_one_joint_settles_with_a_moderate_overshoot():
    boards = ServoBoards(
        ("elbow",), BoardConfig(kp_nm_rad=(200.0,), kd_nm_s_rad=(4.0,), torque_cap_nm=(8.0,))
    )
    boards.reset(0.0, [0.0])
    inertia = 0.05  # kg*m^2: kp/kd above give w_n 63 rad/s, zeta 0.63 before the encoder's delay and filter
    boards.schedule(ServoCommand(name=("elbow",), position=np.array([0.1])), latch_t=0.0)
    q = qd = 0.0
    trace = []
    for k in range(int(0.5 / DT)):
        tau = boards.step(k * DT, [q])[0]
        qd += DT * tau / inertia
        q += qd * DT
        trace.append(q)
    trace = np.array(trace)
    overshoot = (trace.max() - 0.1) / 0.1
    assert 0.02 < overshoot < 0.30
    settled_after = np.flatnonzero(np.abs(trace - 0.1) > 0.002)[-1] * DT
    assert settled_after < 0.15
    assert abs(trace[-1] - 0.1) < 2 * 2 * math.pi / 2**14


def test_board_matches_bam_pid_tunning_controller_tick_tick_for_tick():
    """Independent oracle: bam_pid_tunning's ServoSim (the rack tuner's moteus position loop,
    `controller_tick`, tested tick for tick against moteus_servo.step_moteus_controller) plays a tape on a
    frictionless joint with the contract's encoder (0.5 ms, 14 bits, 200 Hz). Our board on the same joint must
    ask for the same torque and see the same position at every command."""
    if not BAM_PID_TUNNING_SRC.is_dir():
        pytest.skip(f"bam_pid_tunning is not on disk at {BAM_PID_TUNNING_SRC}")
    sys.path.insert(0, str(BAM_PID_TUNNING_SRC))
    try:
        from bam_pid_tunning import tapes
        from bam_pid_tunning.plants.servo_sim import PREROLL_S, Truth, simulate
        from bam_pid_tunning.types import Controller, Envelope, Gains, Servo, Tape
    except ImportError as error:
        pytest.skip(f"bam_pid_tunning is not importable: {error}")
    finally:
        sys.path.remove(str(BAM_PID_TUNNING_SRC))

    kp, kd, cap, inertia, command_hz = 200.0, 4.0, 1.0, 0.05, 100.0  # a 1 N*m cap so the clamp is exercised
    servo = Servo(
        name="elbow",
        gains=Gains(kp=kp, kd=kd),
        torque_limit_nm=cap,
        command_rate_hz=command_hz,
        envelope=Envelope(travel_rad=1.0, max_speed_rad_s=5.0, max_torque_nm=cap, origin_rad=0.0),
    )
    truth = Truth(
        inertia_kg_m2=inertia,
        viscous_nm_s_rad=0.0,
        coulomb_nm=0.0,
        torque_bandwidth_hz=1e12,
        velocity_filter_hz=200.0,
        encoder_counts_per_output_rev=2.0**14,
        measurement_delay_s=0.0005,
        loop_hz=4000.0,
    )
    tape = Tape(name="steps", kind="waypoints", legs=((0.3, 0.25, 0.3), (-0.2, 0.3, 0.2)))
    trace = simulate(servo, truth, tape, Controller(Gains(kp=kp, kd=kd)))
    assert trace.reason == "complete"

    rows = len(trace.t)
    q_ref, qd_ref, _ = tapes.render(tape, np.arange(rows) / command_hz)
    preroll = int(round(PREROLL_S * command_hz))
    ticks_per_command = int(round(4000.0 / command_hz))
    boards = ServoBoards(
        ("elbow",), BoardConfig(kp_nm_rad=(kp,), kd_nm_s_rad=(kd,), torque_cap_nm=(cap,), watchdog_s=100.0)
    )
    boards.reset(0.0, [0.0])
    q = qd = tau_applied = 0.0
    seen_q, asked_tau = [], []
    for tick in range((preroll + rows) * ticks_per_command):
        command_index, phase = divmod(tick, ticks_per_command)
        row = command_index - preroll
        if phase == 0:
            position, velocity = (q_ref[row], qd_ref[row]) if row >= 0 else (0.0, 0.0)
            boards.schedule(
                ServoCommand(
                    name=("elbow",),
                    position=np.array([position]),
                    velocity=np.array([velocity]),
                    feedforward_effort=np.array([0.0]),
                ),
                latch_t=tick * DT,
            )
        tau = boards.step(tick * DT, [q])[0]
        if phase == 0 and row >= 0:
            seen_q.append(boards.measured_position[0])
            asked_tau.append(tau)
        # the oracle's joint with every friction term zero (same arithmetic: a 1.0 lag, semi-implicit Euler)
        tau_applied += 1.0 * (tau - tau_applied)
        qd = qd + DT * tau_applied / inertia
        q += qd * DT
    asked_tau = np.array(asked_tau)
    assert np.max(np.abs(asked_tau)) == pytest.approx(cap)  # the clamp was reached
    np.testing.assert_allclose(np.array(seen_q), trace.q, rtol=0, atol=1e-12)
    np.testing.assert_allclose(asked_tau, trace.tau_nm, rtol=0, atol=1e-9)
