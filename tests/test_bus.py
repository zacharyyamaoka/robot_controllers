"""Amendment 4, §4.2, beyond the acceptance file: the board's per-joint latch queues, the plant's first-in
first-out rule when a jittered command would overtake the one before it, the jitter's own generator, and
HostTiming's validation."""

from __future__ import annotations

import numpy as np
import pytest
from robot_msgs.control import ServoCommand
from robot_msgs.std import Header

from robot_controllers.bench.board import ServoBoards
from robot_controllers.bench.plant import PlanarArmPlant
from robot_controllers.bench.types import HostTiming, Scene

JOINTS = ("shoulder", "elbow")
DT = 0.00025


def command(position, stamp=0.0):
    return ServoCommand(name=JOINTS, position=np.asarray(position, float), header=Header(stamp=stamp))


def test_the_board_latches_each_joint_at_its_own_time_and_refuses_an_out_of_order_joint():
    boards = ServoBoards(JOINTS)
    boards.reset(0.0, [0.0, 0.0])
    boards.schedule(command([0.1, 0.2]), [0.001, 0.003], tag=7)
    seen = []
    for k in range(20):
        boards.step(k * DT, [0.0, 0.0])
        seen.append(tuple(boards.latched_tag))
    assert seen[3] == (-1, -1) and seen[4] == (7, -1)  # shoulder at 1 ms
    assert seen[11] == (7, -1) and seen[12] == (7, 7)  # elbow at 3 ms
    boards.schedule(command([0.0, 0.0]), [0.010, 0.010], tag=8)
    with pytest.raises(ValueError):
        boards.schedule(command([0.0, 0.0]), [0.011, 0.009], tag=9)  # the elbow would overtake tag 8


def _plant(timing, seed=0):
    plant = PlanarArmPlant()
    plant.reset(Scene(), seed, timing=timing, duration_s=0.5)
    return plant


def test_a_command_that_would_overtake_the_one_sent_before_it_latches_on_the_same_tick():
    plant = _plant(HostTiming(command_delay_s=(0.004, 0.004)))
    first = plant.write(command(plant.start_q, stamp=0.000))
    # the bus cannot reorder: shrink the delay for the second command by changing the plant's delays mid-run
    plant._delays = (0.001, 0.001)
    second = plant.write(command(plant.start_q, stamp=0.001))
    assert first == pytest.approx(0.004) and second == pytest.approx(0.004)


def test_the_jitter_draws_from_its_own_generator_in_whole_board_ticks():
    timing = HostTiming(command_delay_jitter_s=0.002)
    latches = []
    for seed in (5, 5, 6):
        plant = _plant(timing, seed)
        latches.append([plant.write(command(plant.start_q, stamp=0.01 * k)) - 0.01 * k for k in range(40)])
    np.testing.assert_array_equal(latches[0], latches[1])  # seeded
    assert latches[0] != latches[2]
    delays = np.asarray(latches[0])
    assert np.all(np.abs(delays - 0.0055) <= 0.002 + 1e-12)
    ticks = (delays - 0.0055) / DT
    np.testing.assert_allclose(ticks, np.round(ticks), atol=1e-6)  # whole ticks
    assert np.std(delays) > 0.0005


def test_host_timing_validates_and_spells_its_delays():
    assert HostTiming().command_delays(2) == (0.0055, 0.0055)
    assert HostTiming(command_delay_s=[0.002, 0.008]).command_delay_s == (0.002, 0.008)  # hashable
    with pytest.raises(ValueError):
        HostTiming(command_delay_s=(0.002, 0.008)).command_delays(6)
    for bad in ({"command_delay_s": -0.001}, {"command_delay_s": ()}, {"command_delay_jitter_s": -1e-3}):
        with pytest.raises(ValueError):
            HostTiming(**bad)
    with pytest.raises(ValueError):
        _plant(HostTiming(command_delay_s=(0.001, 0.002, 0.003)))  # the planar arm has two joints
