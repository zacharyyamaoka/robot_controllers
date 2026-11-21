"""Lane A's own checks (CONTRACT amendment 5): the fixed targets, the hard stops every plant honours, the bus
delay steps and the monitor's recovery. The acceptance file is tests/test_timing_acceptance.py (the dispatcher's)."""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from robot_controllers.bench.types import HostTiming, Scene
from robot_controllers.timing import BusTiming

V1 = ("trajectory", "trajectory", "board_gains", "board")


def _targets(arm_name):
    from robot_controllers.bench.arms import arm

    return {t.id: t for t in arm(arm_name).targets()}


@pytest.mark.parametrize("arm_name", ["planar_arm", "ur5e"])
def test_the_three_paths_are_one_path_time_scaled_and_nominal_is_the_base_rung(arm_name):
    from robot_controllers.bench.arms import arm

    targets = _targets(arm_name)
    home = np.asarray(arm(arm_name).home_q)
    base = targets[f"{arm_name}/path@nominal"]
    nominal = base.rung().inputs(home)[0][1]
    assert base.params["base_rung"] in ("R2.FastFreeSpace", "U2.FastFreeSpace")
    for speed, scale in (("slow", 2.0), ("fast", 0.9)):
        scaled = targets[f"{arm_name}/path@{speed}"].rung().inputs(home)[0][1]
        np.testing.assert_array_equal(scaled.positions, nominal.positions)
        np.testing.assert_allclose(scaled.t, np.asarray(nominal.t) * scale, rtol=0, atol=1e-12)
        np.testing.assert_allclose(scaled.velocities, np.asarray(nominal.velocities) / scale, rtol=1e-12)
        model = arm(arm_name).feedforward("rigid_body", None)
        np.testing.assert_allclose(scaled.effort, model(scaled.positions, scaled.velocities, scaled.accelerations))


def test_path_at_nominal_runs_exactly_like_r2():
    from robot_controllers.bench.layouts import layout_spec
    from robot_controllers.bench.loop import run_episode
    from robot_controllers.bench.rungs import rung

    spec = layout_spec(*V1, lookahead_s=0.0055)
    exact = dataclasses.replace(rung("R2.FastFreeSpace"), timing=HostTiming(include_compute=False))
    target = dataclasses.replace(_targets("planar_arm")["planar_arm/path@nominal"].rung(), timing=exact.timing)
    np.testing.assert_array_equal(run_episode(spec, exact).q, run_episode(spec, target).q)


@pytest.mark.parametrize("arm_name", ["planar_arm", "ur5e"])
def test_the_calibration_sweep_moves_every_joint_and_starts_and_ends_at_rest_at_home(arm_name):
    from robot_controllers.bench.arms import arm

    home = np.asarray(arm(arm_name).home_q)
    sweep = _targets(arm_name)[f"{arm_name}/calibration"].rung().inputs(home)[0][1]
    assert np.all(np.degrees(np.ptp(sweep.positions, axis=0)) >= 5.0)
    for row in (0, -1):
        np.testing.assert_allclose(sweep.positions[row], home, atol=1e-12)
        np.testing.assert_allclose(sweep.velocities[row], 0.0, atol=1e-12)
    # the analytic velocity is the position's derivative (central differences on the 4 ms grid)
    numeric = np.gradient(np.asarray(sweep.positions), np.asarray(sweep.t), axis=0)
    assert np.max(np.abs(numeric[2:-2] - np.asarray(sweep.velocities)[2:-2])) < 2e-3


@pytest.mark.parametrize("arm_name", ["planar_arm", "ur5e"])
def test_the_collision_stop_holds_and_its_torque_is_logged(arm_name):
    from robot_controllers.bench.layouts import layout_spec
    from robot_controllers.bench.loop import run_episode

    collision = _targets(arm_name)[f"{arm_name}/collision"].rung()
    (joint, angle, side), = collision.scene.stops
    assert side == collision.params["stop_side"] and joint == collision.params["stop_joint_index"]
    log = run_episode(layout_spec(*V1, lookahead_s=0.0055, interpolation="secant_mid_ff", arm=arm_name), collision)
    touches = [e for e in log.events if e["kind"] == "stop_touch"]
    assert len(touches) == 1 and touches[0]["joint"] == log.joint_names[joint]
    beyond = (log.q[:, joint] - angle) if side == "upper" else (angle - log.q[:, joint])
    assert math.degrees(float(beyond.max())) < 0.5  # a stiff wall: the joint never gets half a degree past it
    pressed = log.tau_stop[:, joint]
    assert np.all(pressed <= 0.0) if side == "upper" else np.all(pressed >= 0.0)
    assert np.count_nonzero(pressed) > 0 and float(log.t[np.flatnonzero(pressed)[0]]) == touches[0]["t"]
    others = np.delete(log.tau_stop, joint, axis=1)
    assert not np.any(others)


def test_a_start_past_a_stop_is_refused_and_a_rung_without_stops_logs_none():
    from robot_controllers.bench.plant import PlanarArmPlant

    plant = PlanarArmPlant()
    with pytest.raises(ValueError, match="past its upper stop"):
        plant.reset(Scene(stops=((1, math.radians(-100.0), "upper"),)))
    plant.reset(Scene())
    plant.advance(0.01)
    assert "tau_stop" not in plant.samples()
    with pytest.raises(ValueError):
        Scene(stops=((0, 0.0, "sideways"),))


def test_delay_steps_are_validated_and_the_defaults_have_none():
    assert HostTiming().command_delay_steps == ()
    timing = HostTiming(command_delay_steps=[(1.0, 0.009), (2.0, [0.002, 0.004])])
    assert timing.delay_steps(2) == ((1.0, (0.009, 0.009)), (2.0, (0.002, 0.004)))
    with pytest.raises(ValueError):
        timing.delay_steps(3)
    with pytest.raises(ValueError):
        HostTiming(command_delay_steps=((2.0, 0.009), (1.0, 0.005)))
    with pytest.raises(ValueError):
        HostTiming(command_delay_steps=((1.0, -0.001),))


def _lagged_log(slow_s, from_s, until_s):
    """A synthetic log: a 0.5 Hz sine reference, followed 5.5 ms late, and `slow_s` late in [from_s, until_s)."""
    sent = np.round(np.arange(0.0, 8.0, 0.01), 12)
    ref = np.sin(2 * np.pi * 0.5 * sent)[:, None]
    t = np.round(np.arange(0.0, 8.0, 0.00025), 12)
    lag = np.where((t >= from_s) & (t < until_s), slow_s, 0.0055)
    q = np.interp(t - lag, sent, ref[:, 0])[:, None]
    return SimpleNamespace(t=t, q=q, sent_t=sent, ref_position=ref)


def test_the_monitor_recovers_when_the_bus_comes_back_and_ignores_a_still_joint():
    from robot_controllers.monitor import LagMonitor

    log = _lagged_log(0.0105, 3.0, 5.0)
    monitor = LagMonitor(BusTiming(0.0055), window_s=1.0, step_s=0.25, alert_s=0.002, confirm=2)
    events = []
    for start in np.arange(0.0, 8.0, 0.2):
        rows = (log.t >= start) & (log.t < start + 0.2)
        ticks = (log.sent_t >= start) & (log.sent_t < start + 0.2)
        events += monitor.update(log.t[rows], log.q[rows], log.sent_t[ticks], log.ref_position[ticks])
    kinds = [(e.kind, round(e.measured_s * 1e3, 2)) for e in events]
    assert [k for k, _ in kinds] == ["drift", "recovered"], kinds
    assert 3.0 < events[0].t <= 4.5 and 5.0 < events[1].t <= 6.5
    assert monitor.alerted == (False,)
    still = LagMonitor(BusTiming(0.0055), window_s=1.0, step_s=0.25)
    assert still.update(log.t, np.zeros_like(log.q), log.sent_t, np.zeros_like(log.ref_position)) == []
    assert all(e.measured_s is None for e in still.estimates)


def test_the_calibrate_cli_prints_measured_against_nominal(capsys):
    from robot_controllers.bench.cli import main

    assert main(["calibrate", "--bus-ms", "2", "8", "--nominal-ms", "2", "8", "--friction"]) == 0
    out = capsys.readouterr().out
    assert "planar_arm/calibration·bus2/8ms" in out and "WITHIN" in out
    assert out.count("ms") > 8 and "shoulder" in out and "elbow" in out
