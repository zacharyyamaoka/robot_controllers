"""Amendment 4, §4.4, beyond the acceptance file: the variants compose, keep the base untouched, refuse what
they cannot re-time, and R10 at every study speed is reachable with its window on the cruise."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robot_controllers.bench.arm import fk
from robot_controllers.bench.rungs import bus_variant, rate_variant, rung, speed_variant, sync_variant


def test_variants_compose_and_name_every_step():
    r2 = rung("R2.FastFreeSpace")
    composed = bus_variant(rate_variant(speed_variant(r2, 0.5), 25), (0.0015, 0.0037), 0.001)
    assert composed.id == "R2.FastFreeSpace·move500ms·host25Hz·bus1.5/3.7ms±1ms"
    assert composed.timing.rate_hz == 25.0 and composed.timing.command_delay_s == (0.0015, 0.0037)
    assert composed.timing.command_delay_jitter_s == 0.001
    assert {(g.metric, g.value) for g in composed.gates if g.metric == "latency_p99_ms"} == {
        ("latency_p99_ms", 40.0)
    }
    assert composed.params["move_s"] == 0.5 and composed.params["host_rate_hz"] == 25.0
    assert composed.params["command_delay_ms"] == "1.5/3.7"
    assert r2.timing.command_delay_s == 0.0055 and r2.params["move_s"] == 0.3  # the base is untouched


def test_speed_variant_refuses_a_script_it_cannot_retime():
    with pytest.raises(ValueError):
        speed_variant(rung("R5.RapidReplan"), 0.4)  # a replan chain's move_s is not a free-space path
    with pytest.raises(ValueError):
        speed_variant(rung("R4.SidewaysPush"), 0.4)  # no move_s at all


def test_speed_variant_peak_speeds_match_the_analytic_tool_velocity():
    from robot_controllers.bench.arm import jacobian

    variant = speed_variant(rung("R1.FreeSpace"), 0.5)
    trajectory = variant.inputs(np.zeros(2))[0][1]
    tool_v = np.einsum("nij,nj->ni", jacobian(trajectory.positions), trajectory.velocities)
    peak = np.linalg.norm(tool_v, axis=1).max()
    assert variant.params["peak_tool_speed_m_s"] == pytest.approx(peak, rel=2e-3)
    assert variant.params["peak_joint_speed_deg_s"] == pytest.approx(
        math.degrees(np.abs(trajectory.velocities).max())
    )


@pytest.mark.parametrize("speed", [0.10, 0.25, 0.50, 1.00])
def test_the_sync_rung_at_every_study_speed(speed):
    sync = sync_variant(speed)
    assert sync.id == f"R10.BeltSync@{speed:.2f}m/s" and sync.family == "sync" and not sync.scene.belt
    start, end = sync.params["sync_start_s"], sync.params["sync_end_s"]
    assert end - start == pytest.approx(0.20 / speed)
    trajectory = sync.inputs(np.asarray(sync.scene.start_q))[0][1]
    x, z = fk(trajectory.positions)
    t = np.asarray(trajectory.t)
    assert fk(np.asarray(sync.scene.start_q)) == pytest.approx((0.11, 0.20), abs=1e-12)
    assert (x[0], x[-1]) == pytest.approx((0.11, 0.40), abs=1e-12)
    np.testing.assert_allclose(z, 0.20, atol=1e-9)
    speeds = np.gradient(x, t)
    assert speeds.max() <= speed * (1 + 1e-3) and speeds.min() >= -1e-9  # never faster than the item
    assert sync.duration_s > t[-1]
