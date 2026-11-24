"""The simulated bus calibration is a property of the bus, not of the host running it (2026-11-24)."""

from __future__ import annotations

import robot_controllers.calibration as calibration
from robot_controllers.bench import population
from robot_controllers.calibration import datasheet_bus


def test_scenario_calibration_runs_with_the_compute_time_left_out(monkeypatch):
    seen = []

    def spy(spec_at, rung, **kwargs):
        seen.append(rung.timing.include_compute)
        raise RuntimeError("stop after the rung is built")

    monkeypatch.setattr(calibration, "calibrate_bus", spy)
    population.scenario_calibration.cache_clear()
    try:
        population.scenario_calibration("planar_arm", datasheet_bus(2), 50.0)
    except RuntimeError:
        pass
    finally:
        population.scenario_calibration.cache_clear()
    assert seen == [False]
