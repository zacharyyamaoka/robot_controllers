"""Amendment 4, §4.3, beyond the acceptance file: the lag estimate's tie-break and its None cases, the
calibration's median fill and clip, and the sync metrics on a hand-made window."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_controllers.metrics import lag


def _log(shift_s, still=(False, False), joints=2):
    sent_t = np.round(np.arange(0.0, 1.5, 0.01), 12)
    ref = np.stack([0.3 * np.sin(2 * np.pi * (0.9 + 0.3 * j) * sent_t) for j in range(joints)], axis=1)
    for j, flat in enumerate(still):
        if flat:
            ref[:, j] = 0.1
    t = np.round(np.arange(0.0, 1.5, 0.00025), 12)
    shifts = np.broadcast_to(shift_s, (joints,))
    q = np.stack([np.interp(t - shifts[j], sent_t, ref[:, j]) for j in range(joints)], axis=1)
    return SimpleNamespace(t=t, q=q, sent_t=sent_t, ref_position=ref)


def test_per_joint_shifts_come_back_one_each_and_the_whole_arm_lies_between():
    log = _log(np.array([0.002, 0.008]))
    assert lag.estimate_lag_s(log, per_joint=True) == pytest.approx((0.002, 0.008), abs=1e-9)
    assert 0.002 <= lag.estimate_lag_s(log) <= 0.008


def test_a_hold_and_a_log_without_reference_have_no_lag():
    assert lag.estimate_lag_s(_log(0.003, still=(True, True))) is None
    assert lag.estimate_lag_s(_log(0.003, still=(True, True)), per_joint=True) == (None, None)
    bare = SimpleNamespace(t=np.zeros(3), q=np.zeros((3, 2)), sent_t=np.zeros(3), ref_position=None)
    assert lag.estimate_lag_s(bare) is None and lag.estimate_lag_s(bare, per_joint=True) == (None, None)


def test_a_flat_residual_ties_to_the_smallest_shift():
    shifts = lag._shifts(0.001, 0.00025)
    assert shifts[0] == 0.0 and abs(shifts[1]) == pytest.approx(0.00025) and shifts.size == 9


def test_calibration_clips_at_zero_and_fills_a_still_joint_with_the_median(monkeypatch):
    import robot_controllers.bench.loop as loop

    logs = iter([_log(np.array([-0.002, 0.004, 0.006]), still=(False, False, True), joints=3)])
    monkeypatch.setattr(loop, "run_episode", lambda spec, rung, seed=0: next(logs))
    asked = []
    calibrated = lag.calibrate_lookahead(lambda la: asked.append(la) or "spec", SimpleNamespace(id="R"))
    assert asked == [0.0]
    assert calibrated == pytest.approx((0.0, 0.004, 0.002), abs=1e-9)  # median of (0, 4) ms


def test_sync_metrics_measure_a_tool_trailing_its_target_by_a_known_time():
    speed, behind_s = 0.25, 0.004
    sent_t = np.round(np.arange(0.0, 2.0, 0.01), 12)
    ref_tool = np.stack([0.1 + speed * sent_t, np.zeros_like(sent_t), np.full_like(sent_t, 0.2)], axis=1)
    t = np.round(np.arange(0.0, 2.0, 0.00025), 12)
    tool = np.stack([0.1 + speed * (t - behind_s), np.zeros_like(t), np.full_like(t, 0.2)], axis=1)
    values = lag.sync_metrics(t, tool, sent_t, ref_tool, 0.5, 1.5, speed)
    assert values["sync_timing_ms"] == pytest.approx(4.0, abs=1e-9)
    assert values["sync_error_p95_mm"] == pytest.approx(1.0, abs=1e-9)  # 4 ms at 0.25 m/s
    assert lag.sync_metrics(t, tool, sent_t, ref_tool, 5.0, 6.0, speed) == {
        "sync_error_p95_mm": None, "sync_timing_ms": None
    }
