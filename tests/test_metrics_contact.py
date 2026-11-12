"""Contact and push metrics: a hand-built force trace and tool path give known numbers."""

import numpy as np
import pytest
from test_metrics_helpers import (
    DWELL,
    belt_log,
    belt_rung,
    belt_trace,
    make_log,
    make_rung,
    push_log,
    push_path,
    push_rung,
)

from robot_controllers.metrics.contact import contact_metrics, debounced, push_metrics
from robot_controllers.metrics.episode import episode_metrics, episode_metrics_and_notes


def test_a_40_n_spike_then_a_10_n_dwell():
    t, force, in_contact, tool = belt_trace()
    out = contact_metrics(t, force, in_contact, tool, DWELL, 10.0)
    assert out["contact_peak_n"] == pytest.approx(40.0)
    assert out["contact_max_n"] == pytest.approx(50.0)  # the later knock: the peak window must not reach it
    assert out["dwell_force_mean_n"] == pytest.approx(10.0)
    assert out["dwell_force_std_n"] == pytest.approx(0.0)
    assert out["dwell_force_error_n"] == pytest.approx(0.0)
    assert out["dwell_contact_fraction"] == pytest.approx(100.0)
    assert out["dwell_bounces"] == 0
    assert out["touch_speed_m_s"] == pytest.approx(0.05)


def test_bounces_are_debounced_2_ms():
    t, force, in_contact, tool = belt_trace()
    flicker = (t >= 0.8) & (t < 0.801)  # 1 ms out of contact: chatter, not a bounce
    loss = (t >= 1.0) & (t < 1.005)  # 5 ms out of contact: a bounce
    in_contact = in_contact & ~flicker & ~loss
    force[flicker | loss, 2] = 0.0
    out = contact_metrics(t, force, in_contact, tool, DWELL, 10.0)
    assert out["dwell_bounces"] == 1  # without the debounce it would be 2
    in_dwell = (t >= DWELL[0]) & (t <= DWELL[1])
    expected_fraction = 100.0 * np.mean(in_contact[in_dwell])
    assert out["dwell_contact_fraction"] == pytest.approx(expected_fraction)
    assert out["dwell_force_mean_n"] == pytest.approx(10.0 * expected_fraction / 100.0)
    assert out["dwell_force_error_n"] == pytest.approx(10.0 - out["dwell_force_mean_n"])


def test_a_bounce_outside_the_dwell_does_not_count():
    t, force, in_contact, tool = belt_trace()
    in_contact = in_contact & ~((t >= 0.52) & (t < 0.53))  # bounces off the belt before the dwell opens
    assert contact_metrics(t, force, in_contact, tool, DWELL, 10.0)["dwell_bounces"] == 0


def test_debounced_signal_changes_only_after_holding():
    t = np.arange(0.0, 0.02, 0.001)
    raw = np.zeros(t.size, dtype=bool)
    raw[3:4] = True  # 1 ms blip
    raw[8:15] = True  # 7 ms on
    state, changes = debounced(t, raw, 0.002)
    assert list(changes) == pytest.approx([0.008, 0.015])
    assert not state[3] and state[10] and not state[16]


def test_dwell_slip_is_the_rms_error_along_the_belt_in_mm():
    t, force, in_contact, tool = belt_trace()
    tool[t >= DWELL[0], 0] += 0.002  # dragged 2 mm along the belt for the whole dwell
    out = contact_metrics(
        t,
        force,
        in_contact,
        tool,
        DWELL,
        10.0,
        ref_tool_t=np.array([0.0, 2.0]),
        ref_tool_pos=np.array([[0.30, 0, 0.15]] * 2),
    )
    assert out["dwell_slip_mm"] == pytest.approx(2.0)


def test_never_touching_is_zero_newtons_measured_and_no_transient():
    t, force, in_contact, tool = belt_trace()
    out = contact_metrics(t, np.zeros_like(force), np.zeros_like(in_contact), tool, DWELL, 10.0)
    assert out["contact_max_n"] == 0.0 and out["contact_peak_n"] is None and out["touch_speed_m_s"] is None
    assert out["dwell_contact_fraction"] == 0.0 and out["dwell_force_error_n"] == pytest.approx(10.0)


def test_contact_through_episode_metrics_uses_the_meta_dwell_and_target():
    metrics = episode_metrics(belt_log(), belt_rung())
    assert metrics["contact_peak_n"] == pytest.approx(40.0)
    assert metrics["contact_max_n"] == pytest.approx(50.0)
    assert metrics["dwell_force_mean_n"] == pytest.approx(10.0)


def test_a_free_space_rung_reports_contact_as_not_measured_not_zero():
    metrics, notes = episode_metrics_and_notes(make_log(), make_rung())
    for name in (
        "touch_speed_m_s",
        "contact_peak_n",
        "contact_max_n",
        "dwell_force_mean_n",
        "dwell_bounces",
        "dwell_contact_fraction",
        "dwell_slip_mm",
    ):
        assert metrics[name] is None
    assert "no belt" in notes["contact_max_n"]


def test_push_deflection_recovery_and_overshoot():
    t, tool = push_path()
    out, note = push_metrics(t, tool, 0.5, 0.9, np.array([1.0, 0.0, 0.0]))
    assert out["push_deflection_mm"] == pytest.approx(3.0)
    assert out["push_overshoot_mm"] == pytest.approx(2.0)
    # 1.00 s is inside 1 mm but leaves after 20 ms; recovery needs 50 ms inside, from 1.05 s
    assert out["push_recovery_s"] == pytest.approx(0.15)
    assert note == ""


def test_never_recovering_is_none_with_a_note():
    t, tool = push_path()
    tool[t >= 0.9, 0] = 0.30 + 0.004
    out, note = push_metrics(t, tool, 0.5, 0.9, np.array([1.0, 0.0, 0.0]))
    assert out["push_recovery_s"] is None and "never back within 1 mm" in note


def test_push_through_episode_metrics_from_meta_or_from_the_scene():
    log = push_log()
    from_scene = episode_metrics(log, push_rung())
    assert from_scene["push_recovery_s"] == pytest.approx(0.15)
    assert from_scene["push_deflection_mm"] == pytest.approx(3.0)
    log.meta["push"] = (0.5, 0.9, 0)
    from_meta = episode_metrics(log, make_rung("R4.SidewaysPush", family="disturbance"))
    assert from_meta["push_overshoot_mm"] == pytest.approx(2.0)
    # read as a -x push, the -2 mm after release is along the push and the largest +x excursion (1.5 mm at
    # 1.02-1.05 s) is the overshoot
    log.meta["push"] = (0.5, 0.9, "-x")
    assert episode_metrics(log, push_rung())["push_overshoot_mm"] == pytest.approx(1.5)
