"""Timing metrics: known gaps, latencies and seqs give known numbers; the black-box core reads only stamps."""

import numpy as np
import pytest
from test_metrics_helpers import make_log, make_rung, trajectory_script

from robot_controllers.metrics.episode import episode_metrics, episode_metrics_and_notes
from robot_controllers.metrics.timing import schedule_slip_ms, timing_metrics, watchdog_trips


def ticks_with_gap(gap_start=0.30, gap_s=0.5, period=0.01, end=1.5):
    before = np.arange(0.0, gap_start + 1e-9, period)
    after = np.arange(gap_start + gap_s, end, period)
    return np.concatenate([before, after])


def test_a_half_second_gap_is_a_500_ms_stall_and_49_missed_cycles_at_100_hz():
    sent = ticks_with_gap()
    out = timing_metrics(sent, sent - 0.003, np.arange(sent.size), 0.01)
    assert out["worst_stall_ms"] == pytest.approx(500.0)
    assert out["missed_cycles"] == 49
    assert out["period_jitter_max_ms"] == pytest.approx(490.0)


def test_the_same_gap_through_episode_metrics():
    log = make_log(sent_t=ticks_with_gap(end=1.0), duration_s=1.0)
    metrics = episode_metrics(log, make_rung())
    assert metrics["worst_stall_ms"] == pytest.approx(500.0)
    assert metrics["missed_cycles"] == 49


def test_constant_latency_of_3_ms_is_3_ms_at_every_percentile():
    sent = np.arange(0.0, 1.0, 0.01)
    out = timing_metrics(sent, sent - 0.003, np.arange(sent.size), 0.01)
    assert out["latency_p50_ms"] == pytest.approx(3.0)
    assert out["latency_p99_ms"] == pytest.approx(3.0)
    assert out["latency_max_ms"] == pytest.approx(3.0)


def test_jitter_is_the_absolute_deviation_not_the_signed_one():
    # 70 % of periods 3 ms early (7 ms), 30 % 1 ms late (11 ms): |dev| is 3 ms for most ticks. A signed jitter
    # would give p50 = -3 ms and p99 = +1 ms; the absolute one gives 3 ms for both.
    periods = np.array([0.007] * 70 + [0.011] * 30)
    rng = np.random.default_rng(0)
    rng.shuffle(periods)
    sent = np.concatenate([[0.0], np.cumsum(periods)])
    out = timing_metrics(sent, sent - 0.001, np.arange(sent.size), 0.01)
    assert out["period_jitter_p50_ms"] == pytest.approx(3.0)
    assert out["period_jitter_p99_ms"] == pytest.approx(3.0)
    assert out["period_jitter_max_ms"] == pytest.approx(3.0)


def test_missed_cycles_round_half_up():
    # a 25 ms gap at 100 Hz skipped the ticks at +10 and +20 ms: 2 missed (numpy's half-even round says 1)
    sent = np.array([0.0, 0.01, 0.035, 0.045])
    out = timing_metrics(sent, sent, np.arange(4), 0.01)
    assert out["missed_cycles"] == 2
    # a 14 ms late tick is not a missed cycle
    assert (
        timing_metrics(np.array([0.0, 0.014, 0.024]), np.zeros(3), np.arange(3), 0.01)["missed_cycles"] == 0
    )


def test_stale_replies_count_ticks_whose_seq_did_not_advance():
    sent = np.arange(7) * 0.01
    out = timing_metrics(sent, sent - 0.002, np.array([0, 1, 1, 2, 2, 2, 3]), 0.01)
    assert out["stale_replies"] == 3


def test_open_loop_ticks_are_skipped_in_latency_and_all_open_loop_is_not_measured():
    sent = np.arange(5) * 0.01
    stamps = sent - 0.004
    stamps[[1, 3]] = np.nan
    assert timing_metrics(sent, stamps, np.arange(5), 0.01)["latency_max_ms"] == pytest.approx(4.0)
    none = timing_metrics(sent, np.full(5, np.nan), np.arange(5), 0.01)
    assert none["latency_p50_ms"] is None and none["latency_max_ms"] is None


def test_one_tick_has_no_period_so_jitter_is_not_measured_rather_than_zero():
    out = timing_metrics(np.array([0.0]), np.array([-0.001]), np.array([0]), 0.01)
    assert (
        out["period_jitter_p50_ms"] is None and out["worst_stall_ms"] is None and out["missed_cycles"] is None
    )
    assert out["latency_p50_ms"] == pytest.approx(1.0)


def test_schedule_slip_is_first_done_tick_minus_schedule_end():
    sent = np.arange(0.0, 1.0, 0.01)
    status = np.where(sent >= 0.5 + 0.12 - 1e-9, "done", "running")
    slip, note = schedule_slip_ms(sent, status, np.zeros(sent.size, dtype=int), 0.5)
    assert slip == pytest.approx(120.0) and note == ""
    never, never_note = schedule_slip_ms(sent, np.array(["running"] * sent.size), np.zeros(sent.size), 0.5)
    assert never is None and "never declared" in never_note
    late, late_note = schedule_slip_ms(sent, status, np.zeros(sent.size), 5.0)
    assert late is None and "after the run" in late_note


def test_schedule_slip_through_episode_rebuilds_the_schedule_from_the_rung_script():
    log = make_log(duration_s=1.0)  # the script's trajectory lasts 0.6 s
    log.status = np.where(log.sent_t >= 0.65 - 1e-9, "done", "running").astype(object)
    metrics, notes = episode_metrics_and_notes(log, make_rung(inputs=trajectory_script(0.6)))
    assert metrics["schedule_slip_ms"] == pytest.approx(50.0)
    log.meta["schedule_end_s"] = 0.62  # a schedule end the loop recorded wins over the rebuild
    assert episode_metrics(log, make_rung())["schedule_slip_ms"] == pytest.approx(30.0)


def test_no_slip_is_measured_against_an_input_the_controller_refused():
    log = make_log(duration_s=1.0)
    log.input_index = np.where(log.sent_t >= 0.3, 1, 0)
    log.events = [{"t": 0.3, "kind": "refused", "index": 1}]
    log.status = np.where(log.sent_t >= 0.6, "done", "running").astype(object)
    log.meta["schedule_end_s"] = 0.9
    metrics, notes = episode_metrics_and_notes(log, make_rung())
    assert metrics["schedule_slip_ms"] is None  # not -300 ms "early"
    assert "refused the input" in notes["schedule_slip_ms"]


def test_watchdog_trips_count_entries_per_board():
    modes = np.zeros((10, 2), dtype=int)
    modes[2:4, 0] = 11  # one entry on joint 0
    modes[6:8, 0] = 11  # a second entry on joint 0
    modes[3:9, 1] = 11  # one long entry on joint 1
    assert watchdog_trips(modes) == 3
