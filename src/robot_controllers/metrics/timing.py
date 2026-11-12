"""Timing metrics: how regular, how fresh and how late the host's commands are.

`timing_metrics` is the black-box core: it reads only the stamps every command carries (`sent_t`,
`state_stamp`, `state_seq`) and the nominal period, so the same function scores a sim log, a rig log or a
ros2_control statistics stream. `schedule_slip_ms` and `watchdog_trips` read two more observations (the
controller's declared `status`, the board's mode) and live in their own functions.

Every function here is pure on arrays and returns plain floats/ints/None. WHY pure: an online monitor can call
the same functions on a rolling window (or keep the running sums they reduce to) without an EpisodeLog.
"""

from __future__ import annotations

import math

import numpy as np

WATCHDOG_MODE = 11  # moteus mode 11: the command watchdog expired (CONTRACT §3 board model)


def _ms(seconds: float) -> float:
    return float(seconds) * 1000.0


def timing_metrics(
    sent_t: np.ndarray,
    state_stamp: np.ndarray,
    state_seq: np.ndarray,
    nominal_period_s: float,
) -> dict[str, float | int | None]:
    """Jitter, missed cycles, worst stall, latency and stale replies from the stamped commands alone.

    - period = the gap between two sent commands; jitter = |period - nominal| (absolute: a late tick and an
      early one are both irregular, and a signed median would let them cancel).
    - missed cycles per gap = max(0, round(period / nominal) - 1), rounding half up: a 2.5x gap skipped two
      ticks and counts two (numpy's round-half-even would round 2.5 down and count one).
    - latency = sent_t - state_stamp (sensor-to-command); an open-loop tick (NaN stamp) is skipped.
    - a stale reply = a tick whose state seq did not advance past the previous tick's.
    """
    sent = np.asarray(sent_t, dtype=float).reshape(-1)
    stamps = np.asarray(state_stamp, dtype=float).reshape(-1)
    seqs = np.asarray(state_seq, dtype=float).reshape(-1)
    out: dict[str, float | int | None] = {
        "period_jitter_p50_ms": None,
        "period_jitter_p99_ms": None,
        "period_jitter_max_ms": None,
        "missed_cycles": None,
        "worst_stall_ms": None,
        "latency_p50_ms": None,
        "latency_p99_ms": None,
        "latency_max_ms": None,
        "stale_replies": None,
    }
    if sent.size >= 2 and nominal_period_s > 0:
        periods = np.diff(sent)
        jitter = np.abs(periods - nominal_period_s)
        out["period_jitter_p50_ms"] = _ms(np.percentile(jitter, 50.0))
        out["period_jitter_p99_ms"] = _ms(np.percentile(jitter, 99.0))
        out["period_jitter_max_ms"] = _ms(np.max(jitter))
        cycles = np.floor(periods / nominal_period_s + 0.5)
        out["missed_cycles"] = int(np.sum(np.maximum(0.0, cycles - 1.0)))
        out["worst_stall_ms"] = _ms(np.max(periods))
    latency = sent - stamps if stamps.size == sent.size else np.array([])
    latency = latency[np.isfinite(latency)]
    if latency.size:
        out["latency_p50_ms"] = _ms(np.percentile(latency, 50.0))
        out["latency_p99_ms"] = _ms(np.percentile(latency, 99.0))
        out["latency_max_ms"] = _ms(np.max(latency))
    if seqs.size >= 1:
        finite = np.isfinite(seqs)
        previous, current = seqs[:-1], seqs[1:]
        both = finite[:-1] & finite[1:]
        out["stale_replies"] = int(np.sum(current[both] <= previous[both]))
    return out


def schedule_slip_ms(
    sent_t: np.ndarray,
    status: np.ndarray,
    input_index: np.ndarray,
    schedule_end_s: float | None,
) -> tuple[float | None, str]:
    """How late the controller declares its input finished vs the input's schedule, in ms, plus a note.

    The controller's own clock "ends" at the first tick of the last active input whose `status` is no longer
    `running` (done or aborted); the schedule ends at `schedule_end_s` (the active input's start plus its
    duration). A time-driven controller skips ahead after a late tick, so it ends within a period of the
    schedule; a tick-counting one ends late by every tick it missed.

    None when the schedule did not end inside the run, or the controller never declared an end (the note says
    which: a lower bound is never reported as the value, because it could pass a gate it should fail).
    """
    if schedule_end_s is None or not math.isfinite(schedule_end_s):
        return None, "not measured: no input schedule"
    sent = np.asarray(sent_t, dtype=float).reshape(-1)
    states = np.asarray(status).reshape(-1)
    indices = np.asarray(input_index).reshape(-1)
    if sent.size == 0 or schedule_end_s > sent[-1]:
        return None, "not measured: the input's schedule ends after the run"
    last_input = indices[-1] if indices.size else 0
    on_last = indices == last_input if indices.size == sent.size else np.ones(sent.size, dtype=bool)
    ended = on_last & (states != "running")
    if not ended.any():
        lower_bound = _ms(sent[-1] - schedule_end_s)
        return None, f"never declared its input done (slip > {lower_bound:.0f} ms)"
    first_end = sent[int(np.argmax(ended))]
    # WHY signed: an early end (negative) is reported as it is, never clamped to 0.
    return _ms(first_end - schedule_end_s), ""


def watchdog_trips(board_mode: np.ndarray) -> int:
    """Entries into moteus mode 11, per board (joint), summed; a log that starts in mode 11 counts one."""
    modes = np.asarray(board_mode)
    if modes.ndim == 1:
        modes = modes[:, None]
    if modes.shape[0] == 0:
        return 0
    in_watchdog = modes == WATCHDOG_MODE
    entries = in_watchdog[1:] & ~in_watchdog[:-1]
    return int(np.sum(entries) + np.sum(in_watchdog[0]))
