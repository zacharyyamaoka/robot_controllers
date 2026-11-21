"""The live bus-delay monitor: notice when the bus stops being what it was calibrated to (CONTRACT amendment 5, §5.2).

    monitor = LagMonitor(calibration.measured, lookahead_s=layout.lookahead_s)
    for chunk in stream:                                   # whatever arrived since the last call
        for event in monitor.update(chunk.t, chunk.q, chunk.sent_t, chunk.ref_position):
            alert(event)                                   # LagEvent(t, joint, expected_s, measured_s, "drift")

Zach, 2026-11-20: "and of course if that begins to change a lot we should be alerted."

Every `step_s` of data (window ends on multiples of `step_s`, once a whole `window_s` has arrived) it estimates each
moving joint's lag over the last `window_s` the way the one-run calibration does (`metrics.lag.lags_in_window`: the
shift that best lines the measured joint up with its reference), with one more term fitted beside the shift: a
friction-shaped error that flips with the direction of motion. WHY: on slow moves the gearbox's dry friction, which
no feed-forward model carries, reads as up to 2.75 ms of fake lag with a pure shift (R1, 2026-11-20), enough to
alert on a bus that never changed; with the term a 1 s window lands within 0.5 ms of the bus. The measured *delay*
is that lag plus the lookahead
the controller is using: a controller that looks ahead by the true delay shows no lag. It emits a `drift` event when
|measured - expected| > `alert_s` for `confirm` windows in a row, and `recovered` when it is back inside for
`confirm` windows in a row. A joint whose reference moved less than 0.1° in a window is skipped: that window neither
counts toward an alert nor breaks a run of them.

Pure and streaming: no clock, no I/O, no randomness; it only keeps the last window of samples. It never changes the
controller, it reports (re-calibrating automatically is a later decision; an alert is what Zach asked for).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from robot_controllers.metrics.lag import WINDOW_START_S, lags_in_window
from robot_controllers.timing import BusTiming

TIME_TOLERANCE_S = 1e-9
MAX_LAG_S = 0.030  # the shift search range, as estimate_lag_s


@dataclass(frozen=True)
class LagEvent:
    """One alert: at window end `t` (s), joint index `joint` measured `measured_s` against `expected_s`."""

    t: float
    joint: int
    expected_s: float
    measured_s: float
    kind: Literal["drift", "recovered"]


@dataclass(frozen=True)
class LagEstimate:
    """One window's measurement of one joint (None = it did not move, skipped)."""

    t: float
    joint: int
    measured_s: float | None


class LagMonitor:
    """Streaming drift detector on one arm's command delay (module docstring)."""

    def __init__(
        self,
        expected: BusTiming,
        *,
        lookahead_s: float | tuple[float, ...] = 0.0,
        window_s: float = 2.0,
        step_s: float = 0.5,
        alert_s: float = 0.002,
        confirm: int = 2,
    ) -> None:
        for name, value in (("window_s", window_s), ("step_s", step_s)):
            if not (math.isfinite(float(value)) and float(value) > 0.0):
                raise ValueError(f"{name} must be finite and > 0 s, got {value!r}")
        if not (math.isfinite(float(alert_s)) and float(alert_s) >= 0.0):
            raise ValueError(f"alert_s must be finite and >= 0 s, got {alert_s!r}")
        if int(confirm) < 1:
            raise ValueError(f"confirm must be >= 1, got {confirm!r}")
        self.expected = expected
        self.lookahead_s = lookahead_s
        self.window_s = float(window_s)
        self.step_s = float(step_s)
        self.alert_s = float(alert_s)
        self.confirm = int(confirm)
        self.estimates: list[LagEstimate] = []  # every window's measurement, for a page or a log
        self.events: list[LagEvent] = []  # every event emitted so far
        self._joints: int | None = None
        self._t = np.zeros(0)
        self._q = np.zeros((0, 0))
        self._sent = np.zeros(0)
        self._ref = np.zeros((0, 0))
        self._first_sent: float | None = None
        self._first_t: float | None = None
        self._next_end: float | None = None

    # ── state ──────────────────────────────────────────────────────────────────────────────────────────────

    def _start(self, joints: int) -> None:
        self._joints = joints
        self._expected = np.asarray(self.expected.per_joint(joints), dtype=float)
        if np.ndim(self.lookahead_s) == 0:
            self._lookahead = np.full(joints, float(self.lookahead_s))
        else:
            if len(self.lookahead_s) != joints:
                raise ValueError(f"lookahead_s has {len(self.lookahead_s)} values for {joints} joints")
            self._lookahead = np.asarray(self.lookahead_s, dtype=float)
        self._q = np.zeros((0, joints))
        self._ref = np.zeros((0, joints))
        self._over = [0] * joints  # windows in a row outside the alert band
        self._inside = [0] * joints  # windows in a row back inside it
        self._alerted = [False] * joints

    @property
    def alerted(self) -> tuple[bool, ...]:
        """Per joint: is a drift alert standing (emitted and not yet recovered)?"""
        return tuple(self._alerted) if self._joints is not None else ()

    # ── the stream ─────────────────────────────────────────────────────────────────────────────────────────

    def update(self, t: Any, q: Any, sent_t: Any, ref_position: Any) -> list[LagEvent]:
        """Take the plant samples (t (N,), q (N, J)) and host ticks (sent_t (K,), ref_position (K, J)) that arrived
        since the last call; return the events of every window that is now complete."""
        t = np.asarray(t, dtype=float).reshape(-1)
        sent = np.asarray(sent_t, dtype=float).reshape(-1)
        q = np.asarray(q, dtype=float).reshape(t.size, -1) if t.size else np.zeros((0, self._joints or 0))
        ref = (
            np.asarray(ref_position, dtype=float).reshape(sent.size, -1)
            if sent.size
            else np.zeros((0, self._joints or 0))
        )
        if self._joints is None:
            joints = q.shape[1] if t.size else (ref.shape[1] if sent.size else 0)
            if joints == 0:
                return []
            self._start(joints)
        if t.size:
            self._t = np.concatenate([self._t, t])
            self._q = np.concatenate([self._q, q])
            if self._first_t is None:
                self._first_t = float(t[0])
        if sent.size:
            self._sent = np.concatenate([self._sent, sent])
            self._ref = np.concatenate([self._ref, ref])
            if self._first_sent is None:
                self._first_sent = float(sent[0])
        if self._first_t is None or self._first_sent is None or not self._t.size or not self._sent.size:
            return []
        if self._next_end is None:
            first_full = max(self._first_t, self._first_sent) + self.window_s
            self._next_end = math.ceil(first_full / self.step_s - TIME_TOLERANCE_S) * self.step_s
        events: list[LagEvent] = []
        # a window is complete once both streams have reached its end (the last plant sample sits one board
        # tick short of a chunk edge, so its end is "reached" within that tick)
        horizon = min(float(self._t[-1]) + self._board_tick(), float(self._sent[-1]) + TIME_TOLERANCE_S)
        while self._next_end <= horizon + TIME_TOLERANCE_S:
            events += self._window(self._next_end)
            self._next_end += self.step_s
        self._trim(self._next_end - self.window_s - MAX_LAG_S - 0.1)
        self.events += events
        return events

    def _board_tick(self) -> float:
        if self._t.size < 2:
            return 0.0
        return float(self._t[-1] - self._t[-2])

    def _trim(self, before: float) -> None:
        keep_t = self._t >= before
        keep_sent = self._sent >= before
        self._t, self._q = self._t[keep_t], self._q[keep_t]
        self._sent, self._ref = self._sent[keep_sent], self._ref[keep_sent]

    def _window(self, end: float) -> list[LagEvent]:
        start = max(end - self.window_s, float(self._first_sent) + WINDOW_START_S)
        ticks = self._sent <= end + TIME_TOLERANCE_S
        lags = lags_in_window(
            self._t,
            self._q,
            self._sent[ticks],
            self._ref[ticks],
            start_s=start,
            end_s=end,
            max_s=MAX_LAG_S,
            friction=True,
        )
        events = []
        for joint, lag in enumerate(lags):
            if lag is None:
                self.estimates.append(LagEstimate(end, joint, None))
                continue  # WHY skipped, not reset: a still joint says nothing about the bus
            measured = float(lag) + float(self._lookahead[joint])
            expected = float(self._expected[joint])
            self.estimates.append(LagEstimate(end, joint, measured))
            if abs(measured - expected) > self.alert_s:
                self._over[joint] += 1
                self._inside[joint] = 0
                if not self._alerted[joint] and self._over[joint] >= self.confirm:
                    self._alerted[joint] = True
                    events.append(LagEvent(end, joint, expected, measured, "drift"))
            else:
                self._inside[joint] += 1
                self._over[joint] = 0
                if self._alerted[joint] and self._inside[joint] >= self.confirm:
                    self._alerted[joint] = False
                    events.append(LagEvent(end, joint, expected, measured, "recovered"))
        return events
