"""Feed-forward and feedback as pure functions, and the layout that says where and how often each is evaluated
(CONTRACT amendment 3, §3.1-3.2).

Zach, 2026-11-20: "The feedforward models it should accept as pure functions. the feedback model as well.
Success to me looks like: Given a feed forward model, Given a feedback model. what is the best way you
basically lay these things out...?"

    from robot_controllers.models import PD, InertiaScaledPD, Layout, ZERO_FEEDFORWARD
    ff = lambda q, qd, qdd: 3.0 * np.cos(q)            # a FeedforwardModel: (q, qd, qdd) -> tau
    fb = PD((300.0, 200.0), (7.5, 4.0))                 # a FeedbackModel the board can close: it has gains()
    layout = Layout("plan", "board")                    # ff once per trajectory sample, fb on the servo board

Conventions for both kinds of model: numpy float arrays in the trajectory's joint order, SI (rad, rad/s,
rad/s², N·m); shape `(n,)` -> `(n,)` and `(N, n)` -> `(N, n)`. **Pure:** no state, no clock, no I/O, no
randomness; equal inputs give equal outputs and the inputs are never written. A feedback that a servo board
can close also has `gains(q_ref) -> (kp, kd)` and is exactly `kp·(q_ref − q) + kd·(qd_ref − qd)` with those
gains; one without `gains` runs on the host only.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

FeedforwardModel = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
"""`ff(q, qd, qdd) -> tau`: the torque that makes the arm follow (q, qd, qdd), from a model of it."""


@runtime_checkable
class FeedbackModel(Protocol):
    """`fb(q_ref, qd_ref, q, qd) -> tau`: the torque that pulls the measured state toward the reference."""

    def __call__(
        self, q_ref: np.ndarray, qd_ref: np.ndarray, q: np.ndarray, qd: np.ndarray
    ) -> np.ndarray: ...


@runtime_checkable
class BoardFeedback(FeedbackModel, Protocol):
    """A feedback a servo board can close: exactly `kp·(q_ref − q) + kd·(qd_ref − qd)` with `gains(q_ref)`."""

    def gains(self, q_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...


def has_gains(feedback: Any) -> bool:
    """Whether `feedback` can move onto the board (it says which kp, kd it is)."""
    return feedback is not None and callable(getattr(feedback, "gains", None))


def _per_joint(value: Any, name: str) -> float | tuple[float, ...]:
    """A gain as a float or a tuple of floats, finite and >= 0 (the board only takes non-negative gains)."""
    if np.ndim(value) == 0:
        number = float(value)
        if not (math.isfinite(number) and number >= 0.0):
            raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        return number
    numbers = tuple(float(v) for v in value)
    if not numbers or not all(math.isfinite(v) and v >= 0.0 for v in numbers):
        raise ValueError(f"{name} values must be finite and >= 0, got {value!r}")
    return numbers


@dataclass(frozen=True)
class PD:
    """Constant-gain joint PD, `tau = kp·(q_ref − q) + kd·(qd_ref − qd)`; one value, or one per joint."""

    kp_nm_rad: float | tuple[float, ...]
    kd_nm_s_rad: float | tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "kp_nm_rad", _per_joint(self.kp_nm_rad, "kp_nm_rad"))
        object.__setattr__(self, "kd_nm_s_rad", _per_joint(self.kd_nm_s_rad, "kd_nm_s_rad"))
        kp = np.asarray(self.kp_nm_rad, dtype=np.float64)
        kd = np.asarray(self.kd_nm_s_rad, dtype=np.float64)
        kp.setflags(write=False)
        kd.setflags(write=False)
        # WHY cached arrays outside the fields: the law runs every tick, and a tuple -> array conversion per
        # call would be most of its cost. Not fields, so equality, hashing and the ledger see only the gains.
        object.__setattr__(self, "_kp", kp)
        object.__setattr__(self, "_kd", kd)

    def __call__(self, q_ref: Any, qd_ref: Any, q: Any, qd: Any) -> np.ndarray:
        return self._kp * (np.asarray(q_ref, dtype=np.float64) - q) + self._kd * (
            np.asarray(qd_ref, dtype=np.float64) - qd
        )

    def gains(self, q_ref: Any) -> tuple[np.ndarray, np.ndarray]:
        shape = np.shape(q_ref)
        return np.broadcast_to(self._kp, shape).copy(), np.broadcast_to(self._kd, shape).copy()


@dataclass(frozen=True)
class InertiaScaledPD:
    """Computed-torque feedback: `tau = M(q)·(ωn²·e + 2ζωn·ė)`, e = q_ref − q, ωn = 2π·bandwidth.

    On the host it uses the full mass matrix at the measured q. `gains(q_ref)` is the diagonal part a board
    can render, `diag(M(q_ref))·ωn²` and `diag(M(q_ref))·2ζωn`; the coupling is dropped there, which is that
    layout's measured
    cost. `mass_matrix(q) -> (n, n)` must itself be pure.
    """

    bandwidth_hz: float
    damping_ratio: float
    mass_matrix: Callable[[np.ndarray], np.ndarray]

    def __post_init__(self) -> None:
        bandwidth, ratio = float(self.bandwidth_hz), float(self.damping_ratio)
        if not (math.isfinite(bandwidth) and bandwidth > 0.0):
            raise ValueError(f"bandwidth_hz must be finite and > 0, got {self.bandwidth_hz!r}")
        if not (math.isfinite(ratio) and ratio >= 0.0):
            raise ValueError(f"damping_ratio must be finite and >= 0, got {self.damping_ratio!r}")
        if not callable(self.mass_matrix):
            raise TypeError("mass_matrix must be a function q -> (n, n)")
        object.__setattr__(self, "bandwidth_hz", bandwidth)
        object.__setattr__(self, "damping_ratio", ratio)

    @property
    def natural_frequency_rad_s(self) -> float:
        return 2.0 * math.pi * self.bandwidth_hz

    def __call__(self, q_ref: Any, qd_ref: Any, q: Any, qd: Any) -> np.ndarray:
        wn = self.natural_frequency_rad_s
        q = np.asarray(q, dtype=np.float64)
        demand = wn**2 * (np.asarray(q_ref, dtype=np.float64) - q) + 2.0 * self.damping_ratio * wn * (
            np.asarray(qd_ref, dtype=np.float64) - qd
        )
        if q.ndim == 1:
            return np.asarray(self.mass_matrix(q), dtype=np.float64) @ demand
        return np.stack(
            [
                np.asarray(self.mass_matrix(row), dtype=np.float64) @ d
                for row, d in zip(q, demand, strict=True)
            ]
        )

    def gains(self, q_ref: Any) -> tuple[np.ndarray, np.ndarray]:
        wn = self.natural_frequency_rad_s
        q_ref = np.asarray(q_ref, dtype=np.float64)
        if q_ref.ndim == 1:
            diagonal = np.diag(np.asarray(self.mass_matrix(q_ref), dtype=np.float64)).copy()
        else:
            diagonal = np.stack(
                [np.diag(np.asarray(self.mass_matrix(row), dtype=np.float64)) for row in q_ref]
            )
        return diagonal * wn**2, diagonal * (2.0 * self.damping_ratio * wn)


class _ZeroFeedforward:
    """`tau = 0`, the shape of q. A named singleton (not a lambda) so its repr in a ledger is stable."""

    __slots__ = ()

    def __call__(self, q: Any, qd: Any, qdd: Any) -> np.ndarray:
        return np.zeros(np.shape(q), dtype=np.float64)

    def __repr__(self) -> str:
        return "ZERO_FEEDFORWARD"


ZERO_FEEDFORWARD = _ZeroFeedforward()


# ── the layout ─────────────────────────────────────────────────────────────────────────────────────────────

FeedforwardAt = Literal["trajectory", "none", "plan", "host_reference", "host_measured"]
FeedbackAt = Literal["board", "board_scheduled", "host", "split"]
FEEDFORWARD_AT: tuple[str, ...] = ("trajectory", "none", "plan", "host_reference", "host_measured")
FEEDBACK_AT: tuple[str, ...] = ("board", "board_scheduled", "host", "split")
# placements that evaluate the feed-forward model (the others never call it)
MODEL_FEEDFORWARD_AT: tuple[str, ...] = ("plan", "host_reference", "host_measured")


@dataclass(frozen=True)
class Layout:
    """Where each model is evaluated and how often (CONTRACT §3.2).

    - `feedforward_at`: `trajectory` (the trajectory's own effort column, 0 when absent: today's JTC), `none`
      (0), `plan` (ff on every trajectory sample once, at build and at replace_input; the tick samples that
      column), `host_reference` (ff at the sampled reference every tick), `host_measured` (ff at the measured
      position and velocity with the reference acceleration, every tick).
    - `feedback_at`: `board` (position/velocity targets, constant gains from the first sample; no feedback
      model = the board's configured gains), `board_scheduled` (gains(q_ref) every tick), `host` (the whole
      feedback as a
      torque from the measured state, board gains 0), `split` (host spring kp·(q_ref − q), board damper kd).
    - `lookahead_s`: the reference is sampled at clock + lookahead (the bus delay a board-side law lags by).
    """

    feedforward_at: FeedforwardAt = "trajectory"
    feedback_at: FeedbackAt = "board"
    lookahead_s: float = 0.0

    def __post_init__(self) -> None:
        if self.feedforward_at not in FEEDFORWARD_AT:
            raise ValueError(f"feedforward_at must be one of {FEEDFORWARD_AT}, got {self.feedforward_at!r}")
        if self.feedback_at not in FEEDBACK_AT:
            raise ValueError(f"feedback_at must be one of {FEEDBACK_AT}, got {self.feedback_at!r}")
        lookahead = float(self.lookahead_s)
        if not (math.isfinite(lookahead) and lookahead >= 0.0):
            raise ValueError(f"lookahead_s must be finite and >= 0 s, got {self.lookahead_s!r}")
        object.__setattr__(self, "lookahead_s", lookahead)

    @property
    def name(self) -> str:
        """`plan+board` (with `+lookahead=0.008s` when set): how diagnostics and pages name it."""
        text = f"{self.feedforward_at}+{self.feedback_at}"
        return text if self.lookahead_s == 0.0 else f"{text}+lookahead={self.lookahead_s:g}s"

    def check(self, feedforward: Any, feedback: Any) -> None:
        """Raise ValueError when this layout cannot place these models (CONTRACT §3.2's invalid pairs)."""
        if self.feedforward_at in MODEL_FEEDFORWARD_AT and feedforward is None:
            raise ValueError(
                f"feedforward_at={self.feedforward_at!r} evaluates a feed-forward model; none was given"
            )
        if feedforward is not None and not callable(feedforward):
            raise TypeError("feedforward must be a function ff(q, qd, qdd) -> tau")
        if feedback is not None and not callable(feedback):
            raise TypeError("feedback must be a function fb(q_ref, qd_ref, q, qd) -> tau")
        if self.feedback_at in ("host", "board_scheduled", "split") and feedback is None:
            raise ValueError(f"feedback_at={self.feedback_at!r} needs a feedback model; none was given")
        if (
            self.feedback_at in ("board", "board_scheduled", "split")
            and feedback is not None
            and not has_gains(feedback)
        ):
            raise ValueError(
                f"feedback_at={self.feedback_at!r} needs a feedback the board can close (with gains(q_ref)); "
                f"{feedback!r} has none, so it can only run on the host"
            )
