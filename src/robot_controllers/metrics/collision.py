"""Collision metrics: how soon and how hard a hard stop is felt (CONTRACT amendment 5, §5.6; the collision target only).

    from robot_controllers.metrics.collision import collision_metrics
    collision_metrics(log_with_stop, log_without_stop)
    # {"collision_detect_ms": 18.5, "collision_peak_nm": 42.0, "collision_margin": 3.1}

Zach, 2026-11-20: "perhaps if you want like to really optimize for less impact, or better collision detection".

- **The residual** at each plant sample: the feedback torque (the board's PD term + the host feedback riding in the
  feed-forward field, as `metrics.decomposition`) per joint, low-passed at 50 Hz (first order), over the joint's
  torque cap; the max over joints. It is what a controller can watch without any sensor beyond its own boards.
- **The threshold** = 1.25 × the largest residual of the same config on the same scenario without the stop (the free
  run). WHY the controller's own nominal run sets it: a better feed-forward leaves less residual in free motion, so the
  same stop stands out sooner; that is the link between feed-forward quality and collision detection.
- `collision_detect_ms`: first contact (the first sample the stop pushes back, `tau_stop` != 0) -> the first sample
  at or after it whose residual crosses the threshold. None = never (a fail).
- `collision_peak_nm`: the stop's largest constraint torque (|tau_stop|, max over joints) from first contact up to and
  including the detecting sample; over the whole contact when it is never detected. The impact.
- `collision_margin`: the largest residual while in contact (any |tau_stop| > 0) / the threshold: higher = easier to
  tell the stop apart from free motion.

Pure on arrays (the `*_from_arrays` functions), plain numbers out, so an online detector can reuse the residual.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from robot_controllers.metrics.decomposition import host_feedback_per_sample, low_pass

THRESHOLD_FACTOR = 1.25  # §5.6: the threshold is 1.25 x the free run's largest residual
LOW_PASS_HZ = 50.0  # §5.6: the residual is low-passed at 50 Hz
CONTACT_NM = 1.0e-9  # |tau_stop| above this is contact (the stop's torque is exactly 0 off the stop)
NAMES: tuple[str, ...] = ("collision_detect_ms", "collision_peak_nm", "collision_margin")


def residual(
    t: np.ndarray,
    tau_fb_board: np.ndarray,
    host_feedback: np.ndarray,
    caps_nm: Sequence[float],
    cutoff_hz: float = LOW_PASS_HZ,
) -> np.ndarray:
    """(N,) the residual: |low-pass(board PD term + host feedback)| / cap per joint, the max over joints."""
    feedback = np.atleast_2d(np.asarray(tau_fb_board, dtype=float)) + np.atleast_2d(
        np.asarray(host_feedback, dtype=float)
    )
    caps = np.asarray(caps_nm, dtype=float).reshape(-1)
    if caps.size != feedback.shape[1] or not np.all(caps > 0.0):
        raise ValueError(f"caps_nm must be one positive cap per joint ({feedback.shape[1]}), got {caps_nm!r}")
    filtered = low_pass(np.asarray(t, dtype=float), feedback, cutoff_hz)
    return np.max(np.abs(filtered) / caps[None, :], axis=1)


def collision_from_arrays(
    t: np.ndarray,
    residual_stop: np.ndarray,
    tau_stop: np.ndarray,
    threshold: float,
) -> dict[str, float | None]:
    """The three metrics from the collision run's residual and stop torque and the threshold (free max × 1.25)."""
    out: dict[str, float | None] = {name: None for name in NAMES}
    times = np.asarray(t, dtype=float).reshape(-1)
    stop = np.max(np.abs(np.atleast_2d(np.asarray(tau_stop, dtype=float))), axis=1)
    contact = stop > CONTACT_NM
    if not contact.any() or not math.isfinite(threshold) or threshold <= 0.0:
        return out
    first = int(np.argmax(contact))
    values = np.asarray(residual_stop, dtype=float).reshape(-1)
    crossed = np.flatnonzero(values[first:] > threshold)
    if crossed.size:
        detect = first + int(crossed[0])
        out["collision_detect_ms"] = 1000.0 * float(times[detect] - times[first])
        out["collision_peak_nm"] = float(np.max(stop[first : detect + 1]))
    else:
        out["collision_peak_nm"] = float(np.max(stop[contact]))
    out["collision_margin"] = float(np.max(values[contact])) / threshold
    return out


def log_residual(log: Any) -> np.ndarray | None:
    """The residual of an EpisodeLog (None when it has no board torque split or no board caps in its meta)."""
    if log.tau_fb_board is None or log.latched_tick is None:
        return None
    caps = (log.meta.get("board") or {}).get("torque_cap_nm")
    if caps is None:
        return None
    host = host_feedback_per_sample(log.latched_tick, log.board_mode, log.split_fb_host)
    return residual(log.t, log.tau_fb_board, host, caps)


def collision_metrics(log: Any, free: Any | None) -> tuple[dict[str, float | None], dict[str, str]]:
    """(metrics, notes) for a run with a hard stop (`log.tau_stop`) and the same config's run without it (`free`).

    Every value is None, with a note saying why, when the run has no stop, no free run was given, either log has no
    torque split, or the stop was never touched."""
    metrics: dict[str, float | None] = {name: None for name in NAMES}
    if log.tau_stop is None:
        return metrics, {name: "not measured: no hard stop on this rung" for name in NAMES}
    if free is None:
        why = "not measured: the threshold needs the same config's run without the stop (grade.run_scenario runs it)"
        return metrics, {name: why for name in NAMES}
    stopped, unstopped = log_residual(log), log_residual(free)
    if stopped is None or unstopped is None or unstopped.size == 0:
        return metrics, {name: "not measured: a log has no board torque split or no board caps" for name in NAMES}
    threshold = THRESHOLD_FACTOR * float(np.max(unstopped))
    metrics = collision_from_arrays(log.t, stopped, log.tau_stop, threshold)
    notes: dict[str, str] = {}
    if metrics["collision_peak_nm"] is None:
        notes = {name: "not measured: the arm never touched the stop" for name in NAMES}
    elif metrics["collision_detect_ms"] is None:
        notes["collision_detect_ms"] = (
            f"never detected: the residual never crossed {threshold:.3g} (1.25 x the free run's largest)"
        )
    return metrics, notes
