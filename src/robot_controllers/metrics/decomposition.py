"""Decomposition metrics: how the applied torque splits into feed-forward and feedback (CONTRACT amendment 3,
§3.4).

Zach, 2026-11-11: "low feedback torque, where most of it is coming from feedforward terms". Every physics step
the board adds the latched feed-forward field (`tau_ff`) to its own PD term (`tau_fb_board`); a host-side
feedback rides inside that feed-forward field, and the controller's `last_split()` says how much of each host
tick's field it was (`split_fb_host`). `latched_tick` maps each physics step to the host tick whose command
the board ran, so

    host feedback = split_fb_host[latched_tick]    (0 before the first command, in watchdog, or without one)
    feedback      = tau_fb_board + host feedback
    feed-forward  = tau_ff - host feedback          (what the feed-forward model carried)

Pure on arrays, plain numbers out. WHY pure: an online monitor can keep the same numbers per window.
"""

from __future__ import annotations

import numpy as np

WATCHDOG_MODE = 11  # board_mode 11: the board brakes and adds no feed-forward


def host_feedback_per_sample(
    latched_tick: np.ndarray, board_mode: np.ndarray, split_fb_host: np.ndarray | None
) -> np.ndarray:
    """(N, J) host feedback torque inside the feed-forward the board added at each physics step.

    `split_fb_host` (K, J) is per host tick (NaN or None = the controller reports none, counted 0); a step
    whose board
    ran no host command (index -1) or was in watchdog mode has none either.
    """
    ticks = np.atleast_2d(np.asarray(latched_tick, dtype=np.int64))
    out = np.zeros(ticks.shape, dtype=float)
    if split_fb_host is None:
        return out
    split = np.atleast_2d(np.asarray(split_fb_host, dtype=float))
    if split.size == 0:
        return out
    modes = np.atleast_2d(np.asarray(board_mode))
    for joint in range(ticks.shape[1]):
        index = ticks[:, joint]
        ran = (index >= 0) & (index < split.shape[0]) & (modes[:, joint] != WATCHDOG_MODE)
        values = split[np.clip(index, 0, split.shape[0] - 1), joint]
        out[:, joint] = np.where(ran & np.isfinite(values), values, 0.0)
    return out


def _rms(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=0))


def decomposition_metrics(
    tau: np.ndarray, tau_ff: np.ndarray, tau_fb_board: np.ndarray, host_feedback: np.ndarray
) -> dict[str, float | None]:
    """fb_share_pct, fb_rms_nm, ff_rms_nm over the plant samples, each the worst joint.

    fb_share_pct is None when the applied torque is zero on every joint (nothing to share).
    """
    applied = np.atleast_2d(np.asarray(tau, dtype=float))
    if applied.shape[0] == 0:
        return {"fb_share_pct": None, "fb_rms_nm": None, "ff_rms_nm": None}
    feedback = np.atleast_2d(np.asarray(tau_fb_board, dtype=float)) + host_feedback
    feedforward = np.atleast_2d(np.asarray(tau_ff, dtype=float)) - host_feedback
    applied_rms, feedback_rms, feedforward_rms = _rms(applied), _rms(feedback), _rms(feedforward)
    if not (np.all(np.isfinite(feedback_rms)) and np.all(np.isfinite(applied_rms))):
        return {"fb_share_pct": None, "fb_rms_nm": None, "ff_rms_nm": None}
    moving = applied_rms > 0.0
    share = float(np.max(100.0 * feedback_rms[moving] / applied_rms[moving])) if moving.any() else None
    return {
        "fb_share_pct": share,
        "fb_rms_nm": float(np.max(feedback_rms)),
        "ff_rms_nm": float(np.max(feedforward_rms)) if np.all(np.isfinite(feedforward_rms)) else None,
    }
