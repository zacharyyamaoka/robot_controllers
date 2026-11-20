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

The share (CONTRACT §3.8) is the arm's, in traj_compare's formula summed over the joints:

    fb_share_pct = 100 · Σ_joints Σ_samples |fb| / (Σ|ff| + Σ|fb|)

WHY not the worst joint's RMS ratio (the first sweep): it picked the elbow, whose ~0.6 N·m at home is mostly
the board's kd times the encoder buzz, and it passed 100 % when feedback opposed the feed-forward.
`fb_share_lf_pct` is the same after a first-order 50 Hz low-pass of ff and fb per joint along time: what a
better model could still take off the feedback, with the buzz (a floor no model removes) filtered out.

Pure on arrays, plain numbers out. WHY pure: an online monitor can keep the same numbers per window.
"""

from __future__ import annotations

import numpy as np

WATCHDOG_MODE = 11  # board_mode 11: the board brakes and adds no feed-forward
LOW_PASS_HZ = 50.0  # fb_share_lf_pct's first-order low-pass (CONTRACT §3.8)


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


def share_pct(feedforward: np.ndarray, feedback: np.ndarray) -> float | None:
    """100 · Σ|fb| / (Σ|ff| + Σ|fb|) over every joint and sample where both are finite (traj_compare's
    fb_share, summed over the joints); None when there is no torque at all."""
    ff, fb = (
        np.atleast_2d(np.asarray(feedforward, dtype=float)),
        np.atleast_2d(np.asarray(feedback, dtype=float)),
    )
    both = np.isfinite(ff) & np.isfinite(fb)
    fb_sum = float(np.sum(np.abs(fb[both])))
    total = float(np.sum(np.abs(ff[both]))) + fb_sum
    return 100.0 * fb_sum / total if total > 0.0 else None


def low_pass(t: np.ndarray, values: np.ndarray, cutoff_hz: float = LOW_PASS_HZ) -> np.ndarray:
    """First-order low-pass per column along time, started at the first sample: y += a (x - y), with
    a = 1 - exp(-2π f dt) for each step's own dt."""
    times = np.asarray(t, dtype=float).reshape(-1)
    x = np.atleast_2d(np.asarray(values, dtype=float))
    out = np.empty_like(x)
    if x.shape[0] == 0:
        return out
    alphas = (1.0 - np.exp(-2.0 * np.pi * cutoff_hz * np.diff(times))).tolist()
    for column in range(x.shape[1]):
        series = x[:, column].tolist()
        state = series[0]
        filtered = [state]
        for alpha, sample in zip(alphas, series[1:], strict=True):
            state += alpha * (sample - state)
            filtered.append(state)
        out[:, column] = filtered
    return out


def decomposition_metrics(
    tau: np.ndarray,
    tau_ff: np.ndarray,
    tau_fb_board: np.ndarray,
    host_feedback: np.ndarray,
    plant_t: np.ndarray | None = None,
) -> dict[str, float | None]:
    """fb_share_pct and fb_share_lf_pct (the arm's share, CONTRACT §3.8; the low-passed one needs `plant_t`),
    fb_rms_nm and ff_rms_nm (worst joint, unfiltered)."""
    applied = np.atleast_2d(np.asarray(tau, dtype=float))
    empty = {"fb_share_pct": None, "fb_share_lf_pct": None, "fb_rms_nm": None, "ff_rms_nm": None}
    if applied.shape[0] == 0:
        return empty
    feedback = np.atleast_2d(np.asarray(tau_fb_board, dtype=float)) + host_feedback
    feedforward = np.atleast_2d(np.asarray(tau_ff, dtype=float)) - host_feedback
    feedback_rms, feedforward_rms = _rms(feedback), _rms(feedforward)
    if not np.all(np.isfinite(feedback_rms)):
        return empty
    low_passed = None
    if plant_t is not None and np.size(plant_t) == feedback.shape[0] and np.all(np.isfinite(feedforward)):
        low_passed = share_pct(low_pass(plant_t, feedforward), low_pass(plant_t, feedback))
    return {
        "fb_share_pct": share_pct(feedforward, feedback),
        "fb_share_lf_pct": low_passed,
        "fb_rms_nm": float(np.max(feedback_rms)),
        "ff_rms_nm": float(np.max(feedforward_rms)) if np.all(np.isfinite(feedforward_rms)) else None,
    }
