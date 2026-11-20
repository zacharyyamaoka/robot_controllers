"""Health metrics: torque saturation, faults and divergence.

Pure on arrays, plain ints/floats out. WHY pure: an online monitor can apply the same checks per window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FAULT_MODE = 1  # board_mode 1 = fault (EpisodeLog: 0 ok, 1 fault, 11 watchdog)


@dataclass(frozen=True)
class DivergenceBounds:
    """What counts as a blown-up episode. WHY this generous: `diverged` flags a numerical blow-up (a NaN, a
    sim explosion, a runaway integrator), not a large but legitimate motion; the planar arm's joints stop well
    inside these, its board caps torque at 12 N·m.
    """

    max_abs_position_rad: float = 4.0 * np.pi  # rad, measured or commanded; two turns either way
    max_abs_velocity_rad_s: float = 100.0  # rad/s, measured or commanded
    max_abs_effort_nm: float = 1.0e3  # N·m, applied torque or commanded feed-forward
    max_tool_distance_m: float = 5.0  # m, tool distance from the world origin


DEFAULT_BOUNDS = DivergenceBounds()


def saturation_share(at_cap: np.ndarray) -> float | None:
    """% of plant samples in which at least one joint's torque is clipped at its cap."""
    flags = np.asarray(at_cap, dtype=bool)
    if flags.ndim == 1:
        flags = flags[:, None]
    if flags.shape[0] == 0:
        return None
    return 100.0 * float(np.mean(np.any(flags, axis=1)))


def torque_rate_rms_nm_s(plant_t: np.ndarray, tau: np.ndarray) -> float | None:
    """RMS of |d tau / dt| over the board samples, worst joint, N·m/s (chatter, a host-rate law's steps)."""
    times = np.asarray(plant_t, dtype=float).reshape(-1)
    torques = np.atleast_2d(np.asarray(tau, dtype=float))
    if times.size < 2:
        return None
    widths = np.diff(times)
    valid = widths > 0
    rates = np.diff(torques, axis=0)[valid] / widths[valid, None]
    if rates.size == 0 or not np.all(np.isfinite(rates)):
        return None
    return float(np.max(np.sqrt(np.mean(rates**2, axis=0))))


def fault_count(board_mode: np.ndarray, status: np.ndarray, abort_events: int = 0) -> int:
    """Board fault entries (per joint, mode 1) plus controller aborts.

    A controller abort is a tick whose status turns `aborted`; `abort_events` (the log's `abort` events) is
    the same fact recorded by the loop, so the larger of the two counts, never their sum (that doubles it).
    """
    modes = np.asarray(board_mode)
    if modes.ndim == 1:
        modes = modes[:, None]
    board_faults = 0
    if modes.shape[0]:
        faulted = modes == FAULT_MODE
        board_faults = int(np.sum(faulted[1:] & ~faulted[:-1]) + np.sum(faulted[0]))
    states = np.asarray(status).reshape(-1)
    aborted = states == "aborted"
    status_aborts = int(np.sum(aborted[1:] & ~aborted[:-1]) + (1 if aborted.size and aborted[0] else 0))
    return board_faults + max(status_aborts, int(abort_events))


def _beyond(values: np.ndarray, bound: float, *, nan_is_bad: bool) -> bool:
    values = np.asarray(values, dtype=float)
    if nan_is_bad and np.any(np.isnan(values)):
        return True
    finite_or_inf = values[~np.isnan(values)]
    return bool(np.any(~np.isfinite(finite_or_inf)) or np.any(np.abs(finite_or_inf) > bound))


def diverged(
    *,
    q: np.ndarray,
    qd: np.ndarray,
    tau: np.ndarray,
    tool_pos: np.ndarray,
    cmd_position: np.ndarray,
    cmd_velocity: np.ndarray,
    cmd_feedforward: np.ndarray,
    cmd_other: tuple[np.ndarray, ...] = (),
    bounds: DivergenceBounds = DEFAULT_BOUNDS,
) -> int:
    """1 if any state went non-finite or out of bounds, or any command went infinite or out of bounds, else 0.

    WHY NaN is allowed in commands: the log writes NaN for a field the controller left None (position NaN is
    moteus's "no position term"), so only +-inf and out-of-bound values are a divergence there. A NaN in the
    plant's state is always one.
    """
    states_bad = (
        _beyond(q, bounds.max_abs_position_rad, nan_is_bad=True)
        or _beyond(qd, bounds.max_abs_velocity_rad_s, nan_is_bad=True)
        or _beyond(tau, bounds.max_abs_effort_nm, nan_is_bad=True)
    )
    tool = np.asarray(tool_pos, dtype=float)
    tool_bad = bool(tool.size) and (
        bool(np.any(~np.isfinite(tool)))
        or bool(np.any(np.linalg.norm(tool, axis=1) > bounds.max_tool_distance_m))
    )
    commands_bad = (
        _beyond(cmd_position, bounds.max_abs_position_rad, nan_is_bad=False)
        or _beyond(cmd_velocity, bounds.max_abs_velocity_rad_s, nan_is_bad=False)
        or _beyond(cmd_feedforward, bounds.max_abs_effort_nm, nan_is_bad=False)
        or any(bool(np.any(np.isinf(np.asarray(values, dtype=float)))) for values in cmd_other)
    )
    return int(states_bad or tool_bad or commands_bad)
