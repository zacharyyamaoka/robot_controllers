"""How far behind its reference the arm runs, the one-run lookahead calibration, and the sync metrics
(CONTRACT amendment 4, §4.3).

    lag_s = estimate_lag_s(log)                         # whole arm, signed: + = the arm lags the reference
    per_joint = estimate_lag_s(log, per_joint=True)     # one per joint, None for a joint that does not move
    lookahead = calibrate_lookahead(lambda la: layout_spec(..., lookahead_s=la), rung)   # one run at 0

Zach, 2026-11-20: "how that can be calibrated on how far the look ahead should be? ... how reliabiliy does
look ahead work?" The estimate reads only what a real arm's log has too (`t`, `q`, `sent_t`,
`ref_position`), so the procedure that calibrates the twin calibrates the rig: run once with no lookahead,
take the shift that best lines the measured joints up with the reference, and send that far ahead.

The shift: τ on a `step_s` grid in [-max_s, +max_s] minimising Σ_j Σ_i (ref_j(t_i - τ) - q_j(t_i))² over the
plant samples t_i in [first sent_t + 0.05 s, last sent_t], `ref` linear in `sent_t` (its ends held). A tie
goes to the smallest |τ|. WHY a grid and not an optimiser: the residual is not smooth in τ at the board's
4 kHz samples, a 0.25 ms grid is the board's own resolution, and a grid's answer is exact and repeatable.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from typing import Any

import numpy as np

WINDOW_START_S = 0.05  # skip the first 50 ms after the first command: the arm starts from the board's hold
STILL_DEG = 0.1  # a joint whose reference moves less than this peak to peak has no lag to measure


def _shifts(max_s: float, step_s: float) -> np.ndarray:
    """The τ grid ordered by |τ| (then τ), so the first minimum is the tie-break the contract names."""
    steps = int(round(max_s / step_s))
    grid = np.arange(-steps, steps + 1) * step_s
    return grid[np.lexsort((grid, np.abs(grid)))]


def lag_residuals(
    log: Any, *, max_s: float = 0.030, step_s: float = 0.00025
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(shifts, residuals (J, S) = Σ_i (ref_j(t_i - τ) - q_j(t_i))², moving (J,) bool), or None when the log
    has no reference or no plant samples in the window. `shifts` are in tie-break order."""
    if getattr(log, "ref_position", None) is None:
        return None
    sent = np.asarray(log.sent_t, dtype=float).reshape(-1)
    if sent.size < 2:
        return None
    ref = np.atleast_2d(np.asarray(log.ref_position, dtype=float))
    t = np.asarray(log.t, dtype=float).reshape(-1)
    q = np.atleast_2d(np.asarray(log.q, dtype=float))
    window = (t >= sent[0] + WINDOW_START_S) & (t <= sent[-1])
    if not window.any():
        return None
    times, measured = t[window], q[window]
    shifts = _shifts(max_s, step_s)
    shifted = times[None, :] - shifts[:, None]  # (S, N): ref_j is read at t_i - τ
    joints = ref.shape[1]
    residuals = np.full((joints, shifts.size), np.nan)
    moving = np.zeros(joints, dtype=bool)
    in_window = (sent >= times[0]) & (sent <= times[-1])
    for j in range(joints):
        column = ref[:, j]
        if not np.all(np.isfinite(column)):
            continue
        seen = column[in_window] if in_window.any() else column
        moving[j] = math.degrees(float(np.ptp(seen))) >= STILL_DEG
        error = np.interp(shifted, sent, column) - measured[:, j][None, :]
        residuals[j] = np.einsum("sn,sn->s", error, error)
    return shifts, residuals, moving


def estimate_lag_s(
    log: Any, *, per_joint: bool = False, max_s: float = 0.030, step_s: float = 0.00025
) -> float | tuple[float | None, ...] | None:
    """The shift τ (s) that best lines the measured joints up with the reference (module docstring); positive
    means the arm lags the reference. `per_joint=True` gives one τ per joint, None for a joint whose
    reference moves less than 0.1° peak to peak. The whole-arm τ sums every joint with a reference and is None
    when none of them moves (a hold has no lag to measure) or the log has no reference."""
    found = lag_residuals(log, max_s=max_s, step_s=step_s)
    if found is None:
        if not per_joint:
            return None
        joints = np.atleast_2d(np.asarray(log.q)).shape[1]
        return (None,) * joints
    shifts, residuals, moving = found
    if per_joint:
        return tuple(
            float(shifts[int(np.argmin(residuals[j]))]) if moving[j] else None for j in range(len(moving))
        )
    if not moving.any():
        return None
    total = np.nansum(residuals, axis=0)
    return float(shifts[int(np.argmin(total))])


def calibrate_lookahead(spec_at: Callable[[float], Any], rung: Any, *, seed: int = 0) -> tuple[float, ...]:
    """The one-run calibration (§4.3): run `spec_at(0.0)` once on `rung` and return each joint's lag (s),
    clipped at >= 0 (a lookahead cannot look back), with a joint that did not move given the median of the
    others. Joint order = the log's (the plant's = the rung's trajectory's)."""
    from robot_controllers.bench.loop import run_episode

    log = run_episode(spec_at(0.0), rung, seed=seed)
    lags = estimate_lag_s(log, per_joint=True)
    measured = [max(0.0, lag) for lag in lags if lag is not None]
    if not measured:
        raise ValueError(f"calibrate_lookahead: no joint moved {STILL_DEG:g} deg on {rung.id}")
    median = float(statistics.median(measured))
    return tuple(median if lag is None else max(0.0, float(lag)) for lag in lags)


# ── sync: the tool against a point riding the belt ─────────────────────────────────────────────────────────

SYNC_PARAMS = ("sync_start_s", "sync_end_s", "belt_speed_m_s")


def sync_metrics(
    plant_t: np.ndarray,
    tool_pos: np.ndarray,
    sent_t: np.ndarray,
    ref_tool_pos: np.ndarray,
    sync_start_s: float,
    sync_end_s: float,
    belt_speed_m_s: float,
) -> dict[str, float | None]:
    """`sync_error_p95_mm` and `sync_timing_ms` over the plant samples in [sync_start_s, sync_end_s]: the
    target point is `ref_tool_pos` linear in `sent_t`; timing = 1000 · mean(target x - tool x) / belt speed,
    positive = the tool is behind the item."""
    t = np.asarray(plant_t, dtype=float).reshape(-1)
    window = (t >= sync_start_s) & (t <= sync_end_s)
    if not window.any() or belt_speed_m_s == 0.0:
        return {"sync_error_p95_mm": None, "sync_timing_ms": None}
    sent = np.asarray(sent_t, dtype=float).reshape(-1)
    reference = np.asarray(ref_tool_pos, dtype=float)
    target = np.stack([np.interp(t[window], sent, reference[:, i]) for i in range(3)], axis=1)
    tool = np.asarray(tool_pos, dtype=float)[window]
    timing_ms = 1000.0 * float(np.mean(target[:, 0] - tool[:, 0])) / float(belt_speed_m_s)
    error_mm = 1000.0 * float(np.percentile(np.linalg.norm(target - tool, axis=1), 95))
    return {"sync_error_p95_mm": error_mm, "sync_timing_ms": timing_ms}
