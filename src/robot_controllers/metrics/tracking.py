"""Tracking and replanning metrics: how far the measured motion is from the intended one, and how smooth the
commands stay when the input changes.

Tracking error = reference - measured, at each host tick's sent time (CONTRACT §4). The reference is the
active input sampled on the intended schedule (so a controller that slips shows error); the measured position
is the plant's sample interpolated at that same instant, never the delayed state the controller saw, so the
bus delay is part of the error only through what the arm actually did.

Every function here is pure on arrays and returns plain floats/None. WHY pure: an online monitor can feed the
same functions a rolling window of ticks and plant samples.
"""

from __future__ import annotations

import math

import numpy as np
from robot_msgs.compare import error_stats

SETTLE_TOLERANCE_DEG = 0.5  # names.py: settle = every joint stays within 0.5 deg
# WHY 1e-6 rad: a trajectory sampled at 4 ms and held at its end is exactly constant after its end, and its
# last min-jerk sample is ~4e-7 rad from the end; anything looser would call a slow final creep "ended".
REFERENCE_HELD_TOLERANCE_RAD = 1.0e-6


def interpolate_columns(sample_t: np.ndarray, values: np.ndarray, at_t: np.ndarray) -> np.ndarray:
    """Linear interpolation of each column of `values` (N, C) sampled at `sample_t` (N,), at `at_t` (K,).

    Outside the sampled span the end values are held (np.interp); NaN samples propagate into their neighbours.
    """
    sample_t = np.asarray(sample_t, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)
    at_t = np.asarray(at_t, dtype=float).reshape(-1)
    column_values = values[:, None] if values.ndim == 1 else values
    out = np.empty((at_t.size, column_values.shape[1]))
    for column in range(column_values.shape[1]):
        out[:, column] = np.interp(at_t, sample_t, column_values[:, column])
    return out


def joint_errors(
    sent_t: np.ndarray, ref_position: np.ndarray, plant_t: np.ndarray, plant_q: np.ndarray
) -> np.ndarray:
    """(K, J) joint position error in rad: reference at each sent time minus the plant interpolated there."""
    measured = interpolate_columns(plant_t, plant_q, sent_t)
    return np.asarray(ref_position, dtype=float) - measured


def joint_tracking_metrics(
    errors_rad: np.ndarray, sent_t: np.ndarray | None = None
) -> dict[str, float | None]:
    """RMS, p95 and max |error| (worst joint) and the final |error| (worst joint), in degrees.

    Per joint through robot_msgs.compare.error_stats (NaN samples, where there is no reference, are skipped);
    the worst joint is the reported number, because one joint far off is a failed motion however well the
    other one tracks.
    """
    errors = np.atleast_2d(np.asarray(errors_rad, dtype=float))
    stats = [error_stats(errors[:, joint], sent_t) for joint in range(errors.shape[1])]
    stats = [s for s in stats if s.n > 0]
    if not stats:
        return {
            "tracking_rms_deg": None,
            "tracking_p95_deg": None,
            "tracking_max_deg": None,
            "final_error_deg": None,
        }
    return {
        "tracking_rms_deg": math.degrees(max(s.rms for s in stats)),
        "tracking_p95_deg": math.degrees(max(s.p95_abs for s in stats)),
        "tracking_max_deg": math.degrees(max(s.max_abs for s in stats)),
        "final_error_deg": math.degrees(max(abs(s.final) for s in stats)),
    }


def tool_tracking_metrics(
    sent_t: np.ndarray, ref_tool_pos: np.ndarray, plant_t: np.ndarray, tool_pos: np.ndarray
) -> dict[str, float | None]:
    """RMS and p95 of |ref tool position - tool position| (3-D distance) at each sent time, in mm."""
    measured = interpolate_columns(plant_t, tool_pos, sent_t)
    distance = np.linalg.norm(np.asarray(ref_tool_pos, dtype=float) - measured, axis=1)
    stats = error_stats(distance)
    if stats.n == 0:
        return {"tool_rms_mm": None, "tool_p95_mm": None}
    return {"tool_rms_mm": stats.rms * 1000.0, "tool_p95_mm": stats.p95_abs * 1000.0}


def reference_end_time(
    sent_t: np.ndarray, ref_position: np.ndarray, tolerance_rad: float = REFERENCE_HELD_TOLERANCE_RAD
) -> float | None:
    """The first sent time from which the reference holds its final value (every joint within `tolerance_rad`)
    to the end of the run; None if it is not held for at least two ticks (it never ended inside the run).

    WHY the last change, not the input's nominal duration: a trajectory that ends with a rest has stopped
    moving before its last sample, and settling is measured from when the motion stopped.
    """
    sent = np.asarray(sent_t, dtype=float).reshape(-1)
    ref = np.atleast_2d(np.asarray(ref_position, dtype=float))
    if sent.size < 2:
        return None
    final = ref[-1]
    if not np.all(np.isfinite(final)):
        return None
    off = np.any(~(np.abs(ref - final) <= tolerance_rad), axis=1)  # NaN counts as off
    if not off.any():
        return float(sent[0])
    first_held = int(np.max(np.nonzero(off)[0])) + 1
    if first_held > sent.size - 2:
        return None
    return float(sent[first_held])


def settle_time(
    reference_end_s: float | None,
    final_reference: np.ndarray,
    plant_t: np.ndarray,
    plant_q: np.ndarray,
    tolerance_deg: float = SETTLE_TOLERANCE_DEG,
) -> tuple[float | None, str]:
    """Seconds after `reference_end_s` until every joint is within `tolerance_deg` of the final reference and
    stays there to the end of the run (plant samples, full rate), plus a note when it is not measured.

    0.0 when it never leaves the band after the reference ends; None when it is still outside at the last
    sample (never settled: reporting the time left would pass a gate it should fail).
    """
    if reference_end_s is None:
        return None, "not measured: the reference never ended inside the run"
    plant_t = np.asarray(plant_t, dtype=float).reshape(-1)
    after = plant_t >= reference_end_s
    if not after.any():
        return None, "not measured: no plant samples after the reference ended"
    q_after = np.atleast_2d(np.asarray(plant_q, dtype=float))[after]
    error_deg = np.degrees(np.abs(np.asarray(final_reference, dtype=float)[None, :] - q_after))
    outside = np.any(~(error_deg <= tolerance_deg), axis=1)
    if not outside.any():
        return 0.0, ""
    last_outside = int(np.max(np.nonzero(outside)[0]))
    t_after = plant_t[after]
    if last_outside == t_after.size - 1:
        return None, f"never settled within {tolerance_deg:g} deg before the run ended"
    return float(t_after[last_outside + 1] - reference_end_s), ""


def command_jump_max_deg(cmd_position: np.ndarray) -> float | None:
    """Largest |change| of the commanded position between two consecutive ticks, worst joint, in degrees.

    Only pairs where both ticks command a position count; a controller that never commands one (position NaN,
    the host-torque variants) has no command jump to measure, so None.
    """
    positions = np.atleast_2d(np.asarray(cmd_position, dtype=float))
    if positions.shape[0] < 2:
        return None
    steps = np.abs(np.diff(positions, axis=0))
    steps = steps[np.isfinite(steps)]
    return math.degrees(float(np.max(steps))) if steps.size else None


def torque_rate_max_nm_s(plant_t: np.ndarray, tau: np.ndarray) -> float | None:
    """Largest |d tau / dt| over the plant samples, any joint, in N·m/s."""
    times = np.asarray(plant_t, dtype=float).reshape(-1)
    torques = np.atleast_2d(np.asarray(tau, dtype=float))
    if times.size < 2:
        return None
    widths = np.diff(times)
    valid = widths > 0
    rates = np.abs(np.diff(torques, axis=0)[valid] / widths[valid, None])
    rates = rates[np.isfinite(rates)]
    return float(np.max(rates)) if rates.size else None
