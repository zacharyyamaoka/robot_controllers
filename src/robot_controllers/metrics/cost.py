"""Cost metrics: what one controller tick costs the host (compute) and the bus (command fields, wire bytes).

Pure on arrays, plain numbers out. WHY pure: an online monitor can keep the same numbers per window.
"""

from __future__ import annotations

import numpy as np

# moteus position-mode frame (CONTRACT §4): a 2-byte register header per joint, one float32 per field set
REGISTER_HEADER_BYTES = 2
FLOAT32_BYTES = 4


def compute_metrics(compute_s: np.ndarray) -> dict[str, float | None]:
    """p50, p99 and max wall time of one controller tick, in microseconds (finite samples only)."""
    seconds = np.asarray(compute_s, dtype=float).reshape(-1)
    seconds = seconds[np.isfinite(seconds)]
    if seconds.size == 0:
        return {"compute_us_p50": None, "compute_us_p99": None, "compute_us_max": None}
    micros = seconds * 1.0e6
    return {
        "compute_us_p50": float(np.percentile(micros, 50.0)),
        "compute_us_p99": float(np.percentile(micros, 99.0)),
        "compute_us_max": float(np.max(micros)),
    }


def command_fields(field_arrays: tuple[np.ndarray, ...]) -> float | None:
    """Mean count, per joint per tick, of the command fields that were set (not NaN).

    `field_arrays` are the (K, J) position, velocity, feedforward, stiffness, damping and max_effort columns.
    A field sent as 0 (stiffness 0 switches the board's spring off) is set: it travels on the bus.
    """
    if not field_arrays:
        return None
    stacked = np.stack([np.atleast_2d(np.asarray(values, dtype=float)) for values in field_arrays])
    if stacked.shape[1] == 0:
        return None
    return float(np.mean(np.sum(~np.isnan(stacked), axis=0)))


def wire_bytes_per_tick(joint_count: int, fields_per_joint: float | None) -> float | None:
    """joints x (2 + 4 x fields): the estimated CAN-FD payload of one tick for all joints."""
    if fields_per_joint is None:
        return None
    return float(joint_count) * (REGISTER_HEADER_BYTES + FLOAT32_BYTES * float(fields_per_joint))


def host_rate_hz(nominal_period_s: float) -> float | None:
    return 1.0 / float(nominal_period_s) if nominal_period_s and nominal_period_s > 0 else None
