"""Contact and disturbance metrics: the belt impact (ISO/TS 15066's transient and quasi-static phases) and the
sideways push.

Contact reads the plant's contact force on the tool (normal = +z, the belt's top face), the in-contact flag
and the dwell window and press target the rung put in `log.meta`; the push reads the tool position and the
push window. Every function is pure on arrays and returns plain floats/ints/None. WHY pure: the same functions
can run on a rolling window online (a force monitor on the rig) without an EpisodeLog.
"""

from __future__ import annotations

import numpy as np

TRANSIENT_WINDOW_S = 0.030  # CONTRACT §4: transient = [touch, touch + 30 ms]
BOUNCE_DEBOUNCE_S = 0.002  # a contact change counts only once it has held 2 ms
PUSH_RECOVERY_BAND_M = 0.001  # back within 1 mm of the pre-push tool position...
PUSH_RECOVERY_HOLD_S = 0.050  # ...and staying there 50 ms


def first_touch_index(in_contact: np.ndarray) -> int | None:
    flags = np.asarray(in_contact, dtype=bool).reshape(-1)
    return int(np.argmax(flags)) if flags.any() else None


def debounced(times: np.ndarray, flags: np.ndarray, hold_s: float) -> tuple[np.ndarray, np.ndarray]:
    """A boolean signal debounced by `hold_s`: the state changes only once the raw signal has held the new
    value for at least `hold_s`. Returns (debounced state per sample, change times): each change is stamped at
    the raw edge that started the run that held.
    """
    times = np.asarray(times, dtype=float).reshape(-1)
    raw = np.asarray(flags, dtype=bool).reshape(-1)
    state = np.empty_like(raw)
    if raw.size == 0:
        return state, np.array([])
    # runs of equal raw values: [start, end) index pairs
    edges = np.nonzero(raw[1:] != raw[:-1])[0] + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [raw.size]))
    current = bool(raw[0])
    change_times: list[float] = []
    for start, end in zip(starts, ends, strict=True):
        value = bool(raw[start])
        # duration the run held: to the next run's first sample, or to the last sample for the final run
        run_end_t = times[end] if end < raw.size else times[-1]
        held = run_end_t - times[start]
        if value != current and (held >= hold_s - 1e-12):
            current = value
            change_times.append(float(times[start]))
            state[start:end] = current
        else:
            state[start:end] = current
    return state, np.asarray(change_times)


def contact_metrics(
    t: np.ndarray,
    contact_force: np.ndarray,
    in_contact: np.ndarray,
    tool_pos: np.ndarray,
    dwell: tuple[float, float] | None,
    press_target_n: float | None,
    ref_tool_t: np.ndarray | None = None,
    ref_tool_pos: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Every contact metric of a belt rung.

    - normal force = contact_force[:, 2] (the belt pushes the tool up, +z).
    - first touch = the first sample in contact; touch speed = |tool z velocity| over the step into it.
    - contact_peak_n = largest normal force in [touch, touch + 30 ms]; contact_max_n = largest at any time
      (0 when it never touched: measured, and the dwell gates fail it).
    - dwell = samples with t in [t0, t1]: force mean, std, |mean - target|, the share in contact, bounces
      (in contact -> not in contact, debounced 2 ms, the edge inside the dwell) and the RMS tool error along
      the belt (x) against the reference tool position (or against the tool at the dwell start without one).
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    normal = np.asarray(contact_force, dtype=float)[:, 2]
    flags = np.asarray(in_contact, dtype=bool).reshape(-1)
    tool = np.asarray(tool_pos, dtype=float)
    out: dict[str, float | int | None] = {
        "touch_speed_m_s": None,
        "contact_peak_n": None,
        "contact_max_n": float(np.max(normal)) if normal.size else None,
        "dwell_force_mean_n": None,
        "dwell_force_std_n": None,
        "dwell_force_error_n": None,
        "dwell_bounces": None,
        "dwell_contact_fraction": None,
        "dwell_slip_mm": None,
    }
    touch = first_touch_index(flags)
    if touch is not None:
        if touch > 0 and t[touch] > t[touch - 1]:
            out["touch_speed_m_s"] = abs(
                float((tool[touch, 2] - tool[touch - 1, 2]) / (t[touch] - t[touch - 1]))
            )
        window = (t >= t[touch]) & (t <= t[touch] + TRANSIENT_WINDOW_S + 1e-12)
        out["contact_peak_n"] = float(np.max(normal[window]))
    if dwell is None:
        return out
    t0, t1 = float(dwell[0]), float(dwell[1])
    in_dwell = (t >= t0) & (t <= t1)
    if not in_dwell.any():
        return out
    dwell_force = normal[in_dwell]
    mean_force = float(np.mean(dwell_force))
    out["dwell_force_mean_n"] = mean_force
    out["dwell_force_std_n"] = float(np.std(dwell_force))
    if press_target_n is not None:
        out["dwell_force_error_n"] = abs(mean_force - float(press_target_n))
    out["dwell_contact_fraction"] = 100.0 * float(np.mean(flags[in_dwell]))
    state, change_times = debounced(t, flags, BOUNCE_DEBOUNCE_S)
    losses = [
        change_t
        for change_t in change_times
        if t0 <= change_t <= t1 and not state[int(np.searchsorted(t, change_t))]
    ]
    out["dwell_bounces"] = len(losses)
    if ref_tool_t is not None and ref_tool_pos is not None and np.size(ref_tool_t):
        reference_x = np.interp(
            t[in_dwell], np.asarray(ref_tool_t, dtype=float), np.asarray(ref_tool_pos, dtype=float)[:, 0]
        )
    else:
        reference_x = np.full(int(np.sum(in_dwell)), tool[in_dwell][0, 0])
    slip = tool[in_dwell, 0] - reference_x
    slip = slip[np.isfinite(slip)]
    if slip.size:
        out["dwell_slip_mm"] = float(np.sqrt(np.mean(slip**2))) * 1000.0
    return out


def push_metrics(
    t: np.ndarray,
    tool_pos: np.ndarray,
    push_start_s: float,
    push_end_s: float,
    push_direction: np.ndarray,
) -> tuple[dict[str, float | None], str]:
    """Deflection, recovery and overshoot of the tool around one push, plus a note if recovery is unmeasured.

    - displacement = tool - the tool at the last sample before the push starts (strictly before: the sample at
      the start instant may already carry the push).
    - push_deflection_mm = largest |displacement| (3-D) during [start, end].
    - push_recovery_s = first time at or after the push ends from which |displacement| stays < 1 mm for 50 ms,
      minus the push end; None when that never happens inside the run.
    - push_overshoot_mm = largest displacement against `push_direction` after the end (0 if it never crosses).
    """
    t = np.asarray(t, dtype=float).reshape(-1)
    tool = np.asarray(tool_pos, dtype=float)
    direction = np.asarray(push_direction, dtype=float).reshape(3)
    direction = direction / np.linalg.norm(direction)
    out: dict[str, float | None] = {
        "push_deflection_mm": None,
        "push_recovery_s": None,
        "push_overshoot_mm": None,
    }
    before = np.nonzero(t < push_start_s)[0]
    if before.size == 0:
        return out, "not measured: no sample before the push"
    displacement = tool - tool[before[-1]]
    distance = np.linalg.norm(displacement, axis=1)
    during = (t >= push_start_s) & (t <= push_end_s)
    if during.any():
        out["push_deflection_mm"] = float(np.max(distance[during])) * 1000.0
    after = np.nonzero(t >= push_end_s)[0]
    if after.size == 0:
        return out, "not measured: the run ended before the push did"
    out["push_overshoot_mm"] = max(0.0, float(np.max(-(displacement[after] @ direction)))) * 1000.0
    inside = distance < PUSH_RECOVERY_BAND_M
    # the index of the next sample outside the band, from each sample on (N means none later)
    next_outside = np.full(t.size + 1, t.size)
    for index in range(t.size - 1, -1, -1):
        next_outside[index] = index if not inside[index] else next_outside[index + 1]
    for index in after:
        if not inside[index]:
            continue
        hold_until = t[index] + PUSH_RECOVERY_HOLD_S
        if hold_until > t[-1] + 1e-12:
            break  # not enough run left to see it hold
        stays = next_outside[index] >= t.size or t[next_outside[index]] > hold_until + 1e-12
        if stays:
            out["push_recovery_s"] = float(t[index] - push_end_s)
            return out, ""
    return out, f"never back within {PUSH_RECOVERY_BAND_M * 1000:g} mm for {PUSH_RECOVERY_HOLD_S * 1000:g} ms"
