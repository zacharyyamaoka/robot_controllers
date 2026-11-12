"""One episode -> every metric, its gates and the RunResult the results JSON stores (CONTRACT §4, §5).

    from robot_controllers.metrics.episode import run_result
    result = run_result(log, rung)          # metrics, gate rows, passed, objective, trace, events

`episode_metrics` always returns every name in `metrics.names.METRICS`: a number, or None when this run could
not measure it (contact on a free-space rung, tracking on a sequence rung). Never 0 for "not measured". The
per-family functions it calls (timing, tracking, contact, health, cost, sequence) are pure on arrays, so an
online monitor can reuse them on a rolling window; this module is the offline assembly over an EpisodeLog.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from robot_controllers.bench.types import EpisodeLog, GateResult, Rung, RunResult, gate_rows, judge
from robot_controllers.metrics import contact, cost, health, sequence, timing, tracking
from robot_controllers.metrics.names import BY_NAME, METRICS

Metrics = dict[str, float | int | None]

TRACE_RATE_HZ = 200.0  # CONTRACT §5: traces downsampled to 200 Hz...
CONTACT_TRACE_RATE_HZ = 1000.0  # ...except the contact force around the first touch, at 1 kHz
CONTACT_TRACE_BEFORE_S = 0.020  # [touch - 20 ms, touch + 80 ms]
CONTACT_TRACE_AFTER_S = 0.080
TRACE_SIGNIFICANT_DIGITS = 4


# ── metrics ─────────────────────────────────────────────────────────────────────────────────────────────────


def episode_metrics(log: EpisodeLog, rung: Rung) -> Metrics:
    """Every metric in names.METRICS for one episode (None = not measured)."""
    return episode_metrics_and_notes(log, rung)[0]


def episode_metrics_and_notes(log: EpisodeLog, rung: Rung) -> tuple[Metrics, dict[str, str]]:
    """(metrics, notes): notes say why a metric is None, or qualify a number a gate will judge."""
    metrics: Metrics = {m.name: None for m in METRICS}
    notes: dict[str, str] = {}

    def note(name: str, text: str) -> None:
        if text:
            notes[name] = text

    # timing: the black-box stamps, then the controller's declared end and the board's watchdog
    metrics.update(timing.timing_metrics(log.sent_t, log.state_stamp, log.state_seq, log.nominal_period_s))
    metrics["watchdog_trips"] = timing.watchdog_trips(log.board_mode)
    last_input = int(np.asarray(log.input_index).reshape(-1)[-1]) if np.size(log.input_index) else 0
    refused_last = any(e.get("kind") == "refused" and e.get("index") == last_input for e in log.events)
    if log.ref_position is not None and refused_last:
        # WHY not measured: the controller's declared end belongs to an input it kept, not to the refused one
        # the schedule describes (a tape feeder on the replan rung would read as finishing 450 ms "early")
        note(
            "schedule_slip_ms",
            "not measured: the controller refused the input whose schedule it would end on",
        )
    elif log.ref_position is not None:
        schedule_end, why = _schedule_end_s(log, rung)
        slip, slip_note = timing.schedule_slip_ms(log.sent_t, log.status, log.input_index, schedule_end)
        metrics["schedule_slip_ms"] = slip
        note("schedule_slip_ms", why or slip_note)
    else:
        note("schedule_slip_ms", "not measured: this input has no time schedule")

    # tracking
    settle: float | None = None
    settle_note = "not measured: no reference"
    if log.ref_position is not None and np.any(np.isfinite(log.ref_position)):
        errors = tracking.joint_errors(log.sent_t, log.ref_position, log.t, log.q)
        metrics.update(tracking.joint_tracking_metrics(errors))
        reference_end = tracking.reference_end_time(log.sent_t, log.ref_position)
        settle, settle_note = tracking.settle_time(
            reference_end, np.asarray(log.ref_position)[-1], log.t, log.q
        )
        metrics["settle_time_s"] = settle
        note("settle_time_s", settle_note)
    if log.ref_tool_pos is not None:
        metrics.update(tracking.tool_tracking_metrics(log.sent_t, log.ref_tool_pos, log.t, log.tool_pos))

    # replanning: command smoothness is measured on every rung; replans only where the script has them
    metrics["command_jump_max_deg"] = tracking.command_jump_max_deg(log.cmd_position)
    if metrics["command_jump_max_deg"] is None:
        note("command_jump_max_deg", "not measured: the controller never commands a position")
    metrics["torque_rate_max_nm_s"] = tracking.torque_rate_max_nm_s(log.t, log.tau)
    accepted, refused = _replan_counts(log)
    if accepted is not None:
        metrics["replans_accepted"] = accepted
        if refused:
            note("replans_accepted", f"unsupported: refused {refused} replan{'s' if refused != 1 else ''}")
        metrics["replan_settle_s"] = settle
        note("replan_settle_s", settle_note)

    # contact (belt rungs)
    if rung.scene.belt:
        dwell = log.meta.get("dwell")
        press_target = log.meta.get("press_target_n")
        metrics.update(
            contact.contact_metrics(
                log.t,
                log.contact_force,
                log.in_contact,
                log.tool_pos,
                None if dwell is None else (float(dwell[0]), float(dwell[1])),
                None if press_target is None else float(press_target),
                ref_tool_t=log.sent_t if log.ref_tool_pos is not None else None,
                ref_tool_pos=log.ref_tool_pos,
            )
        )
        if dwell is None:
            for name in (
                "dwell_force_mean_n",
                "dwell_force_std_n",
                "dwell_force_error_n",
                "dwell_bounces",
                "dwell_contact_fraction",
                "dwell_slip_mm",
            ):
                note(name, "not measured: the log has no meta['dwell'] window")
        elif press_target is None:
            note("dwell_force_error_n", "not measured: the log has no meta['press_target_n']")
        if metrics["contact_peak_n"] is None:
            note("contact_peak_n", "not measured: the tool never touched the belt")
    else:
        for name in ("contact_peak_n", "contact_max_n"):
            note(name, "not measured: no belt on this rung")

    # disturbance (push rungs)
    push = _push_window(log, rung)
    if push is not None:
        push_start, push_end, direction = push
        values, push_note = contact.push_metrics(log.t, log.tool_pos, push_start, push_end, direction)
        metrics.update(values)
        note("push_recovery_s", push_note)

    # health
    metrics["saturation_share"] = health.saturation_share(log.at_cap)
    metrics["faults"] = health.fault_count(
        log.board_mode, log.status, abort_events=len(log.event_times("abort"))
    )
    metrics["diverged"] = int(
        health.diverged(
            q=log.q,
            qd=log.qd,
            tau=log.tau,
            tool_pos=log.tool_pos,
            cmd_position=log.cmd_position,
            cmd_velocity=log.cmd_velocity,
            cmd_feedforward=log.cmd_feedforward,
            cmd_other=(log.cmd_stiffness, log.cmd_damping, log.cmd_max_effort),
        )
        or bool(log.event_times("diverged"))
    )

    # cost
    metrics.update(cost.compute_metrics(log.compute_s))
    fields = cost.command_fields(
        (
            log.cmd_position,
            log.cmd_velocity,
            log.cmd_feedforward,
            log.cmd_stiffness,
            log.cmd_damping,
            log.cmd_max_effort,
        )
    )
    metrics["command_fields"] = fields
    metrics["wire_bytes_per_tick"] = cost.wire_bytes_per_tick(len(log.joint_names), fields)
    metrics["host_rate_hz"] = cost.host_rate_hz(log.nominal_period_s)

    # sequence (JointCommandSequence rungs)
    if _is_sequence_rung(log, rung):
        metrics["steps_done"] = sequence.steps_done(log.events)
        steps_total = log.meta.get("steps_total")
        metrics["steps_total"] = None if steps_total is None else int(steps_total)
        if steps_total is None:
            note("steps_total", "not measured: the log has no meta['steps_total']")
        finished, finished_note = sequence.sequence_time_s(log.sent_t, log.status)
        metrics["sequence_time_s"] = finished
        note("sequence_time_s", finished_note)
        reached, reached_note = sequence.reached_error_deg(
            log.events, log.joint_names, log.tick_t, log.cmd_position, log.t, log.q
        )
        metrics["reached_error_deg"] = reached
        note("reached_error_deg", reached_note)

    return {name: _plain(value) for name, value in metrics.items()}, notes


def _plain(value: Any) -> float | int | None:
    """JSON-safe: Python int/float, None for NaN/inf (results JSON is written with allow_nan=False)."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = float(value)
    return number if math.isfinite(number) else None


def _schedule_end_s(log: EpisodeLog, rung: Rung) -> tuple[float | None, str]:
    """When the active input's schedule ends: `meta["schedule_end_s"]` if the loop recorded it, else rebuilt
    from the rung's input script (the active message's script time plus its duration `t[-1]`)."""
    if log.meta.get("schedule_end_s") is not None:
        return float(log.meta["schedule_end_s"]), ""
    start_q = np.asarray(log.meta.get("start_q", log.q[0] if len(log.q) else ()), dtype=float)
    try:
        script = list(rung.inputs(start_q))
    except Exception as error:  # noqa: BLE001 - the slip is one number; say why it is missing, keep the rest
        return None, f"not measured: could not rebuild the input script ({type(error).__name__}: {error})"
    if not script:
        return None, "not measured: the input script is empty"
    active = int(np.asarray(log.input_index).reshape(-1)[-1]) if np.size(log.input_index) else 0
    active = min(max(active, 0), len(script) - 1)
    start_t, message = script[active]
    times = getattr(message, "t", None)
    if times is None or not np.size(times):
        return None, "not measured: the input has no time axis"
    return float(start_t) + float(np.asarray(times, dtype=float).reshape(-1)[-1]), ""


def _replan_counts(log: EpisodeLog) -> tuple[int | None, int]:
    """(replans accepted, refused); accepted is None when the episode had no replans to take."""
    accepted = len(log.event_times("replan"))
    refused = len(log.event_times("refused"))
    if accepted or refused:
        return accepted, refused
    indices = np.asarray(log.input_index).reshape(-1)
    switches = int(np.sum(np.diff(indices) > 0)) if indices.size > 1 else 0
    if switches:
        return switches, 0
    return None, 0


def _push_window(log: EpisodeLog, rung: Rung) -> tuple[float, float, np.ndarray] | None:
    """(start, end, unit direction) of the first push, from `meta["push"] = (t0, t1, axis)` or the scene."""
    scene_force = np.asarray(rung.scene.pushes[0].force_n, dtype=float) if rung.scene.pushes else None
    recorded = log.meta.get("push")
    if recorded is not None:
        start, end = float(recorded[0]), float(recorded[1])
        axis = recorded[2] if len(recorded) > 2 else 0
        return start, end, _direction(axis, scene_force)
    if scene_force is not None:
        first = rung.scene.pushes[0]
        return float(first.start_s), float(first.start_s + first.duration_s), _direction(None, scene_force)
    return None


def _direction(axis: Any, scene_force: np.ndarray | None) -> np.ndarray:
    """A unit push direction from an axis index (0/1/2), an axis name ("x", "-z"), a 3-vector, or the scene's
    force. The sign of an index or a bare name comes from the scene's force on that axis (+ without one)."""
    if axis is None:
        if scene_force is None or not np.any(scene_force):
            return np.array([1.0, 0.0, 0.0])
        return scene_force / np.linalg.norm(scene_force)
    if isinstance(axis, str):
        text = axis.strip().lower()
        sign = -1.0 if text.startswith("-") else 1.0
        index = "xyz".index(text.lstrip("+-"))
        explicit = text[0] in "+-"
    elif np.ndim(axis) == 1 and np.size(axis) == 3:
        vector = np.asarray(axis, dtype=float)
        return vector / np.linalg.norm(vector)
    else:
        index, sign, explicit = int(axis), 1.0, False
    if not explicit and scene_force is not None and scene_force[index] < 0:
        sign = -1.0
    direction = np.zeros(3)
    direction[index] = sign
    return direction


def _is_sequence_rung(log: EpisodeLog, rung: Rung) -> bool:
    return (
        rung.family == "sequence"
        or "steps_total" in log.meta
        or any(e.get("kind") == "step_done" for e in log.events)
    )


# ── gates and the run result ───────────────────────────────────────────────────────────────────────────────


def evaluate(
    rung: Rung, metrics: Mapping[str, float | int | None], notes: Mapping[str, str] | None = None
) -> list[GateResult]:
    """Judge every gate of `rung` on `metrics` (bench.types.judge: a None value fails as "not measured").

    `notes` (from `episode_metrics_and_notes`) replaces the bare "not measured" with the reason, and qualifies
    a measured value ("unsupported: refused 8 replans"). A gate on a name outside the catalogue is a contract
    error and raises.
    """
    results = []
    for gate in rung.gates:
        if gate.metric not in BY_NAME:
            raise KeyError(f"rung {rung.id}: gate on unknown metric {gate.metric!r}")
        results.append(judge(gate, metrics.get(gate.metric), (notes or {}).get(gate.metric, "")))
    return results


def run_result(log: EpisodeLog, rung: Rung, *, trace: bool = True) -> RunResult:
    """The scored run: metrics, gate rows, passed (every gate), objective = metrics[rung.objective], the trace
    (CONTRACT §5) and the episode's events, all JSON-safe. The rung id comes from `rung`, not the log."""
    metrics, notes = episode_metrics_and_notes(log, rung)
    if rung.objective not in BY_NAME:
        raise KeyError(f"rung {rung.id}: objective {rung.objective!r} is not a metric")
    gate_results = evaluate(rung, metrics, notes)
    return RunResult(
        controller=log.controller,
        rung=rung.id,
        seed=int(log.seed),
        metrics=metrics,
        gates=gate_rows(gate_results),
        passed=all(result.passed for result in gate_results),
        objective=metrics[rung.objective],
        trace=episode_trace(log, rung) if trace else {},
        events=[_json_value(dict(event)) for event in log.events],
    )


# ── the trace ───────────────────────────────────────────────────────────────────────────────────────────────


def episode_trace(log: EpisodeLog, rung: Rung) -> dict[str, list[float | None]]:
    """Plot series (CONTRACT §5): plant series at 200 Hz, the contact force at 1 kHz around the first touch
    (with its own `contact_t`), compute per host tick (with `tick_t`); 4 significant digits, NaN -> None.

    The 200 Hz series pick real plant samples (the first at or after each 5 ms grid time), never interpolated
    ones, so a plotted value is one the plant had; the reference is interpolated from the host ticks.
    """
    plant_t = np.asarray(log.t, dtype=float)
    picks = _grid_indices(
        plant_t,
        plant_t[0] if plant_t.size else 0.0,
        plant_t[-1] if plant_t.size else 0.0,
        1.0 / TRACE_RATE_HZ,
    )
    times = plant_t[picks]
    out: dict[str, list[float | None]] = {"t": _rounded(times)}
    q = np.atleast_2d(np.asarray(log.q, dtype=float))
    tau = np.atleast_2d(np.asarray(log.tau, dtype=float))
    ref = None if log.ref_position is None else np.atleast_2d(np.asarray(log.ref_position, dtype=float))
    for joint, name in enumerate(log.joint_names):
        out[f"q_deg.{name}"] = _rounded(np.degrees(q[picks, joint]))
        if ref is not None:
            ref_here = np.interp(times, np.asarray(log.sent_t, dtype=float), ref[:, joint])
            out[f"ref_deg.{name}"] = _rounded(np.degrees(ref_here))
            out[f"err_deg.{name}"] = _rounded(np.degrees(ref_here - q[picks, joint]))
        out[f"tau_nm.{name}"] = _rounded(tau[picks, joint])
    tool = np.asarray(log.tool_pos, dtype=float)
    out["tool_x_mm"] = _rounded(tool[picks, 0] * 1000.0)
    out["tool_z_mm"] = _rounded(tool[picks, 2] * 1000.0)
    if rung.scene.belt:
        contact_picks = picks
        touch = contact.first_touch_index(log.in_contact)
        if touch is not None:
            touch_t = plant_t[touch]
            window_start = max(plant_t[0], touch_t - CONTACT_TRACE_BEFORE_S)
            window_end = min(plant_t[-1], touch_t + CONTACT_TRACE_AFTER_S)
            dense = _grid_indices(plant_t, window_start, window_end, 1.0 / CONTACT_TRACE_RATE_HZ)
            outside = (times < window_start) | (times > window_end)
            contact_picks = np.union1d(picks[outside], dense)
        out["contact_t"] = _rounded(plant_t[contact_picks])
        out["contact_fn_n"] = _rounded(np.asarray(log.contact_force, dtype=float)[contact_picks, 2])
    out["tick_t"] = _rounded(np.asarray(log.tick_t, dtype=float))
    out["compute_us"] = _rounded(np.asarray(log.compute_s, dtype=float) * 1.0e6)
    return out


def _grid_indices(sample_t: np.ndarray, start_s: float, end_s: float, step_s: float) -> np.ndarray:
    """Indices of the first sample at or after each grid time start, start + step, ... <= end (unique)."""
    if sample_t.size == 0:
        return np.array([], dtype=int)
    grid = np.arange(start_s, end_s + 1e-9, step_s)
    picks = np.searchsorted(sample_t, grid - 1e-9)
    return np.unique(np.clip(picks, 0, sample_t.size - 1))


def round_significant(value: float, digits: int = TRACE_SIGNIFICANT_DIGITS) -> float | None:
    number = float(value)
    return float(f"{number:.{digits}g}") if math.isfinite(number) else None


def _rounded(values: np.ndarray) -> list[float | None]:
    return [round_significant(v) for v in np.asarray(values, dtype=float).reshape(-1)]


def _json_value(value: Any) -> Any:
    """Events as JSON: numpy scalars/arrays -> Python, tuples -> lists, NaN/inf -> None."""
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if value is None or isinstance(value, str):
        return value
    return repr(value)
