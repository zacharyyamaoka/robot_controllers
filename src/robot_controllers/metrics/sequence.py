"""Sequence metrics (JointCommandSequence box): steps done, time to finish, error at the last Reached.

Reads the log's `step_done` events (`{"t", "kind": "step_done", "outcome", ...}`, outcome as robot_msgs'
StepEvent: "Reached", "Settled", "Elapsed", "Stall j2", or "timeout" / "guard" when it aborted),
`meta["steps_total"]`, the status column, the commanded positions and the plant. Pure on arrays and event
lists. WHY pure: an online monitor can count steps as events arrive.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from robot_controllers.metrics.tracking import interpolate_columns

ABORT_OUTCOMES = frozenset({"timeout", "guard"})  # robot_msgs StepEvent outcomes that abort the sequence


def step_events(events: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [e for e in events if e.get("kind") == "step_done"]


def steps_done(events: Sequence[Mapping[str, Any]]) -> int:
    """Steps that finished on their done-when (an abort outcome is not a finished step)."""
    return sum(
        1 for e in step_events(events) if str(e.get("outcome", "")).split(" ")[0] not in ABORT_OUTCOMES
    )


def sequence_time_s(sent_t: np.ndarray, status: np.ndarray) -> tuple[float | None, str]:
    """The sent time of the first tick whose status is no longer `running` (done or aborted)."""
    sent = np.asarray(sent_t, dtype=float).reshape(-1)
    finished = np.asarray(status).reshape(-1) != "running"
    if sent.size == 0 or not finished.any():
        return None, "the sequence did not finish inside the run"
    return float(sent[int(np.argmax(finished))]), ""


def reached_error_deg(
    events: Sequence[Mapping[str, Any]],
    joint_names: Sequence[str],
    tick_t: np.ndarray,
    cmd_position: np.ndarray,
    plant_t: np.ndarray,
    plant_q: np.ndarray,
) -> tuple[float | None, str]:
    """|target - position| at the last `Reached` step_done event, worst joint, in degrees.

    Target: the event's `target` (a joint -> rad mapping, or values in joint order) when the loop recorded it;
    otherwise the commanded position of the tick BEFORE the one that fired it: `Reached` only fires after
    every watched joint's setpoint has sat at its target for its dwell (robot_msgs.control.Reached), and the
    firing tick's own command already belongs to the next step. Position: the plant interpolated at the event
    time. Joints without a target (NaN, limp) are left out.
    """
    reached = [e for e in step_events(events) if str(e.get("outcome", "")).startswith("Reached")]
    if not reached:
        return None, "not measured: no step finished on Reached"
    event = reached[-1]
    event_t = float(event["t"])
    names = list(joint_names)
    target = np.full(len(names), np.nan)
    recorded = event.get("target")
    if isinstance(recorded, Mapping):
        for joint, value in recorded.items():
            if joint in names:
                target[names.index(joint)] = float(value)
    elif recorded is not None:
        target[:] = np.asarray(recorded, dtype=float).reshape(-1)[: len(names)]
    else:
        ticks = np.asarray(tick_t, dtype=float).reshape(-1)
        firing_tick = int(np.searchsorted(ticks, event_t + 1e-9, side="right")) - 1
        if firing_tick < 1:
            return None, "not measured: no tick before the Reached event to read the setpoint from"
        target = np.atleast_2d(np.asarray(cmd_position, dtype=float))[firing_tick - 1].copy()
    measured = interpolate_columns(plant_t, plant_q, np.array([event_t]))[0]
    error = np.abs(target - measured)
    error = error[np.isfinite(error)]
    if error.size == 0:
        return None, "not measured: no joint had a target at the Reached event"
    return math.degrees(float(np.max(error))), ""
