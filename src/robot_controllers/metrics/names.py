"""The metric catalogue: every number a rung reports, its unit, family and which way is better.

These names are the contract between the metrics (which compute them), the rungs (whose gates and objectives
name them), the W&B projects (summary keys, the same in every box) and the site (columns). Units live in the
key. Angles are shown in degrees (Zach, 2026-11-11); stored data elsewhere stays SI.

A metric that a run could not measure is `None` ("not measured"), never 0: a stepped sim has no wall-clock
jitter of its own, a free-space rung has no contact force.

Sources: the controller-metrics prior-art note (vault, 2026-11-11) rows 1-30; ros2_control's statistics names
(`execution_time`, `periodicity`); robot_msgs.compare's ErrorStats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Family = Literal["tracking", "contact", "disturbance", "replanning", "timing", "health", "cost", "sequence"]
Better = Literal["lower", "higher", "none"]


@dataclass(frozen=True)
class MetricInfo:
    name: str
    unit: str  # display unit: "deg", "mm", "N", "ms", "us", "s", "%", "count", "bytes", ""
    family: Family
    better: Better
    description: str


def _m(name: str, unit: str, family: Family, better: Better, description: str) -> MetricInfo:
    return MetricInfo(name, unit, family, better, description)


METRICS: tuple[MetricInfo, ...] = (
    # tracking: reference = the rung's active input sampled on the intended schedule (so a slip is an error)
    _m("tracking_rms_deg", "deg", "tracking", "lower", "RMS joint position error over the run, worst joint."),
    _m("tracking_p95_deg", "deg", "tracking", "lower", "95th percentile |joint position error|, worst joint."),
    _m("tracking_max_deg", "deg", "tracking", "lower", "Largest |joint position error|, worst joint."),
    _m("tool_rms_mm", "mm", "tracking", "lower", "RMS tool position error against the reference's forward kinematics."),
    _m("tool_p95_mm", "mm", "tracking", "lower", "95th percentile tool position error."),
    _m("final_error_deg", "deg", "tracking", "lower", "|joint error| at the end of the run, worst joint."),
    _m("settle_time_s", "s", "tracking", "lower", "Time after the reference ends until every joint stays within 0.5 deg."),
    # contact (belt rungs; ISO/TS 15066's two phases)
    _m("touch_speed_m_s", "m/s", "contact", "none", "Tool speed normal to the belt at first contact (context, not a score)."),
    _m("contact_peak_n", "N", "contact", "lower", "Transient peak: largest normal force in the 30 ms after first touch."),
    _m("contact_max_n", "N", "contact", "lower", "Largest normal force at any time in the run."),
    _m("dwell_force_mean_n", "N", "contact", "none", "Quasi-static force: mean normal force over the dwell."),
    _m("dwell_force_std_n", "N", "contact", "lower", "Spread of the normal force over the dwell."),
    _m("dwell_force_error_n", "N", "contact", "lower", "|mean dwell force - the rung's target press force|."),
    _m("dwell_bounces", "count", "contact", "lower", "Times contact was lost during the dwell."),
    _m("dwell_contact_fraction", "%", "contact", "higher", "Share of the dwell spent in contact."),
    _m("dwell_slip_mm", "mm", "contact", "lower", "RMS tool error along the belt during the dwell (dragged by the belt)."),
    # disturbance (sideways push)
    _m("push_deflection_mm", "mm", "disturbance", "none", "Largest tool displacement during the push (softness, not a score)."),
    _m("push_recovery_s", "s", "disturbance", "lower", "Time after the push ends until the tool is back within 1 mm."),
    _m("push_overshoot_mm", "mm", "disturbance", "lower", "Largest tool displacement opposite the push after release."),
    # replanning
    _m("command_jump_max_deg", "deg", "replanning", "lower", "Largest change in commanded position between two ticks."),
    _m("torque_rate_max_nm_s", "N·m/s", "replanning", "lower", "Largest rate of change of motor torque."),
    _m("replan_settle_s", "s", "replanning", "lower", "Time after the last replan's trajectory ends to settle within 0.5 deg."),
    _m("replans_accepted", "count", "replanning", "higher", "Replans the controller took (a refusal counts 0)."),
    # timing (from the stamped log: sent times, state stamps)
    _m("period_jitter_p50_ms", "ms", "timing", "lower", "Median |actual period - nominal|."),
    _m("period_jitter_p99_ms", "ms", "timing", "lower", "99th percentile |actual period - nominal|."),
    _m("period_jitter_max_ms", "ms", "timing", "lower", "Largest |actual period - nominal|."),
    _m("missed_cycles", "count", "timing", "lower", "Sum over periods of max(0, round(period / nominal) - 1)."),
    _m("worst_stall_ms", "ms", "timing", "lower", "Longest gap between two sent commands."),
    _m("latency_p50_ms", "ms", "timing", "lower", "Median sensor-to-command latency: sent time - state sample stamp."),
    _m("latency_p99_ms", "ms", "timing", "lower", "99th percentile sensor-to-command latency."),
    _m("latency_max_ms", "ms", "timing", "lower", "Largest sensor-to-command latency."),
    _m("stale_replies", "count", "timing", "lower", "Ticks whose state seq did not advance (the same sample used twice)."),
    _m("schedule_slip_ms", "ms", "timing", "lower", "How late the controller's own clock ends vs the input's schedule."),
    _m("watchdog_trips", "count", "timing", "lower", "Times a board's command watchdog expired (moteus mode 11)."),
    # health
    _m("saturation_share", "%", "health", "lower", "Share of board samples with a joint at its torque cap."),
    _m("faults", "count", "health", "lower", "Board faults plus controller aborts."),
    _m("diverged", "count", "health", "lower", "1 if any command or state went non-finite or out of bounds, else 0."),
    # cost
    _m("compute_us_p50", "us", "cost", "lower", "Median wall time of one controller tick (measured, single BLAS thread)."),
    _m("compute_us_p99", "us", "cost", "lower", "99th percentile wall time of one controller tick."),
    _m("compute_us_max", "us", "cost", "lower", "Largest wall time of one controller tick."),
    _m("command_fields", "count", "cost", "lower", "ServoCommand fields set per joint per tick (bus load proxy)."),
    _m("wire_bytes_per_tick", "bytes", "cost", "lower", "Estimated CAN-FD payload per tick for all joints (moteus registers)."),
    _m("host_rate_hz", "Hz", "cost", "lower", "The host loop rate this run used."),
    # sequence (JointCommandSequence box)
    _m("steps_done", "count", "sequence", "higher", "Steps that finished on their done-when."),
    _m("steps_total", "count", "sequence", "none", "Steps in the sequence."),
    _m("sequence_time_s", "s", "sequence", "lower", "Time until the sequence finished (done or aborted)."),
    _m("reached_error_deg", "deg", "sequence", "lower", "|target - position| when the last Reached fired, worst joint."),
)

BY_NAME: dict[str, MetricInfo] = {m.name: m for m in METRICS}
if len(BY_NAME) != len(METRICS):
    raise RuntimeError("duplicate metric names")

FAMILIES: tuple[Family, ...] = ("tracking", "contact", "disturbance", "replanning", "timing", "health", "cost", "sequence")


def info(name: str) -> MetricInfo:
    try:
        return BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown metric {name!r}") from None
