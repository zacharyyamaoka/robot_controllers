"""Run requests for the real arms, in arm_motion_bench's bundle terms (CONTRACT amendment 5, §5.9).

    from robot_controllers.real.requests import run_request, write_requests
    request = run_request("bam_fivebar")           # dict: the target, the four real-safe configs, the rate, the bus
    write_requests()                               # results/grade/requests/<station>.json, one per real station

robot_controllers never drives hardware and never queues a job: a request is what the rig's owner (arm_motion_bench's
Trajectory Tracking Agent, through `scripts/bam_fb_real_jtc.py` and `rackq`) needs to build one `fullarm_samples`
bundle per config and run it. Each config is spelled the way that tooling spells a bundle:

- `effort`: `as_built` (the tape's own column, P2) or `model_at_plan` (the plan-time model column `modelff` writes:
  inertial + offset + viscous + J1 gravity), `+ coulomb` for P5 ("coulomb held in the approach direction");
- `lookahead_ms`: per joint (the calibrated best lookahead from the grade ledger's twin calibration at that rate and
  bus), plus `lookahead_ms_scalar` (their mean) because `augment_bundle` takes one `shift_s` for every joint;
- `interpolation`: `tangent` (the JTC as it runs) or `secant_mid_ff`, which a bundle carries as a column transform at
  the JTC's own tick period T: `qd <- (q(t + T) - q(t)) / T` and `effort <- effort(t + T / 2)`.

Only `population.REAL_SAFE` goes real (P2, P4, P4s, P5s): every host-side feedback law saturated the boards at 50 Hz
in every sim. The logger ask of §5.9 rides along: stamp each exchange's start and the board order (or each reply's
time), so one run calibrates the bus by xcorr; until then the real calibration is the sweep.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robot_controllers.bench.population import REAL_SAFE, config

SCHEMA = "robot-controllers/real-request-1"
OWNER = "Trajectory Tracking Agent (arm_motion_bench)"
LOGGER_ASK = (
    "stamp each exchange's start and the order the boards are served in (or each reply's own time): with the "
    "exchange's end alone, one run bounds the delay only to one exchange (~17 ms), so the calibration is the sweep"
)


def _effort(config_id: str) -> str:
    member = config(config_id)
    if member.ff_model == "trajectory":
        return "as_built"
    if member.ff_model == "rigid_body_coulomb":
        return "model_at_plan + coulomb"
    return "model_at_plan"


def _calibration(ledger: Mapping[str, Any] | None, arm: str, rate_hz: float, bus_label: str) -> dict[str, Any] | None:
    if ledger is None:
        return None
    for row in ledger.get("rows", []):
        if (
            row.get("source", "sim") == "sim"
            and row["arm"] == arm
            and float(row["host_rate_hz"]) == float(rate_hz)
            and row["bus"]["label"] == bus_label
            and row.get("calibration")
        ):
            return row["calibration"]
    return None


def run_request(
    arm_name: str,
    *,
    rate_hz: float = 50.0,
    target: str = "path@nominal",
    configs: tuple[str, ...] = REAL_SAFE,
    ledger: Mapping[str, Any] | None = None,
    bus_label: str = "stagger",
) -> dict[str, Any]:
    """One station's request: run `target` with each of `configs` at `rate_hz`. The calibrated lookahead comes from
    `ledger`'s twin calibration at that rate on `bus_label` (default the rig as measured; None = left for the
    sweep on the rig)."""
    from robot_controllers.bench.arms import arm as registered

    spec = registered(arm_name)
    if spec.real_station is None:
        raise ValueError(f"{arm_name} has no real station")
    unsafe = [c for c in configs if c not in REAL_SAFE]
    if unsafe:
        raise ValueError(f"{unsafe} are not real-safe; only {REAL_SAFE} run on a real arm (§5.5)")
    targets = {t.id: t for t in spec.targets()}
    target_id = f"{arm_name}/{target}"
    params = dict(targets[target_id].params)
    calibration = _calibration(ledger, arm_name, rate_hz, bus_label)
    entries = []
    for cid in configs:
        member = config(cid)
        lookahead = None
        if member.lookahead_mode == "calibrated" and calibration is not None:
            lookahead = list(calibration["best_lookahead_ms"])
        elif member.lookahead_mode == "none":
            lookahead = [0.0] * len(spec.joint_names)
        entries.append({
            "config": cid,
            "title": member.title,
            "effort": _effort(cid),
            "feedback": "the boards' PD (fullarm_jtc's gains)",
            "lookahead_mode": member.lookahead_mode,
            "lookahead_ms": lookahead,
            "lookahead_ms_scalar": None if lookahead is None else round(sum(lookahead) / len(lookahead), 3),
            "lookahead_note": None if lookahead is not None
            else "calibrate on the rig: the sweep of §5.2 on a real log",
            "interpolation": member.interpolation,
            "interpolation_in_bundle": None if member.interpolation == "tangent" else (
                "qd <- (q(t + T) - q(t)) / T and effort <- effort(t + T/2), T = the JTC's tick period"
            ),
        })
    return {
        "schema": SCHEMA,
        "station": spec.real_station,
        "arm": arm_name,
        "owner": OWNER,
        "target": {
            "id": target_id,
            "base_rung": params.get("base_rung"),
            "time_scale": params.get("time_scale"),
            "peak_joint_speed_deg_s": params.get("peak_joint_speed_deg_s"),
            "what": targets[target_id].rung().description,
        },
        "rate_hz": float(rate_hz),
        "configs": entries,
        "calibration": calibration,
        "score_against": "the bundle's q_reference on the JTC's leg clock (robot_controllers.real.import_fullarm)",
        "logger_ask": LOGGER_ASK,
        "why": "robot_controllers never drives hardware (CONTRACT §5.9): the rig's owner builds and queues these",
    }


def write_requests(out_dir: Path | None = None, *, ledger: Mapping[str, Any] | None = None) -> list[Path]:
    """One request per real station of the registered arms: `<out_dir>/<station>.json` (default
    results/grade/requests/)."""
    from robot_controllers.bench.arms import arms
    from robot_controllers.grade import default_out_dir

    out = out_dir or default_out_dir() / "requests"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in arms():
        if spec.real_station is None:
            continue
        path = out / f"{spec.real_station}.json"
        path.write_text(json.dumps(run_request(spec.name, ledger=ledger), indent=1) + "\n")
        paths.append(path)
    return paths
