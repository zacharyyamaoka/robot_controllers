"""Run requests for the real arms, in arm_motion_bench's bundle terms (CONTRACT amendment 5, §5.9).

    from robot_controllers.real.requests import run_request, write_requests
    request = run_request("bam_fivebar")           # dict: the target, P2 + the top 3 real-safe configs, the rate
    write_requests()                               # results/grade/requests/<station>.json, one per real station

robot_controllers never drives hardware and never queues a job: a request is what the rig's owner (arm_motion_bench's
Trajectory Tracking Agent, through `scripts/bam_fb_real_jtc.py` and `rackq`) needs to build one `fullarm_samples`
bundle per config and run it. Each config is spelled the way that tooling spells a bundle:

- `effort`: `as_built` (the tape's own column, P2) or `model_at_plan` (the plan-time model column `modelff` writes:
  inertial + offset + viscous + J1 gravity), `+ coulomb` for P5 ("coulomb held in the approach direction"),
  `model_at_host` for P6 (evaluated each tick on the measured state: not a bundle column);
- `feedback`: the boards' PD at fullarm_jtc's gains, or for P7 computed-torque gains re-sent with every command
  (moteus kp_scale / kd_scale);
- `lookahead_ms`: per joint (the calibrated lookahead, the friction-aware bus delay of §5.2, from the grade ledger's
  twin calibration at that rate and bus), plus `lookahead_ms_scalar` (their mean) because `augment_bundle` takes one `shift_s` for every joint;
- `interpolation`: `tangent` (the JTC as it runs) or `secant_mid_ff`, which a bundle carries as a column transform at
  the JTC's own tick period T: `qd <- (q(t + T) - q(t)) / T` and `effort <- effort(t + T / 2)`.

Which configs (`choose_configs`): the as-built P2 (what the arm runs today) plus the three best `population.REAL_SAFE`
configs for accuracy at the arm's base scenario in the grade ledger, each with a one-line WHY. Only REAL_SAFE goes real
(§5.5: the board closes the feedback and there is a feed-forward). The logger ask of §5.9 rides along: stamp each exchange's start and the board order (or each reply's
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


def _feedback(config_id: str, arm_name: str) -> str:
    member = config(config_id)
    if member.fb_model == "board_gains":
        return "the boards' PD at the station's own gains (fullarm_jtc's on bam_fb)"
    if member.fb_model == "ctc" and member.fb_at == "board_scheduled":
        from robot_controllers.bench.layouts import feedback_model

        bandwidth = feedback_model("ctc", arm_name).bandwidth_hz
        return (
            f"computed-torque gains re-sent with every command: kp = M_jj(q_ref)·ωn², kd = 2ζωn·M_jj(q_ref) "
            f"(ζ 0.9, {bandwidth:.3g} Hz), as moteus kp_scale / kd_scale over the boards' configured gains"
        )
    raise ValueError(f"{config_id}'s feedback ({member.fb_model}@{member.fb_at}) is not real-safe")


def _effort_text(config_id: str) -> str:
    return "model_at_host (measured state, each tick)" if config(config_id).ff_at == "host_measured" else _effort(
        config_id
    )


def choose_configs(
    arm_name: str, ledger: Mapping[str, Any], *, top: int = 3
) -> list[tuple[str, str]]:
    """[(config, why)]: the as-built P2, then the `top` best REAL_SAFE configs for accuracy at the arm's base scenario
    (`grade.rank` over the ledger's passing rows)."""
    from robot_controllers.grade import rank

    rows = [r for r in ledger.get("rows", []) if r.get("source", "sim") == "sim" and r["arm"] == arm_name
            and r["axis"] == "base"]
    if not rows:
        raise ValueError(f"the ledger has no base scenario for {arm_name}: grade it first")
    order = [r for r in rank([r for r in rows if r["config"] in REAL_SAFE], "accuracy")]
    by = {r["config"]: r for r in rows}

    def numbers(row: Mapping[str, Any]) -> str:
        m = row["metrics"]
        return f"tool p95 {m['tool_p95_mm']:.3g} mm, joint p95 {m['tracking_p95_deg']:.3g} deg"

    base = rows[0]["scenario"]
    out = [("P2", f"the as-built baseline (the tape's own column, board PD, no lookahead: what the arm runs today); "
                  f"twin at {base}: {numbers(by['P2'])}")]
    picked = [r for r in order if r["config"] != "P2"][:top]
    for k, row in enumerate(picked, start=1):
        out.append((row["config"], f"real-safe #{k} for accuracy of {len(order)} passing at {base}: {numbers(row)}"))
    return out


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
    configs: tuple[str, ...] | None = None,
    ledger: Mapping[str, Any] | None = None,
    bus_label: str = "stagger",
) -> dict[str, Any]:
    """One station's request: run `target` with each config at `rate_hz`. `configs` default: `choose_configs` on
    `ledger` (default: the committed grade ledger). The calibrated lookahead comes from the ledger's twin calibration at
    that rate on `bus_label` (default the rig as measured; none there = left for the sweep on the rig)."""
    from robot_controllers.bench.arms import arm as registered

    if ledger is None:
        from robot_controllers.grade import load_ledger

        ledger = load_ledger()
    if configs is None:
        chosen = choose_configs(arm_name, ledger)
    else:
        chosen = [(cid, "asked for") for cid in configs]

    spec = registered(arm_name)
    if spec.real_station is None:
        raise ValueError(f"{arm_name} has no real station")
    unsafe = [c for c, _why in chosen if c not in REAL_SAFE]
    if unsafe:
        raise ValueError(f"{unsafe} are not real-safe; only {REAL_SAFE} run on a real arm (§5.5)")
    targets = {t.id: t for t in spec.targets()}
    target_id = f"{arm_name}/{target}"
    params = dict(targets[target_id].params)
    calibration = _calibration(ledger, arm_name, rate_hz, bus_label)
    entries = []
    for cid, why in chosen:
        member = config(cid)
        lookahead = None
        source = None
        if member.lookahead_mode == "calibrated" and calibration is not None:
            lookahead = list(calibration["lookahead_ms"])
            source = f"the twin's calibrated bus delay at {rate_hz:g} Hz on the {bus_label} bus (§5.2)"
        elif member.lookahead_mode == "nominal":
            from robot_controllers.calibration import datasheet_bus

            datasheet = datasheet_bus(len(spec.joint_names))
            lookahead = [round(1e3 * v, 6) for v in datasheet.per_joint(len(spec.joint_names))]
            source = datasheet.source
        elif member.lookahead_mode == "none":
            lookahead = [0.0] * len(spec.joint_names)
            source = "none"
        entries.append({
            "config": cid,
            "title": member.title,
            "why": why,
            "effort": _effort_text(cid),
            "feedback": _feedback(cid, arm_name),
            "lookahead_mode": member.lookahead_mode,
            "lookahead_ms": lookahead,
            "lookahead_source": source,
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
