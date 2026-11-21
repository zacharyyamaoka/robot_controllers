"""A real `fullarm_jtc` run as an EpisodeLog the bench's metrics score unchanged (CONTRACT amendment 5, §5.9).

    from robot_controllers.real.import_fullarm import import_fullarm_run
    log, rung = import_fullarm_run(run_dir, bundle_dir)     # run_dir: report.json, jtc_ticks.npy, jtc_tick_columns.json
    episode_metrics(log, rung)["tracking_p95_deg"]          # 0.664 on run_jtc-20261121T021137Z (j2, the worst joint)

What it reads (read-only; arm_motion_bench and the rack's state are never written):

- `jtc_ticks.npy` + `jtc_tick_columns.json`: one row per bus exchange, the command sent (`cmd<CAN>_*`) and every
  board's reply (`can<CAN>_*`). **The tape** is the tick phase with the most rows (fullarm_jtc's PHASES: prelude 0,
  tape 1, end 2); its time is the JTC's own `leg_clock_s`, the clock it samples the plan at.
- `report.json`'s `run.board_map`: board = sign × joint + offset per drive (arm_motion_bench
  `baselines/bam_fb_real.py` `trace_from_ticks`, ~125-160), so q = sign · (board − offset), qd and torque × sign.
- the bundle's `samples.npz`: the reference is `q_reference` (the unshifted plan: a lookahead bundle commands the
  samples read early, and a run is scored against the plan itself) sampled at the leg clock, held at its end; a bundle
  without it is scored against its `q`.

Into the EpisodeLog: one host tick and one plant sample per exchange (both at the leg clock), the commands in the
plan's joint frame, `ref_position` the reference above, `tau` the boards' measured torque, the torque split as the
names catalogue defines it on a moteus (`tau_ff` = the sent feed-forward, `tau_fb_board` = measured − sent, each
exchange's own command latched), `board_mode` 1 in moteus fault mode (1), 11 in timeout (11), else 0; a running
board's fault register (its active limit) goes to `meta['limit_flags']`. The tool
point is the registry arm's FK when the arm is registered (the five-bar's encoder FK in the same joint frame).

WHY the send times are the exchange's, not each board's (§5.9): the logger stamps the exchange's end and the boards
answer at different points inside it, so a one-run xcorr calibration on this log bounds the delay only to one exchange
(about 17 ms); the sweep calibration is the one that works on it.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from robot_msgs.trajectory import JointTrajectory

from robot_controllers.bench.types import EpisodeLog, Gate, HostTiming, Rung

MOTEUS_POSITION_MODE = 10
MOTEUS_FAULT_MODE = 1
MOTEUS_TIMEOUT_MODE = 11
ARM_FOR_ROBOT = {"bam_fb_v2": "bam_fivebar"}
SOURCE = "real"


@dataclass(frozen=True)
class ReferenceScript:
    """The plan's reference as the rung's one input (its time from 0, `duration_s`), so schedule-based metrics
    rebuild it."""

    t: tuple[float, ...]
    positions: tuple[tuple[float, ...], ...]
    velocities: tuple[tuple[float, ...], ...]
    joint_names: tuple[str, ...]

    def __call__(self, start_q: np.ndarray) -> list[tuple[float, Any]]:
        return [(0.0, JointTrajectory(t=np.asarray(self.t), joint_names=self.joint_names,
                                      positions=np.asarray(self.positions), velocities=np.asarray(self.velocities)))]

    @property
    def duration_s(self) -> float:
        return float(self.t[-1])


def _table(run_dir: Path) -> tuple[np.ndarray, dict[str, int]]:
    table = np.load(run_dir / "jtc_ticks.npy")
    columns = json.loads((run_dir / "jtc_tick_columns.json").read_text())
    return np.asarray(table, dtype=float), {name: index for index, name in enumerate(columns)}


def _columns(rows: np.ndarray, col: dict[str, int], pattern: str, can_ids: list[int]) -> np.ndarray:
    return np.column_stack([rows[:, col[pattern.format(c)]] for c in can_ids])


def _registered(arm: str | None) -> Any:
    if arm is None:
        return None
    try:
        from robot_controllers.bench.arms import arm as registered

        return registered(arm)
    except KeyError:
        return None


def import_fullarm_run(run_dir: Path | str, bundle_dir: Path | str) -> tuple[EpisodeLog, Rung]:
    """(EpisodeLog, Rung) of one fullarm_jtc run against its bundle (the module docstring)."""
    run_dir, bundle_dir = Path(run_dir), Path(bundle_dir)
    report = json.loads((run_dir / "report.json").read_text())
    board_map = report["run"]["board_map"]
    drives = tuple(str(d) for d in board_map["drives"])
    can_ids = [int(c) for c in board_map["can_ids"]]
    sign = np.asarray(board_map["sign"], dtype=float)
    offset = np.asarray(board_map["offset_rad"], dtype=float)
    table, col = _table(run_dir)
    phases = Counter(table[:, col["phase"]].tolist())
    tape_phase = max(phases, key=lambda p: phases[p])
    rows = table[table[:, col["phase"]] == tape_phase]
    clock = rows[:, col["leg_clock_s"]]
    if np.any(np.diff(clock) < -1e-9):
        raise ValueError(f"{run_dir}: leg_clock_s runs backwards inside the tape")
    count, joints = rows.shape[0], len(drives)

    def to_joint(board: np.ndarray) -> np.ndarray:
        return sign * (board - offset)

    q = to_joint(_columns(rows, col, "can{}_position_rad", can_ids))
    qd = sign * _columns(rows, col, "can{}_velocity_rad_s", can_ids)
    tau = sign * _columns(rows, col, "can{}_torque_nm", can_ids)
    cmd_q = to_joint(_columns(rows, col, "cmd{}_position_rad", can_ids))
    cmd_qd = sign * _columns(rows, col, "cmd{}_velocity_rad_s", can_ids)
    cmd_ff = sign * _columns(rows, col, "cmd{}_feedforward_nm", can_ids)
    modes = _columns(rows, col, "can{}_mode", can_ids)
    faults = _columns(rows, col, "can{}_fault", can_ids)
    # WHY the mode, not the fault register: a running moteus (mode 10) reports its active limit there (CAN 12 reads
    # 104 for most of run_jtc-20261121T021137Z's tape), which is not a fault; mode 1 is
    board_mode = np.where(modes == MOTEUS_FAULT_MODE, 1, np.where(modes == MOTEUS_TIMEOUT_MODE, 11, 0)).astype(int)
    limit_flags = {
        drives[j]: {str(int(code)): int(n) for code, n in zip(*np.unique(faults[:, j], return_counts=True), strict=True)
                    if code != 0}
        for j in range(joints)
    }

    plan = json.loads((bundle_dir / "plan.json").read_text())
    with np.load(bundle_dir / plan.get("samples", "samples.npz")) as data:
        t_ref = np.asarray(data["t"], dtype=float)
        q_ref = np.asarray(data["q_reference"] if "q_reference" in data.files else data["q"], dtype=float)
        qd_ref = np.asarray(data["qd_reference"] if "qd_reference" in data.files else data["qd"], dtype=float)
        shifted = "q_reference" in data.files
    if list(plan.get("drives", drives)) != list(drives):
        raise ValueError(f"the plan's drives {plan.get('drives')} are not the run's {drives}")
    at = np.minimum(clock, t_ref[-1])
    ref = np.column_stack([np.interp(at, t_ref, q_ref[:, k]) for k in range(joints)])
    ref_qd = np.column_stack([np.interp(at, t_ref, qd_ref[:, k]) for k in range(joints)])
    ref_qd[clock > t_ref[-1]] = 0.0

    arm_name = ARM_FOR_ROBOT.get(str(plan.get("robot")))
    spec = _registered(arm_name)
    caps = None
    if spec is not None:
        caps = list(spec.board_config().torque_cap_nm)
        tool = np.asarray(spec.tool_position(q), dtype=float)
        ref_tool = np.asarray(spec.tool_position(ref), dtype=float)
    else:
        tool, ref_tool = np.zeros((count, 3)), None
    at_cap = (
        np.abs(tau) >= 0.999 * np.asarray(caps, dtype=float)[None, :]
        if caps is not None
        else np.zeros((count, joints), dtype=bool)
    )
    dt = np.diff(clock)
    period = float(np.median(dt[dt > 1e-9])) if np.any(dt > 1e-9) else 0.017
    nan = np.full((count, joints), np.nan)
    run_id = str(report.get("run_id", run_dir.name))
    meta = {
        "plant": f"real:{plan.get('robot')}",
        "source": SOURCE,
        "run_dir": str(run_dir),
        "bundle_dir": str(bundle_dir),
        "run_id": run_id,
        "utc": report.get("utc"),
        "law": report.get("law"),
        "board_map": board_map,
        "board": {"torque_cap_nm": caps},
        "tape_phase": tape_phase,
        "limit_flags": {name: flags for name, flags in limit_flags.items() if flags},
        "phase_rows": {str(k): int(v) for k, v in phases.items()},
        "reference": "q_reference (the unshifted plan)" if shifted else "q (the plan as sent)",
        "plan_why": plan.get("why"),
        "speed_fraction": plan.get("speed_fraction"),
        "host_period_median_s": period,
        "late_ticks": int(np.sum(rows[:, col["late"]])) if "late" in col else None,
        "schedule_end_s": float(t_ref[-1]),
        "send_times": "the exchange's end (the logger's stamp), not each board's (§5.9)",
    }
    log = EpisodeLog(
        controller=f"real:{report.get('law', 'fullarm_jtc')}",
        rung=f"real:{run_id}",
        seed=0,
        joint_names=drives,
        nominal_period_s=period,
        tick_t=clock.copy(),
        sent_t=clock.copy(),
        state_stamp=clock.copy(),
        state_seq=np.arange(count),
        compute_s=np.full(count, np.nan),
        cmd_position=cmd_q,
        cmd_velocity=cmd_qd,
        cmd_feedforward=cmd_ff,
        cmd_stiffness=nan.copy(),
        cmd_damping=nan.copy(),
        cmd_max_effort=nan.copy(),
        ref_position=ref,
        ref_velocity=ref_qd,
        ref_tool_pos=ref_tool,
        status=np.array(["running"] * (count - 1) + ["done"]),
        input_index=np.zeros(count, dtype=int),
        t=clock.copy(),
        q=q,
        qd=qd,
        tau=tau,
        at_cap=at_cap,
        board_mode=board_mode,
        tool_pos=tool,
        contact_force=np.zeros((count, 3)),
        in_contact=np.zeros(count, dtype=bool),
        external_force=np.zeros((count, 3)),
        events=[{"t": 0.0, "kind": "trip", "what": trip} for trip in report.get("trips") or []],
        meta=meta,
        tau_ff=cmd_ff.copy(),
        tau_fb_board=tau - cmd_ff,
        latched_tick=np.tile(np.arange(count)[:, None], (1, joints)),
        split_ff=None,
        split_fb_host=None,
        tau_stop=None,
    )
    rung = real_rung(run_id, plan, t_ref, q_ref, qd_ref, drives, period, arm_name)
    return log, rung


def real_rung(
    run_id: str,
    plan: dict[str, Any],
    t_ref: np.ndarray,
    q_ref: np.ndarray,
    qd_ref: np.ndarray,
    drives: tuple[str, ...],
    period_s: float,
    arm_name: str | None,
) -> Rung:
    """The rung a real run is scored on: its own reference as the input, the arm's path gates (R2's: tracking p95 <=
    3 deg, saturation <= 5 %, nothing diverged) and the timing gates at its own host period."""
    from robot_controllers.bench.rungs import timing_gates
    from robot_controllers.registry import JOINT_TRAJECTORY

    rate = 1.0 / period_s
    gates = (
        Gate("tracking_p95_deg", "<=", 3.0, "tracks the tape within 3 degrees (R2's gate)"),
        Gate("saturation_share", "<=", 5, "the motion stays mostly inside the boards' torque"),
        Gate("diverged", "==", 0, "nothing went non-finite or out of bounds"),
        *timing_gates(rate),
    )
    script = ReferenceScript(
        t=tuple(float(v) for v in t_ref),
        positions=tuple(tuple(float(x) for x in row) for row in q_ref),
        velocities=tuple(tuple(float(x) for x in row) for row in qd_ref),
        joint_names=drives,
    )
    return Rung(
        id=f"real:{run_id}",
        title=f"Real {plan.get('robot')} run {run_id}",
        box=JOINT_TRAJECTORY,
        family="free_space",
        description=f"A real fullarm_jtc run of {', '.join(t['task_id'] for t in plan.get('tasks', []))} at "
        f"{plan.get('speed_fraction')}. Scored against the plan's reference on the JTC's own clock.",
        duration_s=float(t_ref[-1]),
        inputs=script,
        timing=HostTiming(rate_hz=rate, command_delay_s=0.0, reply_delay_s=0.0),
        gates=gates,
        objective="tracking_p95_deg",
        params={"source": SOURCE, "run_id": run_id, "speed_fraction": plan.get("speed_fraction") or math.nan,
                "host_rate_hz": rate},
        plant=arm_name or "real",
    )


def real_row(run_dir: Path | str, bundle_dir: Path | str, *, config: str | None = None) -> dict[str, Any]:
    """A `source: real` ledger row for one run (its metrics, passed, failed gates; `config` = which population
    member it ran, when known)."""
    from robot_controllers.metrics.episode import run_result

    log, rung = import_fullarm_run(run_dir, bundle_dir)
    result = run_result(log, rung, trace=False)
    arm_name = rung.plant if rung.plant != "real" else None
    return {
        "arm": arm_name,
        "scenario": rung.id,
        "axis": "real",
        "target": f"{arm_name}/path@nominal" if arm_name else None,
        "kind": "path",
        "speed": "nominal",
        "host_rate_hz": round(float(rung.timing.rate_hz), 4),
        "config": config,
        "controller": log.controller,
        "metrics": {k: (float(f"{v:.6g}") if isinstance(v, float) else v) for k, v in result.metrics.items()},
        "passed": bool(result.passed),
        "failed_gates": [f"{g['metric']} {g['op']} {g['value']:g}" for g in result.gates if not g["passed"]],
        "source": SOURCE,
        "run_dir": str(run_dir),
        "bundle_dir": str(bundle_dir),
        "plan_why": log.meta.get("plan_why"),
    }
