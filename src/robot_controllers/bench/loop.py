"""The host loop: one controller on one rung for one seed -> an EpisodeLog (CONTRACT §3 "Host loop").

    log = run_episode(controller("joint_trajectory_controller"), rung, seed=0)

1. Reset the plant with the rung's scene (start at `scene.start_q` or home); build the input script from the
   start joints; the controller is `spec.build(first message)`.
2. The host wakes at k * period, plus seeded gaussian jitter clipped to +-period/2, and never inside a stall
   window (and never before its previous command went out).
3. On a wake: script messages due at or before now are delivered first (`replace_input`; a controller without
   it, or one that raises, gives a `refused` event). Then `state = plant.read()` (the newest board sample that
   has arrived) and `controller.tick(state.joint_state(), dt)` is timed with perf_counter_ns (dt = this wake -
   the previous wake, 0 on the first).
4. The command is stamped (header.stamp = sent time = wake + compute when `include_compute`, state_stamp,
   state_seq) and written; the boards latch it command_delay later. One host-tick row is recorded; the plant
   records every physics step.
5. The reference is the rung's active script message sampled at (sent time - its scheduled start), the end
   held: the intended schedule, so a controller that slips shows it as tracking error.

`meta` carries what the metrics need beyond the arrays: `dwell`, `press_target_n`, `push`, `steps_total`,
`targets_rad`, `schedule_end_s`, `start_q`, the timing, the board and plant configuration, the wall time, and
`compute_cpu_s` (each tick's thread CPU time) with `load_average_1m`: a slow controller vs a busy host.

The torque split (amendment 3): after each tick, outside its timing, a controller with `last_split()` reports
how much of its command's feed-forward field was feed-forward and how much host feedback (`split_ff`,
`split_fb_host`, one row per host tick); the plant logs per physics step what the board added and its own PD
term (`tau_ff`, `tau_fb_board`) and which host tick's command it ran (`latched_tick`). At the end, a
controller whose `diagnostics()` reports `plan_compute_s` (plan-time CPU) has it copied to
`meta["plan_compute_s"]`.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
from robot_msgs.control import JointCommandSequence, ServoCommand
from robot_msgs.trajectory import JointTrajectory

from robot_controllers.bench.arm import tool_position
from robot_controllers.bench.plant import PlanarArmPlant
from robot_controllers.bench.types import EpisodeLog, Rung
from robot_controllers.core import ControllerSpec

SCRIPT_TOLERANCE_S = 1e-9  # a message scheduled at 0.15 s is due on a wake computed as 0.15000000000000002
COMMAND_FIELDS = ("position", "velocity", "feedforward_effort", "stiffness", "damping", "max_effort")


def run_episode(
    spec: ControllerSpec,
    rung: Rung,
    *,
    seed: int = 0,
    plant_factory: Callable[[], Any] | None = None,
) -> EpisodeLog:
    """Run `spec`'s controller on `rung` and return everything that happened (see the module docstring).

    `plant_factory` builds the plant (default: the plant `rung.plant` names, `plant_for`); anything with
    reset/read/write/advance/samples/ meta/events/joint_names/start_q works, and its `tool_position(q)` (when
    it has one) turns the reference into `ref_tool_pos` (amendment 4, §4.5).
    """
    # WHY here: the contract times tick compute on one BLAS thread; this reaches BLAS libraries loaded after
    # it (numpy's own pool is fixed at import, so a runner should also export it before Python starts).
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    wall_start = time.perf_counter()
    load_average = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    timing = rung.timing
    period = timing.period_s
    duration = float(rung.duration_s)
    plant = plant_factory() if plant_factory is not None else plant_for(rung.plant)
    plant.reset(rung.scene, seed, timing=timing, duration_s=duration)
    joint_names = tuple(plant.joint_names)
    joint_count = len(joint_names)
    start_q = np.asarray(plant.start_q, dtype=float)

    script = sorted(
        ((float(t), message) for t, message in rung.inputs(start_q.copy())), key=lambda item: item[0]
    )
    if not script or abs(script[0][0]) > SCRIPT_TOLERANCE_S:
        raise ValueError(f"rung {rung.id}: the input script must start with a message at t = 0")
    controller = spec.build(script[0][1], **_rate_overrides(spec, period))
    references = [_Reference.of(message, joint_names) for _, message in script]

    events: list[dict[str, Any]] = []
    for stall_start, stall_length in timing.stalls:
        if stall_start < duration:
            events.append({"t": float(stall_start), "kind": "stall_start", "duration_s": float(stall_length)})
            events.append({"t": float(stall_start + stall_length), "kind": "stall_end"})

    rows = _Rows(joint_count, capacity=int(math.ceil(duration / period)) + 2)
    rng = np.random.default_rng(seed)
    next_script = 1
    active = 0
    previous_wake: float | None = None
    previous_sent = 0.0
    seen_step_events = 0
    compute_cpu: list[float] = []  # each tick's thread CPU time (the scored compute_s)
    compute_wall: list[float] = []  # the same ticks' wall time: includes preemption by other processes
    last_status = getattr(controller, "status", "running")
    controller_failed = False

    for wake_index in range(int(math.ceil(duration / period)) + 1):
        nominal = wake_index * period
        jitter = 0.0
        if timing.jitter_s > 0.0:
            # WHY draw for every k, even skipped ones: a stall window then does not shift later wakes' jitter
            jitter = float(np.clip(rng.normal(0.0, timing.jitter_s), -period / 2.0, period / 2.0))
        wake = max(nominal + jitter, 0.0, previous_sent)
        if wake >= duration - SCRIPT_TOLERANCE_S:
            break
        if any(start <= wake < start + length for start, length in timing.stalls):
            continue
        if controller_failed:
            continue
        plant.advance(wake)

        # script messages due now: delivered before the tick
        while next_script < len(script) and script[next_script][0] <= wake + SCRIPT_TOLERANCE_S:
            scheduled_t, message = script[next_script]
            event = {"t": wake, "index": next_script, "scheduled_t": scheduled_t}
            replace = getattr(controller, "replace_input", None)
            if not callable(replace):
                events.append({**event, "kind": "refused", "why": "the controller has no replace_input"})
            else:
                replan_start = time.perf_counter_ns()
                try:
                    replace(message)
                except Exception as error:  # noqa: BLE001 - a refusal is a finding, recorded, never fatal
                    events.append({**event, "kind": "refused", "why": f"{type(error).__name__}: {error}"})
                else:
                    compute = (time.perf_counter_ns() - replan_start) * 1e-9
                    events.append({**event, "kind": "replan", "compute_s": compute})
            active = next_script  # the rung's intended input, whether or not the controller took it
            next_script += 1

        state = plant.read()
        dt = 0.0 if previous_wake is None else wake - previous_wake
        tick_cpu_start = time.thread_time_ns()
        tick_start = time.perf_counter_ns()
        try:
            command, defines = controller.tick(state.joint_state(), dt)
        except Exception as error:  # noqa: BLE001 - the run records the abort and the plant runs on unattended
            events.append(
                {"t": wake, "kind": "abort", "why": f"controller.tick raised {type(error).__name__}: {error}"}
            )
            controller_failed = True
            continue
        # WHY thread CPU time is the scored cost (2026-11-12): this machine runs peer fleets at load ~100, and wall
        # time then counts other processes' turns as 100+ ms "ticks" that fail timing gates at random. CPU time is
        # the controller's own work; wall time stays in meta["compute_wall_s"] for diagnosis.
        compute_wall.append((time.perf_counter_ns() - tick_start) * 1e-9)
        compute_s = (time.thread_time_ns() - tick_cpu_start) * 1e-9
        compute_cpu.append(compute_s)
        sent = wake + compute_s if timing.include_compute else wake
        command = command.replace(
            header=command.header.replace(stamp=sent),
            state_stamp=state.header.stamp,
            state_seq=state.seq,
        )
        defines = dict(defines or {})
        plant.write(command, defines or None)
        split = _last_split(controller)
        for joint, value in defines.items():
            events.append({"t": wake, "kind": "define", "joint": joint, "value": float(value)})

        status = str(getattr(controller, "status", "running"))
        if status == "aborted" and last_status != "aborted":
            events.append(
                {"t": wake, "kind": "abort", "why": str(getattr(controller, "message", "")) or "aborted"}
            )
        last_status = status
        step_events = getattr(controller, "events", None)
        if isinstance(step_events, list) and len(step_events) > seen_step_events:
            for step_event in step_events[seen_step_events:]:
                events.append(_step_done(step_event, wake, script[0][1]))
            seen_step_events = len(step_events)

        reference = references[active]
        ref_position, ref_velocity = reference.at(sent - script[active][0])
        rows.add(
            wake,
            sent,
            state.header.stamp,
            state.seq,
            compute_s,
            command,
            joint_names,
            ref_position,
            ref_velocity,
            status,
            active,
            split,
        )
        previous_wake = wake
        previous_sent = sent

    plant.advance(duration)
    samples = plant.samples()
    events.extend(plant.events)
    events.sort(key=lambda event: float(event["t"]))
    has_reference = any(reference.has_schedule for reference in references)
    ref_position = rows.array("ref_position") if has_reference else None
    meta = _meta(rung, plant, script, seed, timing)
    meta["wall_s"] = time.perf_counter() - wall_start
    meta["ticks"] = rows.count
    # WHY both clocks: compute_s (perf_counter, the contract's) also counts time the OS ran other processes;
    # on a loaded machine that shows up as fake multi-ms ticks, missed cycles and latency. The CPU time and
    # the load average let a reader tell a slow controller from a busy host.
    meta["compute_cpu_s"] = compute_cpu
    meta["compute_wall_s"] = compute_wall
    meta["compute_clock"] = "thread_cpu"
    meta["load_average_1m"] = load_average
    plan_compute = _plan_compute_s(controller)
    if plan_compute is not None:
        meta["plan_compute_s"] = plan_compute
    has_split = rows.has_split
    return EpisodeLog(
        controller=spec.name,
        rung=rung.id,
        seed=int(seed),
        joint_names=joint_names,
        nominal_period_s=period,
        tick_t=rows.array("tick_t"),
        sent_t=rows.array("sent_t"),
        state_stamp=rows.array("state_stamp"),
        state_seq=rows.array("state_seq").astype(np.int64),
        compute_s=rows.array("compute_s"),
        cmd_position=rows.array("cmd_position"),
        cmd_velocity=rows.array("cmd_velocity"),
        cmd_feedforward=rows.array("cmd_feedforward"),
        cmd_stiffness=rows.array("cmd_stiffness"),
        cmd_damping=rows.array("cmd_damping"),
        cmd_max_effort=rows.array("cmd_max_effort"),
        ref_position=ref_position,
        ref_velocity=rows.array("ref_velocity") if has_reference else None,
        ref_tool_pos=_tool_positions(plant, ref_position) if has_reference else None,
        status=np.array(rows.status, dtype="<U7"),
        input_index=np.array(rows.input_index, dtype=np.int64),
        t=samples["t"],
        q=samples["q"],
        qd=samples["qd"],
        tau=samples["tau"],
        at_cap=samples["at_cap"],
        board_mode=samples["board_mode"],
        tool_pos=samples["tool_pos"],
        contact_force=samples["contact_force"],
        in_contact=samples["in_contact"],
        external_force=samples["external_force"],
        events=events,
        meta=meta,
        tau_ff=samples.get("tau_ff"),
        tau_fb_board=samples.get("tau_fb_board"),
        latched_tick=samples.get("latched_tick"),
        split_ff=rows.array("split_ff") if has_split else None,
        split_fb_host=rows.array("split_fb_host") if has_split else None,
    )


def plant_for(name: str) -> Any:
    """A fresh plant by `Rung.plant` name: any arm in the registry (`bench.arms`, amendment 5 §5.1), on its own
    nominal boards. An unknown name is a ValueError."""
    if name == "planar_arm":
        return PlanarArmPlant()  # WHY direct: the ladder's plant, without importing every registered arm
    from robot_controllers.bench.arms import arm

    try:
        spec = arm(name)
    except KeyError as error:
        raise ValueError(f"unknown plant {name!r}: {error.args[0]}") from None
    return spec.plant()


def _tool_positions(plant: Any, positions: np.ndarray) -> np.ndarray:
    """The reference's tool path on the plant's own kinematics; a plant without them is the planar arm's."""
    kinematics = getattr(plant, "tool_position", None)
    return kinematics(positions) if callable(kinematics) else tool_position(positions)


def _last_split(controller: Any) -> tuple[Any, Any] | None:
    """The controller's (tau_ff, tau_fb_host) of the tick just sent, or None when it does not report one."""
    report = getattr(controller, "last_split", None)
    if not callable(report):
        return None
    feedforward, host_feedback = report()
    return feedforward, host_feedback


def _plan_compute_s(controller: Any) -> float | None:
    diagnostics = getattr(controller, "diagnostics", None)
    if not callable(diagnostics):
        return None
    try:
        value = diagnostics().get("plan_compute_s")
    except Exception:  # noqa: BLE001 - a self-report that fails is simply not there
        return None
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


# ── the reference on the intended schedule ─────────────────────────────────────────────────────────────────


class _Reference:
    """One script message as a reference: a JointTrajectory sampled linearly with its end held (as
    `JointTrajectory.joint_state_at(t, extrapolate="clamp")`, but np.interp per joint: it runs every host
    tick). Any other input (a JointCommandSequence) has no time schedule, so its reference is NaN."""

    def __init__(
        self,
        times: np.ndarray | None,
        positions: np.ndarray | None,
        velocities: np.ndarray | None,
        joint_count: int,
    ) -> None:
        self.times, self.positions, self.velocities = times, positions, velocities
        self.joint_count = joint_count

    @property
    def has_schedule(self) -> bool:
        return self.times is not None

    @classmethod
    def of(cls, message: Any, joint_names: Sequence[str]) -> _Reference:
        if not isinstance(message, JointTrajectory):
            return cls(None, None, None, len(joint_names))
        columns = [message.joint_index(name) if name in message.joint_names else None for name in joint_names]
        positions = np.full((len(message), len(joint_names)), np.nan)
        velocities = np.full_like(positions, np.nan)
        message_velocities = message.velocities if message.velocities is not None else None
        for out, column in enumerate(columns):
            if column is None:
                continue
            positions[:, out] = message.positions[:, column]
            velocities[:, out] = 0.0 if message_velocities is None else message_velocities[:, column]
        return cls(np.asarray(message.t, dtype=float), positions, velocities, len(joint_names))

    def at(self, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        if self.times is None:
            nan = np.full(self.joint_count, np.nan)
            return nan, nan
        position = np.array(
            [np.interp(time_s, self.times, self.positions[:, j]) for j in range(self.joint_count)]
        )
        if time_s >= self.times[-1]:  # the held end is at rest
            velocity = np.where(np.isnan(self.velocities[-1]), np.nan, 0.0)
        else:
            velocity = np.array(
                [np.interp(time_s, self.times, self.velocities[:, j]) for j in range(self.joint_count)]
            )
        return position, velocity


# ── host-tick rows ─────────────────────────────────────────────────────────────────────────────────────────


class _Rows:
    def __init__(self, joint_count: int, capacity: int) -> None:
        self.joint_count = joint_count
        self.count = 0
        self._scalars = {
            name: np.zeros(capacity) for name in ("tick_t", "sent_t", "state_stamp", "state_seq", "compute_s")
        }
        self._vectors = {
            name: np.full((capacity, joint_count), np.nan)
            for name in (
                "cmd_position",
                "cmd_velocity",
                "cmd_feedforward",
                "cmd_stiffness",
                "cmd_damping",
                "cmd_max_effort",
                "ref_position",
                "ref_velocity",
                "split_ff",
                "split_fb_host",
            )
        }
        self.has_split = False
        self.status: list[str] = []
        self.input_index: list[int] = []

    def _grow(self) -> None:
        for table in (self._scalars, self._vectors):
            for name, array in table.items():
                bigger = np.full((array.shape[0] * 2, *array.shape[1:]), np.nan if array.ndim == 2 else 0.0)
                bigger[: array.shape[0]] = array
                table[name] = bigger

    def add(
        self,
        wake: float,
        sent: float,
        stamp: float,
        seq: int,
        compute_s: float,
        command: ServoCommand,
        joint_names: Sequence[str],
        ref_position: np.ndarray,
        ref_velocity: np.ndarray,
        status: str,
        active: int,
        split: tuple[Any, Any] | None = None,
    ) -> None:
        if self.count >= len(self._scalars["tick_t"]):
            self._grow()
        row = self.count
        for name, value in (
            ("tick_t", wake),
            ("sent_t", sent),
            ("state_stamp", stamp),
            ("state_seq", seq),
            ("compute_s", compute_s),
        ):
            self._scalars[name][row] = value
        order = {name: i for i, name in enumerate(command.name)}
        for field, key in zip(
            COMMAND_FIELDS,
            (
                "cmd_position",
                "cmd_velocity",
                "cmd_feedforward",
                "cmd_stiffness",
                "cmd_damping",
                "cmd_max_effort",
            ),
            strict=True,
        ):
            values = getattr(command, field)
            if values is None:
                continue
            target = self._vectors[key][row]
            for j, name in enumerate(joint_names):
                i = order.get(name)
                if i is not None:
                    target[j] = values[i]
        self._vectors["ref_position"][row] = ref_position
        self._vectors["ref_velocity"][row] = ref_velocity
        if split is not None:
            # the split arrays are in the command's joint order (the controller's input), like its fields
            self.has_split = True
            for key, values in zip(("split_ff", "split_fb_host"), split, strict=True):
                target = self._vectors[key][row]
                values = np.asarray(values, dtype=float).reshape(-1)
                for j, name in enumerate(joint_names):
                    i = order.get(name)
                    if i is not None and i < values.size:
                        target[j] = values[i]
        self.status.append(status)
        self.input_index.append(active)
        self.count += 1

    def array(self, name: str) -> np.ndarray:
        table = self._scalars if name in self._scalars else self._vectors
        return table[name][: self.count].copy()


# ── events and meta ────────────────────────────────────────────────────────────────────────────────────────


def _step_done(step_event: Any, wake: float, first_message: Any) -> dict[str, Any]:
    """A robot_msgs StepEvent as a log event; `target` = the step's position-mode targets (what Reached waited
    for), because the command on the tick that fired already belongs to the next step."""
    event = {
        "t": wake,
        "kind": "step_done",
        "index": int(step_event.index),
        "name": str(step_event.name),
        "outcome": str(step_event.outcome),
        "started": float(step_event.started),
        "finished": float(step_event.finished),
    }
    if isinstance(first_message, JointCommandSequence) and 0 <= step_event.index < len(first_message.steps):
        step = first_message.steps[step_event.index]
        targets = {
            c.joint: float(c.position)
            for c in step.commands
            if c.mode == "position" and c.position is not None
        }
        if targets:
            event["target"] = targets
    return event


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _meta(
    rung: Rung, plant: Any, script: Sequence[tuple[float, Any]], seed: int, timing: Any
) -> dict[str, Any]:
    scene = rung.scene
    meta: dict[str, Any] = dict(plant.meta()) if hasattr(plant, "meta") else {}
    meta.update(
        {
            "rung": rung.id,
            "seed": int(seed),
            "duration_s": float(rung.duration_s),
            "host_rate_hz": float(timing.rate_hz),
            "timing": _json_value(asdict(timing)),
            "rung_params": _json_value(dict(rung.params)),
            "scene": {
                "belt": bool(scene.belt),
                "belt_speed_m_s": float(scene.belt_speed_m_s),
                "contact_stiffness_n_m": float(scene.contact_stiffness_n_m),
                "contact_damping_n_s_m": float(scene.contact_damping_n_s_m),
                "payload_kg": float(scene.payload_kg),
                "pushes": [_json_value(asdict(push)) for push in scene.pushes],
            },
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "script": [
                {
                    "t": t,
                    "type": type(message).__name__,
                    "duration_s": float(message.t[-1]) if isinstance(message, JointTrajectory) else None,
                }
                for t, message in script
            ],
        }
    )
    for (
        key,
        value,
    ) in scene.extras.items():  # the rung's windows and targets: dwell, press_target_n, targets_rad
        meta[str(key)] = _json_value(value)
    if scene.pushes:
        first = scene.pushes[0]
        axis = int(np.argmax(np.abs(first.force_n)))
        sign = "+" if first.force_n[axis] >= 0 else "-"
        meta["push"] = [float(first.start_s), float(first.start_s + first.duration_s), f"{sign}{'xyz'[axis]}"]
    first_message = script[0][1]
    if isinstance(first_message, JointCommandSequence):
        meta["steps_total"] = len(first_message.steps)
    last_t, last_message = script[-1]
    if isinstance(last_message, JointTrajectory):
        meta["schedule_end_s"] = float(last_t + last_message.t[-1])
    return meta


def _rate_overrides(spec: Any, period: float) -> dict[str, float]:
    """Configure an index-driven controller for the rung's host rate (`control_period_s`), as a deployment would.

    WHY: the tape feeder assumes its tick period; built with the 100 Hz default it plays a 500 Hz rung 5x fast,
    which measures a mis-configuration, not the controller (2026-11-12).
    """
    config_class = getattr(spec, "config_class", None)
    if not callable(config_class) or not getattr(spec, "config_target", None):
        return {}
    import dataclasses

    names = {f.name for f in dataclasses.fields(config_class())}
    return {"control_period_s": period} if "control_period_s" in names else {}
