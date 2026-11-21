"""Fixed target trajectories per arm: one frozen path at three speeds, a calibration sweep and a collision
(CONTRACT amendment 5, §5.4; lane A gives the planar arm and the UR5e theirs).

    from robot_controllers.bench.arms import arm
    targets = {t.id: t for t in arm("planar_arm").targets()}
    rung = targets["planar_arm/path@fast"].rung()        # a Rung on the arm's plant, 100 Hz host
    targets["planar_arm/path@fast"].params["peak_joint_speed_deg_s"]

Zach, 2026-11-20: "fixed target trajectories ... a bunch of different controller/driver configurations (that's the
population) and we basically want to grade them on all of these scenarios".

Per arm (`<arm>/...`):

- `path@slow`, `path@nominal`, `path@fast`: one frozen path time-scaled (`TimeScaledScript`): every sample's time ×
  k (`SPEED_TIME_SCALES`: 2.0, 1.0, 0.9), velocity / k, acceleration / k², and the effort column refilled with the
  arm's nominal `rigid_body` model at the new motion (as every rung fills it). k = 1 is the base path untouched, so
  `path@nominal` runs exactly like its base rung (planar: R2, UR5e: U2). The base rung's gates.
- `calibration`: a smooth multi-joint sine sweep around home (`SineSweepScript`): every joint at once, 8° amplitude
  under a minimum-jerk envelope, a chirp from 0.4 to 1.6 Hz with each joint a quarter turn of phase apart, so every
  joint moves >= 5° (16° peak to peak) and its lag is measurable. The base rung's gates.
- `collision`: `path@nominal` with a hard stop on its most-moving joint (largest peak to peak), half way along that
  joint's travel ((min + max) / 2), on the side away from where the joint starts (`collision_stop`); the stop is
  `Scene.stops`, which every plant honours and logs (`EpisodeLog.tau_stop`, a `stop_touch` event). Gates: the base
  rung's minus tracking and saturation (an arm pressed against a wall is not tracking). The same run without the
  stop is `params["free_target"]` (= `path@nominal`).

`Target.rung()` builds once per process and caches; `Target.params` is its rung's params (`target`, `kind`, `speed`,
`time_scale`, `base_rung`, `peak_joint_speed_deg_s`, `peak_tool_speed_m_s`, and the collision's `stop_*`). Scripts
and builders are frozen dataclasses, so targets and their rungs pickle into a worker process. `arm_targets()` builds
the five for any registered arm from one path script, which is what lane B1's arms can use.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from robot_msgs.trajectory import JointTrajectory

from robot_controllers.bench.types import Gate, HostTiming, Rung, Scene

TargetKind = Literal["path", "calibration", "collision"]
KINDS: tuple[str, ...] = ("path", "calibration", "collision")
SPEEDS: tuple[str, ...] = ("slow", "nominal", "fast")
# WHY 0.9 for fast (measured 2026-11-20 on the nominal rigid_body model): the largest round time scale whose
# feed-forward stays under every joint's cap on both arms, the planar shoulder at 98 % and the UR5e pan at 94 %;
# 0.85 needs 105 %. Slow is half speed.
SPEED_TIME_SCALES: Mapping[str, float] = {"slow": 2.0, "nominal": 1.0, "fast": 0.9}
SAMPLE_PERIOD_S = 0.004  # the calibration sweep is sampled every 4 ms, as every rung
TAIL_S = 0.6  # the run continues this long after the script ends (as R1/R2/U1/U2)
TRACKING_METRICS = ("tracking_p95_deg", "saturation_share")  # the gates a collision run drops


# ── the target ─────────────────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Target:
    """One fixed target of one arm: `id` (`<arm>/path@slow|nominal|fast`, `<arm>/calibration`, `<arm>/collision`),
    `arm`, `kind`, `speed` (slow | nominal | fast for the paths and the collision, None for the calibration) and
    `build` (called once: the target's Rung). `rung()` and `params` are what a grader reads."""

    id: str
    arm: str
    kind: TargetKind
    speed: str | None
    build: Callable[[], Rung] = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"Target.kind must be one of {KINDS}, got {self.kind!r}")
        if self.speed is not None and self.speed not in SPEEDS:
            raise ValueError(f"Target.speed must be one of {SPEEDS} or None, got {self.speed!r}")

    @functools.cached_property
    def _rung(self) -> Rung:
        return self.build()

    def rung(self) -> Rung:
        """The target's rung: its plant, script, scene, gates and duration; the host at 100 Hz unless the arm's
        base rung says otherwise (vary it with `rungs.rate_variant` / `bus_variant`)."""
        return self._rung

    @property
    def params(self) -> Mapping[str, Any]:
        return self.rung().params


# ── scripts ────────────────────────────────────────────────────────────────────────────────────────────────


def _rigid_body(arm: str) -> Any:
    from robot_controllers.bench.arms import arm as registered

    return registered(arm).feedforward("rigid_body", None)


@functools.lru_cache(maxsize=64)
def _time_scaled(base: Any, time_scale: float, arm: str, start_q: tuple[float, ...]) -> JointTrajectory:
    trajectory = base(np.asarray(start_q, dtype=float))[0][1]
    if time_scale == 1.0:
        return trajectory
    k = float(time_scale)
    t = np.asarray(trajectory.t, dtype=float) * k
    q = np.asarray(trajectory.positions, dtype=float)
    qd = None if trajectory.velocities is None else np.asarray(trajectory.velocities, dtype=float) / k
    if trajectory.accelerations is not None:
        qdd = np.asarray(trajectory.accelerations, dtype=float) / k**2
    elif qd is not None:
        qdd = np.gradient(qd, t, axis=0)
    else:
        qdd = None
    effort = None
    if trajectory.effort is not None:
        if qd is None or qdd is None:
            raise ValueError("a time-scaled effort column needs the base trajectory's velocities")
        effort = _rigid_body(arm)(q, qd, qdd)
    return JointTrajectory(
        t=t,
        joint_names=trajectory.joint_names,
        positions=q,
        velocities=qd,
        accelerations=None if trajectory.accelerations is None else qdd,
        effort=effort,
    )


@dataclass(frozen=True)
class TimeScaledScript:
    """`base`'s one JointTrajectory with every time × `time_scale` (velocity / k, acceleration / k², the effort
    column refilled by `arm`'s nominal rigid_body model); k = 1 returns the base trajectory itself."""

    base: Any  # an input script returning one JointTrajectory at t = 0 (a frozen dataclass)
    time_scale: float
    arm: str

    def __post_init__(self) -> None:
        if not (math.isfinite(float(self.time_scale)) and float(self.time_scale) > 0.0):
            raise ValueError(f"time_scale must be finite and > 0, got {self.time_scale!r}")
        object.__setattr__(self, "time_scale", float(self.time_scale))

    def __call__(self, start_q: np.ndarray) -> list[tuple[float, Any]]:
        start = tuple(float(v) for v in np.asarray(start_q, dtype=float))
        return [(0.0, _time_scaled(self.base, self.time_scale, self.arm, start))]

    @property
    def duration_s(self) -> float:
        return float(self.base.duration_s) * self.time_scale


@functools.lru_cache(maxsize=16)
def _sweep(script: SineSweepScript, start_q: tuple[float, ...]) -> JointTrajectory:
    from robot_controllers.bench.arms import arm as registered
    from robot_controllers.bench.rungs import min_jerk

    spec = registered(script.arm)
    joints = len(spec.joint_names)
    centre = np.asarray(start_q, dtype=float)
    amplitude = np.full(joints, math.radians(script.amplitude_deg))
    total = script.duration_s
    count = int(round(total / SAMPLE_PERIOD_S))
    t = np.arange(count + 1) * SAMPLE_PERIOD_S
    t[-1] = total
    local = np.clip(t - script.hold_s, 0.0, script.sweep_s)
    inside = (t > script.hold_s - 1e-12) & (t < script.hold_s + script.sweep_s - 1e-12)
    # the envelope: a minimum-jerk ramp up over ramp_s, 1, and the mirror ramp down
    ramp = script.ramp_s
    up, up_d, up_dd = min_jerk(local / ramp)
    down, down_d, down_dd = min_jerk((script.sweep_s - local) / ramp)
    envelope = up * down
    envelope_d = (up_d / ramp) * down - up * (down_d / ramp)
    envelope_dd = (up_dd / ramp**2) * down - 2.0 * (up_d / ramp) * (down_d / ramp) + up * (down_dd / ramp**2)
    # the chirp: phase(t) = 2π (f0 t + (f1 - f0) t² / (2 T)), each joint a quarter turn apart
    f0, f1, span = script.start_hz, script.end_hz, script.sweep_s
    phase = 2.0 * math.pi * (f0 * local + (f1 - f0) * local**2 / (2.0 * span))
    rate = 2.0 * math.pi * (f0 + (f1 - f0) * local / span)
    rate_d = 2.0 * math.pi * (f1 - f0) / span
    offsets = 0.5 * math.pi * np.arange(joints)
    angle = phase[:, None] + offsets[None, :]
    wave, wave_d = np.sin(angle), np.cos(angle) * rate[:, None]
    wave_dd = -np.sin(angle) * rate[:, None] ** 2 + np.cos(angle) * rate_d
    e, e_d, e_dd = (np.where(inside, v, 0.0)[:, None] for v in (envelope, envelope_d, envelope_dd))
    q = centre + amplitude * e * wave
    qd = amplitude * (e_d * wave + e * wave_d)
    qdd = amplitude * (e_dd * wave + 2.0 * e_d * wave_d + e * wave_dd)
    return JointTrajectory(
        t=t,
        joint_names=spec.joint_names,
        positions=q,
        velocities=qd,
        accelerations=qdd,
        effort=spec.feedforward("rigid_body", None)(q, qd, qdd),
    )


@dataclass(frozen=True)
class SineSweepScript:
    """One JointTrajectory at t = 0: hold, then every joint swept around the start at `amplitude_deg` under a
    minimum-jerk envelope (`ramp_s` up and down) with a chirp from `start_hz` to `end_hz` over `sweep_s`, joint j's
    phase j·90° on; then hold. Effort = the arm's nominal rigid_body model."""

    arm: str
    amplitude_deg: float = 8.0
    start_hz: float = 0.4
    end_hz: float = 1.6
    sweep_s: float = 3.0
    ramp_s: float = 0.5
    hold_s: float = 0.3

    def __call__(self, start_q: np.ndarray) -> list[tuple[float, Any]]:
        return [(0.0, _sweep(self, tuple(float(v) for v in np.asarray(start_q, dtype=float))))]

    @property
    def duration_s(self) -> float:
        return 2.0 * self.hold_s + self.sweep_s


# ── the collision stop ─────────────────────────────────────────────────────────────────────────────────────


def collision_stop(trajectory: JointTrajectory, start_q: Sequence[float] | None = None) -> tuple[int, float, str]:
    """The collision target's stop (§5.4): (joint index, angle rad, side) on the joint that moves most (largest
    peak to peak), at the middle of its travel, on the side away from where it starts (`start_q`, default the
    trajectory's first sample). A joint starting at the middle gets the stop on the side it first moves toward."""
    q = np.asarray(trajectory.positions, dtype=float)
    joint = int(np.argmax(np.ptp(q, axis=0)))
    column = q[:, joint]
    middle = 0.5 * (float(column.min()) + float(column.max()))
    start = float(column[0] if start_q is None else start_q[joint])
    if abs(start - middle) > 1e-9:
        return joint, middle, "upper" if start < middle else "lower"
    moved = column[np.abs(column - start) > 1e-9]
    return joint, middle, "upper" if moved.size and moved[0] > start else "lower"


# ── building a target's rung ───────────────────────────────────────────────────────────────────────────────


def _peak_speeds(arm: str, trajectory: JointTrajectory) -> dict[str, float]:
    from robot_controllers.bench.arms import arm as registered

    times = np.asarray(trajectory.t, dtype=float)
    joint_speed = float(np.max(np.abs(np.asarray(trajectory.velocities))))
    tool = np.asarray(registered(arm).tool_position(np.asarray(trajectory.positions)))
    tool_speed = float(np.max(np.linalg.norm(np.gradient(tool, times, axis=0), axis=1)))
    return {"peak_joint_speed_deg_s": math.degrees(joint_speed), "peak_tool_speed_m_s": tool_speed}


@dataclass(frozen=True)
class TargetRung:
    """What one target's Rung is made of; calling it builds the Rung, with the reference's peak joint and tool
    speeds and (`collision=True`) the stop measured on the script itself."""

    target: str  # the target id, also the rung id
    arm: str
    kind: TargetKind
    speed: str | None
    title: str
    description: str
    script: Any  # an input script with `duration_s`
    gates: tuple[Gate, ...]
    timing: HostTiming = field(default_factory=HostTiming)
    family: str = "free_space"
    order: int = 0
    collision: bool = False
    params: tuple[tuple[str, Any], ...] = ()

    def __call__(self) -> Rung:
        from robot_controllers.bench.arms import arm as registered
        from robot_controllers.registry import JOINT_TRAJECTORY

        home = np.asarray(registered(self.arm).home_q, dtype=float)
        trajectory = self.script(home)[0][1]
        params: dict[str, Any] = {"target": self.target, "kind": self.kind}
        if self.speed is not None:
            params["speed"] = self.speed
        params.update(dict(self.params))
        params.update(_peak_speeds(self.arm, trajectory))
        scene = Scene()
        gates = self.gates
        if self.collision:
            joint, angle, side = collision_stop(trajectory, home)
            scene = Scene(stops=((joint, angle, side),))
            gates = tuple(g for g in gates if g.metric not in TRACKING_METRICS)
            params.update(
                {
                    "stop_joint": registered(self.arm).joint_names[joint],
                    "stop_joint_index": joint,
                    "stop_angle_rad": angle,
                    "stop_side": side,
                    "free_target": f"{self.arm}/path@nominal",
                }
            )
        return Rung(
            id=self.target,
            title=self.title,
            box=JOINT_TRAJECTORY,
            family=self.family,
            description=self.description,
            duration_s=round(float(self.script.duration_s) + TAIL_S, 6),
            inputs=self.script,
            scene=scene,
            timing=self.timing,
            gates=gates,
            objective="tracking_p95_deg",
            params=params,
            order=self.order,
            plant=self.arm,
        )


def arm_targets(
    arm: str,
    path: Any,
    *,
    gates: tuple[Gate, ...],
    timing: HostTiming | None = None,
    base_rung: str = "",
    path_text: str = "the arm's frozen path",
    time_scales: Mapping[str, float] = SPEED_TIME_SCALES,
    calibration: Any = None,
) -> tuple[Target, ...]:
    """The five fixed targets of `arm` from one path script (one JointTrajectory at t = 0 with `duration_s`): the
    path time-scaled to each speed, the calibration sweep (`calibration`, default `SineSweepScript(arm)`) and the
    collision (the nominal path with `collision_stop`). `gates` and `timing` are the base rung's."""
    timing = timing or HostTiming()
    common = {"arm": arm, "gates": tuple(gates), "timing": timing}
    base = (("base_rung", base_rung),) if base_rung else ()
    out = []
    for order, speed in enumerate(SPEEDS, start=1):
        scale = float(time_scales[speed])
        script = TimeScaledScript(path, scale, arm)
        target_id = f"{arm}/path@{speed}"
        builder = TargetRung(
            target=target_id,
            kind="path",
            speed=speed,
            title=f"{arm} path, {speed}",
            description=f"{path_text[0].upper()}{path_text[1:]}, every time x{scale:g} ({speed}). Tests tracking "
            "on a fixed target at this speed.",
            script=script,
            order=order,
            params=(*base, ("time_scale", scale)),
            **common,
        )
        out.append(Target(target_id, arm, "path", speed, builder))
    sweep = calibration or SineSweepScript(arm)
    calibration_id = f"{arm}/calibration"
    out.append(
        Target(
            calibration_id,
            arm,
            "calibration",
            None,
            TargetRung(
                target=calibration_id,
                kind="calibration",
                speed=None,
                title=f"{arm} calibration sweep",
                description=f"Every joint swept {sweep.amplitude_deg:g} deg around home with a {sweep.start_hz:g}-"
                f"{sweep.end_hz:g} Hz chirp. Gives each joint's bus delay in one run.",
                script=sweep,
                order=4,
                params=(
                    *base,
                    ("amplitude_deg", sweep.amplitude_deg),
                    ("chirp_hz", f"{sweep.start_hz:g}-{sweep.end_hz:g}"),
                    ("sweep_s", sweep.sweep_s),
                ),
                **common,
            ),
        )
    )
    collision_id = f"{arm}/collision"
    nominal_scale = float(time_scales["nominal"])
    out.append(
        Target(
            collision_id,
            arm,
            "collision",
            "nominal",
            TargetRung(
                target=collision_id,
                kind="collision",
                speed="nominal",
                title=f"{arm} collision with a hard stop",
                description=f"{path_text[0].upper()}{path_text[1:]} at nominal speed with a hard stop half way "
                "along its most-moving joint's travel. Tests how soon and how hard the stop is felt.",
                script=TimeScaledScript(path, nominal_scale, arm),
                family="collision",
                order=5,
                collision=True,
                params=(*base, ("time_scale", nominal_scale)),
                **common,
            ),
        )
    )
    return tuple(out)


# ── the planar arm's and the UR5e's ────────────────────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=1)
def planar_targets() -> tuple[Target, ...]:
    """R2's path (home -> out -> up-in -> home, 0.3 s per move) at three speeds, the sweep and the collision."""
    from robot_controllers.bench.rungs import rung

    base = rung("R2.FastFreeSpace")
    return arm_targets(
        "planar_arm",
        base.inputs,
        gates=base.gates,
        timing=base.timing,
        base_rung=base.id,
        path_text="R2's tool path (home -> out -> up-in -> home, 0.3 s moves and rests)",
    )


@functools.lru_cache(maxsize=1)
def ur5e_targets() -> tuple[Target, ...]:
    """U2's joint path (W0 -> W1 -> W2 -> W0, 0.45 s per move) at three speeds, the sweep and the collision."""
    from robot_controllers.bench.ur5e import ur5e_rungs

    base = {r.id: r for r in ur5e_rungs()}["U2.FastFreeSpace"]
    return arm_targets(
        "ur5e",
        base.inputs,
        gates=base.gates,
        timing=base.timing,
        base_rung=base.id,
        path_text="U2's joint path (W0 -> W1 -> W2 -> W0, 0.45 s moves, 0.3 s rests)",
    )
