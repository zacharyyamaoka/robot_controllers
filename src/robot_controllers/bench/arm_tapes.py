"""Fixed targets for the vendored arms (CONTRACT amendment 5, §5.4 / §5.8): arm_motion_bench's frozen tapes
resampled onto the bench's grid, and the arm's five targets built from one of them by `bench.targets`.

    from robot_controllers.bench.arm_tapes import TapeScript, load_tape, resample, tape_targets
    tape = load_tape("bam_fb/workspace-dev-000__s0.5.npz")          # the 100 Hz JointTrajectory, as frozen
    t, q, qd, qdd = resample(tape, 0.004)                           # every 4 ms, the end included
    targets = tape_targets("bam_fivebar", TapeScript("bam_fb/workspace-dev-000__s0.5.npz", "bam_fivebar"),
    ...)

- **The tapes** live in `models/tapes/<robot>/`, copied byte for byte from arm_motion_bench's
  `outputs/datasets/<robot>/workspace/v2/tapes/` (`models/tapes/tapes.json` has each one's source path and
  sha256, which equals its dataset manifest's entry; `load_tape` refuses any other bytes).
- **Resampling** is quintic Hermite through each 10 ms interval's end samples (position, velocity and
  acceleration). WHY quintic Hermite: the tapes are bam_ruckig profiles, piecewise constant jerk, so position
  is a cubic inside every interval that holds no jerk switch, and the quintic through a cubic's end values is
  that cubic. The bench's 4 ms samples then lie on the curve the real arm was sent (it plays the same tape),
  so a real log of the tape compares one to one; linear interpolation would cut the corners by up to a·h²/8.
- **The targets** are `bench.targets.arm_targets` on the tape (the path time-scaled ×2.0 / ×1.0 / ×0.9, lane
  A's sine sweep around home, the collision stop), with one change: a path or collision run starts where the
  tape starts (`Scene.start_q`), not at the arm's home, and the collision stop's side is taken from that
  start. WHY: arm_motion_bench's tapes start where its tour put the arm (the five-bar's station home, the
  pendulum's task 006 end), and a jump from home to the tape's first sample would be a step no real run has.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

TAPES_DIR = Path(__file__).resolve().parent / "models" / "tapes"
SAMPLE_PERIOD_S = 0.004  # every bench trajectory is sampled every 4 ms (as the planar and UR5e rungs)
TIME_TOLERANCE_S = 1e-9


@functools.lru_cache(maxsize=1)
def _records() -> dict[str, dict[str, Any]]:
    data = json.loads((TAPES_DIR / "tapes.json").read_text())
    return {record["file"]: record for record in data["tapes"]}


def tape_record(name: str) -> dict[str, Any]:
    """The provenance record of vendored tape `name` (`<robot>/<file>.npz`): source, sha256, speed."""
    return dict(_records()[f"tapes/{name}"])


@functools.lru_cache(maxsize=16)
def load_tape(name: str) -> Any:
    """The vendored tape `name` (`<robot>/<file>.npz`) as robot_msgs' JointTrajectory; refused unless its
    bytes are the ones `tapes.json` (= arm_motion_bench's manifest) lists."""
    from robot_msgs._core import load_npz

    path = TAPES_DIR / name
    record = tape_record(name)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise ValueError(
            f"{path}: sha256 {digest[:12]} is not the vendored {record['sha256'][:12]} (an edited tape)"
        )
    return load_npz(path)


def _hermite5(s: np.ndarray, h: float, p0, v0, a0, p1, v1, a1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quintic Hermite on one interval of length h at s = (t - t0) / h in [0, 1], per joint: (p, v, a)."""
    s = s[:, None]
    s2, s3, s4, s5 = s * s, s**3, s**4, s**5
    basis = (
        1 - 10 * s3 + 15 * s4 - 6 * s5,
        h * (s - 6 * s3 + 8 * s4 - 3 * s5),
        h * h * (0.5 * s2 - 1.5 * s3 + 1.5 * s4 - 0.5 * s5),
        10 * s3 - 15 * s4 + 6 * s5,
        h * (-4 * s3 + 7 * s4 - 3 * s5),
        h * h * (0.5 * s3 - s4 + 0.5 * s5),
    )
    d1 = (
        (-30 * s2 + 60 * s3 - 30 * s4) / h,
        1 - 18 * s2 + 32 * s3 - 15 * s4,
        h * (s - 4.5 * s2 + 6 * s3 - 2.5 * s4),
        (30 * s2 - 60 * s3 + 30 * s4) / h,
        -12 * s2 + 28 * s3 - 15 * s4,
        h * (1.5 * s2 - 4 * s3 + 2.5 * s4),
    )
    d2 = (
        (-60 * s + 180 * s2 - 120 * s3) / (h * h),
        (-36 * s + 96 * s2 - 60 * s3) / h,
        1 - 9 * s + 18 * s2 - 10 * s3,
        (60 * s - 180 * s2 + 120 * s3) / (h * h),
        (-24 * s + 84 * s2 - 60 * s3) / h,
        3 * s - 12 * s2 + 10 * s3,
    )
    ends = (p0, v0, a0, p1, v1, a1)
    return tuple(sum(b * e for b, e in zip(column, ends, strict=True)) for column in (basis, d1, d2))  # type: ignore[return-value]


def resample(
    tape: Any, period_s: float = SAMPLE_PERIOD_S
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(t, q, qd, qdd) of `tape` every `period_s` from its first sample to its last (the end included),
    quintic Hermite through the tape's own (position, velocity, acceleration) samples; the knots are
    reproduced exactly."""
    knots = np.asarray(tape.t, dtype=float)
    positions = np.asarray(tape.positions, dtype=float)
    velocities = np.asarray(tape.velocities, dtype=float)
    accelerations = np.asarray(tape.accelerations, dtype=float)
    start, end = float(knots[0]), float(knots[-1])
    count = int(math.floor((end - start) / period_s + TIME_TOLERANCE_S))
    t = start + np.arange(count + 1) * period_s
    if end - t[-1] > TIME_TOLERANCE_S:
        t = np.append(t, end)
    t[-1] = end
    index = np.clip(np.searchsorted(knots, t, side="right") - 1, 0, len(knots) - 2)
    q = np.empty((len(t), positions.shape[1]))
    qd, qdd = np.empty_like(q), np.empty_like(q)
    for interval in np.unique(index):
        rows = index == interval
        h = float(knots[interval + 1] - knots[interval])
        s = (t[rows] - knots[interval]) / h
        p, v, a = _hermite5(
            s,
            h,
            positions[interval],
            velocities[interval],
            accelerations[interval],
            positions[interval + 1],
            velocities[interval + 1],
            accelerations[interval + 1],
        )
        q[rows], qd[rows], qdd[rows] = p, v, a
    return t - start, q, qd, qdd


def tape_trajectory(name: str, effort: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> Any:
    """The vendored tape `name` on the bench's 4 ms grid as a JointTrajectory starting at t = 0, effort =
    `effort` (the arm's nominal model) of the resampled reference."""
    from robot_msgs.trajectory import JointTrajectory

    tape = load_tape(name)
    t, q, qd, qdd = resample(tape)
    return JointTrajectory(
        t=t,
        joint_names=tuple(tape.joint_names),
        positions=q,
        velocities=qd,
        accelerations=qdd,
        effort=effort(q, qd, qdd),
    )


@functools.lru_cache(maxsize=16)
def _tape_trajectory(name: str, arm: str) -> Any:
    from robot_controllers.bench.arms import arm as registered

    return tape_trajectory(name, registered(arm).feedforward("rigid_body", None))


@dataclass(frozen=True)
class TapeScript:
    """One JointTrajectory at t = 0: vendored tape `tape` on the 4 ms grid, effort = `arm`'s nominal
    `rigid_body` model. It starts where the tape does, whatever start it is handed. Frozen and compared by
    value, so a rung pickles into a worker process and `bench.targets` caches its time-scaled copies."""

    tape: str
    arm: str

    def __call__(self, start_q: np.ndarray) -> list[tuple[float, Any]]:
        return [(0.0, _tape_trajectory(self.tape, self.arm))]

    @property
    def duration_s(self) -> float:
        times = np.asarray(load_tape(self.tape).t, dtype=float)
        return float(times[-1] - times[0])

    @property
    def start_q(self) -> tuple[float, ...]:
        return tuple(float(v) for v in np.asarray(load_tape(self.tape).positions, dtype=float)[0])


@dataclass(frozen=True)
class FromTapeStart:
    """A target's rung builder (`bench.targets.TargetRung`) whose plant starts at the path's first sample, the
    collision stop's side taken from there (`collision_stop` with that start)."""

    inner: Any

    def __call__(self) -> Any:
        import dataclasses

        from robot_controllers.bench.arms import arm as registered
        from robot_controllers.bench.targets import collision_stop

        rung = self.inner()
        trajectory = rung.inputs(np.asarray(registered(rung.plant).home_q, dtype=float))[0][1]
        start = tuple(float(v) for v in np.asarray(trajectory.positions, dtype=float)[0])
        scene = dataclasses.replace(rung.scene, start_q=start)
        params = {**rung.params, "start_q_rad": "(" + ", ".join(f"{v:.6g}" for v in start) + ")"}
        if rung.scene.stops:
            joint, angle, side = collision_stop(trajectory, start)
            scene = dataclasses.replace(scene, stops=((joint, angle, side),))
            params.update(
                {
                    "stop_joint": registered(rung.plant).joint_names[joint],
                    "stop_joint_index": joint,
                    "stop_angle_rad": angle,
                    "stop_side": side,
                }
            )
        return dataclasses.replace(rung, scene=scene, params=params)


def tape_targets(arm: str, path: TapeScript, *, gates: tuple, timing: Any = None, path_text: str) -> tuple:
    """`arm`'s five fixed targets (`bench.targets.arm_targets`) on tape script `path`, the path and collision
    runs starting where the tape does."""
    import dataclasses

    from robot_controllers.bench.targets import arm_targets

    out = []
    for target in arm_targets(arm, path, gates=gates, timing=timing, path_text=path_text):
        if target.kind in ("path", "collision"):
            target = dataclasses.replace(target, build=FromTapeStart(target.build))
        out.append(target)
    return tuple(out)
