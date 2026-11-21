"""Amendment 5, §5.4 / §5.8: the vendored tapes and their resampling onto the bench's grid, and the
calibration sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from robot_controllers.bench.arm_tapes import (
    TAPES_DIR,
    _hermite5,
    load_tape,
    resample,
    tape_record,
)

ARM_MOTION_BENCH = Path("/home/bam/arm_motion_bench")
VENDORED = ("bam_fb/workspace-dev-000__s0.5.npz", "pendulum/workspace-dev-007__s0.5.npz")


@pytest.mark.parametrize("name", VENDORED)
def test_each_vendored_tape_is_byte_for_byte_its_sidecar_and_its_manifest(name):
    record = tape_record(name)
    assert hashlib.sha256((TAPES_DIR / name).read_bytes()).hexdigest() == record["sha256"]
    manifest = Path(record["manifest"])
    if not manifest.exists():
        pytest.skip("arm_motion_bench is not on this machine")
    files = json.loads(manifest.read_text())["files"]
    assert files[Path(name).name] == record["sha256"]


def test_quintic_hermite_reproduces_a_cubic_and_its_derivatives():
    rng = np.random.default_rng(3)
    coefficients = rng.normal(size=(4, 2))
    h = 0.01

    def cubic(t):
        p = sum(coefficients[k] * t[:, None] ** k for k in range(4))
        v = sum(k * coefficients[k] * t[:, None] ** (k - 1) for k in range(1, 4))
        a = sum(k * (k - 1) * coefficients[k] * t[:, None] ** (k - 2) for k in range(2, 4))
        return p, v, a

    ends = cubic(np.array([0.0, h]))
    s = np.linspace(0.0, 1.0, 11)
    p, v, a = _hermite5(s, h, ends[0][0], ends[1][0], ends[2][0], ends[0][1], ends[1][1], ends[2][1])
    expected = cubic(s * h)
    for got, want in zip((p, v, a), expected, strict=True):
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-9)


@pytest.mark.parametrize(
    "name", ["bam_fb/workspace-dev-000__s0.5.npz", "pendulum/workspace-dev-007__s0.5.npz"]
)
def test_resampling_lands_on_every_tape_sample_and_stays_smooth(name):
    tape = load_tape(name)
    t, q, qd, qdd = resample(tape, 0.004)
    assert t[0] == 0.0 and t[-1] == pytest.approx(float(tape.t[-1]) - float(tape.t[0]), abs=1e-12)
    np.testing.assert_allclose(np.diff(t), 0.004, atol=1e-9)
    # every 10 ms tape sample that falls on the 4 ms grid (every 20 ms) is reproduced exactly
    knots = np.asarray(tape.t) - float(tape.t[0])
    on_grid = np.flatnonzero(np.abs(np.round(knots / 0.004) * 0.004 - knots) < 1e-9)
    rows = np.round(knots[on_grid] / 0.004).astype(int)
    np.testing.assert_allclose(q[rows], np.asarray(tape.positions)[on_grid], atol=1e-12)
    np.testing.assert_allclose(qd[rows], np.asarray(tape.velocities)[on_grid], atol=1e-12)
    # the velocity is the derivative of the position (no corner cut between knots)
    np.testing.assert_allclose(np.gradient(q, t, axis=0)[1:-1], qd[1:-1], atol=2e-3)
    assert np.all(qd[0] == 0.0) and np.all(qd[-1] == 0.0)
