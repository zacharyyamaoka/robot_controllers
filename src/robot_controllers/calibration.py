"""The bus-delay calibration routine: measure how late commands reach each board, and compare it with the datasheet
(CONTRACT amendment 5, §5.2).

    from robot_controllers.calibration import calibrate_bus
    result = calibrate_bus(lambda la: layout_spec(..., lookahead_s=la), rung, nominal=datasheet_bus)
    result.measured.command_delay_s     # (2.0e-3, 8.0e-3): one per joint
    result.delta_s, result.within       # measured - nominal per joint; every |delta| <= tolerance_s
    print(result.why)                   # one plain line

Zach, 2026-11-20: "this controller then should register a routine to calibrate the delay ... our representation of
the CANFD should have a data sheet nominal value for the delay and stuff. and then we can compare that against
reality".

Two methods:

- `xcorr`: one run at zero lookahead; each joint's delay is the shift that best lines its measured position up with
  its reference (`metrics.lag.estimate_lag_s(per_joint=True)`, clipped at >= 0, a joint that did not move given the
  median of the others). It needs the log's send times to be the real ones: the twin's are; the real arm's logger
  stamps the exchange's end today, which bounds the delay only to one exchange (§5.9).
- `sweep`: one run per scalar lookahead (default: the nominal's mean + {-3, -1.5, 0, +1.5, +3} ms, clipped at 0);
  the measured delay is the lookahead with the lowest `tracking_p95_deg`, the same for every joint. It works on any
  log the bench can score, real ones included.

WHY a routine and not only live (§5.2): a clean run gives the delay at once; the monitor (`monitor.LagMonitor`) then
only has to notice a change. Neither method measures the reply delay or the jitter: the measured `BusTiming`
carries the nominal's.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from robot_controllers.timing import BusTiming

METHODS: tuple[str, ...] = ("xcorr", "sweep")
SWEEP_OFFSETS_S: tuple[float, ...] = (-0.003, -0.0015, 0.0, 0.0015, 0.003)  # around the nominal's mean (§5.2)


@dataclass(frozen=True)
class Calibration:
    """A measured bus next to the nominal it was checked against.

    `delta_s` is measured - nominal per joint; `within` = every |delta| <= the tolerance. `lags_s` is what the run(s)
    measured before filling and clipping (xcorr: each joint's lag, None = it did not move); `curve` is the sweep's
    (lookahead_s, tracking_p95_deg) per run, in lookahead order (empty for xcorr).
    """

    measured: BusTiming
    nominal: BusTiming
    delta_s: tuple[float, ...]
    within: bool
    method: Literal["xcorr", "sweep"]
    why: str
    tolerance_s: float = 0.002
    rung: str = ""
    lags_s: tuple[float | None, ...] = ()
    curve: tuple[tuple[float, float | None], ...] = ()


def _ms(values: Sequence[float]) -> str:
    return "/".join(f"{1e3 * float(v):.2f}" for v in values) + " ms"


def _signed_ms(values: Sequence[float]) -> str:
    return "/".join(f"{1e3 * float(v):+.2f}" for v in values) + " ms"


def _episode(spec: Any, rung: Any, seed: int, plant_factory: Callable[[], Any] | None) -> Any:
    from robot_controllers.bench.loop import run_episode

    return run_episode(spec, rung, seed=seed, plant_factory=plant_factory)


def _sweep_lookaheads(nominal: BusTiming, lookaheads_s: Sequence[float] | None) -> tuple[float, ...]:
    if lookaheads_s is not None:
        values = [float(v) for v in lookaheads_s]
    else:
        delay = nominal.command_delay_s
        mean = statistics.fmean(delay) if isinstance(delay, tuple) else float(delay)
        values = [mean + offset for offset in SWEEP_OFFSETS_S]
    out = sorted({round(max(0.0, v), 9) for v in values})  # WHY rounded: 5.5 ms - 1.5 ms twice is one lookahead
    if not out or not all(math.isfinite(v) for v in out):
        raise ValueError(f"lookaheads_s must be finite seconds, got {lookaheads_s!r}")
    return tuple(out)


def calibrate_bus(
    spec_at: Callable[[float | tuple[float, ...]], Any],
    rung: Any,
    *,
    nominal: BusTiming,
    method: str = "xcorr",
    lookaheads_s: Sequence[float] | None = None,
    tolerance_s: float = 0.002,
    seed: int = 0,
    plant_factory: Callable[[], Any] | None = None,
) -> Calibration:
    """Measure the bus `rung` runs on with `spec_at(lookahead)` (a callable lookahead -> ControllerSpec), and
    compare it with `nominal` (the module docstring has the two methods). `plant_factory` is passed to
    `run_episode` (a scenario with other board gains, §5.3); the default is the rung's own plant."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    tolerance = float(tolerance_s)
    if not (math.isfinite(tolerance) and tolerance >= 0.0):
        raise ValueError(f"tolerance_s must be finite and >= 0, got {tolerance_s!r}")
    rung_id = str(getattr(rung, "id", rung))
    if method == "xcorr":
        from robot_controllers.metrics.lag import STILL_DEG, estimate_lag_s

        log = _episode(spec_at(0.0), rung, seed, plant_factory)
        lags = tuple(estimate_lag_s(log, per_joint=True))
        moved = [max(0.0, float(v)) for v in lags if v is not None]
        if not moved:
            raise ValueError(f"calibrate_bus: no joint moved {STILL_DEG:g} deg on {rung_id}")
        median = float(statistics.median(moved))
        delays = tuple(median if v is None else max(0.0, float(v)) for v in lags)
        count = len(delays)
        measured = BusTiming(
            command_delay_s=delays,
            reply_delay_s=nominal.reply_delay_s,
            jitter_s=nominal.jitter_s,
            source=f"calibrated:{rung_id}:xcorr",
        )
        curve: tuple[tuple[float, float | None], ...] = ()
        how = "one run at zero lookahead, each joint's best-aligning shift"
        still = [j for j, v in enumerate(lags) if v is None]
        if still:
            how += f" (joint {', '.join(str(j) for j in still)} did not move: given the others' median)"
    else:
        from robot_controllers.bench.loop import run_episode
        from robot_controllers.metrics.episode import episode_metrics

        points: list[tuple[float, float | None]] = []
        count = 0
        for lookahead in _sweep_lookaheads(nominal, lookaheads_s):
            log = run_episode(spec_at(lookahead), rung, seed=seed, plant_factory=plant_factory)
            count = len(log.joint_names)
            value = episode_metrics(log, rung).get("tracking_p95_deg")
            points.append((lookahead, None if value is None else float(value)))
        scored = [(v, la) for la, v in points if v is not None and math.isfinite(v)]
        if not scored:
            raise ValueError(f"calibrate_bus: no sweep run on {rung_id} measured tracking_p95_deg")
        # WHY ties go to the smaller lookahead: sorted() order; a smaller lookahead risks less when the bus speeds up
        best_value, best = min(scored)
        delays = (best,) * count
        lags = ()
        curve = tuple(points)
        measured = BusTiming(
            command_delay_s=best,
            reply_delay_s=nominal.reply_delay_s,
            jitter_s=nominal.jitter_s,
            source=f"calibrated:{rung_id}:sweep",
        )
        how = (
            f"{len(points)} runs at lookahead {_ms([la for la, _ in points])}, the lowest tracking p95 "
            f"({best_value:.3g} deg)"
        )
    expected = nominal.per_joint(count)
    delta = tuple(float(m) - float(e) for m, e in zip(delays, expected, strict=True))
    within = all(abs(d) <= tolerance + 1e-12 for d in delta)
    verdict = (
        f"within the {1e3 * tolerance:g} ms tolerance"
        if within
        else f"off by {_signed_ms(delta)}, beyond the {1e3 * tolerance:g} ms tolerance"
    )
    why = (
        f"{method} on {rung_id} ({how}): measured {_ms(delays)} vs nominal {_ms(expected)} "
        f"({nominal.source}): {verdict}"
    )
    return Calibration(
        measured=measured,
        nominal=nominal,
        delta_s=delta,
        within=within,
        method=method,  # type: ignore[arg-type]
        why=why,
        tolerance_s=tolerance,
        rung=rung_id,
        lags_s=tuple(None if v is None else float(v) for v in lags),
        curve=curve,
    )

