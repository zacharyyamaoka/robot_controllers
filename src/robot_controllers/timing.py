"""Bus timing: the datasheet's nominal, a calibration's measurement, and the per-joint delays a controller looks ahead by
(CONTRACT amendment 5, §5.2).

Zach, 2026-11-20: "our representation of the CANFD should have a data sheet nominal value for the delay and stuff. and
then we can compare that against reality and of course if that begins to change a lot we should be alerted."

The nominal comes from robotparts (the adapter's datasheet exchange timing and the driver's `command_latency`), passed in
as plain numbers so this package never imports robotparts (leaves don't import each other). WHY per joint: one adapter
serves its boards in turn (BAM's fdcanusb: seven boards in ~15.5 ms), so each joint's command lands at a different time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _seconds(value: float, name: str) -> float:
    number = float(value)
    if not (math.isfinite(number) and number >= 0.0):
        raise ValueError(f"{name} must be finite and >= 0 s, got {value!r}")
    return number


@dataclass(frozen=True)
class BusTiming:
    """How late a command reaches each board, and a reply reaches the host. SI seconds.

    `command_delay_s` is one value for every joint or one per joint (the trajectory's joint order). `source` says where the
    numbers came from: "datasheet:robotparts:<id>@<release>", "calibrated:<log or rung id>", "twin" or "assumed".
    """

    command_delay_s: float | tuple[float, ...] = 0.0055
    reply_delay_s: float = 0.0015
    jitter_s: float = 0.0
    source: str = "twin"

    def __post_init__(self) -> None:
        if isinstance(self.command_delay_s, (list, tuple)):
            object.__setattr__(
                self, "command_delay_s", tuple(_seconds(v, "command_delay_s") for v in self.command_delay_s)
            )
        else:
            object.__setattr__(self, "command_delay_s", _seconds(self.command_delay_s, "command_delay_s"))
        object.__setattr__(self, "reply_delay_s", _seconds(self.reply_delay_s, "reply_delay_s"))
        object.__setattr__(self, "jitter_s", _seconds(self.jitter_s, "jitter_s"))

    def per_joint(self, count: int) -> tuple[float, ...]:
        """One command delay per joint."""
        if isinstance(self.command_delay_s, tuple):
            if len(self.command_delay_s) != count:
                raise ValueError(f"{len(self.command_delay_s)} command delays for {count} joints")
            return self.command_delay_s
        return (self.command_delay_s,) * count

    @classmethod
    def from_datasheet(
        cls,
        exchange_time_s: float,
        devices: int,
        *,
        slot_order: Sequence[int] | None = None,
        command_latency_s: float = 0.0,
        reply_delay_s: float = 0.0015,
        source: str = "datasheet",
    ) -> BusTiming:
        """The adapter serves `devices` boards in turn in `exchange_time_s`; joint j's board is served in slot
        `slot_order[j]` (0 = first; default: the joint order). Its command lands `command_latency_s + exchange_time_s ·
        (slot + 1) / devices` after the host starts the exchange."""
        exchange = _seconds(exchange_time_s, "exchange_time_s")
        latency = _seconds(command_latency_s, "command_latency_s")
        if devices < 1:
            raise ValueError(f"devices must be >= 1, got {devices!r}")
        slots = tuple(range(devices)) if slot_order is None else tuple(int(s) for s in slot_order)
        if any(not 0 <= s < devices for s in slots):
            raise ValueError(f"slot_order {slots} outside 0..{devices - 1}")
        return cls(
            command_delay_s=tuple(latency + exchange * (s + 1) / devices for s in slots),
            reply_delay_s=reply_delay_s,
            source=source,
        )

    @classmethod
    def from_calibration(cls, lags_s: Sequence[float], *, reply_delay_s: float = 0.0015, source: str) -> BusTiming:
        return cls(command_delay_s=tuple(max(0.0, float(v)) for v in lags_s), reply_delay_s=reply_delay_s, source=source)
