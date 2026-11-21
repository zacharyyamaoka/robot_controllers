"""The arm registry: one record per arm, so the grader, the calibration and `recommend` are arm-agnostic
(CONTRACT amendment 5, §5.1).

Each arm's own module builds its `ArmSpec` and calls `register`; `arm(name)` imports the arm modules on first use.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_controllers.bench.board import BoardConfig
from robot_controllers.models import FeedforwardModel, ModelError

# The modules that register an arm, in display order. WHY a list, not discovery: the order is the page's order, and
# a missing module is an error, never a silently shorter ledger.
ARM_MODULES: tuple[tuple[str, str], ...] = (
    ("planar_arm", "robot_controllers.bench.arm_planar"),
    ("ur5e", "robot_controllers.bench.ur5e"),
)


@dataclass(frozen=True)
class ArmSpec:
    name: str  # "planar_arm" | "ur5e" | "bam_fivebar" | "pendulum"
    title: str  # "Planar twin (2 joints)"
    joint_names: tuple[str, ...]
    home_q: tuple[float, ...]
    plant: Callable[..., Any]  # (board_config: BoardConfig | None = None) -> a plant with PlanarArmPlant's interface
    board_config: Callable[[], BoardConfig]  # the arm's nominal boards (gains, caps, encoder)
    feedforward: Callable[[str, ModelError | None], FeedforwardModel]  # "gravity" | "rigid_body" | "rigid_body_coulomb"
    mass_matrix: Callable[[ModelError | None], Callable[[np.ndarray], np.ndarray]]
    tool_position: Callable[[np.ndarray], np.ndarray]
    targets: Callable[[], tuple[Any, ...]]  # the fixed target trajectories (§5.4), `bench.targets.Target`
    real_station: str | None = None  # arm_motion_bench station id; None = sim only


_REGISTRY: dict[str, ArmSpec] = {}
_LOADED = False


def register(spec: ArmSpec) -> ArmSpec:
    if spec.name in _REGISTRY and _REGISTRY[spec.name] is not spec:
        raise ValueError(f"arm {spec.name!r} registered twice")
    _REGISTRY[spec.name] = spec
    return spec


def _load() -> None:
    global _LOADED
    if not _LOADED:
        for _name, module in ARM_MODULES:
            importlib.import_module(module)
        _LOADED = True


def arms() -> tuple[ArmSpec, ...]:
    _load()
    order = {name: i for i, (name, _module) in enumerate(ARM_MODULES)}
    return tuple(sorted(_REGISTRY.values(), key=lambda s: order.get(s.name, len(order))))


def arm(name: str) -> ArmSpec:
    _load()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"no arm {name!r}; registered: {sorted(_REGISTRY)}") from None
