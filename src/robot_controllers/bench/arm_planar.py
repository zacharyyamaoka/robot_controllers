"""The planar twin in the arm registry (CONTRACT amendment 5, §5.1): `arm.py`'s plant and `arm_models.py`'s models
behind one `ArmSpec`, so the grader, the calibration and `recommend` treat it like any other arm.

    from robot_controllers.bench.arms import arm
    planar = arm("planar_arm")
    ff = planar.feedforward("rigid_body", ModelError(0.1))      # what a controller with a 10 % wrong model is told
    plant = planar.plant()                                      # PlanarArmPlant on its nominal boards

Nothing here changes behaviour: `feedforward(name, None)` and `feedforward(name, ModelError())` are the same objects
`layouts.feedforward_model` built before the registry (`planar_arm_feedforward(terms)`).
"""

from __future__ import annotations

from typing import Any

from robot_controllers.bench.arm import HOME_Q, JOINT_NAMES, tool_position
from robot_controllers.bench.arm_models import (
    RIGID_BODY,
    PlanarArmFeedforward,
    PlanarArmMassMatrix,
    planar_arm_feedforward,
    planar_arm_mass_matrix,
)
from robot_controllers.bench.arms import ArmSpec, register
from robot_controllers.bench.board import BoardConfig
from robot_controllers.bench.plant import PlanarArmPlant
from robot_controllers.models import ModelError

# The registry's feed-forward names (§5.1) as the planar arm's dynamics terms
FEEDFORWARD_TERMS: dict[str, tuple[str, ...]] = {
    "gravity": ("gravity",),
    "rigid_body": RIGID_BODY,
    "rigid_body_coulomb": (*RIGID_BODY, "coulomb"),
}


def planar_feedforward(name: str, model_error: ModelError | None = None) -> PlanarArmFeedforward:
    """The planar feed-forward named `name` (`gravity`, `rigid_body`, `rigid_body_coulomb`) with `model_error`."""
    try:
        terms = FEEDFORWARD_TERMS[name]
    except KeyError:
        raise ValueError(f"unknown planar feed-forward {name!r}; known: {tuple(FEEDFORWARD_TERMS)}") from None
    return planar_arm_feedforward(terms, model_error=model_error)


def planar_mass_matrix(model_error: ModelError | None = None) -> PlanarArmMassMatrix:
    return planar_arm_mass_matrix(model_error=model_error)


def planar_plant(board_config: BoardConfig | None = None) -> PlanarArmPlant:
    """A fresh planar plant (its nominal boards unless `board_config` says otherwise)."""
    return PlanarArmPlant(board_config=board_config)


def planar_targets() -> tuple[Any, ...]:
    """The planar arm's fixed targets (§5.4): R2's path at three speeds, a calibration sweep and a collision."""
    from robot_controllers.bench.targets import planar_targets as targets

    return targets()


PLANAR_ARM = register(
    ArmSpec(
        name="planar_arm",
        title="Planar twin (2 joints)",
        joint_names=JOINT_NAMES,
        home_q=HOME_Q,
        plant=planar_plant,
        board_config=BoardConfig,
        feedforward=planar_feedforward,
        mass_matrix=planar_mass_matrix,
        tool_position=tool_position,
        targets=planar_targets,
    )
)
