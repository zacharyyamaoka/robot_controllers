"""robot_controllers — robot controllers as black boxes: a robot_msgs input in, ServoCommand out.

Import a controller through the registry, or its module directly::

    from robot_controllers import controller
    jtc = controller("joint_trajectory_controller").build(trajectory)
    command, defines = jtc.tick(joint_state, dt)
"""

from robot_controllers.core import (
    HUB_PROJECT,
    WANDB_ENTITY,
    Box,
    Controller,
    ControllerSpec,
    Replannable,
    SelfReport,
)
from robot_controllers.registry import BOXES, CONTROLLERS, box_by_slug, controller, in_box

__version__ = "0.1.0"

__all__ = [
    "BOXES",
    "CONTROLLERS",
    "HUB_PROJECT",
    "WANDB_ENTITY",
    "Box",
    "Controller",
    "ControllerSpec",
    "Replannable",
    "SelfReport",
    "box_by_slug",
    "controller",
    "in_box",
]
