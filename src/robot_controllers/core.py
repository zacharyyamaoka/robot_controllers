"""The controller contract: a controller is a black box keyed by its input and output message types.

Zach, 2026-11-12: "controllers are just black boxes as well ... the controllers should be grouped by their like
input and output messages. if they are the same types then they are the same black box!"

- A **Box** is (input type, output type). Every controller in a box can be swapped for every other one, so they
  are compared with each other on the same rungs, and each box is one W&B project.
- A **Controller** is the robot_msgs `JointController` protocol, unchanged: `tick(state, dt) -> (ServoCommand,
  defines)`. It is built from one input message (the static description: a JointTrajectory, a
  JointCommandSequence, ...), holds no clock and talks to no hardware, so the same object runs on a twin, on the
  rack, in the caller's thread or its own.
- The loop around it, never the controller, stamps the command: `header.stamp` = when it is sent,
  `state_stamp` / `state_seq` = the `ServoState` it came from (robot_msgs.control.ServoCommand.latency).
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from robot_msgs.control import ServoCommand
from robot_msgs.sensor import JointState

Status = Literal["running", "done", "aborted"]
RunsOn = Literal["board", "host", "split"]
Maturity = Literal["built", "planned"]
# WHY (amendment 2, Zach 2026-11-15): one black box, several source implementations ("Python today ... now
# C++"); each implementation is its own spec on the same rungs, so the cost of the language is a number.
Implementation = Literal["python", "cpp"]

# W&B project names may not contain these (wandb 0.31 sdk/wandb_settings.py), and must stay <= 128 characters.
_WANDB_FORBIDDEN = re.compile(r"[/\\#?%:]")
# A controller name: snake_case, or (amendment 3, §3.6) a layout sweep point
# `ff=<model>@<at>+fb=<model>@<at>`, whose name is its coordinates in the (feed-forward model, placement,
# feedback model, placement) space, plus `+la=<ms>ms` for a non-zero lookahead (§3.8.2).
_SNAKE_CASE = r"[a-z][a-z0-9_]*"
_LOOKAHEAD = r"(?:\+la=[0-9]+(?:\.[0-9]+)?ms)?"
_LAYOUT_POINT = rf"ff={_SNAKE_CASE}@{_SNAKE_CASE}\+fb={_SNAKE_CASE}@{_SNAKE_CASE}{_LOOKAHEAD}"
_CONTROLLER_NAME = re.compile(rf"{_SNAKE_CASE}|{_LAYOUT_POINT}")


@runtime_checkable
class Controller(Protocol):
    """robot_msgs.control.sequence_controller.JointController, so `JointCommandSequenceController` already is one.

    `tick` gets the latest measured joints and the measured period since the last tick (s), and returns the
    command to send plus any positions to define this tick (homing; empty for most controllers). `status` is
    `running`, `done` (the input finished; it keeps commanding its end) or `aborted` (it stopped on a guard).
    """

    status: Status

    def tick(self, state: JointState, dt: float) -> tuple[ServoCommand, Mapping[str, float]]: ...


@runtime_checkable
class Replannable(Protocol):
    """A controller that accepts a new input of its own box while running (the rapid-replanning rung).

    The new message starts at the controller's current time (ros2_control JTC's replace: the new trajectory's
    t = 0 is "now"). A controller without it refuses replanning, and that rung reports it as unsupported (a
    failed gate), never as skipped.
    """

    def replace_input(self, message: Any) -> None: ...


@runtime_checkable
class SelfReport(Protocol):
    """Optional controller self-report (diagnostics): never used by the outside metrics, shown beside them."""

    def diagnostics(self) -> Mapping[str, float | str]: ...


@dataclass(frozen=True)
class Box:
    """A black box: every controller that takes `input_type` and emits `output_type`."""

    input_type: type
    output_type: type = ServoCommand

    @property
    def input_name(self) -> str:
        return self.input_type.__name__

    @property
    def output_name(self) -> str:
        return self.output_type.__name__

    @property
    def name(self) -> str:
        """`JointTrajectory → ServoCommand`, as the site's sidebar heads the group."""
        return f"{self.input_name} → {self.output_name}"

    @property
    def slug(self) -> str:
        """`jointtrajectory-to-servocommand`: the site URL and the results folder."""
        return f"{self.input_name.lower()}-to-{self.output_name.lower()}"

    @property
    def wandb_project(self) -> str:
        """`Controller · JointTrajectory to ServoCommand`.

        WHY this form: the house names W&B projects in Title Case words and sub-projects `<Family> · <member>`
        (`Sim2Real · wrist rig`, proven on this account); "to" instead of an arrow because ">" URL-encodes and "→"
        is untested on this account, and ":" or "/" are illegal.
        """
        name = f"Controller · {self.input_name} to {self.output_name}"
        if _WANDB_FORBIDDEN.search(name) or len(name) > 128:
            raise ValueError(f"not a legal W&B project name: {name!r}")
        return name


HUB_PROJECT = "Controllers"  # cross-box reports, like the Sim2Real hub
WANDB_ENTITY = "zach-yamaoka-independent"  # the entity every BAM project logs to (sim2real, bam_trackio)


@dataclass(frozen=True)
class ControllerSpec:
    """One registered controller: what it is, where it runs, how to build it.

    `target` is "module:Class", imported on first use so the registry never imports the bench or a heavy
    controller. The class is built as `Class(message, config)`, where `config` is an instance of the
    `config_target` class (or None for the class's own defaults). A `planned` spec has no target yet; the site
    lists it as not built.
    """

    name: str  # snake_case id, unique: "joint_trajectory_controller"
    title: str  # "Joint trajectory controller"
    box: Box
    runs_on: RunsOn  # where the feedback law closes: on the servo board, on the host, or split between them
    summary: str  # one sentence: what it does
    maturity: Maturity = "built"
    target: str | None = None  # "robot_controllers.controllers.joint_trajectory:JointTrajectoryController"
    config_target: str | None = None  # "robot_controllers.controllers.joint_trajectory:JointTrajectoryConfig"
    prior_art: tuple[str, ...] = ()  # "ros2_control joint_trajectory_controller", "CRISP CartesianController"
    variant_of: str | None = None  # another spec's name when this is the same law split differently
    default_config: Mapping[str, Any] = field(default_factory=dict)  # fields the bench sets on the config
    implementation: Implementation = "python"  # the law's language; a "cpp" spec names its twin in variant_of

    def __post_init__(self) -> None:
        if not _CONTROLLER_NAME.fullmatch(self.name):
            raise ValueError(
                f"controller name must be snake_case (or ff=<model>@<at>+fb=<model>@<at>): {self.name!r}"
            )
        if self.maturity == "built" and not self.target:
            raise ValueError(f"{self.name}: a built controller needs a target")
        if self.implementation not in ("python", "cpp"):
            raise ValueError(f"{self.name}: implementation must be python or cpp: {self.implementation!r}")

    def controller_class(self) -> type:
        return _load(self.target)

    def config_class(self) -> type | None:
        return _load(self.config_target) if self.config_target else None

    def make_config(self, **overrides: Any) -> Any:
        """The config this spec runs with on the bench: the class defaults, then `default_config`, then
        `overrides`."""
        cls = self.config_class()
        return None if cls is None else cls(**{**self.default_config, **overrides})

    def build(self, message: Any, config: Any = None, **overrides: Any) -> Controller:
        """`Class(message, config)`; without `config`, `make_config(**overrides)`."""
        if self.maturity != "built":
            raise NotImplementedError(f"{self.name} is planned, not built")
        if config is None:
            config = self.make_config(**overrides)
        cls = self.controller_class()
        return cls(message) if config is None else cls(message, config)


def _load(target: str | None) -> Any:
    if not target or ":" not in target:
        raise ValueError(f"expected 'module:attribute', got {target!r}")
    module, attribute = target.split(":", 1)
    return getattr(importlib.import_module(module), attribute)
