"""The frozen contract: boxes, registry, metric names, gates."""

import pytest
from robot_msgs.control import JointCommandSequence, ServoCommand
from robot_msgs.trajectory import JointTrajectory

from robot_controllers import BOXES, CONTROLLERS, Box, controller, in_box
from robot_controllers.bench.types import Gate, judge
from robot_controllers.core import _WANDB_FORBIDDEN
from robot_controllers.metrics.names import BY_NAME, FAMILIES, METRICS


def test_a_box_is_its_input_and_output_types():
    box = Box(JointTrajectory)
    assert box.output_type is ServoCommand
    assert box.name == "JointTrajectory → ServoCommand"
    assert box.slug == "jointtrajectory-to-servocommand"
    assert box.wandb_project == "Controller · JointTrajectory to ServoCommand"
    assert Box(JointTrajectory) == box and Box(JointCommandSequence) != box


@pytest.mark.parametrize("box", BOXES, ids=lambda b: b.slug)
def test_every_wandb_project_name_is_legal(box):
    name = box.wandb_project
    assert not _WANDB_FORBIDDEN.search(name) and len(name) <= 128


def test_every_registered_controller_belongs_to_a_known_box_and_names_are_unique():
    assert {s.box for s in CONTROLLERS} <= set(BOXES)
    assert len({s.name for s in CONTROLLERS}) == len(CONTROLLERS)
    assert controller("tape_feeder").variant_of == "joint_trajectory_controller"
    with pytest.raises(KeyError):
        controller("nope")


def test_the_sequence_controller_is_robot_msgs_own():
    spec = controller("joint_command_sequence_controller")
    from robot_msgs.control.sequence_controller import JointCommandSequenceController

    assert spec.controller_class() is JointCommandSequenceController
    assert spec.config_target is None


def test_planned_controllers_cannot_be_built():
    planned = [s for s in CONTROLLERS if s.maturity == "planned"]
    assert planned
    with pytest.raises(NotImplementedError):
        planned[0].build(None)


def test_every_box_with_a_built_controller_has_at_least_two_or_says_why():
    built = {b.slug: len(in_box(b, built_only=True)) for b in BOXES}
    assert built["jointtrajectory-to-servocommand"] >= 5
    assert built["jointcommandsequence-to-servocommand"] == 1


def test_metric_catalogue_is_complete_and_typed():
    assert len(BY_NAME) == len(METRICS)
    assert {m.family for m in METRICS} == set(FAMILIES)
    for m in METRICS:
        assert m.better in ("lower", "higher", "none")
        assert m.description.endswith(".") or m.description.endswith(")")


def test_a_gate_on_an_unmeasured_metric_fails_and_says_so():
    gate = Gate("tracking_p95_deg", "<=", 1.0, "tracks the path")
    assert judge(gate, 0.5).passed
    assert not judge(gate, 1.5).passed
    unmeasured = judge(gate, None)
    assert not unmeasured.passed and unmeasured.note == "not measured"
    assert judge(Gate("dwell_bounces", "==", 0, "x"), 0).passed
    assert judge(Gate("replans_accepted", ">=", 8, "x"), 8).passed
