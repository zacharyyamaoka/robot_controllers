"""Every controller robot_controllers knows, grouped by black box. The site's sidebar and the bench read this.

Built controllers have a `target`; planned ones are the standard set from the controller-metrics prior-art note
(ros2_control, CRISP) that is not built yet, listed so the enumeration is honest about what is missing.
"""

from __future__ import annotations

from robot_msgs.control import JointCommandSequence, ServoCommand
from robot_msgs.trajectory import JointTrajectory, PoseTrajectory

from robot_controllers.core import Box, ControllerSpec

JOINT_TRAJECTORY = Box(JointTrajectory)
JOINT_COMMAND_SEQUENCE = Box(JointCommandSequence)
POSE_TRAJECTORY = Box(PoseTrajectory)
SERVO_COMMAND = Box(ServoCommand)

BOXES: tuple[Box, ...] = (JOINT_TRAJECTORY, JOINT_COMMAND_SEQUENCE, POSE_TRAJECTORY, SERVO_COMMAND)

_JT = "robot_controllers.controllers.joint_trajectory"
_JI = "robot_controllers.controllers.joint_impedance"

CONTROLLERS: tuple[ControllerSpec, ...] = (
    # ── JointTrajectory → ServoCommand ────────────────────────────────────────────────────────────────────
    ControllerSpec(
        name="joint_trajectory_controller",
        title="Joint trajectory controller",
        box=JOINT_TRAJECTORY,
        runs_on="board",
        summary="Samples the trajectory at the time since it started and streams position, velocity and "
        "feed-forward to the board's own PD; a late tick skips ahead, so it stays on schedule.",
        target=f"{_JT}:JointTrajectoryController",
        config_target=f"{_JT}:JointTrajectoryConfig",
        prior_art=("ros2_control joint_trajectory_controller",),
    ),
    ControllerSpec(
        name="tape_feeder",
        title="Tape feeder",
        box=JOINT_TRAJECTORY,
        runs_on="board",
        summary="Sends the trajectory's next sample every tick, whatever the clock says (bam_controllers v1); "
        "a late tick slips the rest of the trajectory and it refuses a new trajectory while running.",
        target=f"{_JT}:TapeFeeder",
        config_target=f"{_JT}:TapeFeederConfig",
        prior_art=("bam_controllers v1 (bam_ws, 2026-07-28)",),
        variant_of="joint_trajectory_controller",
    ),
    ControllerSpec(
        name="joint_impedance_board",
        title="Joint impedance on the board",
        box=JOINT_TRAJECTORY,
        runs_on="board",
        summary="Renders a joint spring and damper on the servo board: sends stiffness and damping with every "
        "setpoint, so the board's fast PD closes the impedance (moteus kp_scale / kd_scale).",
        target=f"{_JI}:JointImpedanceController",
        config_target=f"{_JI}:JointImpedanceConfig",
        default_config={"location": "board"},
        prior_art=("CRISP joint impedance", "bam_impedance_controller JOINT mode"),
    ),
    ControllerSpec(
        name="joint_impedance_host",
        title="Joint impedance on the host",
        box=JOINT_TRAJECTORY,
        runs_on="host",
        summary="Computes the same spring and damper on the host as a torque, with the board's gains at zero; "
        "the law closes at the host rate, behind the bus delay.",
        target=f"{_JI}:JointImpedanceController",
        config_target=f"{_JI}:JointImpedanceConfig",
        default_config={"location": "host"},
        prior_art=("CRISP torque-mode impedance", "bam_impedance_controller TORQUE mode"),
        variant_of="joint_impedance_board",
    ),
    ControllerSpec(
        name="joint_impedance_host_spring",
        title="Host spring, board damper",
        box=JOINT_TRAJECTORY,
        runs_on="split",
        summary="Puts the spring on the host as a torque and the damper on the board (kp 0, kd > 0), so damping "
        "acts at board rate while the host shapes stiffness.",
        target=f"{_JI}:JointImpedanceController",
        config_target=f"{_JI}:JointImpedanceConfig",
        default_config={"location": "host_spring"},
        prior_art=("bam_impedance_controller HOST_SPRING mode",),
        variant_of="joint_impedance_board",
    ),
    ControllerSpec(
        name="cartesian_impedance_controller",
        title="Cartesian impedance controller",
        box=JOINT_TRAJECTORY,
        runs_on="split",
        maturity="planned",
        summary="A Cartesian spring and damper mapped to the joints through the Jacobian (needs a kinematics "
        "model); the unmerged bam_impedance_controller lane has four ways to split it.",
        prior_art=("CRISP CartesianController", "bam_impedance_controller (unmerged lane)"),
    ),
    ControllerSpec(
        name="cartesian_admittance_controller",
        title="Cartesian admittance controller",
        box=JOINT_TRAJECTORY,
        runs_on="host",
        maturity="planned",
        summary="Bends the trajectory by a virtual mass-spring-damper driven by a measured wrench; needs a force "
        "signal the rack does not have yet.",
        prior_art=("ros2_controllers admittance_controller", "CRISP admittance"),
    ),
    # ── JointCommandSequence → ServoCommand ───────────────────────────────────────────────────────────────
    ControllerSpec(
        name="joint_command_sequence_controller",
        title="Joint command sequence controller",
        box=JOINT_COMMAND_SEQUENCE,
        runs_on="board",
        summary="Runs a JointCommandSequence step by step (hold, move by, move to, limp) until each step's "
        "done-when fires, with a speed guard that sends every joint limp.",
        target="robot_msgs.control.sequence_controller:JointCommandSequenceController",
        prior_art=("ros2 motion_primitives_controllers (nearest)", "bam_fivebar_recovery home button"),
    ),
    # ── PoseTrajectory → ServoCommand ─────────────────────────────────────────────────────────────────────
    ControllerSpec(
        name="operational_space_controller",
        title="Operational space controller",
        box=POSE_TRAJECTORY,
        runs_on="host",
        maturity="planned",
        summary="Tracks a tool pose trajectory with task-space torques through the operational-space inertia.",
        prior_art=("CRISP OSC", "Khatib 1987"),
    ),
    # ── ServoCommand → ServoCommand ───────────────────────────────────────────────────────────────────────
    ControllerSpec(
        name="forward_command_controller",
        title="Forward command controller",
        box=SERVO_COMMAND,
        runs_on="board",
        maturity="planned",
        summary="Passes the latest servo command through, with limits; the baseline for teleop and tests.",
        prior_art=("ros2_controllers forward_command_controller",),
    ),
)

_BY_NAME = {spec.name: spec for spec in CONTROLLERS}
if len(_BY_NAME) != len(CONTROLLERS):
    raise RuntimeError("duplicate controller names in the registry")


def controller(name: str) -> ControllerSpec:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"no controller named {name!r}; known: {sorted(_BY_NAME)}") from None


def in_box(box: Box, *, built_only: bool = False) -> tuple[ControllerSpec, ...]:
    return tuple(s for s in CONTROLLERS if s.box == box and (s.maturity == "built" or not built_only))


def box_by_slug(slug: str) -> Box:
    for box in BOXES:
        if box.slug == slug:
            return box
    raise KeyError(f"no box {slug!r}; known: {[b.slug for b in BOXES]}")
