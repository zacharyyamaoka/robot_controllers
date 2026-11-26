# robot_controllers

**Landing page: [zacharyyamaoka.github.io/robot-controllers](https://zacharyyamaoka.github.io/robot-controllers/)**

Robot controllers as black boxes: a robot_msgs message in, a `ServoCommand` out, each one benchmarked on a ladder of rungs for performance and cost.

1. **Grouped by what goes in and out.** A controller is anything that loops reading joint state and emitting a `ServoCommand`. Controllers that take the same input message and emit the same output are one black box, so they are swapped and compared with each other on the same rungs.
2. **A ladder of rungs.** Free-space motion, fast motion, impact into a moving conveyor belt at three speeds, a sideways push, rapid replanning, a host stall, and the host loop at 50 and 500 Hz. Every rung has fixed gates and one objective.
3. **Performance and cost.** Tracking error, contact force, loop timing and health beside compute time per tick and bus load, so for a given input type you can pick the controller that passes the most rungs for the least cost.
4. **Board or host.** The same impedance law runs on the servo board (stiffness and damping sent to moteus), on the host as a torque, or split (host spring, board damper), and the ladder shows what each costs.
5. **A functional core.** numpy and robot_msgs only: `controller.tick(state, dt)` runs in the caller's thread, in its own, on a twin or on the rack.

```python
from robot_controllers import controller

jtc = controller("joint_trajectory_controller").build(trajectory)   # a robot_msgs JointTrajectory
command, defines = jtc.tick(joint_state, dt)                         # every tick, any thread
```

See `docs/CONTRACT.md` for the build contract and `results/` for the latest ladder results.
