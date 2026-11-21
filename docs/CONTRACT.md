# robot_controllers — build contract (v1, frozen 2026-11-12; amendment 1 the same day)

Zach, 2026-11-12: *"controllers are just black boxes ... grouped by their like input and output messages. if they
are the same types then they are the same black box ... a w&B project for each controller type ... a bunch of
different inputs. they can be a rung. simplest could be like free space motion, more complex is like impact into a
conveyor belt, one that requires rapid replanning ... compare like doing impedance control ON moteus vs on the
computer ... evaluate their performance ... and also their cost (how much computation time) ... for a given input
type we can basically know like what is the ideal controller we should be using!"*

This file is the seam the build lanes share. Change it only as a barrier (alone, before dependent lanes run).

## 1. Layout and ownership

| Path | Owner | What |
|---|---|---|
| `src/robot_controllers/core.py`, `registry.py`, `bench/types.py`, `metrics/names.py`, this file | contract (frozen) | Box, Controller protocol, ControllerSpec, registry, EpisodeLog, Rung, Gate, RunResult, metric names |
| `src/robot_controllers/controllers/` | lane **controllers** | the built controllers named in `registry.py` |
| `src/robot_controllers/bench/{plant,board,arm,loop,rungs}.py` | lane **bench** | the planar-arm plant, the servo-board model, the host loop, the rung ladder |
| `src/robot_controllers/metrics/{timing,tracking,contact,health,cost,sequence,episode,summary}.py` | lane **metrics** | EpisodeLog → metrics dict; gates; per-box summary, Pareto, recommendation |
| `src/robot_controllers/tracking/` | lane **wandb** | one W&B project per box, runs, views, reports |
| `website/` | lane **site** | the robotcontrollers landing page + docs, generated from the registry and results JSON |
| `src/robot_controllers/bench/run.py`, `bench/cli.py`, `results/` | integration | run every built controller on every rung of its box, write results JSON |

Dependencies: core = numpy + robot_msgs. `bench` extra = MuJoCo 3.11.0. `tracking` extra = wandb 0.31.0,
wandb-workspaces 0.4.13, weave 0.54.1. `site` extra = matplotlib, pillow. **Lanes never add dependencies or
touch `pyproject.toml`/`uv.lock`; ask the dispatcher.** Run tests with
`env -u VIRTUAL_ENV uv run --frozen --directory /home/bam/robot_controllers --extra test pytest -q <files>`.

## 2. The controller contract (`core.py`)

- `Controller.tick(state: JointState, dt: float) -> (ServoCommand, Mapping[str, float])` — robot_msgs'
  `JointController`, unchanged. `status` ∈ running | done | aborted.
- Built as `Class(message, config)`; the config is a frozen dataclass whose fields are the controller's
  parameters (SI units, documented per field: the site prints them).
- Optional `replace_input(message)` (Replannable): the new message's t = 0 is the controller's "now".
- Optional `diagnostics() -> Mapping[str, float | str]` (SelfReport).
- The controller never stamps time; the loop sets `header.stamp`, `state_stamp`, `state_seq` on the command.
- Joint order: the command's `name` = the input's joint names; the state may carry more joints (select by name).

### Built controllers (registry targets)

| name | class | config | runs_on | law |
|---|---|---|---|---|
| `joint_trajectory_controller` | `controllers.joint_trajectory:JointTrajectoryController` | `JointTrajectoryConfig` | board | time-driven: own clock += dt, sample the trajectory (linear in position/velocity between samples, `JointTrajectory.joint_state_at`), hold the end; command position, velocity, feedforward = trajectory effort (or 0); stiffness/damping None (board gains). Late tick skips. `replace_input` restarts the clock at 0 on the new trajectory. Optional path tolerance → `aborted` + hold at measured (default off). |
| `tape_feeder` | `controllers.joint_trajectory:TapeFeeder` | `TapeFeederConfig` | board | index-driven: sample k = tick count × control_period, regardless of dt (bam_controllers v1); refuses `replace_input` (raises `NotReplannable`); `diagnostics()["schedule_slip_s"]` = elapsed dt sum − own clock. |
| `joint_impedance_board` | `controllers.joint_impedance:JointImpedanceController` | `JointImpedanceConfig(location="board")` | board | position/velocity = reference, `stiffness` = K, `damping` = D, feedforward = reference effort. |
| `joint_impedance_host` | same | `location="host"` | host | `stiffness` = 0, `damping` = 0, `position` = NaN, feedforward = K(q_ref − q) + D(qd_ref − qd) + effort_ref, from the measured state. |
| `joint_impedance_host_spring` | same | `location="host_spring"` | split | `stiffness` = 0, `damping` = D, `position` = NaN, `velocity` = qd_ref, feedforward = K(q_ref − q) + effort_ref. |

`JointImpedanceConfig`: `location`, `stiffness_nm_rad` (per joint or scalar, default (40, 25)), `damping_ratio`
(0.8), `inertia_kg_m2` (the per-joint inertia the damping is designed on, default (0.10, 0.03): D = 2ζ√(K·I)),
`damping_nm_s_rad` (explicit D overrides the ratio), `max_effort_nm` (None = board cap). All impedance variants
sample the trajectory exactly like the JTC (time-driven, replannable).

`joint_command_sequence_controller` is robot_msgs' `JointCommandSequenceController`, built as `Class(sequence)`.

## 3. The bench (lane bench)

### Plant: `planar_arm` (MuJoCo 3.11.0, MJCF built in code, no external assets)

- Two revolute joints `shoulder`, `elbow` about world **y**; the arm moves in the x–z plane. Base at (0, 0, 0.50).
  q = 0 points the link along +x; positive rotates toward −z (down) is NOT assumed — document the sign you choose
  and give `fk(q) -> (x, z)` and `ik(x, z) -> q` (elbow-up, analytic) helpers.
- Links L1 = 0.30 m, L2 = 0.25 m, uniform rods m1 = 1.2 kg, m2 = 0.8 kg; tool sphere r = 0.015 m, 0.3 kg at the tip.
- Gearbox: armature 0.020 / 0.010 kg·m², frictionloss 0.30 / 0.20 N·m, damping 0.05 N·m·s/rad.
- Belt (scene.belt): a box whose top is at z = 0.15 m, sliding along x at `belt_speed_m_s` (a slide joint driven by
  a stiff velocity actuator, or `geom/surfacevel`), friction 0.6. Contact force on the tool is read every physics
  step (normal = +z). The belt contact is calibrated to `contact_stiffness_n_m` / `contact_damping_n_s_m`
  (MuJoCo's negative solref is acceleration units: calibrate, and test the calibration).
- Pushes: `xfrc_applied` on the tool body.
- Physics and board rate 4 kHz (dt 0.25 ms). `nominal_dynamics(q, qd, qdd) -> tau` (mj_inverse on the nominal
  model, no payload) so input scripts can fill trajectory `effort` (gravity + inertia feed-forward).
- Plant joint names `("shoulder", "elbow")`; home q puts the tool at (0.30, 0.25).

### Servo board model (`board.py`, numpy, per joint) — moteus position mode

- Configured gains `kp_nm_rad` (300, 200), `kd_nm_s_rad` (7.5, 4.0), torque cap (12, 8) N·m, at 4 kHz.
- τ = clip(kp·(p − q) + kd·(v − qd) + ff, ±min(cap, max_effort)); `stiffness`/`damping` None → configured gains,
  else the absolute values sent; `position` NaN → no position term.
- Encoder: 0.5 ms measurement delay, velocity from a 200 Hz first-order filter; quantisation 2π/2¹⁴.
- Watchdog 0.25 s: no command latched for that long → mode 11, τ = −kd_brake·qd (kd_brake = configured kd) until the
  next command. Each entry into mode 11 is a `watchdog` event.
- Cross-check: if `bam_pid_tunning` is importable, a test compares one joint's response with its `controller_tick`
  (independent oracle); otherwise that test skips with the reason.

### Host loop (`loop.py`)

`run_episode(spec, rung, *, seed=0, plant_factory=None) -> EpisodeLog`

1. Reset the plant with `rung.scene`; start q = scene.start_q or home. Build the input script; the controller is
   `spec.build(first_message)`.
2. Host wakes at k·period (+ seeded jitter, clipped ±period/2; no wake inside a stall window).
3. On wake: `state = plant.read()` — the newest board sample that has arrived (sampled ≤ now − reply_delay), as a
   `ServoState` (stamp = sample time, seq increments per board reply at the host rate). Pass `state.joint_state()`.
4. Time `controller.tick(state, dt)` with `time.perf_counter_ns` (dt = this wake − previous wake). Script events at
   or before now are delivered before the tick (`replace_input`; a refusal is a `refused` event).
5. Stamp and send: `header.stamp` = sent_t = wake (+ compute if `include_compute`), `state_stamp`, `state_seq`.
   The board latches it at sent_t + command_delay.
6. Record one host-tick row; the plant records every physics step.
7. Reference: the active script message sampled at (sent_t − its start time), extrapolate = hold the end; that is
   the intended schedule, so a slipping controller shows tracking error.
8. Stop at `rung.duration_s`. Set `OMP_NUM_THREADS=1` before timing.

### The rung ladder (`rungs.py`) — the ruler, frozen with the gates below

All JointTrajectory rungs build trajectories with `effort` = `nominal_dynamics` (feed-forward), sampled at 4 ms.
Timing gates on every rung except the stall rungs: `missed_cycles == 0`, `latency_p99_ms <= 1000/rate_hz`.

| order | id | title | what | gates (besides timing) | objective |
|---|---|---|---|---|---|
| 1 | `R1.FreeSpace` | Free-space motion | three min-jerk moves between tool points (0.30,0.25)→(0.42,0.20)→(0.25,0.32)→(0.30,0.25), 0.8 s each, 0.3 s rests | `tracking_p95_deg <= 1.0`, `saturation_share <= 2`, `diverged == 0`, `faults == 0` | `tracking_p95_deg` ↓ |
| 2 | `R2.FastFreeSpace` | Fast free-space motion | the same path, 0.3 s per move | `tracking_p95_deg <= 3.0`, `saturation_share <= 5`, `diverged == 0` | `tracking_p95_deg` ↓ |
| 3 | `R3.BeltImpact@0.05m/s` | Belt impact at 0.05 m/s | from 40 mm above the belt at x = 0.30, straight down, constant `touch_speed` for the last 20 mm, aiming 10 mm below the surface; dwell 0.8 s on the belt moving 0.25 m/s; lift 40 mm | `contact_peak_n <= 60`, `dwell_bounces == 0`, `dwell_force_error_n <= 10` (target press force 10 N), `dwell_contact_fraction >= 95`, `diverged == 0` | `contact_max_n` ↓ |
| 4 | `R3.BeltImpact@0.10m/s` | Belt impact at 0.10 m/s | same, 0.10 m/s | same | `contact_max_n` ↓ |
| 5 | `R3.BeltImpact@0.20m/s` | Belt impact at 0.20 m/s | same, 0.20 m/s | same | `contact_max_n` ↓ |
| 6 | `R4.SidewaysPush` | Sideways push | hold at home; push +15 N along x for 0.4 s at t = 0.5 s | `push_recovery_s <= 0.5`, `push_overshoot_mm <= 5`, `diverged == 0` | `push_recovery_s` ↓ |
| 7 | `R5.RapidReplan` | Rapid replanning | a new trajectory every 0.15 s, 8 times, each from the previous one's state at that instant (position and velocity) to a seeded target within ±20° of home, 0.6 s, ending at rest | `replans_accepted >= 8`, `command_jump_max_deg <= 2`, `replan_settle_s <= 0.5`, `diverged == 0` | `tracking_p95_deg` ↓ |
| 8 | `R6.HostStall` | Host stall | R1's path with jitter 0.3 ms and one 0.5 s stall at t = 0.8 s (the F09 hang); the board watchdog brakes | `diverged == 0`, `schedule_slip_ms <= 20`, `settle_time_s <= 1.0` | `tracking_p95_deg` ↓ |
| 9 | `R7.BeltImpact@0.10m/s·host50Hz` | Belt impact, host at 50 Hz | rung 4 with the host at 50 Hz | rung 4's gates, timing gates at 50 Hz | `contact_max_n` ↓ |
| 10 | `R7.BeltImpact@0.10m/s·host500Hz` | Belt impact, host at 500 Hz | rung 4 with the host at 500 Hz | rung 4's gates, timing gates at 500 Hz | `contact_max_n` ↓ |

JointCommandSequence rungs (same plant):

| order | id | title | what | gates | objective |
|---|---|---|---|---|---|
| 1 | `S1.MoveTo` | Move to a pose | `to` both joints (speed 1.5 rad/s, max effort = cap) until `Reached` (1°), then hold 0.5 s | `steps_done >= steps_total`, `reached_error_deg <= 1.0`, `faults == 0` | `sequence_time_s` ↓ |
| 2 | `S2.SeekHardStop` | Seek a hard stop | `move_by` the elbow toward its joint limit (a hard stop in the MJCF) at 0.5 rad/s, max effort 3 N·m, until `Stall`; define that position | `steps_done >= steps_total`, `faults == 0` | `sequence_time_s` ↓ |
| 3 | `S3.HostStall` | Host stall mid-move | S1 with a 0.5 s stall at t = 0.3 s | `steps_done >= steps_total`, `diverged == 0` | `sequence_time_s` ↓ |

`rungs.ladder(box) -> tuple[Rung, ...]`, ordered. Gate numbers above are the ruler: never move them after seeing
results; a controller that fails is a finding.

## 4. Metrics (lane metrics)

`episode_metrics(log: EpisodeLog, rung: Rung) -> dict[str, float | int | None]` returns **every** name in
`metrics.names.METRICS` (None = not measured, e.g. contact on a free-space rung, timing jitter is still measured
from sent times). Formulas are the descriptions in `names.py`; definitions that need more:

- tracking error = ref_position − measured position at the host tick's sent time, interpolated from the plant
  samples (not the delayed state); `robot_msgs.compare` for ErrorStats where it fits. Worst joint, degrees.
- tool error = |fk(ref) − tool_pos| (ref_tool_pos is in the log).
- contact: first touch = first sample with in_contact; transient window [touch, touch + 30 ms]; dwell = the rung's
  dwell window from `log.meta["dwell"] = (t0, t1)`; a bounce = an in_contact → not-in-contact transition inside
  the dwell, debounced 2 ms; target press force from `log.meta["press_target_n"]`.
- push window from `log.meta["push"] = (t0, t1, axis)`; recovery = first time after t1 that |tool − pre-push
  tool| stays < 1 mm for 50 ms.
- timing: from sent_t and state_stamp only (black box). nominal period = `log.nominal_period_s`.
- cost: compute from compute_s; `command_fields` = mean count of non-NaN fields among position, velocity,
  feedforward, stiffness, damping, max_effort per joint; `wire_bytes_per_tick` = joints × (2 + 4 × fields)
  (one register header + float32 values, moteus position-mode frame).
- sequence: from `log.events` (`step_done` events with outcome) and `log.meta["steps_total"]`.

`evaluate(rung, metrics) -> list[GateResult]` (uses `bench.types.judge`); `run_result(log, rung) -> RunResult`.

`summarize(box, rungs, results) -> dict` (results-1 `summary`): per controller `gates_passed`, `gates_total`,
`rungs_passed`, `rungs_total`, `objective` per rung and its rank among the box, `compute_us_p99` (median over
rungs); `pareto` = per belt rung, the non-dominated set on (R1 `tracking_p95_deg`, that rung's `contact_max_n`,
`compute_us_p99`); `recommendation` = the controller with the most rungs passed, then most gates passed, then the
lowest `compute_us_p99`, with a one-line `why` naming those numbers; per-rung `best` = lowest (or highest)
objective among controllers that passed that rung's gates (None if nobody did).

## 5. Results JSON (`results/<box-slug>/results.json`, schema `robot-controllers/results-1`)

```json
{
  "schema": "robot-controllers/results-1",
  "box": {"slug": "...", "name": "JointTrajectory → ServoCommand", "input": "JointTrajectory",
          "output": "ServoCommand", "wandb_project": "Controller · JointTrajectory to ServoCommand"},
  "generated": {"commit": "...", "dirty": false, "robot_msgs": "0.2.0 @ sha", "plant": "planar_arm",
                "host": "hostname", "python": "3.12.3", "mujoco": "3.11.0", "date": "2026-11-12"},
  "rungs": [{"id": "...", "title": "...", "family": "...", "order": 1, "description": "...", "duration_s": 3.0,
             "params": {}, "timing": {"rate_hz": 100, ...}, "scene": {...},
             "gates": [{"metric": "...", "op": "<=", "value": 1.0, "why": "..."}],
             "objective": "tracking_p95_deg", "minimize": true}],
  "controllers": [{"name": "...", "title": "...", "runs_on": "board", "config": {...}, "variant_of": null}],
  "results": [RunResult.to_dict(), ...],
  "summary": {"per_controller": {...}, "per_rung": {...}, "pareto": {...}, "recommendation": {...}},
  "wandb": {"project_url": null, "reports": {}}
}
```

`RunResult.trace` keys (downsampled to 200 Hz from the plant samples unless noted): `t`; `q_deg.<joint>`;
`ref_deg.<joint>` (host ticks interpolated); `err_deg.<joint>`; `tau_nm.<joint>`; `tool_x_mm`, `tool_z_mm`;
`contact_fn_n` (**1 kHz** inside [touch − 20 ms, touch + 80 ms], 200 Hz elsewhere: the key `contact_t` carries its
own time base); `compute_us` (per host tick) with `tick_t`. Floats rounded to 4 significant digits.

The results JSON is the ledger (the record); W&B gets a copy and the site renders it.

## 6. W&B (lane wandb)

- Entity `zach-yamaoka-independent`. One project per box: `Box.wandb_project`; hub `Controllers` for cross-box
  reports. Online (sim data, no rig data leaves the machine).
- One run per (controller, rung, seed): id = `sha1("robot_controllers|box|controller|rung|seed|digest")[:16]`
  (digest = sha1 of the run's metrics JSON) so identical numbers resume, new numbers make new runs and the old
  ones get `evaluation_state = superseded` (never deleted). name `"<controller> @ <rung>"` (≤ 128 bytes),
  group = controller, job_type = `rung`, tags {rung family, "sim", "latest"}.
- config: box, input_type, output_type, controller, runs_on, controller config fields, rung, rung params, host
  timing, plant, commit, dirty, robot_msgs version, evaluation_state. summary: every metric (units in the key),
  `gate/<metric>` text ("pass 0.62 <= 1.0" / "FAIL ..." / "not measured"), `passed`, `objective`.
  History: the trace's series at their own step (so line plots show the run).
- Views and reports built in code, saved in place via a committed registry JSON (`tracking/wandb_views.json`):
  per box a Scoreboard (controller × rung pass table), Performance vs cost (scatter tracking vs compute,
  contact vs compute), Contact (peak and dwell force on belt rungs), Timing; per rung a saved view.
  Rewrite forge.coreweave.com URLs to wandb.ai. Weave leaderboard per box via bam_trackio's `bbo.py` loaded by
  file (never import bam_trackio), optional (`--weave`).
- Pure record building (`records(results) -> list[dict]`) is testable without wandb; wandb is imported only inside
  functions that talk to W&B. Logging refuses a dirty tree unless `--allow-dirty`.

## 7. Site (lane site)

Copy and adapt `/home/bam/robot_msgs/website` (chrome byte-identical where possible: site.css, site.js,
highlight.py, serve.py, generic tests). Generated from `robot_controllers.registry` + `metrics.names` +
`results/*/results.json`; nothing written per controller. Port **4494**. Pages:

- `/` landing: what a controller is here (black box: input message → ServoCommand), the boxes with their
  controllers, the recommendation per box, install + a 10-line same-thread example.
- `/boxes/<slug>/`: the box: input/output chips linking to robotmsgs.com type pages, the rung ladder with gates,
  the leaderboard (controller × rung: objective value, pass/fail, best highlighted), Pareto plots, the
  recommendation and why, the W&B project link.
- `/controllers/<slug>/<name>/`: summary, I/O chips, where it runs (an inline SVG of host ↔ bus ↔ board showing
  where K and D live), parameters table (config fields, defaults, units), an example (same-thread loop), per-rung
  results table with gate verdicts, per-rung plots (tracking traces in degrees, contact force trace), cost
  (compute µs p50/p99, fields, bytes), self-report keys, prior art, W&B run links. Planned controllers get a page
  saying not built, with prior art.
- `/rungs/` the ladder for every box, `/metrics/` the catalogue.
- Sidebar: search, Overview (Home, Rungs, Metrics), then one group per box headed `Input → Output`, listing its
  controllers (planned ones marked).
- Angles in degrees everywhere a human reads them.

## Amendment 1 (2026-11-12, after the build lanes; a barrier, applied alone)

1. **Compute cost is thread CPU time**, not wall time (`EpisodeLog.compute_s`, `compute_us_*`, and the simulated send
   delay when `include_compute`). The shared machine ran at load ~100; wall time counted other processes' turns as
   100+ ms ticks and failed timing gates at random. Wall time stays in `meta["compute_wall_s"]`.
2. **`Gate.applies = "if_measured"`**: a metric that does not exist for a controller by design passes as "n/a"
   instead of failing as "not measured". Used once: R5's `command_jump_max_deg` (host-torque impedance never
   commands a position). Every other gate keeps "coverage counts".
3. **Index-driven controllers are configured for the rung's host rate**: the loop sets `control_period_s` when the
   config has it (the tape feeder), as a deployment would.
4. Sequence `steps_done >= steps_total` gates are the literal step count of each rung's sequence; S1 and S2 carry
   the timing gates, S3 (the stall rung) does not.
5. Belt friction is the plant's own Coulomb law (μ 0.6 on the previous step's normal force, smoothed below 0.02 m/s
   slip): MuJoCo's soft friction chattered a lightly pressed tool off the sliding belt.
6. The first tick after build or `replace_input` samples t = 0; the tape feeder accepts a new input once `done`.


## Amendment 2 (2026-11-15: several source implementations of one black box; a barrier, applied alone)

Zach, 2026-11-15: *"a controller template with several source implementations of the same black box: Python today
(robot_controllers), now C++"*.

1. **`ControllerSpec.implementation`** ∈ `python` | `cpp` (default `python`). A second implementation of a law is its
   own spec, named `<reference>_<language>`, with `variant_of` = the reference spec and the reference's `runs_on` and
   `config_target`; it is built the same way (`Class(message, config)`), gets its own rows in the results JSON and its
   own W&B runs, and every test parametrized over the box's built controllers runs on it.
2. **Parity is its acceptance.** On randomized trajectories, late and irregular dt, the hold at the end,
   `replace_input` mid-run and in every status, and the path-tolerance abort, every tick's ServoCommand equals the
   reference's to 1e-12 and the status and diagnostics match (`tests/test_cpp_parity.py`; bit-exact today:
   `PARITY_TOLERANCE=0` passes). On the ladder every metric equals the reference's except compute.
3. **`joint_trajectory_controller_cpp`**: the law is `controllers/cpp/include/robot_controllers/joint_trajectory_controller.hpp`
   (C++17, header-only; no allocation, lock or I/O in `tick`; at most 64 joints; built with `-ffp-contract=off` so
   `(1 - w)·a + w·b` rounds as numpy does). The Python class loads `controllers/cpp/jtc_capi.cpp`'s C ABI with ctypes
   (stdlib): no new dependency, `pyproject.toml` untouched. It compiles the shim on first use into the cache
   (`$ROBOT_CONTROLLERS_CACHE`, else `~/.cache/robot_controllers`; needs a C++17 compiler); `ROBOT_CONTROLLERS_JTC_LIB`
   points at a library built elsewhere. `diagnostics()` adds `implementation` and `law_cpu_ns` (the law's thread CPU
   time inside C++, without the ctypes and ServoCommand cost that `compute_s` includes).
4. C++ users take the header through `controllers/cpp/CMakeLists.txt` (`robot_controllers::headers`). The bam_ws
   moteus controller node runs this header in its real-time thread, so the law ranked here is the law on the arm.
   **Erratum (2026-11-20):** the node has run only on the twin; on a real bus it is hold-only (its README: "Nothing here
   has touched the arm"). The real arm's pick cycle streams its own sampler (bam_ws `repick.PanDriver` + `PlanClock`), and
   the real-arm JTC is arm_motion_bench's lane J (`fullarm_jtc.py`, in progress). The sentence above is the goal, not yet a fact.
5. One deliberate difference: a state missing one of the trajectory's joints raises the same KeyError, but before the
   clock advances (the Python law advances first).


## Amendment 3 (2026-11-20: feed-forward and feedback as pure functions, and where and how often each runs; a barrier, applied alone)

Zach, 2026-11-20 (Daily Note, `# Controllers`): *"The feedforward models it should accept as pure functions. the feedback
model as well. Success to me looks like: Given a feed forward model, Given a feedback model. what is the best way you
basically lay these things out...? computation cost, time, etc."* · *"how and where you do the feedforward control or the
feedback control ... when and where and how often do we do thoose computations?"* · *"Surely thats just one that like just
fowrwards the joint trajectory its given, and feedback runs on the driver. that is just alike a first step"*.

A controller in the JointTrajectory box is now three choices on top of one trajectory: a **feed-forward model**, a
**feedback model** and a **layout** (where each one is evaluated and how often). The built controllers are points in that
space; the composed controller below reaches all of them and the ones between, so the question "given this FF and this FB,
which layout?" is a sweep, not a new controller per answer.

### 3.1 The pure-function types (`src/robot_controllers/models.py`, contract)

- **`FeedforwardModel`**: `ff(q, qd, qdd) -> tau`. numpy float arrays in the trajectory's joint order, SI (rad, rad/s,
  rad/s², N·m). Shape `(n,)` → `(n,)`, and `(N, n)` → `(N, n)` (plan-time evaluation calls it once on every sample).
  **Pure:** no state, no clock, no I/O, no randomness; equal inputs give equal outputs and the inputs are never written.
- **`FeedbackModel`**: `fb(q_ref, qd_ref, q, qd) -> tau`, the same conventions. Pure as above (no integrator state: an
  integral is a later amendment with its own `integral_share_pct`). A feedback a servo board can close also has
  **`gains(q_ref) -> (kp, kd)`** (N·m/rad, N·m·s/rad per joint) and is exactly `kp·(q_ref − q) + kd·(qd_ref − qd)` with
  those gains: that is what lets the layout move it onto the board. A feedback without `gains` can only run on the host.
- Built models (`models.py`):
  - `PD(kp_nm_rad, kd_nm_s_rad)`: constant gains; `gains` returns them.
  - `InertiaScaledPD(bandwidth_hz, damping_ratio, mass_matrix)`: computed-torque feedback,
    `tau = M(q)·(ωn²·e + 2ζωn·ė)` with ωn = 2π·bandwidth, `mass_matrix(q) -> (n, n)` a pure function. On the host it uses
    the full M(q) at the measured q; `gains(q_ref)` returns `diag(M(q_ref))·ωn²` and `diag(M(q_ref))·2ζωn` (the diagonal
    part a board can render; the coupling is dropped there, which is the layout's measured cost).
  - `ZERO_FEEDFORWARD`: `tau = 0`.
- Planar-arm models (`bench/arm_models.py`, the bench's own robot; numpy only, no MuJoCo):
  `planar_arm_feedforward(terms, arm=ARM, payload_kg=0.0)` with `terms` ⊆ {`gravity`, `inertia`, `coriolis`, `viscous`,
  `coulomb`} (`coulomb` = frictionloss·tanh(qd / 0.01 rad/s)), and `planar_arm_mass_matrix(arm=ARM)`. Closed-form
  two-link rigid-body dynamics with the armature. **Oracle:** with terms {gravity, inertia, coriolis, viscous} it equals
  `arm.nominal_dynamics` (MuJoCo `mj_inverse`) to 1e-9 N·m on random (q, qd, qdd); M(q) equals `mj_fullM`.

### 3.2 The layout (`src/robot_controllers/models.py`)

`Layout(feedforward_at, feedback_at, lookahead_s=0.0)`, frozen:

| field | value | means | evaluated |
|---|---|---|---|
| `feedforward_at` | `trajectory` | use the trajectory's own `effort` column as given (0 when absent): today's JTC | never |
| | `none` | feed-forward 0 | never |
| | `plan` | `ff` on every trajectory sample once, at build and at `replace_input`; the tick only samples the column | plan time |
| | `host_reference` | `ff(q_ref, qd_ref, qdd_ref)` at the sampled reference, every tick | host rate |
| | `host_measured` | `ff(q, qd, qdd_ref)`: position and velocity terms at the measured state, acceleration from the reference | host rate |
| `feedback_at` | `board` | position = q_ref, velocity = qd_ref; stiffness/damping = `fb.gains(q_ref)` sampled at build time from the first sample (constant gains); `fb = None` sends None = the board's configured gains | board rate |
| | `board_scheduled` | as `board`, but `fb.gains(q_ref)` every tick, sent as stiffness/damping | host rate (gains), board rate (law) |
| | `host` | position NaN, velocity None, stiffness 0, damping 0; feed-forward += `fb(q_ref, qd_ref, q, qd)` from the measured state (as `joint_impedance_host`) | host rate |
| | `split` | host spring + board damper: position NaN, velocity = qd_ref, stiffness 0, damping = kd of `fb.gains(q_ref)` at build; feed-forward += kp·(q_ref − q) (as `joint_impedance_host_spring`) | host (K), board (D) |
| `lookahead_s` | ≥ 0 | the reference is sampled at clock + lookahead (the bus delay a board-side law lags by; 2026-11-12 finding) | |

Invalid pairs raise `ValueError` at build: `board`, `board_scheduled` or `split` with a feedback without `gains`;
`host`, `board_scheduled` or `split` with `fb = None`; `plan` / `host_*` with `ff = None`.

### 3.3 The composed controller (`src/robot_controllers/controllers/layout.py`)

`LayoutController(trajectory, LayoutConfig(feedforward=None, feedback=None, layout=Layout("trajectory", "board"),
max_effort_nm=None, path_tolerance_rad=None))`. Time-driven exactly as the JTC (`TrajectoryClock`, `TrajectorySampler`,
the hold at rest after the end, `status`, the path-tolerance abort and hold, `replace_input` with t = 0 at the next
tick). Velocity/acceleration of the reference: the sampler's velocity; the acceleration is the trajectory's own
`accelerations` when it has them, else the finite difference of its velocities on its own grid (`np.gradient`), sampled
linearly like the rest and held at the ends. `plan` evaluates `ff` on the trajectory's samples and the tick interpolates
that column linearly, so it equals `host_reference` exactly on the sample grid and differs between samples by the
interpolation error (part of what the sweep measures).

Self-report (`diagnostics()`, plus `last_split() -> (tau_ff, tau_fb_host)` arrays of the last tick): `layout`,
`feedforward_at`, `feedback_at`, `plan_compute_s` (thread CPU time spent evaluating `ff` at plan time, summed over every
`plan` load). **Acceptance = parity with the built controllers, every tick's ServoCommand equal to 1e-12:**
- `LayoutConfig()` (trajectory FF, board, no fb) ≡ `joint_trajectory_controller`.
- trajectory FF + `PD(K, D)` at `board` ≡ `joint_impedance_board` with explicit K and D; at `host` ≡ `joint_impedance_host`;
  at `split` ≡ `joint_impedance_host_spring`.
- `plan` ≡ `host_reference` in every command when every tick lands on a trajectory sample (to 1e-9, the clock's float
  sum), at different cost.

### 3.4 Metrics added (`metrics/names.py`; new family `decomposition`)

The commanded torque is split three ways at the board, every physics step: `tau_ff` (the feed-forward field latched on
the board), `tau_fb_board` (the board's PD term, `kp(p − q) + kd(v − qd)`), and the host's feedback, which rides inside
the feed-forward field and is taken from the controller's `last_split()` (the only self-report a metric uses; a
controller without `last_split` has `tau_fb_host` = 0). On a real moteus `tau_fb_board` = measured torque − the sent
feed-forward, so the same names hold on the rack.

| name | unit | family | better | definition |
|---|---|---|---|---|
| `fb_share_pct` | % | decomposition | lower | 100 · RMS(tau_fb_board + tau_fb_host) / RMS(tau_applied), worst joint. Zach 2026-11-11: "low feedback torque, where most of it is coming from feedforward terms" |
| `fb_rms_nm` | N·m | decomposition | lower | RMS(tau_fb_board + tau_fb_host), worst joint: what the feed-forward model left for feedback to close |
| `ff_rms_nm` | N·m | decomposition | none | RMS of the feed-forward the feed-forward model carried (context) |
| `plan_compute_ms_per_s` | ms/s | cost | lower | plan-time CPU (`plan_compute_s`) per second of trajectory, ms |
| `host_duty_pct` | % | cost | lower | compute_us_p50 × host rate / 10⁴: share of one core the tick uses |
| `wire_bytes_per_s` | bytes/s | cost | lower | wire_bytes_per_tick × host rate |
| `torque_rate_rms_nm_s` | N·m/s | health | lower | RMS of d(tau_applied)/dt over the board samples, worst joint (chatter, a host-rate law's steps) |

### 3.5 Rungs added (JointTrajectory box; the existing ten unchanged)

| order | id | what | gates | objective |
|---|---|---|---|---|
| 11 | `R8.FastFreeSpace·payload` | R2's path with a 0.3 kg payload on the tool that no feed-forward model knows (model error) | R2's | `tracking_p95_deg` ↓ |
| 12 | `R9.FastFreeSpace·host50Hz` | R2 with the host at 50 Hz | R2's, timing gates at 50 Hz | `tracking_p95_deg` ↓ |
| 13 | `R9.FastFreeSpace·host500Hz` | R2 with the host at 500 Hz | R2's, timing gates at 500 Hz | `tracking_p95_deg` ↓ |

Their gates are R2's numbers, frozen with this amendment, never moved after results. A cost metric's rate is the rung's
host rate.

### 3.6 The layout sweep (`bench/layouts.py`, CLI `robot-controllers layouts`)

The answer to "given FF and FB, which layout?" as a ledger: `results/layouts/results.json`, schema results-1, box
JointTrajectory. Controllers are `LayoutController` configs named `ff=<model>@<at>+fb=<model>@<at>`:
- feed-forward models: `trajectory` (only with `feedforward_at="trajectory"`: the rung's own effort column), `none`
  (only with `feedforward_at="none"`), `gravity`, `rigid_body` (gravity + inertia + coriolis + viscous = the nominal model),
  `rigid_body_coulomb` (+ coulomb);
- feedback models: `board_gains` (`fb = None`: the board's configured kp/kd, the simple JTC), `pd` (`PD` = the board's
  configured kp/kd as a pure function, so it can move to the host or split), `ctc` (`InertiaScaledPD` on
  `planar_arm_mass_matrix`, ζ 0.9, its bandwidth chosen so that its shoulder kp at home equals the board's configured
  shoulder kp, 300 N·m/rad: CTC and PD compared at equal stiffness where the arm starts; the number is recorded);
- every valid (FF placement × FB placement) pair;
- rungs R1, R2, R4, R6, R8 and both R9 (free space, fast at 100 Hz, push, host stall, model error, fast at 50 and
  500 Hz): the host rate is a rung's, as R7 does it, so every number in the sweep is on a frozen rung.

API (the acceptance test calls these): `layout_spec(ff_model, ff_at, fb_model, fb_at) -> ControllerSpec` (built,
`target` = `LayoutController`, its `default_config` carrying the functions and the `Layout`; name as above; `runs_on`
= `board` for board / board_scheduled, `host` for host, `split` for split), and `recommend(candidates) -> dict`, where
each candidate is a mapping with `name`, `passed` (every gate on every swept rung), `tracking_p95_deg` (R2),
`host_duty_pct`, `plan_compute_ms_per_s`, `wire_bytes_per_s`; it returns `{"name": <str | None>, "why": <str>}`.

Summary per (FF model, FB model) → **recommended layout = the cheapest layout (lowest host_duty_pct, then
plan_compute_ms_per_s, then wire_bytes_per_s) among those that pass every gate on every swept rung and whose R2
`tracking_p95_deg` is within 10 % of the best passing layout's**, with a one-line `why` naming the numbers. If none
passes, that is the finding and the summary says so. Every placement except `trajectory` ignores the trajectory's own
`effort` column (the rungs fill it with the nominal model), so the feed-forward is only ever the named model.

### 3.7 Write set

The amendment's builder writes `models.py`, `controllers/layout.py`, `bench/arm_models.py`, `bench/layouts.py`, the
board/loop/log changes for the torque split, `metrics/names.py` + the metric modules, `rungs.py` (two rungs), `registry.py`
(the catalog entries), `bench/cli.py`, `results/layouts/`, and tests. **`tests/test_layout_acceptance.py` is the
dispatcher's, written before the build and read-only to the builder.**

### 3.8 Addendum after the first sweep (2026-11-20, dispatcher; a barrier, applied alone)

Two changes the first sweep (`22c2ffb`) showed were needed. Neither moves a gate.

1. **`fb_share_pct` is the arm's share, in traj_compare's formula.** It was the worst joint's RMS ratio. On R1 that
   picked the elbow, which carries ~0.6 N·m RMS at home, of which ~0.45 N·m is the board's kd times the encoder's
   quantisation buzz (`board.py` models it on purpose; no feed-forward can remove it), and the ratio passed 100 % when
   feedback opposed the feed-forward (154 % for pd@host). Now
   `fb_share_pct = 100 · Σ_joints Σ_samples |fb| / (Σ|ff| + Σ|fb|)`, where ff is what the feed-forward model carried and
   fb is the board's PD term plus the host's feedback. That is traj_compare's per-joint `fb_share`
   (`traj_compare/analysis.py`), summed over the joints, so a run shows the same number in both tools and an idle joint
   can't dominate. New diagnostic **`fb_share_lf_pct`** (%, decomposition, lower): the same after a first-order 50 Hz
   low-pass of ff and fb. That is what a better model could still take off the feedback; the buzz is a floor beneath it.
   `fb_rms_nm` and `ff_rms_nm` stay worst-joint, unfiltered. The acceptance test is unchanged and must pass on this
   definition (dispatcher's check: R1 no feed-forward 99.9 % vs `rigid_body@plan` 14.3 %).
2. **Lookahead is a swept dimension.** The first sweep left `lookahead_s` at 0, so every board-side law tracked a
   reference that reached the board 5.5 ms late (the 2026-11-12 finding). Dispatcher's check on R2:
   `trajectory + board_gains@board` 1.68° → 0.35° with `lookahead_s = 0.0055` (the rung's `command_delay_s`),
   `rigid_body@plan + ctc@board_scheduled` 2.01° → 0.53°, `rigid_body@plan + pd@split` 1.05° → 2.35° (worse: its host
   spring already acts on a stale measurement). `layout_spec(ff_model, ff_at, fb_model, fb_at, lookahead_s=0.0)`; a
   non-zero lookahead adds `+la=<ms>ms` to the name (`ff=trajectory@trajectory+fb=board_gains@board+la=5.5ms`), and
   the name check accepts it. The sweep runs every valid layout at `lookahead_s` ∈ {0, 0.0055}.
3. **Cost in `recommend` is less noisy.** `host_duty_pct` varied 2–4× between identical laws on the shared host. The
   candidates' `host_duty_pct` is now the median over the swept rungs of `compute_us_p50`, expressed at 100 Hz
   (median µs × 100 / 10⁴), so one noisy rung can't flip a pick. Everything else in the rule stays.
4. `joint_impedance_host` and `joint_impedance_host_spring` get `last_split()`, so their host feedback counts as
   feedback on the main ladder.

## Amendment 4 (2026-11-20: host rate, speed, lookahead calibration, a moving target, and a UR5e; a barrier, applied alone)

Zach, 2026-11-20: "why is 50hz a meaning control hz? I would be curious to see more clearly how the results change with
different control frequency … Can you please run examples on the UR robot arm. and also try trajectories at different
speeds. I kinda want to make this a nice stable conclusion." And: "how that can be calibrated on how far the look ahead
should be? how that that impact when you are doing like syncing to catch a moving item where timing is very important?
… how reliabiliy does look ahead work?"

What the dispatcher measured before writing this (scratch runs on `a07fef1`, R2's path, `trajectory@trajectory +
board_gains@board`):

- **Rate.** With `lookahead_s` = 5.5 ms, R2 tracking p95 is 13.7° at 10 Hz, 2.05° at 25 Hz, 0.775° at 50 Hz, 0.347° at
  100 Hz, 0.152° at 200 Hz and 0.098° at 500 Hz. The board closes feedback at 4 kHz whatever the host does, but between
  two commands it walks the setpoint at the sent (tangent) velocity and holds the feed-forward. The error from that grows
  with the square of the period.
- **Interpolation.** If the host sends the *secant* velocity `(p(t+T) − p(t))/T` instead, the walk lands on the next
  sample. The trajectory is known ahead, so this costs nothing. R2 p95 then drops to 0.223° at 50 Hz and 0.979° at
  10 Hz. Adding the effort sampled at the middle of the hold (`t + T/2`) gives 0.201° at 50 Hz and 0.122° at 100 Hz.
  It is flat above 100 Hz (0.116° at 200 Hz, 0.115° at 500 Hz).
- **Lookahead calibration.** The tracking error is V-shaped in the lookahead: lowest at 5.0–5.5 ms, rising about 0.3°
  per ms either side on R2 at 100 Hz. One run at lookahead 0 gives the delay directly: the shift τ that minimises
  `Σ_j RMS(ref_j(t − τ) − q_j(t))` is 5.50 ms, the bus delay.
- **Real bus.** The rig's 50 Hz comes from the bus, not the host. One fdcanusb exchanges with all seven boards in
  ~15.5 ms (p95 16.4 ms, 2026-11-18), so a 10 ms tick cannot hold it (bam_ws PEP 0048, `full_arm.py:77-80`). The
  twin's 5.5 ms down-delay is a model fit (`rack_truths.py:240-250`), not a timed number. On one bus the boards are
  served in turn, so each joint's delay differs. Per-joint delay and lookahead are therefore part of this amendment.

### 4.1 The layout gains interpolation and per-joint lookahead (`models.py`, `controllers/layout.py`)

- `Layout.interpolation` ∈ {`tangent` (default, today's behaviour), `secant`, `secant_mid_ff`}. Let s be the sample time
  (clock + the joint's lookahead) and T the controller's nominal host period. T is `LayoutConfig.control_period_s`,
  which the bench sets from the rung's rate through `_rate_overrides`, as for the tape controllers.
  - **`secant`:** the velocity sent is `(p(s + T) − p(s)) / T`, with p sampled by `TrajectorySampler`. It holds the
    last sample past the end, so the final secant walks onto the end point. The at-rest rule stays: velocity 0 once
    s ≥ end.
  - **`secant_mid_ff`:** additionally, everything the feed-forward reads from the reference is taken at `s + T/2`. That
    covers the effort column for `trajectory`, the plan-time column for `plan`, and the reference sample for
    `host_reference`. `host_measured` still uses the measured q, qd, with the reference qdd at `s + T/2`.
  - **Host and split feedback** are evaluated at the tick as today. Split's board damper gets the secant velocity
    as its target.
  - A non-default interpolation adds `+interp=<value>` to the name (after `+la=…`).
- `Layout.lookahead_s` may be a tuple, one value ≥ 0 per joint in the trajectory's joint order; a scalar applies to all
  joints. Each joint's reference (position, velocity, the feed-forward's sample) is taken at clock + its own lookahead.
  Feedback models that need the whole vector (`InertiaScaledPD`) take each joint's own sample. Name:
  `+la=<ms>/<ms>/…ms`, each `:g`. `Layout.check` refuses a tuple of the wrong length at build.
- `layout_spec(ff_model, ff_at, fb_model, fb_at, lookahead_s=0.0, interpolation="tangent", arm="planar_arm")`; the
  name check in `core.py` accepts the new suffixes.
- The defaults reproduce today's commands bit for bit. `test_layout_acceptance.py` must keep passing unchanged.

### 4.2 The bus gains per-joint delay and jitter (`bench/types.py`, `bench/plant.py`)

- `HostTiming.command_delay_s` may be a tuple, one value per plant joint: joint j's part of a command latches at
  sent + delay_j.
- `HostTiming.command_delay_jitter_s` (default 0): each command's delay gets one draw u ~ U(−jitter, +jitter). It comes
  from its own generator seeded from the episode seed, so the host wake-jitter draws never shift. The draw is added to
  every joint's delay. Latches stay first-in first-out per joint, as on a CAN bus: a command never latches before the
  one sent before it, and if it would, it latches on the same tick.
- With the defaults the episode is identical to today's: same samples, same latches, same metrics.

### 4.3 Lag, sync and the calibration (`metrics/lag.py`, `metrics/names.py`)

- `estimate_lag_s(log, *, per_joint=False, max_s=0.030, step_s=0.00025)` (reads only `log.t`, `log.q`, `log.sent_t`,
  `log.ref_position`): the shift τ ∈ [−max_s, +max_s] on a step_s
  grid that minimises `Σ_j Σ_i (ref_j(t_i − τ) − q_j(t_i))²`.
  - **Data:** the plant samples t_i in [first sent_t + 0.05 s, last sent_t]. `ref` is `log.ref_position`, linear in
    `log.sent_t`. A tie goes to the smallest |τ|.
  - **`per_joint=True`:** returns one τ per joint, or None for a joint whose reference moves less than 0.1° peak to peak.
  - **Sign:** positive means the arm lags the reference.
- `calibrate_lookahead(spec_at, rung, *, seed=0) -> tuple[float, ...]`. It runs `spec_at(0.0)` (a callable lookahead
  → spec) once on `rung`, returns the per-joint `estimate_lag_s` clipped at ≥ 0, and fills a None joint with the median
  of the others. This is the one-run calibration procedure, the same one a real arm's log would go through.
- New metrics:

  | name | unit | family | better | definition |
  |---|---|---|---|---|
  | `tracking_lag_ms` | ms | tracking | none | 1000 · `estimate_lag_s(log)` (whole arm, signed) |
  | `sync_error_p95_mm` | mm | tracking | lower | the sync window's 95th percentile of the tool's distance from the target point (below) |
  | `sync_timing_ms` | ms | tracking | none | 1000 · mean over the sync window of (target x − tool x) / belt speed: signed, positive = behind the item |

  The sync metrics exist only on a rung whose `params` carry `sync_start_s`, `sync_end_s` and `belt_speed_m_s`, and
  are None elsewhere. The target point at t_i is `log.ref_tool_pos` linearly interpolated in `log.sent_t`. The tool is
  `log.tool_pos`.

### 4.4 Rung variants and one new rung (`bench/rungs.py`)

The main ladders are unchanged except for one rung appended to the JointTrajectory ladder. The variants are functions,
not ladder rungs.

- `rate_variant(rung, rate_hz)`:
  - The rung with `timing.rate_hz` replaced, every other timing field kept.
  - id `<id>·host<rate:g>Hz`; gates = the rung's gates without its two timing gates, plus `timing_gates(rate_hz)`;
    `params["host_rate_hz"] = rate_hz`; everything else identical.
  - `rate_variant(R2, 50)` and `R9.FastFreeSpace·host50Hz` have equal timing, gates (as sets of
    (metric, op, value)) and scripts.
- `speed_variant(rung, move_s)`: for rungs with `params["move_s"]` (R1, R2, U1, U2):
  - The same path at `move_s` per move, `duration_s` = script duration + 0.6 s, id `<id>·move<move_s·1000:g>ms`.
  - Gates = the base rung's.
  - Params gain `move_s`, `peak_joint_speed_deg_s` and `peak_tool_speed_m_s`, computed from the script's reference.
  - `speed_variant(R2, 0.8)`'s script equals R1's; `speed_variant(R2, 0.3)`'s equals R2's.
- `bus_variant(rung, command_delay_s, jitter_s=0.0)`: timing with those fields replaced. id
  `<id>·bus<ms>ms[±<ms>ms]`, with a tuple written as `<ms>/<ms>/…`.
- `sync_variant(belt_speed_m_s)`: the rung below at another belt speed.
- **`R10.BeltSync@0.25m/s`** (order 14, family `sync`, JointTrajectory):
  - A virtual item rides the belt at 0.25 m/s along +x, 50 mm above the belt top (tool centre z = 0.20 m). The belt
    itself is off: there is no contact.
  - The tool path: hold at (0.11, 0.20) for 0.3 s; quintic blend to (0.16, 0.20), from rest to v; cruise to
    (0.36, 0.20) at v (the sync window); quintic blend to (0.40, 0.20), from v to rest; hold 0.3 s.
  - Each blend takes 2 · its distance / v. The scene's `start_q` is the elbow-up IK of the first point.
  - Params: `belt_speed_m_s`, `sync_start_s` and `sync_end_s` (the cruise window).
  - Gates: `sync_error_p95_mm` ≤ 5 ("the tool stays within 5 mm of a point riding the belt"),
    `saturation_share` ≤ 5, `diverged` == 0, and the timing gates. Objective `sync_error_p95_mm`.
  - The study's speeds: 0.10, 0.25, 0.50 and 1.00 m/s.

### 4.5 A UR5e plant (`bench/ur5e.py`, `bench/models/ur5e_bare.xml` + its LICENSE)

- **Model.** The links are the MuJoCo Menagerie UR5e's kinematics and inertials (BSD-3, ROS-Industrial). They are
  vendored from bam_ws `bam_pid_tunning/tests/fixtures/ur5e_bare.xml`: no meshes, no actuators, no contacts.
- **Truth.** Six joints `shoulder_pan_joint … wrist_3_joint`, with the values from bam_pid_tunning `arm_truths.ur5e_arm`:
  armature (0.5, 0.5, 0.5, 0.08, 0.08, 0.08) kg·m², frictionloss (4.0 ×3, 0.8 ×3) N·m, damping (2.0 ×3, 0.3 ×3)
  N·m·s/rad, 4 kHz physics.
- **Home and tool.** Home is (0, −1.2, 1.4, −1.6, −1.57, 0) rad. The tool is a site 0.1 m out along the flange (wrist_3
  body, `0 0.1 0`, Menagerie's attachment site). `Scene.payload_kg` adds a point mass there; `Scene.belt` and pushes
  are refused (ValueError).
- **Boards.** The same moteus board model with these values:
  - torque caps (150 ×3, 28 ×3) N·m (UR5e datasheet);
  - encoder 2^14 counts, and every other `BoardConfig` field at its default;
  - gains at the twin shoulder's bandwidth: `kp_j = M_jj(home) · ωn²`, `kd_j = 2 ζ ωn · M_jj(home)`. Here M is the full
    joint-space mass matrix at home, armature included. ωn = √(300 / M₁₁ᵖˡᵃⁿᵃʳ(home)) and ζ = 7.5 / (2 √(300 · M₁₁ᵖˡᵃⁿᵃʳ(home))),
    from the planar arm's shoulder (39.7 rad/s and 0.496). Unrounded.
- **Feed-forward models** (`ur5e_feedforward(terms)`, pure, as §3.1): `gravity`, `rigid_body` (+ inertia, Coriolis and
  viscous: the nominal model the U rungs fill effort with) and `rigid_body_coulomb` (+ frictionloss · tanh(qd / 0.01)).
  Each is MuJoCo `mj_inverse` on its own private copy of the nominal model (no payload).
- **Feedback models.** `ur5e_mass_matrix()` is the mass matrix by `mj_fullM`. `ctc` is `InertiaScaledPD` with ζ 0.9,
  its bandwidth chosen so that its shoulder_lift kp at home equals the board's shoulder_lift kp (as §3.6 does on the
  planar shoulder).
- `layout_spec(..., arm="ur5e")` resolves the model names on the UR5e. The default stays `"planar_arm"`.
- **Module API** (`robot_controllers.bench.ur5e`): `MODEL_PATH`, `UR5E_JOINTS`, `UR5E_HOME_Q`, `UR5E_TRUTH`,
  `ur5e_feedforward(terms)`, `ur5e_mass_matrix()`, `ur5e_board_config()`, `Ur5ePlant` (the plant interface of
  `PlanarArmPlant`, `name = "ur5e"`) and `ur5e_rungs()`.
- **Rungs.** `Rung.plant` (default `"planar_arm"`) names the plant; `run_episode` builds that plant when no factory is
  passed. Tool metrics use the plant's own forward kinematics (`plant.tool_position(q)`). `ur5e_rungs()` returns the two
  UR5e rungs; they are not in `ladder()`.
  - **Path.** A hold of 0.3 s, then minimum-jerk joint moves W0 → W1 → W2 → W0, with a 0.3 s rest after each:
    - W0 = home;
    - W1 = (0.9, −0.8, 1.0, −1.8, −1.57, 0.6);
    - W2 = (−0.6, −1.5, 1.8, −1.9, −1.2, −0.5).
  - **Effort column.** The nominal `rigid_body` model.
  - **`U1.FreeSpace`:** 1.2 s per move, with R1's gates: tracking ≤ 1°, saturation ≤ 2 %, diverged, faults, timing.
  - **`U2.FastFreeSpace`:** 0.45 s per move, with R2's gates. At 0.45 s the dispatcher's sizing puts the pan joint at
    about 80 % of its cap.
  - The study's move times are (2.0, 1.2, 0.8, 0.6, 0.45, 0.35) s. At 0.35 s the pan joint needs 125 % of its cap: the
    arm's edge, kept on purpose.

### 4.6 The study (`bench/study.py`, CLI `robot-controllers study`, ledger `results/study/results.json`, schema `robot-controllers/study-1`)

**The study layouts** (`STUDY_LAYOUTS`, in this order, by name):

1. `ff=none@none+fb=board_gains@board+la=5.5ms`
2. `ff=trajectory@trajectory+fb=board_gains@board`
3. `ff=trajectory@trajectory+fb=board_gains@board+la=5.5ms`
4. `ff=trajectory@trajectory+fb=board_gains@board+la=5.5ms+interp=secant_mid_ff`
5. `ff=rigid_body@plan+fb=pd@split`
6. `ff=rigid_body@host_measured+fb=ctc@host`
7. `ff=rigid_body@plan+fb=pd@host`

**The arms:** the planar arm (base rungs R2, and R10 for sync) and the UR5e (base U2).

**The dimensions**, one row per (arm, dimension, value, layout, seed):

- **`rate`:** `rate_variant(base, r)` for r ∈ (10, 25, 50, 100, 200, 500, 1000) Hz.
- **`speed`:** `speed_variant(base, m)` at 100 Hz. Planar m ∈ (1.2, 0.8, 0.5, 0.3, 0.2) s; UR5e as §4.5.
- **`rate_speed`:** the grid of both, for layouts 2, 3, 4, 6 and 7.
- **`lookahead`:** layouts 3 and 4 with `lookahead_s` ∈ (0, 1, 2, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 8, 10, 12) ms, on the
  base rung at 100 Hz and at 50 Hz.
- **`jitter`:** layout 4 on `bus_variant(base, 0.0055, j)` for j ∈ (0, 0.5, 1, 2, 3) ms, seeds 0–4.
- **`stagger`:** `bus_variant(base, d)` with `d_j = 1.5 + 2.2·j` ms (a one-adapter bus at ~2.2 ms per board), with
  layout 4 four ways:
  - lookahead 5.5 ms (miscalibrated);
  - the scalar mean of d;
  - per-joint d (the truth);
  - per-joint `calibrate_lookahead` (the one-run procedure; the row also carries the calibrated values).

  Each runs at 50 and 100 Hz.
- **`sync`** (planar only): `sync_variant(v)` for the four speeds, all seven layouts, plus layouts 3 and 4 over the
  lookahead list at 0.25 m/s.

**Each row** carries:

- `arm`, `dimension`, `value`, `base_rung`, `rung` (id), `layout` (name) and `seed`;
- every metric, `passed` and the failed gates' labels;
- `per_joint_lag_ms`.

**The summary** answers per arm, from the rows:

- the rate above which layout 4 is within 10 % of its 1 kHz tracking;
- the lookahead that minimises tracking at each rate;
- the tracking cost of ±1 ms of lookahead error;
- the calibrated-vs-true per-joint lookahead error;
- the sync timing and error per layout per belt speed.

**Running it.** Sequential by default; `--workers` for speed, with the cost numbers flagged as in §3.6. The builder runs
it once and commits the ledger.

### 4.7 Write set

The builder writes:

- `models.py`, `controllers/layout.py`;
- `bench/types.py`, `bench/plant.py`, `bench/board.py` (if needed), `bench/loop.py`;
- `bench/ur5e.py`, `bench/models/` (new), `bench/rungs.py`, `bench/layouts.py`, `bench/study.py` (new), `bench/cli.py`;
- `metrics/lag.py` (new), `metrics/names.py`, `metrics/episode.py`;
- `results/study/`, and their own tests.

**`tests/test_study_acceptance.py` is the dispatcher's**, written before the build and read-only to the builder, as is
`tests/test_layout_acceptance.py`.

## Amendment 5 (2026-11-20 night: bus timing, calibration and drift, the grader and `recommend()`, more arms; a barrier, applied in the order below)

Zach, 2026-11-20:
> "this controller then should register a routine to calibrate the delay … our representation of the CANFD should have a
> data sheet nominal value for the delay and stuff. and then we can compare that against reality and of course if that
> begins to change a lot we should be alerted … we start with like feedback given and feedforward given. And we're
> interested in basically like the layout … vary … How good or bad is the feedforward/feedback we are given, Host Rate,
> Speed of trajectories, Whether its running on host/driver, whether its on plan or live … track performance metrics like
> accuracy, impact … fixed target trajectories … a bunch of different controller/driver configurations (that's the
> population) and we basically want to grade them on all of these scenarios … once you tell me what driver you're using,
> and the state of your FF/FB and the host rate … communication rate, I can tell you exactly what the best controller
> is … perhaps if you want to really optimize for less impact, or better collision detection."

After this amendment the controller question is answered by a lookup, not by a new controller: `recommend(setup,
objective)` over a graded ledger.

### 5.1 The arm registry (`bench/arms.py`, committed by the dispatcher with this amendment)

One `ArmSpec` per arm (`bench/arms.py`: `ArmSpec`, `register`, `arm(name)`, `arms()`, `ARM_MODULES`), so every later
section is arm-agnostic. Each arm's module builds its spec and calls `register` at import; `ARM_MODULES` lists them in
display order. The planar arm registers from a new `bench/arm_planar.py` (wrapping `arm.py` / `arm_models.py`), the UR5e
from `bench/ur5e.py`. `layout_spec(..., arm=)`, `run_episode`'s plant choice (`Rung.plant`) and the study read the
registry. Planar and UR5e move onto it with no behaviour change: both acceptance files and both ledgers still reproduce.
`ArmSpec.feedforward(name, model_error)` and `ArmSpec.mass_matrix(model_error)` take a `ModelError` (§5.3); `None` and
`ModelError()` are the nominal model exactly.

### 5.2 Bus timing: nominal, calibrated, monitored (`timing.py`, `metrics/lag.py`, `monitor.py`)

- **`BusTiming`** (`timing.py`, committed with this amendment): `command_delay_s` (one value or one per joint),
  `reply_delay_s`, `jitter_s`, `source`. `BusTiming.from_datasheet(exchange_time_s, devices, *, slot_order,
  command_latency_s, reply_delay_s, source)`: joint j's command lands `command_latency_s + exchange_time_s · (slot_j + 1) /
  devices` after the exchange starts. Plain numbers in: robot_controllers never imports robotparts; the caller passes
  robotparts' datasheet values (the accessor goes in the docstring once the robotparts lane lands it).
  `BusTiming.from_calibration(lags_s, source=)`. Lane A adds `HostTiming.from_bus(bus, rate_hz)` and
  `lookahead_from(bus, count) -> tuple[float, ...]`.
- **The calibration routine** (`calibration.py`: `calibrate_bus(spec_at, rung, *, nominal: BusTiming, method="xcorr",
  lookaheads_s=None, tolerance_s=0.002, seed=0) -> Calibration`):
  - `method="xcorr"`: one run at zero lookahead, `estimate_lag_s(per_joint=True)` → the measured per-joint delay.
    Needs the log's send times to be the real send times (the twin's are; the real arm's logger today stamps the
    exchange's *end*, so on real logs this bounds the delay only to one exchange, about 17 ms: §5.9);
  - `method="sweep"`: one run per scalar lookahead in `lookaheads_s` (default: the nominal's mean + {−3, −1.5, 0, +1.5,
    +3} ms); measured = the lookahead with the lowest `tracking_p95_deg` (scalar, applied to every joint). Works on any
    log the bench can score, real ones included;
  - returns `Calibration(measured: BusTiming, nominal: BusTiming, delta_s: tuple, within: bool, method, why: str)`,
    `within` = every |delta| ≤ `tolerance_s`.
  WHY a routine and not only live: a clean run gives the delay at once; the monitor then only has to notice a change.
- **Two numbers, not one (dispatcher, after lane A).** On a slow servo loop the joint's own friction-driven lag reads as
  delay: UR5e wrists track best looking 10–14 ms ahead on a 5.5 ms bus. So a calibration reports both:
  - **the bus delay** (`calibrate_bus(..., friction=True)`): what gets compared with the datasheet and watched by the
    monitor, because the datasheet says nothing about friction;
  - **the best lookahead** (the plain fit, the default): what the controller samples ahead by, because it tracks best
    (UR5e 0.225° vs 0.338° at the true 5.5 ms).
  ~~The population's "calibrated" lookahead (§5.5) is the best lookahead~~ **Superseded by the first full grade
  (2026-11-20, lane B2, 1296 rows):** at 50 Hz with the secant velocity, the plain fit over-advances and the friction-aware
  bus delay tracks better on 3 of 4 arms (P4s base, tool p95: planar 1.58 → 1.36 mm, UR5e 4.65 → 1.80 mm, pendulum 15.98 →
  11.61 mm; five-bar a wash, 12.05 vs 12.08 mm). Friction lag is not a pure delay, so sampling ahead by it adds error
  elsewhere. So the population's "calibrated" lookahead is the **friction-aware bus delay**; every grade row still records
  both. Known limit: on the pendulum the joint's own lag swamps a 1.25 ms bus (xcorr reads 25 ms plain, 19.75 ms friction-
  aware; the true-bus lookahead tracks 5.48 mm), so a single high-friction joint should be calibrated by the sweep.
- **The live monitor** (`monitor.py`: `LagMonitor(expected: BusTiming, *, lookahead_s=0.0, window_s=2.0, step_s=0.5,
  alert_s=0.002, confirm=2)`): pure and streaming. `update(t, q, sent_t, ref_position) -> list[LagEvent]` takes the new
  plant samples (t, q) and the new host ticks (sent_t, ref_position) since the last call. Every `step_s` it estimates each
  moving joint's lag over the last `window_s`; the measured *delay* is that lag plus the joint's lookahead in use. It emits
  `LagEvent(t, joint, expected_s, measured_s, kind="drift"|"recovered")` when |measured − expected| > `alert_s` for
  `confirm` windows in a row (and "recovered" when it comes back). A joint that did not move in the window (< 0.1° peak
  to peak) is skipped, never alerted on. The monitor never changes the controller: it reports. (Re-calibrating
  automatically is a later decision; an alert is what Zach asked for.)
- **A changing bus in the twin:** `HostTiming.command_delay_steps: tuple[(t_s, delay_s | tuple), ...]` (default ()):
  from each t on, commands sent use that delay. Defaults reproduce today.

### 5.3 What the controller is given, and how good it is (`models.py`, committed with this amendment)

- **`ModelError(scale, seed, friction_scale)`**: each link's mass and inertia in the *model* × `link_factors(count)` (1 ±
  scale, a fixed sign pattern from `seed`), friction × `friction_factor`. On a MuJoCo arm the links are the bodies that
  carry a joint, in model order (`body_mass`, `body_inertia`), friction is `dof_frictionloss` and `dof_damping`, and the
  rotor `armature` stays as it is (it's the drive's, not the link's). The planar arm's links are link 1 and link 2 (the
  tool's mass rides with link 2). The plant (the truth) never changes: only what
  the controller is told. Every arm applies it to its own model (closed form or a private MuJoCo copy), links in its body
  order.
- **Feedback quality:** `gain_scale` multiplies the arm's board kp and kd (`scaled_board(board, gain_scale)`, lane A).
- **`DriverCaps`** (`board_pd`, `per_tick_gains`, `feedforward_field`, `max_rate_hz`; `can_run(layout)`), `MOTEUS` =
  all of it.

### 5.4 Fixed targets and scenarios (`bench/targets.py`, `bench/scenarios.py`)

- **`Target`** (`bench/targets.py`): `id` (`<arm>/path@slow|nominal|fast`, `<arm>/calibration`, `<arm>/collision`),
  `arm`, `kind` ∈ {path, calibration, collision}, `speed`, `rung() -> Rung` (its plant, its script, gates and duration;
  100 Hz host by default) and `params` (peak joint and tool speed). The three path targets are one frozen path time-scaled.
  `calibration` moves every joint at least 5° (a smooth multi-joint sine sweep). `collision` is the nominal path with a
  hard stop (a joint range limit) on its most-moving joint, half way along that joint's travel. Lane A gives planar and
  UR5e their targets (R2's path and U2's path at three speeds); lane B1 gives its two arms theirs.
- **`Scenario`** (`bench/scenarios.py`, frozen): `arm`, `target` (a target id), `host_rate_hz`, `bus: BusTiming`,
  `model_error: ModelError`, `gain_scale`, `axis` (which grid axis it belongs to, "base" for the base) and `rung() ->
  Rung`. Its `id` is readable, e.g. `bam_fivebar/path@nominal·50Hz·bus=stagger±1ms·ff±10%·gains×1`.
  `base_scenario(arm)` and `grid(arm) -> tuple[Scenario, ...]` (the base first).
- **The base scenario per arm** (`axis="base"`): `path@nominal`, 50 Hz (the rig), the datasheet bus, `ModelError(0.1)`,
  gains ×1. The datasheet bus is `BusTiming.from_datasheet(per_device_s × devices, devices, slot_order=joint order,
  source="datasheet:…")` with devices = the arm's joint count and `per_device_s` the fdcanusb's datasheet exchange time
  per device from robotparts (the robotparts lane names the accessor; the source string names its part id and release).
  Until robotparts carries a sourced number, `per_device_s` = 1.25 ms (mjbots: fdcanusb about 800 exchanges/s with one
  moteus) with `source="datasheet:mjbots (fdcanusb ~800 Hz, one device)"`.
- **The grid** (`axis` names in brackets; one axis at a time around the base, plus one 2-D grid):
  - [speed] slow, fast;
  - [rate] 25, 100, 200 Hz;
  - [model_error] 0, 30 %;
  - [gains] ×0.5, ×2;
  - [bus] ideal (5.5 ms every joint), the rig as measured (2.2 ms per board in turn: seven boards in ~15.5 ms), the rig
    + ±1 ms jitter;
  - [event] `collision`; and, planar only, the belt contact R3 at 0.10 m/s;
  - [rate_error] the 2-D grid host rate {25, 50, 100, 200} × model error {0, 10, 30 %} at nominal speed (what
    `recommend` interpolates over).

### 5.5 The population (`bench/population.py`)

`POPULATION`: the 16 config ids in the table's order (`P1`, `P2`, `P2s`, `P3`, `P3s`, …, `P7s`, `P8`, `P9`, `P10`; the `s`
suffix = `secant_mid_ff`), `REAL_SAFE = ("P2", "P4", "P4s", "P5s")`, and `config_spec(config_id, scenario, calibration=None)
-> ControllerSpec`, which builds the `layout_spec` on the scenario's arm with that scenario's model error and gains.
Every config is a `layout_spec` with one more knob, `lookahead_mode` ∈ {"none", "nominal", "calibrated"} (nominal =
the scenario's datasheet `BusTiming`; calibrated = `calibrate_bus` on that scenario's arm, bus and rate, run once and
cached per scenario). The population (each with `interpolation` ∈ {tangent, secant_mid_ff} where it applies):

| # | feed-forward | feedback | lookahead |
|---|---|---|---|
| P1 | none | board PD | calibrated |
| P2 | the trajectory's column (as built) | board PD | none |
| P3 | model at plan time | board PD | nominal |
| P4 | model at plan time | board PD | calibrated |
| P5 | model + Coulomb at plan time | board PD | calibrated |
| P6 | model at the host, measured state | board PD | calibrated |
| P7 | model at plan time | computed-torque gains re-sent each tick | calibrated |
| P8 | model at plan time | host spring + board damper | none |
| P9 | model at the host, measured state | computed torque on the host | none |
| P10 | model at plan time | PD on the host | none |

×2 interpolation for P2–P7 → 16 configs. ~~The real arms run only P2, P4 (tangent and secant) and P5 (secant)~~
**Amended after the grade (2026-11-20):** the real arms run every config whose feedback the board closes and that has a
feed-forward: P2–P7 in both interpolations (`REAL_SAFE`). Never P1 (no feed-forward: the arm sags), P8, P9 or P10 (feedback
on the host, which saturated the boards at 50 Hz in every sim). WHY widened: the first list was a guess; the grade's best
accuracy on three arms was P7 / P7s (gains re-sent each tick, still closed by the board; moteus takes kp/kd scales with
every command) and on the pendulum P3s, neither of which the guess allowed.

### 5.6 The metrics and objectives

- Existing: `tracking_p95_deg`, `tool_p95_mm`, `saturation_share`, `fb_share_lf_pct`, `host_duty_pct`,
  `wire_bytes_per_s`, `contact_peak_n` (planar belt), `tracking_lag_ms`.
- **Collision** (on the collision target only; units: ms, N·m, and "" for the margin): the residual is the feedback torque (board PD term + host feedback) per
  joint over its cap, low-passed at 50 Hz, max over joints. The threshold is 1.25 × the same config's max residual on the
  same scenario *without* the stop.
  - `collision_detect_ms`: first contact → first threshold crossing (None = never: a fail);
  - `collision_peak_nm`: the stop's largest constraint torque before detection (the impact);
  - `collision_margin`: max residual during contact / threshold (higher = easier to tell apart).
  WHY the controller's own nominal run sets the threshold: a better model leaves less residual in free motion, so the
  same stop stands out sooner; that's the link between feed-forward quality and collision detection.
- **Objectives** (`OBJECTIVES`): `accuracy` (tool_p95_mm, then tracking_p95_deg), `impact` (collision_peak_nm, planar
  also contact_peak_n), `collision_detection` (collision_detect_ms, then collision_margin), `cost` (host_duty_pct, then
  wire_bytes_per_s), or a dict of weights over normalised ranks. Gates as today (a config that fails a gate in a
  scenario can't be recommended for it).

### 5.7 The grader and `recommend()` (`grade.py`, CLI `robot-controllers grade` / `robot-controllers recommend`)

- `grade.run_grade(arms=None, *, workers=2)` → `results/grade/results.json` (schema `robot-controllers/grade-1`): `rows`
  (arm, scenario id, axis and fields, config id, metrics, passed, failed gates, the calibration used, `source`:
  "sim" | "real") and `summary[arm]`: the best config per (scenario, objective) and a per-config grade card (how often it
  is best or within 10 %).
- `grade.run_scenario(config_id, scenario) -> dict`: one ledger row (the grader's unit; calibrates first when the config
  needs it, caching per (arm, bus, rate)).
- **`recommend.recommend(setup: Setup, objective="accuracy") -> Recommendation`**:
  - `Setup(arm, host_rate_hz, bus: BusTiming, feedforward: str, model_error: float, driver_caps=MOTEUS, gain_scale=1.0,
    speed="nominal", joint_count=None)` (`joint_count` lets an arm the ledger has never seen be matched);
  - returns `Recommendation(controller: ControllerSpec | None, config: str | None, expected: dict, runner_up, why: str,
    evidence: list)`: the nearest graded scenario on (arm, rate, error, speed, bus), the best passing config the caps
    allow, its metrics there, and one plain-English reason; `controller=None` with the reason when no config fits;
  - an arm not in the ledger → the nearest arm by joint count, and `why` says "nearest".
- CLI: `robot-controllers recommend --arm bam_fivebar --rate 50 --ff-error 0.1 --objective accuracy` prints the answer.

### 5.8 Two more arms: BAM's five-bar and the 1-DOF pendulum (lane B1; additive files only)

Vendored the way the UR5e was: a bare MJCF, the identified constants, a plant on the same board model, the models.
Sources (read them; copy constants, cite file:line in WHY comments):

- **`bam_fivebar`** (`bench/fivebar.py`, `bench/models/bam_fb_v2_bare.xml`): joints `j1 j2 c1 w1 w2 w3` (CAN 11–16).
  - The model: bam_pid_tunning's `plants/bam_fb_v2.py` `reduced_mjcf()` output (bundle 09374d7, geoms removed), the
    five-bar closed by its one `connect` equality (passive `revolute_8` / `revolute_34`), stiffened as `ArmSim` does
    (`plants/mujoco_arm.py`: solref 0.002). The plant commands and logs the six actuated joints only.
  - Truth per drive (`bam_fb_v2.py` DRIVES, lines ~97–129): armature, Coulomb, caps (12.5 N·m five-bar, 1.02 wrist),
    as-found board gains (kp 200/144/200, kd 4/8/4; wrist kp 75, kd 0.75; ki is not modelled), encoder 163840
    counts/rev, gear 10. The twin's bus: 5.5 ms down, 1.5 ms up (`bam_fivebar/plant.py` RACK_BOARD) as the twin
    nominal; the datasheet stagger for the real bus comes from §5.2.
  - Feed-forward: closed-chain inverse dynamics on the actuated joints (τ_act from MuJoCo's `mj_inverse` on the full
    coordinates with the closure solved for the passive joints, as `bam_arm_fb_sim/kin.py` does; least squares over τ
    and λ as `ArmSim` does for gravity). `gravity`, `rigid_body`, `rigid_body_coulomb`; `ModelError` on the private copy.
- **`pendulum`** (`bench/pendulum.py`, its MJCF from `bam_rack_twin/twin.py` `bare_hinge_xml`): one joint `can11`,
  board frame, the twin's truth (`arm_motion_bench/baselines/pendulum_sim.py` ~84–89: G 4.42266, bottom 0.2578, J 0.12011,
  Coulomb 1.12, viscous 0.0874, torque scale 1.04), cap 12.6 N·m, board kp 400 / kd 8 (deployment record), encoder 14-bit
  through 10:1. Feed-forward: `pendulum_ff.FF_MODEL` (G 4.113, bottom 0.253, J 0.1057, Coulomb 0.612·tanh(qd/0.05),
  viscous 0.0814) as `rigid_body_coulomb`; `rigid_body` without the Coulomb term; `gravity` the G term alone.
- **Targets** for both are copies of arm_motion_bench's frozen tapes (the BAM FB workspace v2 `dev000` path; the pendulum
  v2 task the real showcase ran), resampled onto the bench's grid, so a real log of the same tape compares one to one.
- Both register through `ARM_MODULES` (lane B1 adds its two entries; nothing else in the registry changes).

### 5.9 Real runs: only through the rig's own owner

robot_controllers never drives hardware. Lane B2 writes `real/requests.py` (export, per real station, a run request in
arm_motion_bench's bundle terms: the target, the configs, the rate) and `real/import_fullarm.py` (read a `fullarm_jtc` run
folder + its bundle's `samples.npz` into an `EpisodeLog`: q in the plan's joint frame through the report's `board_map`,
ref = the bundle's `q_reference` on the JTC's clock; `import_fullarm_run(run_dir, bundle_dir) -> (EpisodeLog, Rung)`), so
the metrics and the sweep calibration run on real logs
unchanged (`source: real` rows in the grade ledger).

- **Only the board-side configs run on real** (P2, P4 tangent and secant, P5): every host-side feedback law saturated
  the boards at 50 Hz in every sim.
- **The logger needs per-board times for one-run calibration.** The first real log analysed (2026-11-20, `fullarm_jtc`,
  model FF + 5.5 ms lookahead, 0.5 speed, 17.0 ms ticks) bounds the residual lag only to the exchange: −6 to +11 ms,
  because each row is stamped at the exchange's end and the boards answer at different points inside it. Ask: stamp the
  exchange start and the board order (or each reply's time). Until then the real calibration is the sweep.

### 5.10 Lanes, order and write sets

- **Lane A (the barrier, first, robot_controllers `main`)**: §5.1 wiring (`bench/arm_planar.py`, registry use in
  `layouts.py`, `loop.py`, `study.py`, `ur5e.py`), §5.2 (`calibration.py`, `monitor.py`, `HostTiming.command_delay_steps`,
  `HostTiming.from_bus`, `lookahead_from`, `plant.py`), §5.3 per-arm `ModelError` for planar and UR5e (`arm_models.py`,
  `ur5e.py`), `scaled_board`, CLI `robot-controllers calibrate`; and the `Target` type (`bench/targets.py`) with the
  planar and UR5e targets, including the collision target's hard stop (a `Scene` option every plant honours:
  `stops: tuple[(joint_index, angle_rad, side), ...]`, a stiff joint-range limit on one side). Acceptance:
  `tests/test_timing_acceptance.py`.
- **Lane B1 (in parallel with A, its own worktree, additive files only)**: §5.8 (`bench/fivebar.py`, `bench/pendulum.py`,
  `bench/models/*`), plus their two `ARM_MODULES` lines at merge. Acceptance: `tests/test_arms_acceptance.py`.
- **Lane B2 (after A lands)**: §5.4–5.7 and §5.9 (`bench/scenarios.py`, `bench/population.py`,
  `metrics/collision.py`, `metrics/names.py`, `grade.py`, `recommend.py`, `real/`, CLI `grade` / `recommend`,
  `results/grade/`). Runs the grade once every arm is registered. Acceptance: `tests/test_grade_acceptance.py`.
- The three acceptance files, `models.py`'s `ModelError` / `DriverCaps`, `timing.py` and `bench/arms.py` are the
  dispatcher's, committed before the build. Builders extend `timing.py` / `arms.py` only where this section says so.
- **The arm-powered rule binds every lane** (arm_motion_bench CONTRACT v3.8): while
  `/home/bam/arm_motion_bench/outputs/ARM_POWERED` exists, or `rackq list` shows a job RUNNING, no pytest pool, sweep or
  grade starts; finish the step in hand at 1 worker and wait. Always `taskset -c 2-5,16,17`, at most 2 workers.

