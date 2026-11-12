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

