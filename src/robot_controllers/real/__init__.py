"""Real runs, only through the rig's own owner (CONTRACT amendment 5, §5.9): robot_controllers never drives hardware.

- `requests`: per real station, a run request in arm_motion_bench's bundle terms (the target, the configs, the rate)
  for the rig's owner to build and queue.
- `import_fullarm`: a `fullarm_jtc` run folder + its bundle as an `EpisodeLog` and a `Rung`, so the metrics and the
  sweep calibration run on real logs unchanged (`source: real` rows in the grade ledger).
"""
