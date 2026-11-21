"""`robot-controllers` command line.

    robot-controllers run [--box SLUG] [--controller NAME ...] [--rung ID ...] [--seeds 0 1] [--workers N]
    robot-controllers wandb [--views] [--verify] [--weave] [--allow-dirty] [--dry-run]
    robot-controllers layouts [--ff MODEL ...] [--fb MODEL ...] [--rung ID ...] [--lookahead-ms 0 5.5]
                              [--seeds 0] [--workers N]
    robot-controllers study [--arm planar_arm|ur5e ...] [--dimension rate|speed|... ...] [--workers N]
    robot-controllers calibrate [--arm planar_arm] [--target calibration] [--rate 100] [--method xcorr|sweep]
                                [--bus-ms 2 8] [--jitter-ms 0] [--nominal-ms 5.5] [--tolerance-ms 2] [--json]
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="robot-controllers")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the ladder and write results/<box>/results.json")
    run.add_argument("--box", action="append", help="box slug (default: every box with a ladder)")
    run.add_argument("--controller", action="append", help="controller name (default: every built one)")
    run.add_argument("--rung", action="append", help="rung id (default: the whole ladder)")
    run.add_argument("--seeds", type=int, nargs="+", default=[0])
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--no-trace", action="store_true")

    sub.add_parser("wandb", help="log results to W&B (one project per box)", add_help=False)

    layouts = sub.add_parser(
        "layouts", help="sweep every (feed-forward, feedback) layout and write results/layouts/results.json"
    )
    layouts.add_argument("--ff", action="append", help="feed-forward model (default: every one)")
    layouts.add_argument("--fb", action="append", help="feedback model (default: every one)")
    layouts.add_argument("--rung", action="append", help="rung id (default: R1 R2 R4 R6 R8 and both R9)")
    layouts.add_argument("--lookahead-ms", type=float, nargs="+", help="default: 0 and the bus delay")
    layouts.add_argument("--seeds", type=int, nargs="+", default=[0])
    layouts.add_argument("--workers", type=int, default=1, help="processes (default 1: uncontended costs)")

    study = sub.add_parser(
        "study",
        help="rate, speed, lookahead, jitter, stagger and sync on both arms -> results/study/results.json",
    )
    study.add_argument("--arm", action="append", help="planar_arm or ur5e (default: both)")
    study.add_argument("--dimension", action="append", help="rate, speed, rate_speed, lookahead, jitter, "
                       "stagger, sync (default: all)")
    study.add_argument("--workers", type=int, default=1, help="processes (default 1: uncontended costs)")

    calibrate = sub.add_parser(
        "calibrate",
        help="measure the twin's bus delay with the calibration routine and compare it with the datasheet",
    )
    calibrate.add_argument("--arm", default="planar_arm", help="a registered arm (default planar_arm)")
    calibrate.add_argument("--target", default="calibration",
                           help="the arm's target to run: calibration (default), path@nominal, path@fast, ...")
    calibrate.add_argument("--rate", type=float, help="host rate, Hz (default: the target's, 100)")
    calibrate.add_argument("--method", choices=("xcorr", "sweep"), default="xcorr")
    calibrate.add_argument("--bus-ms", type=float, nargs="+",
                           help="the twin's true command delay, one value or one per joint (default: the target's)")
    calibrate.add_argument("--jitter-ms", type=float, default=0.0, help="the twin's bus jitter, +- ms")
    calibrate.add_argument("--nominal-ms", type=float, nargs="+",
                           help="the nominal to compare with (default: the fdcanusb datasheet stagger, §5.4)")
    calibrate.add_argument("--tolerance-ms", type=float, default=2.0)
    calibrate.add_argument("--friction", action="store_true",
                           help="xcorr: fit the dry friction's look-alike out (the bus, not the best lookahead)")
    calibrate.add_argument("--seed", type=int, default=0)
    calibrate.add_argument("--json", action="store_true", help="print the Calibration as JSON")

    args, rest = parser.parse_known_args(argv)
    if args.command == "wandb":
        from robot_controllers.tracking.wandb_log import main as wandb_main

        return wandb_main(rest) or 0
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.command == "calibrate":
        return _calibrate(args)
    if args.command == "study":
        from robot_controllers.bench import study as study_module

        document = study_module.run_study(
            arms=args.arm or study_module.ARMS,
            dimensions=args.dimension or study_module.DIMENSIONS,
            workers=args.workers,
        )
        path = study_module.write_study(document)
        generated = document["generated"]
        print(f"wrote {path}: {generated['episodes']} episodes in {generated['wall_s']} s")
        for arm, summary in document["summary"].items():
            rate = summary.get("rate") or {}
            if rate:
                print(f"  {arm}: {rate['layout']} - {rate['why']}")
            for answer in summary.get("lookahead", []):
                print(
                    f"  {arm}: {answer['layout']} at {answer['rate_hz']:g} Hz: best lookahead "
                    f"{answer['best_lookahead_ms']} ms ({answer['best_tracking_p95_deg']:.3g} deg), "
                    f"+-1 ms costs {answer['cost_of_1ms_error_deg']}"
                )
        return 0
    if args.command == "layouts":
        from robot_controllers.bench import layouts as sweep

        document = sweep.run_layouts(
            ff_models=args.ff or sweep.FEEDFORWARD_MODELS,
            fb_models=args.fb or sweep.FEEDBACK_MODELS,
            rungs=args.rung or sweep.SWEPT_RUNGS,
            lookaheads_s=[ms / 1e3 for ms in args.lookahead_ms] if args.lookahead_ms else sweep.LOOKAHEADS_S,
            seeds=args.seeds,
            workers=args.workers,
        )
        path = sweep.write_layouts(document)
        summary = document["summary"]["layouts"]
        print(f"wrote {path}; ctc bandwidth {summary['ctc']['bandwidth_hz']:.4g} Hz")
        for pair in summary["per_pair"]:
            pick = pair["recommendation"]
            pair_name = f"ff={pair['feedforward_model']} fb={pair['feedback_model']}"
            print(f"  {pair_name}: {pick['name']} - {pick['why']}")
        print(f"overall: {summary['overall']['name']} - {summary['overall']['why']}")
        return 0

    from robot_controllers.bench.run import boxes_with_ladders, run_box, write_results
    from robot_controllers.registry import box_by_slug

    boxes = [box_by_slug(s) for s in args.box] if args.box else boxes_with_ladders()
    for box in boxes:
        document = run_box(box, controllers=args.controller, rungs=args.rung, seeds=args.seeds,
                           workers=args.workers, trace=not args.no_trace)
        path = write_results(document)
        recommendation = document["summary"].get("recommendation") or {}
        print(f"{box.name}: wrote {path}; recommendation: {recommendation.get('controller')} - {recommendation.get('why')}")
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    """`robot-controllers calibrate`: the v1 recipe (the trajectory's effort, board PD, secant_mid_ff) on the arm's
    target, on a twin whose bus is known, through `calibration.calibrate_bus` (CONTRACT §5.2)."""
    import dataclasses
    import json

    from robot_controllers.bench.arms import arm
    from robot_controllers.bench.layouts import layout_spec
    from robot_controllers.bench.rungs import bus_variant, rate_variant
    from robot_controllers.calibration import calibrate_bus, datasheet_bus
    from robot_controllers.timing import BusTiming

    spec_arm = arm(args.arm)
    targets = {t.id: t for t in spec_arm.targets()}
    target_id = args.target if "/" in args.target else f"{args.arm}/{args.target}"
    if target_id not in targets:
        print(f"no target {target_id!r}; {args.arm} has {', '.join(sorted(targets))}", file=sys.stderr)
        return 2
    rung = targets[target_id].rung()
    if args.rate is not None and args.rate != rung.timing.rate_hz:
        rung = rate_variant(rung, args.rate)
    if args.bus_ms is not None or args.jitter_ms:
        delays = rung.timing.command_delay_s if args.bus_ms is None else tuple(v / 1e3 for v in args.bus_ms)
        delays = delays[0] if isinstance(delays, tuple) and len(delays) == 1 else delays
        rung = bus_variant(rung, delays, args.jitter_ms / 1e3)
    joints = len(spec_arm.joint_names)
    if args.nominal_ms is None:
        nominal = datasheet_bus(joints)
    else:
        values = tuple(v / 1e3 for v in args.nominal_ms)
        nominal = BusTiming(values[0] if len(values) == 1 else values, source="assumed:--nominal-ms")

    def spec_at(lookahead: float | tuple[float, ...]) -> object:
        return layout_spec("trajectory", "trajectory", "board_gains", "board", lookahead_s=lookahead,
                           interpolation="secant_mid_ff", arm=args.arm)

    result = calibrate_bus(spec_at, rung, nominal=nominal, method=args.method,
                           tolerance_s=args.tolerance_ms / 1e3, seed=args.seed, friction=args.friction)
    if args.json:
        print(json.dumps(dataclasses.asdict(result), indent=2, default=str))
        return 0
    truth = rung.timing.command_delays(joints)
    print(f"{rung.id} at {rung.timing.rate_hz:g} Hz, {args.method}; nominal = {nominal.source}")
    print(f"  {'joint':<22}{'nominal':>10}{'measured':>10}{'delta':>10}{'twin bus':>10}")
    measured = result.measured.per_joint(joints)
    for j, name in enumerate(spec_arm.joint_names):
        print(f"  {name:<22}{1e3 * nominal.per_joint(joints)[j]:>8.2f}ms{1e3 * measured[j]:>8.2f}ms"
              f"{1e3 * result.delta_s[j]:>+8.2f}ms{1e3 * truth[j]:>8.2f}ms")
    for lookahead, p95 in result.curve:
        print(f"  sweep: lookahead {1e3 * lookahead:5.2f} ms -> tracking p95 {p95:.4g} deg")
    print(("WITHIN: " if result.within else "OUTSIDE: ") + result.why)
    return 0


if __name__ == "__main__":
    sys.exit(main())
