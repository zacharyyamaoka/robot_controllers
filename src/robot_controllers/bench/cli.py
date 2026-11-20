"""`robot-controllers` command line.

    robot-controllers run [--box SLUG] [--controller NAME ...] [--rung ID ...] [--seeds 0 1] [--workers N]
    robot-controllers wandb [--views] [--verify] [--weave] [--allow-dirty] [--dry-run]
    robot-controllers layouts [--ff MODEL ...] [--fb MODEL ...] [--rung ID ...] [--seeds 0] [--workers N]
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
    layouts.add_argument("--seeds", type=int, nargs="+", default=[0])
    layouts.add_argument("--workers", type=int, default=1, help="processes (default 1: uncontended costs)")

    args, rest = parser.parse_known_args(argv)
    if args.command == "wandb":
        from robot_controllers.tracking.wandb_log import main as wandb_main

        return wandb_main(rest) or 0
    if rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    if args.command == "layouts":
        from robot_controllers.bench import layouts as sweep

        document = sweep.run_layouts(
            ff_models=args.ff or sweep.FEEDFORWARD_MODELS,
            fb_models=args.fb or sweep.FEEDBACK_MODELS,
            rungs=args.rung or sweep.SWEPT_RUNGS,
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


if __name__ == "__main__":
    sys.exit(main())
