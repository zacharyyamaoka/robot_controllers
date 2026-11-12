"""Build the robot_controllers site into dist/: every page is generated from the registry and the results ledger.

    env -u VIRTUAL_ENV uv run --frozen --project /home/bam/robot_controllers --extra site python /home/bam/robot_controllers/website/build/build.py
    ... build.py --results-dir /home/bam/robot_controllers/website/tests/fixtures/results   # the synthetic fixture

--results-dir defaults to the repo's results/; when it holds nothing, every page says "not run yet".
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model  # noqa: E402
import pages  # noqa: E402
import plots  # noqa: E402

DIST = ROOT / "dist"
SITE = ROOT / "site"
ASSETS = ("site.css", "site.js", "rc.css")


class Writer:
    def __init__(self, dist: Path) -> None:
        self.dist = dist

    def __call__(self, path: str, content: str | bytes) -> None:
        target = self.dist / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            target.write_bytes(content)
        else:
            target.write_text(content, encoding="utf-8")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the robot_controllers site.")
    parser.add_argument("--results-dir", type=Path, default=model.DEFAULT_RESULTS_DIR,
                        help="folder holding <box-slug>/results.json (default: the repo's results/)")
    parser.add_argument("--dist", type=Path, default=DIST, help="output folder (default: website/dist)")
    # WHY no default origin: the site has no domain yet; a sitemap needs absolute URLs, so it is written only
    # when one is given, instead of inventing a domain.
    parser.add_argument("--origin", default="", help="absolute origin for sitemap.xml, e.g. https://example.com")
    parser.add_argument("--no-examples", action="store_true", help="skip running the per-controller examples")
    return parser.parse_args(argv)


def build_plots(boxes: list[model.BoxInfo], write: Writer) -> dict[str, dict]:
    """Every run figure and each box's Pareto view, as /assets/plots/... paths (and pixel sizes)."""
    files: dict[str, dict] = {}
    for box in boxes:
        mine: dict = {}
        if not box.has_results:
            files[box.slug] = mine
            continue
        joints = ["shoulder", "elbow"]
        for info in box.built:
            for rung in box.rungs:
                run = info.runs.get(rung.id)
                if run is None:
                    continue
                try:
                    figure = plots.run_figure(run, rung, joints)
                except Exception as error:  # noqa: BLE001  one bad trace must not take the site down
                    model.warn(f"{info.name} @ {rung.id}: plot failed ({type(error).__name__}: {error})")
                    continue
                if figure is None:
                    continue
                path = f"/assets/plots/{box.slug}/{info.name}/{rung.anchor}.webp"
                write(path, figure[0])
                mine[f"{info.name}|{rung.id}"] = path
                mine[f"{info.name}|{rung.id}|size"] = figure[1]
        try:
            pareto = plots.pareto_figure(box)
        except Exception as error:  # noqa: BLE001
            model.warn(f"{box.slug}: Pareto plot failed ({type(error).__name__}: {error})")
            pareto = None
        if pareto:
            path = f"/assets/plots/{box.slug}/pareto.webp"
            write(path, pareto[0])
            mine["pareto"], mine["pareto_size"] = path, pareto[1]
        files[box.slug] = mine
    return files


def main(argv: list[str] | None = None) -> list[model.BoxInfo]:
    args = parse_args(argv)
    dist = args.dist.resolve()
    if dist.exists():
        shutil.rmtree(dist)
    shutil.copytree(SITE, dist / "assets")
    write = Writer(dist)
    digest = hashlib.sha256()
    for name in ASSETS:
        digest.update((SITE / name).read_bytes())
    ctx = {"asset_version": digest.hexdigest()[:8], "stamp": model.source_stamp()}

    results_dir = args.results_dir if args.results_dir and Path(args.results_dir).is_dir() else None
    boxes = model.collect(results_dir, run_examples=not args.no_examples)
    plot_files = build_plots(boxes, write)

    write("index.html", pages.home(boxes, **ctx))
    write("rungs/index.html", pages.rungs_page(boxes, **ctx))
    write("metrics/index.html", pages.metrics_page(boxes, **ctx))
    write("404.html", pages.not_found(boxes, **ctx))
    urls = ["/", "/rungs/", "/metrics/"]
    for box in boxes:
        write(f"{box.path}index.html", pages.box_page(box, boxes, plot_files.get(box.slug, {}), **ctx))
        urls.append(box.path)
        for info in box.controllers:
            write(f"{info.path}index.html", pages.controller_page(info, box, boxes, plot_files.get(box.slug, {}), **ctx))
            urls.append(info.path)

    origin = args.origin.rstrip("/")
    if origin:
        write("robots.txt", f"User-agent: *\nAllow: /\nSitemap: {origin}/sitemap.xml\n")
        entries = "".join(f"<url><loc>{origin}{u}</loc></url>" for u in urls)
        write("sitemap.xml", '<?xml version="1.0" encoding="UTF-8"?>\n'
                             f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{entries}</urlset>\n')
    else:
        write("robots.txt", "User-agent: *\nAllow: /\n")
    size = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file())
    with_results = [b.slug for b in boxes if b.has_results]
    print(f"built {len(urls) + 1} pages ({sum(len(b.controllers) for b in boxes)} controller pages), "
          f"results for {with_results or 'no box'} from {results_dir or 'nowhere'}, dist {size / 1024:.0f} KiB at {dist}")
    for message in model.WARNINGS:
        print(f"  note: {message}")
    return boxes


if __name__ == "__main__":
    main()
