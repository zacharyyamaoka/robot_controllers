"""Serve dist/ locally like Cloudflare does: the _headers CORS rule on /assets/viz/ included.

    python3 /home/bam/robot_controllers/website/build/serve.py 4494
    python3 /home/bam/robot_controllers/website/build/serve.py 4494 --directory /abs/path/to/dist
"""

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DIST = Path(__file__).resolve().parent.parent / "dist"


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        if self.path.startswith("/assets/viz/"):
            self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("port", nargs="?", type=int, default=4494)
    # WHY --directory: the dist folder is found from this file, not the cwd, so the command runs from anywhere
    parser.add_argument("--directory", type=Path, default=DIST)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(Handler, directory=str(args.directory)))
    print(f"http://127.0.0.1:{args.port}/")
    server.serve_forever()
