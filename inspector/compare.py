"""Compare Difix reference-selection strategies across result folders, frame by frame.

For every novel view it shows, per run (one row per strategy): the step-0 render, the reference view the
strategy picked, the Difix-fixed image and optionally the fine-tuned model's render. Panels zoom and pan
together; one frame's images can be downloaded as separate PNGs or as one labelled slide image.

    python3 inspector/compare.py
    python3 inspector/compare.py --run pose=/path/a --run overlap=/path/b:"Gaussian overlap"
    # then open the URL it prints

Uses only the Python standard library and the helpers in server.py.
"""

import argparse
import hashlib
import json
import os
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from urllib.parse import parse_qs, unquote, urlparse

from server import IMAGE_TYPES, STATIC_DIR, list_steps, safe_filename, start_server

DEFAULT_RUNS = [
    ("pose", "/home/zack/data/ITRI_campus_difix3d", "Nearest pose", "rotation angle + translation distance"),
    ("normpose", "/home/zack/data/ITRI_campus_difix3d_normpose", "Normalized pose",
     "θ/π + ‖Δt‖ / mean camera radius"),
    ("overlap", "/home/zack/data/ITRI_campus_difix3d_overlap", "Gaussian overlap",
     "most visible Gaussians shared with the view"),
]
RUN_NAME = re.compile(r"^[a-z0-9_-]{1,32}$")
IMAGE_URL = re.compile(r"^/img/([a-z0-9_-]+)/(\d+)/([A-Za-z]+)/(\d+\.png)$")


def parse_run(value):
    """name=dir[:label] -> (name, dir, label, description)."""
    name, _, rest = value.partition("=")
    path, _, label = rest.partition(":")
    if not RUN_NAME.match(name) or not path:
        raise argparse.ArgumentTypeError(f"expected name=dir[:label] with a lowercase name, got {value!r}")
    return name, os.path.abspath(path), label or name, ""


def describe_runs(runs):
    """Per run: its fix step (has Fixed and Ref) and fine-tuned step (render-only), plus the shared views."""
    described, views = [], None
    for name, path, label, description in runs:
        steps = list_steps(os.path.join(path, "renders", "novel"))
        fix = [s for s in steps if {"Pred", "Ref", "Fixed"} <= set(s["types"])]
        final = [s for s in steps if "Fixed" not in s["types"] and "Pred" in s["types"]]
        if not fix:
            raise SystemExit(f"{path}: no renders/novel/<step>/ with Pred, Ref and Fixed")
        described.append({"name": name, "label": label, "description": description, "result_dir": path,
                          "fix_step": fix[0]["step"], "final_step": final[-1]["step"] if final else None})
        run_views = set(fix[0]["views"])
        views = run_views if views is None else views & run_views
    return described, sorted(views)


def make_handler(runs, views, group_size):
    by_name = {r["name"]: r for r in runs}
    digests = {}  # path -> md5, so repeated visits to a view do not re-read its references

    def image_path(url):
        match = IMAGE_URL.match(url)
        if not match or match.group(1) not in by_name or match.group(3) not in IMAGE_TYPES:
            return None
        run, step, folder, name = match.groups()
        path = os.path.join(by_name[run]["result_dir"], "renders", "novel", step, folder, name)
        return path if os.path.isfile(path) else None

    def digest(path):
        if path not in digests:
            with open(path, "rb") as f:
                digests[path] = hashlib.md5(f.read()).hexdigest()
        return digests[path]

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=STATIC_DIR, **kwargs)

        def do_GET(self):
            url = urlparse(self.path)
            path = unquote(url.path)
            if path == "/":
                self.path = "/compare.html"
                return super().do_GET()
            if path == "/api/compare":
                return self.send_json({"runs": runs, "views": views, "group_size": group_size})
            if path == "/api/same_reference":
                return self.send_same_reference(parse_qs(url.query).get("view", [""])[0])
            if path == "/download":
                return self.send_download(parse_qs(url.query))
            if path.startswith("/img/"):
                file_path = image_path(path)
                if file_path is None:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                with open(file_path, "rb") as f:
                    return self.send_bytes(f.read(), "image/png", cache="no-cache")
            return super().do_GET()

        def send_same_reference(self, view):
            """For one view: {run: name of the first earlier run whose reference image is identical, or null}."""
            if view not in views:
                return self.send_error(HTTPStatus.NOT_FOUND)
            result, seen = {}, {}
            for run in runs:
                path = image_path(f"/img/{run['name']}/{run['fix_step']}/Ref/{view}")
                key = digest(path) if path else None
                result[run["name"]] = seen.get(key) if key else None
                seen.setdefault(key, run["name"])
            return self.send_json(result)

        def send_download(self, query):
            """?f=<image url>&n=<file name>: one served image as a PNG attachment."""
            urls, names = query.get("f", []), query.get("n", [])
            if len(urls) != 1 or len(names) != 1:
                return self.send_error(HTTPStatus.BAD_REQUEST)
            file_path = image_path(urls[0])
            if file_path is None:
                return self.send_error(HTTPStatus.NOT_FOUND)
            with open(file_path, "rb") as f:
                body = f.read()
            self.send_bytes(body, "image/png", filename=safe_filename(names[0], ".png"))

        def send_json(self, data):
            self.send_bytes(json.dumps(data).encode(), "application/json")

        def send_bytes(self, body, content_type, filename=None, cache="no-store"):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=parse_run, action="append",
                    help="name=result_dir[:label], one per strategy (default: the three ITRI campus runs).")
    ap.add_argument("--group_size", type=int, default=36, help="Views per trajectory in the view list (0 = no groups).")
    ap.add_argument("--port", type=int, default=8000, help="Preferred port; the next free one is used if it is taken.")
    ap.add_argument("--port_tries", type=int, default=20)
    ap.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to allow access from other machines.")
    args = ap.parse_args()

    runs, views = describe_runs(args.run or DEFAULT_RUNS)
    if not views:
        ap.error("the runs have no novel views in common")
    server = start_server(args.host, args.port, max(args.port_tries, 1), make_handler(runs, views, args.group_size))
    if server is None:
        ap.error(f"ports {args.port}-{args.port + max(args.port_tries, 1) - 1} are all in use; pass another --port.")
    port = server.server_address[1]
    for run in runs:
        print(f"{run['label']:18s} {run['result_dir']}  (fix step {run['fix_step']}, fine-tuned step {run['final_step']})")
    print(f"{len(views)} views in common")
    print(f"Open http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
