"""Browse the novel views of a Difix3D run: alpha map, render, Difix-fixed image and reference per view,
plus the fine-tuned model's final render when the trainer ran with --render-novel-at-end.

Serves a small web page plus the PNGs under <result_dir>/renders/novel/<fix step>/{Alpha,Pred,Fixed,Ref}/.
Uses only the Python standard library, so it runs on the host without the Difix3D container:

    python3 inspector/server.py --result_dir /home/zack/data/ITRI_campus_difix3d
    # then open the URL it prints (http://localhost:8000, or the next free port if 8000 is taken)
"""

import argparse
import errno
import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
# Folder inside renders/novel/<step>/ -> label shown in the page, in display order
IMAGE_TYPES = {"Pred": "Render", "Fixed": "Fixed", "Ref": "Reference", "Alpha": "Alpha"}
IMAGE_NAME = re.compile(r"^\d+\.png$")
IMAGE_URL = re.compile(r"^/img/(\d+)/([A-Za-z]+)/(\d+\.png)$")


def list_steps(novel_dir):
    """Return [{step, views, types}] for every fix step folder, sorted by step."""
    steps = []
    if not os.path.isdir(novel_dir):
        return steps
    for name in os.listdir(novel_dir):
        step_dir = os.path.join(novel_dir, name)
        if not (name.isdigit() and os.path.isdir(step_dir)):
            continue
        types = [t for t in IMAGE_TYPES if os.path.isdir(os.path.join(step_dir, t))]
        views = set()
        for t in types:
            views.update(f for f in os.listdir(os.path.join(step_dir, t)) if IMAGE_NAME.match(f))
        steps.append({"step": int(name), "views": sorted(views), "types": types})
    return sorted(steps, key=lambda s: s["step"])


def describe_run(novel_dir):
    """Return (types, fix steps) for the page.

    A step folder with renders but no Fixed images holds the final model's renders
    (trainer --render-novel-at-end). It is not listed as a fix step; instead the latest one is
    offered on every fix step as a "Fine-tuned model render" panel that points at that folder.
    """
    steps = list_steps(novel_dir)
    fix_steps = [s for s in steps if "Fixed" in s["types"]]
    render_only = [s for s in steps if "Fixed" not in s["types"] and "Pred" in s["types"]]
    types = [{"folder": k, "label": v} for k, v in IMAGE_TYPES.items()]
    if not fix_steps:
        return types, steps
    if render_only:
        final = render_only[-1]
        # Shown right after the Render panel so before and after sit side by side.
        types.insert(1, {"folder": "Final", "label": "Fine-tuned model render",
                         "step": final["step"], "source": "Pred"})
        for s in fix_steps:
            s["types"].append("Final")
    return types, fix_steps


def safe_filename(name, extension):
    """Keep a client-suggested download name to [A-Za-z0-9._-], ending in `extension`."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(name)).strip("._") or "image"
    return name if name.endswith(extension) else name + extension


def make_handler(result_dir):
    novel_dir = os.path.join(result_dir, "renders", "novel")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=STATIC_DIR, **kwargs)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/api/steps":
                types, steps = describe_run(novel_dir)
                return self.send_json({"result_dir": result_dir, "types": types, "steps": steps})
            if path == "/download":
                return self.send_download(parse_qs(urlparse(self.path).query))
            match = IMAGE_URL.match(path)
            if match:
                return self.send_image(*match.groups())
            return super().do_GET()

        def image_path(self, url):
            """Map an /img/<step>/<folder>/<name> URL to its file, or None if it is not a served image."""
            match = IMAGE_URL.match(url)
            if not match or match.group(2) not in IMAGE_TYPES:
                return None
            path = os.path.join(novel_dir, *match.groups())
            return path if os.path.isfile(path) else None

        def send_download(self, query):
            """?f=<image url>&n=<file name>: send one served image as a PNG attachment with that name."""
            urls, names = query.get("f", []), query.get("n", [])
            if len(urls) != 1 or len(names) != 1:
                return self.send_error(HTTPStatus.BAD_REQUEST)
            path = self.image_path(urls[0])
            if path is None:
                return self.send_error(HTTPStatus.NOT_FOUND)
            with open(path, "rb") as f:
                body = f.read()
            return self.send_file(body, "image/png", safe_filename(names[0], ".png"))

        def send_file(self, body, content_type, filename):
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, data):
            body = json.dumps(data).encode()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_image(self, step, folder, name):
            # The URL pattern and the type whitelist keep requests inside renders/novel/.
            path = self.image_path(f"/img/{step}/{folder}/{name}")
            if path is None:
                return self.send_error(HTTPStatus.NOT_FOUND)
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass  # keep the terminal quiet; each view loads up to five images

    return Handler


def start_server(host, port, tries, handler):
    """Bind the first free port in [port, port + tries); return None if all are taken."""
    for candidate in range(port, port + tries):
        try:
            return ThreadingHTTPServer((host, candidate), handler)
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result_dir", required=True, help="Difix3D result directory (contains renders/novel/).")
    ap.add_argument("--port", type=int, default=8000, help="Preferred port; the next free one is used if it is taken.")
    ap.add_argument("--port_tries", type=int, default=20, help="How many consecutive ports to try (1 = fail if --port is taken).")
    ap.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to allow access from other machines.")
    args = ap.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    if not os.path.isdir(os.path.join(result_dir, "renders", "novel")):
        ap.error(f"{result_dir}/renders/novel does not exist.")
    server = start_server(args.host, args.port, max(args.port_tries, 1), make_handler(result_dir))
    if server is None:
        ap.error(f"ports {args.port}-{args.port + max(args.port_tries, 1) - 1} are all in use; pass another --port.")
    port = server.server_address[1]
    if port != args.port:
        print(f"Port {args.port} is in use; using {port} instead.")
    print(f"Inspecting {result_dir}")
    print(f"Open http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
