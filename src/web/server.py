"""A tiny stdlib HTTP server for the web demo. No dependency, no framework.

Serves the single page and its assets, and a small JSON API backed by one `GameSession`.
The session opens on the best model the box can serve (Ouro on a ready GPU, else the stub,
which needs no model and no network). One session per server process is enough for a local
demo; state lives in memory and resets on request.

    python -m web.server            # serves http://127.0.0.1:8000
    python -m web.server --port 842 --no-browser
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from web.game import GameSession, PullJob, default_model_id

STATIC_DIR = Path(__file__).parent / "static"
PORTRAIT_DIR = Path(__file__).resolve().parents[2] / "assets" / "portraits"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".json": "application/json",
}


class DemoHandler(BaseHTTPRequestHandler):
    """Routes the page, the static assets, the portraits, and the JSON API."""

    # One session and one download job shared by the process, set by `build_server`.
    session: GameSession
    pull: PullJob

    def log_message(self, *args: object) -> None:  # keep the console clean; this is a demo
        pass

    # ----------------------------------------------------------------------------- GET

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            self._send_file(STATIC_DIR / "index.html")
        elif path == "/api/snapshot":
            self._send_json(self.session.snapshot())
        elif path == "/api/attribution":
            # The one expensive computation, fetched on demand when the Attribution tab is opened.
            self._send_json(self.session.attribution_live())
        elif path == "/api/pull":
            snap = self.pull.snapshot()
            if snap["state"] == "done":
                # The next snapshot must see the model as ready, not the stale probe.
                self.session.refresh_models()
            self._send_json(snap)
        elif path.startswith("/static/"):
            self._send_static(STATIC_DIR, path[len("/static/"):])
        elif path.startswith("/portraits/"):
            self._send_static(PORTRAIT_DIR, path[len("/portraits/"):])
        else:
            self._send_error(404, "not found")

    # ---------------------------------------------------------------------------- POST

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path not in ("/api/step", "/api/reset", "/api/model", "/api/pull"):
            self._send_error(404, "not found")
            return
        body = self._read_json_body()
        if path == "/api/pull":
            self._send_json(self.pull.start(str(body.get("model") or "")))
            return
        if path == "/api/reset":
            self.session.reset()
        elif path == "/api/model":
            # The result carries whether the switch took; the snapshot reflects the new model.
            status = self.session.set_model(str(body.get("model") or "stub"))
            self._send_json({"status": status, **self.session.snapshot()})
            return
        else:
            self.session.step((body.get("text") or "").strip() or None)
        self._send_json(self.session.snapshot())

    # ------------------------------------------------------------------------- helpers

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length or length > 1_000_000:  # a player line is bytes, not MB; refuse junk
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError):
            return {}

    def _send_static(self, root: Path, relative: str) -> None:
        # Resolve inside the root and refuse anything that escapes it (no path traversal).
        target = (root / relative).resolve()
        if root.resolve() not in target.parents or not target.is_file():
            self._send_error(404, "not found")
            return
        self._send_file(target)

    def _send_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_error(404, "not found")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _CONTENT_TYPES.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, code: int, message: str) -> None:
        data = json.dumps({"error": message}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def build_server(port: int = 8000, default_model: str | None = None) -> ThreadingHTTPServer:
    """Construct the server with a fresh session, without starting it (handy for tests).

    `default_model` pins the opening model; None auto-picks (Ouro on a ready GPU, else the
    stub). A default that fails to load just leaves the stub in place: the demo must open.
    """
    DemoHandler.session = GameSession()
    DemoHandler.pull = PullJob()
    target = default_model or default_model_id()
    if target != "stub":
        print(f"loading {target} (the first load can take a moment)...")
        status = DemoHandler.session.set_model(target)
        if not status["ok"]:
            print(f"could not load {target}: {status.get('error')}. Opening on the stub.")
    return ThreadingHTTPServer(("127.0.0.1", port), DemoHandler)


def serve(port: int = 8000, open_browser: bool = True) -> None:
    server = build_server(port)
    url = f"http://127.0.0.1:{port}"
    model = DemoHandler.session.snapshot()["settings"]["model"]
    print(f"EMBR web demo on {url}  (Ctrl+C to stop). Replies come from: {model}.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab")
    args = parser.parse_args()
    serve(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
