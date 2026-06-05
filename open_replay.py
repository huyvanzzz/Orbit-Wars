from __future__ import annotations

import argparse
import functools
import http.server
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent
REPLAY_DIR = ROOT / "replays"


def find_replay(name: str | None, latest: bool) -> Path:
    if not REPLAY_DIR.exists():
        raise FileNotFoundError(f"Replay folder not found: {REPLAY_DIR}")

    if latest:
        html_files = sorted(
            REPLAY_DIR.glob("*.html"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not html_files:
            raise FileNotFoundError(f"No .html replay files found in: {REPLAY_DIR}")
        return html_files[0]

    replay_name = name or "last_match.html"
    replay_path = Path(replay_name)
    if replay_path.suffix.lower() != ".html":
        replay_path = replay_path.with_suffix(".html")

    if not replay_path.is_absolute():
        replay_path = REPLAY_DIR / replay_path

    if not replay_path.exists():
        raise FileNotFoundError(f"Replay file not found: {replay_path}")

    return replay_path


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def serve_replay(path: Path, port: int | None) -> None:
    port = port or find_free_port()
    relative_name = path.relative_to(REPLAY_DIR).as_posix()
    quoted_name = quote(relative_name)
    url = f"http://127.0.0.1:{port}/{quoted_name}"

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(REPLAY_DIR),
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Serving folder: {REPLAY_DIR}")
    print(f"Replay URL: {url}")
    print("Press Ctrl+C to stop the server.")

    webbrowser.open(url)

    try:
        while True:
            thread.join(timeout=1)
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve and open an Orbit Wars replay HTML file.")
    parser.add_argument(
        "name",
        nargs="?",
        help="Replay file name in replays/. Default: last_match.html",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Open the newest .html file in replays/.",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Local server port. Default: pick a free port.",
    )
    args = parser.parse_args()

    try:
        replay_path = find_replay(args.name, args.latest)
        serve_replay(replay_path, args.port)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
