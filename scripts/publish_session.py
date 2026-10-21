"""Send a processed session to the web app, tagged with who ran it.

After run_session.py has analysed a clip:

    python scripts/publish_session.py --name tuesday_squad \\
        --coach "Jane Smith" --date 2026-10-20 --type squad --court "Court 3" \\
        --theme netplay,consistency

reads outputs/<name>/session_report.json (and session_feedback.json if it
exists) and POSTs it to the app's import endpoint (web/src/app/api/sessions).
The video shown on the session page is outputs/<name>/annotated*_web.mp4 or
annotated*.mp4 if present (it must be H.264 - see scripts/web_video.py), or
--video. Clips uploaded through the app don't need this: scripts/worker.py
analyses and publishes them.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_URL = os.environ.get("APP_URL", "http://localhost:3000")
DEFAULT_TOKEN = os.environ.get("IMPORT_TOKEN", "dev-import-token")


class AppError(RuntimeError):
    pass


def call(url: str, token: str, path: str, body: dict | None = None, method: str = "POST",
         data: bytes | None = None, stream=None, length: int | None = None) -> dict | None:
    """One request to the app with the worker token. JSON in and out;
    `stream` (a file object) with `length` sends a file without reading it
    into memory."""
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if stream is not None:
        data = stream
        headers["Content-Length"] = str(length)
        headers["Content-Type"] = "application/octet-stream"
    request = urllib.request.Request(f"{url.rstrip('/')}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        raise AppError(f"{method} {path} failed ({error.code}): {error.read().decode()[:500]}") from None
    except urllib.error.URLError as error:
        raise AppError(f"Could not reach the app at {url} ({error.reason}) - is it running? (cd web && npm run dev)") from None


def session_body(out_dir: Path) -> dict:
    """The analysis results of one session, as the import endpoint takes them."""
    report_path = out_dir / "session_report.json"
    if not report_path.exists():
        raise AppError(f"No report at {report_path} - run scripts/run_session.py first")
    feedback_path = out_dir / "session_feedback.json"
    return {
        "report": json.loads(report_path.read_text()),
        "feedback": json.loads(feedback_path.read_text()) if feedback_path.exists() else None,
    }


def default_video(out_dir: Path) -> Path | None:
    for pattern in ("annotated*_web.mp4", "annotated*.mp4"):
        found = sorted(out_dir.glob(pattern))
        if found:
            return found[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", required=True, help="the session's folder under outputs/")
    parser.add_argument("--coach", required=True, help="the coach who ran the session")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--type", required=True, choices=["squad", "private"], dest="session_type")
    parser.add_argument("--court")
    parser.add_argument("--theme", default="", help="comma-separated lesson themes")
    parser.add_argument("--title")
    parser.add_argument("--video", type=Path, help="video for the session page (default: the annotated render)")
    parser.add_argument("--exemplary", action="store_true", help="mark it exemplary (sets the benchmark)")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    args = parser.parse_args()

    out_dir = ROOT / "outputs" / args.name
    video = args.video or default_video(out_dir)
    try:
        body = {
            **session_body(out_dir),
            "title": args.title or f"{args.session_type.capitalize()} session - {args.date}",
            "coach": args.coach,
            "date": args.date,
            "session_type": args.session_type,
            "court": args.court,
            "themes": [t.strip() for t in args.theme.split(",") if t.strip()],
            "video_path": str(video.resolve().relative_to(ROOT)) if video else None,
            "exemplary": args.exemplary,
        }
        created = call(args.url, args.token, "/api/sessions", body)
    except AppError as error:
        raise SystemExit(str(error))
    print(f"Published: {args.url.rstrip('/')}/sessions/{created['id']}")
    if not video:
        print("No video found - the session page will show the report without video.")


if __name__ == "__main__":
    main()
