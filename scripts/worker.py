"""Analyse the clips uploaded to the web app, one after another.

Runs on the analysis computer (the Acer server, beside the app, or a Mac
for testing) and talks to the app over HTTP, so it doesn't need to be the
machine the app runs on:

    python scripts/worker.py                         # keep going, check every 30s
    python scripts/worker.py --once                  # one clip, then stop
    APP_URL=https://... IMPORT_TOKEN=... python scripts/worker.py

For each queued upload it downloads the clip, runs run_session.py on it
(reporting each step to the app's Uploads page), sends the annotated video
back, and publishes the report - which marks the upload done and links it
to its session. A failure is reported with the end of the log and the
worker moves on; the head coach can retry from the Uploads page.

Once a session is published its working files here - the downloaded clip,
the normalised copy and outputs/upload_<id>/ - are deleted (--keep-files
keeps them): the app holds the report and video, and footage of students
should not pile up on the analysis computer. A failed clip's files stay so
a retry can reuse them.
"""

from __future__ import annotations

import argparse
import collections
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_session import DEFAULT_TOKEN, DEFAULT_URL, ROOT, AppError, call, session_body  # noqa: E402

sys.path.insert(0, str(ROOT))
from src.session.feedback import default_backend, default_model, feedback_status  # noqa: E402

PY = sys.executable


def download(url: str, token: str, upload_id: int, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(f"{url.rstrip('/')}/api/uploads/{upload_id}/file",
                                     headers={"Authorization": f"Bearer {token}"})
    partial = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(request, timeout=600) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, length=8 * 1024 * 1024)
    partial.rename(dest)


def analyse(job: dict, args: argparse.Namespace) -> None:
    upload_id = job["id"]
    name = f"upload_{upload_id}"
    suffix = Path(job.get("original_name") or "clip.mp4").suffix or ".mp4"
    clip = ROOT / "data" / "uploads" / f"{name}{suffix}"
    if not clip.exists():
        print(f"  downloading {job.get('original_name')} ...", flush=True)
        call(args.url, args.token, f"/api/worker/{upload_id}/progress", {"step": "downloading the video", "steps_done": 0})
        download(args.url, args.token, upload_id, clip)

    command = [PY, "-u", str(ROOT / "scripts" / "run_session.py"), "--input", str(clip), "--name", name,
               "--title", job["title"]]
    if job.get("themes"):
        command += ["--theme", ",".join(job["themes"])]
    if args.no_feedback:
        command.append("--no-feedback")

    # Follow run_session's "[run] <step>" / "[done]" / "[skip]" lines to report progress.
    tail: collections.deque[str] = collections.deque(maxlen=40)
    done = 0
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\n")
        tail.append(line)
        if args.verbose:
            print("   ", line, flush=True)
        if line.startswith("[run]"):
            step = line[len("[run]"):].strip()
            print(f"  {step} ...", flush=True)
            call(args.url, args.token, f"/api/worker/{upload_id}/progress", {"step": step, "steps_done": done})
        elif line.startswith(("[done]", "[skip]")):
            done += 1
    if process.wait() != 0:
        raise AppError(f"run_session.py stopped (exit {process.returncode}):\n" + "\n".join(tail))

    out_dir = ROOT / "outputs" / name
    video_path = None
    video = out_dir / "annotated.mp4"
    if video.exists():
        call(args.url, args.token, f"/api/worker/{upload_id}/progress", {"step": "sending the video", "steps_done": done})
        with open(video, "rb") as handle:
            sent = call(args.url, args.token, f"/api/worker/{upload_id}/video", method="PUT",
                        stream=handle, length=video.stat().st_size)
        video_path = sent["video_path"]

    created = call(args.url, args.token, "/api/sessions", {
        **session_body(out_dir),
        "upload_id": upload_id,
        "title": job["title"],
        "coach": job["coach_name"],
        "date": job["date"],
        "session_type": job["session_type"],
        "court": job.get("court"),
        "themes": job.get("themes") or [],
        "video_path": video_path,
    })
    print(f"  done: {args.url.rstrip('/')}/sessions/{created['id']}", flush=True)
    if not args.keep_files:
        clip.unlink(missing_ok=True)
        (ROOT / "data" / "videos" / f"{name}.mp4").unlink(missing_ok=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def check_feedback() -> None:
    """Stop at start-up if Qwen can't run, rather than publish every session
    without feedback and only say so in a log nobody reads."""
    backend = default_backend()
    model = default_model(backend)
    status = feedback_status(backend, model)
    if status in ("ready", "not downloaded"):  # MLX downloads on first use
        print(f"Feedback: {model} via {backend}", flush=True)
        return
    hints = {"no mlx": "MLX is not installed (Apple silicon only) - set FEEDBACK_BACKEND=ollama",
             "no server": "Ollama is not running - start it (systemctl status ollama)",
             "not pulled": f"the model is not downloaded - ollama pull {model}"}
    raise SystemExit(f"Can't write feedback: {hints.get(status, status)}. Fix that, or run with --no-feedback.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--token", default=DEFAULT_TOKEN)
    parser.add_argument("--once", action="store_true", help="process at most one clip, then stop")
    parser.add_argument("--poll", type=float, default=30, help="seconds between checks when the queue is empty")
    parser.add_argument("--no-feedback", action="store_true", help="skip the AI feedback")
    parser.add_argument("--keep-files", action="store_true", help="keep each clip's working files after publishing")
    parser.add_argument("--name", default=socket.gethostname(), help="this worker's name, shown in the app")
    parser.add_argument("--verbose", action="store_true", help="print run_session's full output")
    args = parser.parse_args()

    print(f"Worker '{args.name}' on {args.url}", flush=True)
    if not args.no_feedback:
        check_feedback()
    while True:
        try:
            job = call(args.url, args.token, "/api/worker/claim", {"worker": args.name})
        except AppError as error:
            print(error, flush=True)
            if args.once:
                raise SystemExit(1)
            time.sleep(args.poll)
            continue
        if job is None:
            if args.once:
                print("Nothing queued.")
                return
            time.sleep(args.poll)
            continue
        print(f"Upload {job['id']}: {job['title']} ({job['coach_name']}, {job['date']})", flush=True)
        try:
            analyse(job, args)
        except Exception as error:  # report anything, keep the worker alive
            print(f"  failed: {error}", flush=True)
            try:
                call(args.url, args.token, f"/api/worker/{job['id']}/fail", {"error": str(error)})
            except AppError as report_error:
                print(f"  (could not report the failure: {report_error})", flush=True)
        if args.once:
            return


if __name__ == "__main__":
    main()
