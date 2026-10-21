"""Raw coaching video in, coach report out - every step, in order.

    python scripts/run_session.py --input ~/Downloads/session.mp4 --name tuesday_squad
    python scripts/run_session.py --input clip.mp4 --name demo --start 12 --end 120

Runs the existing scripts rather than re-implementing them, so each step's
own CLI stays the one place its behaviour is defined, and prints each
command so any step can be re-run by hand. A step whose output already
exists is skipped - a run killed halfway (the detection steps are the slow
ones) picks up where it stopped. --force redoes everything.

    1. normalise    -> data/videos/<name>.mp4            1080p, 25fps
    2. ball + court -> outputs/<name>/replay_cache.pkl    slow: neural nets
    3. people       -> outputs/<name>/person_tracks.pkl   slow: neural net
       pose         -> outputs/<name>/poses.pkl           ~5 min per 2 min of video (tells people from objects)
       players      -> outputs/<name>/players.json
    4. report       -> outputs/<name>/session_report.json
    5. feedback     -> outputs/<name>/session_feedback.json, written by Qwen on this computer through
                       Ollama (or .prompt.txt when Ollama or the model is not installed)
    6. coach page   -> outputs/<name>/coach_report.html
    7. video        -> outputs/<name>/annotated.mp4

"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def _env() -> dict:
    env = dict(os.environ)
    # Model weights download on first use; some Python installs ship without
    # a CA bundle and fail TLS verification (found on this project's Mac).
    try:
        import certifi

        env.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass
    return env


def _short(part: str) -> str:
    """Repo-relative paths in printed commands, so they stay readable and
    can be pasted back in from the repo root."""
    root = str(REPO_ROOT) + os.sep
    return "python" if part == PY else part.replace(root, "")


def step(label: str, output: Path, command: list[str], force: bool) -> None:
    if output.exists() and not force:
        print(f"[skip] {label}: {_short(str(output))} exists")
        return
    print(f"\n[run]  {label}\n       {' '.join(_short(c) for c in command)}", flush=True)
    started = time.time()
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=_env())
    print(f"[done] {label} in {time.time() - started:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="The raw video")
    parser.add_argument("--name", required=True, help="Short name for this session, used for all output paths")
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the video")
    parser.add_argument("--end", type=float, help="seconds into the video (default: the end)")
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--force", action="store_true", help="Redo steps even if their output exists")
    parser.add_argument("--theme", help="What the session was on: netplay, consistency, adjustment, around_backhand "
                        "(comma-separated)")
    parser.add_argument("--title", help="Coach page title (default: from --name). Keep it fixed if the page is published.")
    args = parser.parse_args()

    source = Path(args.input).expanduser()
    if not source.exists():
        raise SystemExit(f"No such video: {source}")
    end = args.end
    if end is None:
        import cv2

        capture = cv2.VideoCapture(str(source))
        end = capture.get(cv2.CAP_PROP_FRAME_COUNT) / (capture.get(cv2.CAP_PROP_FPS) or 25.0)
        capture.release()

    video = REPO_ROOT / "data" / "videos" / f"{args.name}.mp4"
    out = REPO_ROOT / "outputs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    ball_cache = out / "replay_cache.pkl"
    people = out / "person_tracks.pkl"
    report = out / "session_report.json"
    feedback = out / "session_feedback.json"
    s = lambda name: str(REPO_ROOT / "scripts" / name)  # noqa: E731

    step("normalise video", video, [PY, s("normalise_clip.py"), "--input", str(source),
         "--start", str(args.start), "--end", str(end), "--output", str(video)], args.force)
    step("ball + court detection", ball_cache, [PY, s("replay_impacts.py"), "--build",
         "--input", str(video), "--cache", str(ball_cache)], args.force)
    step("person tracking", people, [PY, s("track_people.py"), "--input", str(video),
         "--output", str(people)], args.force)

    poses = out / "poses.pkl"
    # Every frame: swings (fed vs live) need the wrist on every frame - a
    # pose every 2nd frame kept only 57 of 73 swings on Dingles.
    sampled = False
    if poses.exists() and not args.force:
        import pickle

        with open(poses, "rb") as handle:
            sampled = pickle.load(handle).get("every", 1) != 1  # from before swings needed every frame
    step("pose for every person", poses, [PY, s("track_poses.py"), "--input", str(video),
         "--people", str(people), "--output", str(poses), "--every", "1"], args.force or sampled)

    # Bounces (and a second opinion on near-end swings) from E2E-Spot, a video
    # model that needs no ball tracking; its bounces beat ours on Dingles.
    e2e = out / "e2e_spot.json"
    step("E2E-Spot events", e2e, [PY, s("e2e_spot_events.py"), "--video", str(video), "--out", str(e2e)], args.force)

    # Player identity needs the cutaways from a first report; the report is
    # then rebuilt with the players (both report passes are cheap).
    report_cmd = [PY, s("session_report.py"), "--ball-cache", str(ball_cache), "--people", str(people),
                  "--video", str(video), "--name", args.name, "--out", str(report), "--e2e", str(e2e)]
    if args.theme:
        report_cmd += ["--theme", args.theme]
    # The report is cheap and depends on every cache above, so it is always rebuilt.
    step("session report (court, cutaways)", report, report_cmd, force=True)
    players = out / "players.json"
    step("stable player identity", players, [PY, s("assign_players.py"), "--video", str(video),
         "--people", str(people), "--poses", str(poses), "--report", str(report), "--out", str(players)], force=True)
    swings = out / "swings.json"
    step("swings", swings, [PY, s("detect_swings.py"), "--ball-cache", str(ball_cache), "--people", str(people),
         "--poses", str(poses), "--report", str(report), "--players", str(players), "--e2e", str(e2e),
         "--out", str(swings)], force=True)
    # Stroke type per swing, once a model has been trained (docs/gpu_stroke_training.html).
    # Not in the report yet: kept beside it until it is shown to beat BST on labels.
    stroke_weights = REPO_ROOT / "weights" / "stroke_tcn.pt"
    if stroke_weights.exists():
        step("stroke type per swing", out / "strokes_tcn.json", [PY, s("classify_strokes_tcn.py"), "--session", str(out),
             "--weights", str(stroke_weights), "--out", str(out / "strokes_tcn.json")], force=True)
    step("session report (players, fed vs live)", report,
         report_cmd + ["--players", str(players), "--swings", str(swings)], force=True)

    if not args.no_feedback:
        # Always rewritten: the report it reads was just rebuilt.
        step("AI feedback (Qwen on this computer)", feedback, [PY, s("session_feedback.py"), "--report", str(report),
             "--out", str(feedback)], force=True)
    page_cmd = [PY, s("render_coach_report.py"), "--report", str(report), "--feedback", str(feedback),
                "--out", str(out / "coach_report.html")]
    if args.title:
        page_cmd += ["--title", args.title]
    step("coach report page", out / "coach_report.html", page_cmd, force=True)
    if not args.no_video:
        step("annotated video", out / "annotated.mp4", [PY, s("render_from_cache.py"), "--cache", str(ball_cache),
             "--input", str(video), "--people", str(people), "--report", str(report),
             "--players", str(players), "--swings", str(swings),
             # Players, their swings and the ball's flight; no bounce or
             # contact markers, which pile up and clutter the picture.
             "--on-court-only", "--no-track-labels", "--no-impacts", "--no-candidates",
             "--output", str(out / "annotated.mp4")], force=True)

    print(f"\nDone. Open {_short(str(out / 'coach_report.html'))}")


if __name__ == "__main__":
    main()
