"""Re-run impact detection from a cache of a video's detections, and score it
against hand labels - without a GPU, a render, or a second of neural network
time.

Tuning the bounce/contact classifier means changing a threshold and asking
"what did that cost?". Answering that by rendering the video takes minutes
of GPU per attempt and produces an .mp4 that still has to be watched, so in
practice the question gets answered two or three times instead of twenty,
and by eye. Everything downstream of the detectors is cheap arithmetic over
a few hundred points; only the detectors are expensive. So run those once,
cache what they said, and replay:

    # once, with a GPU if you have one (this is the only slow part)
    python scripts/replay_impacts.py --build --input video_input2.mp4 \\
        --cache outputs/video_input2/replay_cache.pkl

    # then as often as you like, on a laptop CPU, in about a second
    python scripts/replay_impacts.py --cache outputs/video_input2/replay_cache.pkl \\
        --labels data/labels/video_input2_impacts.csv
    python scripts/replay_impacts.py --cache ... --labels ... --max-reach-ratio 0.4

The cache holds the ball detector's per-frame candidates and the court
calibrations. `--add-player-boxes` fills in the third input, player boxes,
by running the person detector on the frames where impacts actually
landed - a few dozen frames rather than the whole video, because that is
the only place `classify_touchdowns` reads them.

Scoring is against data/labels/*.csv - see that directory's README for what
those labels are and, importantly, what they are not. The thresholds being
tuned here were derived from those same labels, so a good score is a
consistency check, not evidence of generalization. What it is genuinely
good for is the other direction: showing immediately when a change that
fixes one clip breaks something on the other.
"""

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.court_calibration import CourtCalibration  # noqa: E402
from src.analysis.impact_pipeline import analyze_impacts  # noqa: E402
from src.evaluation.impact_labels import Score, read_labels, score_impacts  # noqa: E402
from src.tracking.candidate_tracker import track_candidates  # noqa: E402

_MARKER = {"bounce": "BOUNCE ", "contact": "contact", "unknown": "  -    "}


def build_cache(args) -> dict:
    """Run the expensive detectors once. Everything else in this file is
    arithmetic over what they produced."""
    from tqdm import tqdm

    from src.analysis.court_calibration import CourtCalibration
    from src.detection.tennis_court_net import TennisCourtNetDetector
    from src.detection.wasb_ball_detector import WASBBallDetector
    from src.video.io import VideoReader

    reader = VideoReader(args.input)
    frames = list(reader.frames())
    if not frames:
        raise SystemExit(f"No frames read from {args.input}")

    ball_detector = WASBBallDetector(args.wasb_weights, device=args.device)
    court_detector = TennisCourtNetDetector(args.court_weights, device=args.device)

    candidates_by_frame = []
    calibrations = {}
    last_good = None
    for i, frame in enumerate(tqdm(frames, desc="detecting")):
        candidates_by_frame.append(ball_detector.detect_candidates(frame))
        named_points = court_detector.detect(frame) or {}
        if len(named_points) >= 4:
            last_good = CourtCalibration.from_keypoints(named_points)
        if last_good is not None:
            calibrations[i] = last_good

    return {
        "video": str(Path(args.input).resolve()),
        "fps": reader.fps,
        "num_frames": len(frames),
        "candidates": candidates_by_frame,
        # plain 3x3 arrays rather than CourtCalibration objects: a cache
        # outlives the code that wrote it, and an array cannot be
        # invalidated by a later change to the class
        "calibrations": {i: c.homography for i, c in calibrations.items()},
        "player_boxes": {},
    }


def add_player_boxes(cache: dict, frame_indices: set[int], args) -> int:
    """Run the person detector on just the frames given.

    `classify_touchdowns` reads player boxes at the impact frame and nowhere
    else, so detecting them for the whole video would spend minutes of
    compute to answer a question about a few dozen frames. Frames already in
    the cache are skipped, which is what makes this incremental: change a
    threshold, get new impact frames, top the cache up with only those.
    """
    from tqdm import tqdm

    from src.detection.player_detector import PlayerDetector
    from src.video.io import VideoReader

    video = args.input or cache.get("video")
    if not video or not Path(video).exists():
        raise SystemExit(
            "--add-player-boxes needs the video the cache was built from; pass --input"
        )

    wanted = sorted(frame_indices - set(cache.get("player_boxes", {})))
    if not wanted:
        return 0

    detector = PlayerDetector(device=args.device)
    calibrations = cache["calibrations"]
    boxes = cache.setdefault("player_boxes", {})
    reader = VideoReader(video)
    wanted_set = set(wanted)
    for i, frame in enumerate(tqdm(reader.frames(), desc="players", total=cache["num_frames"])):
        if i not in wanted_set:
            continue
        players = detector.detect(frame, calibration=calibrations.get(i))
        boxes[i] = [p.bbox for p in players]
    return len(wanted)


def impact_frames_around(impacts, radius: int, num_frames: int) -> set[int]:
    """Impact frames plus a small margin, so a later run whose impacts have
    shifted by a frame or two still finds boxes already cached."""
    wanted = set()
    for impact in impacts:
        for frame in range(impact.frame_idx - radius, impact.frame_idx + radius + 1):
            if 0 <= frame < num_frames:
                wanted.add(frame)
    return wanted


def load_cache(path: Path) -> dict:
    with open(path, "rb") as handle:
        cache = pickle.load(handle)
    cache["calibrations"] = {
        frame: value if isinstance(value, CourtCalibration) else CourtCalibration(homography=value)
        for frame, value in cache["calibrations"].items()
    }
    return cache


def run_analysis(cache: dict, args):
    detections = cache.get("detections")
    if cache.get("candidates") is not None:
        detections = track_candidates(cache["candidates"], max_pixels_per_frame=args.max_jump)
    if detections is None:
        raise SystemExit("cache holds neither 'candidates' nor 'detections'")

    player_boxes = cache.get("player_boxes") or None
    if args.no_player_boxes:
        player_boxes = None
    return analyze_impacts(
        detections,
        cache["fps"],
        calibrations_by_frame=cache["calibrations"],
        player_boxes_by_frame=player_boxes,
        max_pixels_per_frame=args.max_jump,
        max_interpolation_gap=args.interp_gap,
        static_lockon_frames=args.lockon_frames,
        static_lockon_radius=args.lockon_radius,
        use_flight_segments=not args.no_flight_segments,
        frame_edge_margin=args.frame_edge_margin,
        max_reach_ratio=args.max_reach_ratio,
    )


def print_impacts(analysis, fps: float) -> None:
    print(
        f"{len(analysis.impacts)} impacts: {len(analysis.bounces)} bounces, "
        f"{len(analysis.contacts)} contacts, {len(analysis.unattributed)} unattributed"
    )
    by_frame = {td.impact.frame_idx: td for td in analysis.touchdowns}
    for impact in analysis.impacts:
        touchdown = by_frame.get(impact.frame_idx)
        print(f"  {impact.t / fps:6.2f}s  {_MARKER[impact.kind]}  {_measurements(touchdown)}{impact.reason}")


def _measurements(touchdown) -> str:
    """The numbers the verdict was reached from, so a disagreement with a
    hand label can be diagnosed here rather than by re-deriving them."""
    if touchdown is None:
        return ""
    def rate(value):
        return "  n/a" if value is None else f"{value:5.1f}"
    reach = "n/a  " if touchdown.player_reach is None else f"{touchdown.player_reach:.2f} "
    return (
        f"approach {rate(touchdown.approach_before)} ->{rate(touchdown.approach_after)} m/s  "
        f"reach {reach}  "
    )


def report_score(analysis, labels, fps: float) -> Score:
    """Print one line per label, then the counts. The matching itself lives
    in src/evaluation/impact_labels.py, which is where its subtleties are
    explained and tested."""
    score = score_impacts(analysis.impacts, labels, fps)

    print("\nagainst hand labels:")
    for scored in score.scored:
        match, label = scored.match, scored.label
        verdict = {
            "ok": "ok      ",
            "wrong_kind": "WRONG   ",
            "missed": "MISSED  ",
            "false_positive": "FALSE + ",
        }[scored.outcome]
        where = f" {match.kind} at {match.t / fps:.2f}s" if match is not None else " no marker"
        note = f"  ({label.note})" if label.note else ""
        print(f"  {label.seconds:6.2f}s  want {label.kind:8}  {verdict}{where}{note}")

    if score.unclaimed:
        print("\n  markers matching no label (unverified, not counted as errors):")
        for impact in score.unclaimed:
            print(f"    {impact.t / fps:6.2f}s  {impact.kind}  {impact.reason}")

    print(
        f"\n  {score.correct}/{len(labels)} labels correct - {score.wrong_kind} wrong kind, "
        f"{score.missed} missed, {score.false_positives} on nothing, "
        f"{len(score.unclaimed)} unverified markers"
    )
    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, help="Pickle of detector output (see --build)")
    parser.add_argument("--input", help="Source video - only needed for --build/--add-player-boxes")
    parser.add_argument("--labels", help="Hand-label CSV to score against, e.g. data/labels/*.csv")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run the ball and court detectors over --input and write --cache. The only step "
        "that needs a GPU to be quick; everything else replays from the cache.",
    )
    parser.add_argument(
        "--add-player-boxes",
        action="store_true",
        help="Run the person detector on the frames around this run's impacts and add them to "
        "the cache, so the reach gate can fire on the next replay. Incremental - already-cached "
        "frames are skipped.",
    )
    parser.add_argument("--player-box-radius", type=int, default=3, help="Frames either side of an impact to cache boxes for")
    parser.add_argument("--no-player-boxes", action="store_true", help="Replay as if no player boxes were available")
    parser.add_argument("--wasb-weights", default=str(REPO_ROOT / "weights" / "wasb_tennis_pretrained.pth.tar"))
    parser.add_argument("--court-weights", default=str(REPO_ROOT / "weights" / "court_net_pretrained.pt"))
    parser.add_argument("--device", default=None, help="'cpu' to keep off the GPU")
    parser.add_argument("--max-jump", type=float, default=150.0)
    parser.add_argument("--interp-gap", type=int, default=8)
    parser.add_argument("--lockon-frames", type=int, default=10)
    parser.add_argument("--lockon-radius", type=float, default=20.0)
    parser.add_argument("--no-flight-segments", action="store_true")
    parser.add_argument("--frame-edge-margin", type=float, default=150.0)
    parser.add_argument("--max-reach-ratio", type=float, default=0.6)
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if args.build:
        if not args.input:
            raise SystemExit("--build needs --input")
        built = build_cache(args)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as handle:
            pickle.dump(built, handle)
        print(f"Wrote {cache_path} ({built['num_frames']} frames)")
    cache = load_cache(cache_path)

    analysis = run_analysis(cache, args)

    if args.add_player_boxes:
        added = add_player_boxes(
            cache,
            impact_frames_around(analysis.impacts, args.player_box_radius, cache["num_frames"]),
            args,
        )
        with open(cache_path, "wb") as handle:
            pickle.dump(cache, handle)
        print(f"Added player boxes for {added} frames; re-running with them")
        analysis = run_analysis(cache, args)

    print_impacts(analysis, cache["fps"])
    if args.labels:
        report_score(analysis, read_labels(Path(args.labels)), cache["fps"])


if __name__ == "__main__":
    main()
