"""Draw what the pipeline actually saw, straight from a replay cache.

`main.py` can render an annotated video, but it re-runs every detector to
do it - an hour of compute to answer "is it tracking the right blob?".
Everything needed to answer that by eye is already in the cache
`replay_impacts.py --build` wrote: every candidate the detector proposed,
every frame's court calibration, and (once `--add-player-boxes` has run)
the player boxes. So this redraws from the cache and touches no model at
all, which puts a render at video-decode speed rather than GPU speed.

WHAT EACH THING ON SCREEN MEANS:

  small grey dots   every candidate the ball detector proposed this frame,
                    including the ones the tracker rejected. This is the
                    lattice the path search chooses through, so it shows
                    the clutter the tracker had to survive - dead balls on
                    court, a shoe, the net cord.

  coloured circles  one colour per BallTrack, with a fading trail. TWO
                    COLOURS ON SCREEN AT ONCE IS THE POINT: a coaching
                    drill can have two balls genuinely in flight, and the
                    single-path tracker could not represent that (see
                    candidate_tracker's module docstring).

  magenta crosses   bounces, held for the rest of the rally, reprojected
  orange circles    racket contacts, held briefly
                    Both come from the CURRENT default pipeline - the
                    flattened single-ball path - so they show what the
                    impact layer produces today, beside what multi-ball
                    tracking found. An impact the classifier could not
                    attribute ("unknown") is deliberately drawn as
                    neither; see ImpactMarkerDrawer.

  court wireframe   the calibration for THAT frame. Watch it on a camera
                    cut: replay_impacts.py --build fits a calibration per
                    frame with no plausibility guard, so a cutaway gets a
                    confident, wrong court rather than no court.

    python scripts/render_from_cache.py \\
        --cache outputs/dingles_serve_volley/replay_cache.pkl \\
        --input data/videos/dingles_serve_volley.mp4 \\
        --output /tmp/dingles_annotated.mp4 --start 60 --end 75

`--start`/`--end` (seconds) exist because checking one suspicious moment
should not cost a whole clip's render.
"""

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.court_calibration import CourtCalibration  # noqa: E402
from src.analysis.impact_pipeline import analyze_impacts  # noqa: E402
from src.tracking.candidate_tracker import track_ball_paths, track_candidates  # noqa: E402
from src.visualize.draw import CourtOverlayDrawer, ImpactMarkerDrawer  # noqa: E402

# Distinguishable at a glance on a blue court, in BGR. Track 0 is the
# cheapest path (usually the ball in play), so it gets the loudest colour.
#
# Deliberately avoids yellow and orange: draw.py's court overlay is orange
# lines (COURT_LINE_COLOR) with yellow corners (COURT_CORNER_COLOR), and a
# first render with a yellow track 0 was genuinely ambiguous against the
# wireframe. Magenta is also out - ImpactMarkerDrawer uses it for bounces.
_TRACK_COLORS = [
    (255, 255, 0),    # cyan
    (0, 255, 0),      # green
    (255, 255, 255),  # white
    (255, 0, 128),    # violet
    (128, 255, 255),  # pale yellow-white
    (255, 128, 128),  # pale blue
]
_CANDIDATE_COLOR = (120, 120, 120)
_TRAIL_LENGTH = 12


def load_cache(path: Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def calibrations_from_cache(cache: dict) -> dict[int, CourtCalibration]:
    return {
        frame: value if isinstance(value, CourtCalibration) else CourtCalibration(homography=value)
        for frame, value in cache.get("calibrations", {}).items()
    }


def _draw_hud(frame, frame_idx, fps, live_tracks, total_tracks):
    """Frame/time plus how many balls are live RIGHT NOW - the number that
    makes a two-ball stretch legible while scrubbing."""
    lines = [
        f"frame {frame_idx}   t={frame_idx / fps:6.2f}s",
        f"balls live: {live_tracks}   tracks in clip: {total_tracks}",
    ]
    for i, text in enumerate(lines):
        y = 34 + i * 30
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 0, 0), 4)
        cv2.putText(frame, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 1)
    return frame


def render(args) -> None:
    cache = load_cache(Path(args.cache))
    fps = cache["fps"]
    candidates = cache["candidates"]
    calibrations = calibrations_from_cache(cache)

    tracks = track_ball_paths(candidates, max_pixels_per_frame=args.max_jump)
    # Impacts come from the CURRENT pipeline (flattened, single ball), so
    # the markers describe what ships today rather than a preview of what
    # per-track analysis would produce.
    flattened = track_candidates(candidates, max_pixels_per_frame=args.max_jump)
    analysis = analyze_impacts(
        flattened,
        fps,
        calibrations_by_frame=calibrations,
        player_boxes_by_frame=cache.get("player_boxes") or None,
        max_pixels_per_frame=args.max_jump,
    )
    impacts_by_frame = {impact.frame_idx: impact for impact in analysis.impacts}
    print(f"{len(tracks)} ball track(s), {len(analysis.impacts)} impacts")

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.input}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start = int((args.start or 0.0) * fps)
    end = int(args.end * fps) if args.end else len(candidates)
    end = min(end, len(candidates))
    if start:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not open {out_path} for writing")

    court = CourtOverlayDrawer()
    impact_drawer = ImpactMarkerDrawer(fps=fps)
    # Per-track trails, kept here rather than in TrailDrawer because that
    # one is built around a single ball and clears itself on rally gaps.
    trails: dict[int, list[tuple[float, float]]] = {i: [] for i in range(len(tracks))}

    written = 0
    for frame_idx in range(start, end):
        ok, frame = capture.read()
        if not ok:
            break

        if args.court:
            frame = court.draw(frame, calibrations.get(frame_idx))

        if args.candidates:
            for detection in candidates[frame_idx]:
                cv2.circle(frame, (int(detection.x), int(detection.y)), 9, _CANDIDATE_COLOR, 1)

        live = 0
        for i, track in enumerate(tracks):
            color = _TRACK_COLORS[i % len(_TRACK_COLORS)]
            detection = track.detections.get(frame_idx)
            if detection is not None:
                live += 1
                trails[i].append((detection.x, detection.y))
                if len(trails[i]) > _TRAIL_LENGTH:
                    trails[i].pop(0)
            elif trails[i]:
                # Let a trail decay rather than freeze where the ball was
                # last seen - a frozen clump reads as a tracked ball.
                trails[i].pop(0)

            for j, (tx, ty) in enumerate(trails[i]):
                fade = (j + 1) / max(len(trails[i]), 1)
                cv2.circle(frame, (int(tx), int(ty)), max(2, int(5 * fade)), color, -1)
            if detection is not None:
                cv2.circle(frame, (int(detection.x), int(detection.y)), 14, color, 2)
                cv2.putText(
                    frame, str(i), (int(detection.x) + 17, int(detection.y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

        for x1, y1, x2, y2 in (cache.get("player_boxes") or {}).get(frame_idx, []):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 60), 2)

        frame = impact_drawer.draw(
            frame, frame_idx, impacts_by_frame, calibrations.get(frame_idx)
        )
        frame = _draw_hud(frame, frame_idx, fps, live, len(tracks))

        writer.write(frame)
        written += 1

    capture.release()
    writer.release()
    print(f"Wrote {out_path}  ({written} frames, {written / fps:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cache", required=True)
    parser.add_argument("--input", required=True, help="The video the cache was built from")
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=float, default=None, help="seconds")
    parser.add_argument("--end", type=float, default=None, help="seconds")
    parser.add_argument("--max-jump", type=float, default=150.0)
    parser.add_argument("--no-court", dest="court", action="store_false", help="Skip the court wireframe")
    parser.add_argument(
        "--no-candidates", dest="candidates", action="store_false",
        help="Skip the rejected-candidate dots",
    )
    render(parser.parse_args())


if __name__ == "__main__":
    main()
