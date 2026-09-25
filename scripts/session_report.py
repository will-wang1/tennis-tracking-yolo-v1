"""Build a session report (src/session/report.py) for one clip from its
caches - no model runs here, so it takes seconds.

    python scripts/session_report.py \\
        --ball-cache outputs/dingles_serve_volley/replay_cache.pkl \\
        --people outputs/dingles_serve_volley/person_tracks.pkl \\
        --video data/videos/dingles_serve_volley.mp4 \\
        --second-opinion outputs/dingles_serve_volley/tennisproject_ball_track.pkl \\
        --out outputs/dingles_serve_volley/session_report.json

CUTAWAYS ARE FOUND, NOT TYPED IN. With --video, camera cuts are detected
(scene_cuts.py) and each resulting shot is checked for whether the static
court's painted lines are actually in the picture (court_line_contrast): a
shot where they are not is showing some other view, and is excluded from
every time-based metric instead of being counted as idle court. A
fixed fence camera produces no cuts and so no cutaways; this exists for
edited footage like the Dingles clip.
"""

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.court_calibration import (  # noqa: E402
    CourtCalibration,
    static_calibration_from_frames,
)
from src.analysis.impact_pipeline import analyze_impacts  # noqa: E402
from src.analysis.person_tracks import (  # noqa: E402
    attribute_contacts,
    boxes_by_frame,
    coverage_grid,
    foot_positions_by_track,
    summarise_movement,
)
from src.session.report import build_session_report  # noqa: E402
from src.tracking.candidate_tracker import track_ball_paths, track_candidates  # noqa: E402

def find_cutaways(video: str, static: CourtCalibration, num_frames: int, fps: float) -> list[tuple[int, int]]:
    """Shots, between detected camera cuts, whose picture does not contain
    the static court. Tested on the IMAGE (court_line_contrast), not on the
    cached calibrations - those are carried forward through a cutaway and
    agree with the court view while showing something else. Six frames per
    shot is plenty: a shot is one continuous view by definition."""
    import cv2

    from src.analysis.court_calibration import MIN_COURT_LINE_CONTRAST, court_line_contrast
    from src.analysis.scene_cuts import detect_scene_cuts
    from src.video.io import VideoReader

    cuts = detect_scene_cuts(VideoReader(video).frames())
    if not cuts:
        return []
    bounds = [0] + sorted(cuts) + [num_frames]
    capture = cv2.VideoCapture(video)
    cutaways = []
    try:
        for a, b in zip(bounds, bounds[1:]):
            scores = []
            for f in np.linspace(a, b - 1, 6).astype(int):
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(f))
                ok, image = capture.read()
                if ok:
                    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0)
                    scores.append(court_line_contrast(gray, static))
            score = float(np.median(scores)) if scores else 0.0
            is_cutaway = score < MIN_COURT_LINE_CONTRAST
            print(f"  shot {a / fps:6.1f}-{b / fps:6.1f}s  court-line contrast {score:5.1f} -> "
                  f"{'CUTAWAY' if is_cutaway else 'court view'}")
            if is_cutaway:
                cutaways.append((a, b))
    finally:
        capture.release()
    return cutaways


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ball-cache", required=True)
    parser.add_argument("--people", help="scripts/track_people.py cache")
    parser.add_argument("--video", help="Source video, to detect camera cutaways")
    parser.add_argument("--second-opinion", help="scripts/tennisproject_bounces.py cache, to mark agreed bounces")
    parser.add_argument("--name", help="Clip name (default: the cache's folder)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.ball_cache, "rb") as handle:
        cache = pickle.load(handle)
    fps, num_frames = cache["fps"], cache["num_frames"]
    per_frame = {
        f: (v if isinstance(v, CourtCalibration) else CourtCalibration(homography=v))
        for f, v in cache["calibrations"].items()
    }
    static = static_calibration_from_frames(per_frame)
    calibrations = {f: static for f in range(num_frames)}

    cutaways = find_cutaways(args.video, static, num_frames, fps) if args.video else []
    skip = {f for a, b in cutaways for f in range(a, b)}

    people_cache = None
    if args.people:
        with open(args.people, "rb") as handle:
            people_cache = pickle.load(handle)
    # Person-tracking boxes when available - every frame, from the same pass
    # that measures movement - else whatever the ball cache holds.
    player_boxes = boxes_by_frame(people_cache["people"]) if people_cache else (cache.get("player_boxes") or None)

    ball_tracks = track_ball_paths(cache["candidates"])
    impacts = analyze_impacts(
        track_candidates(cache["candidates"]),
        fps,
        calibrations_by_frame=calibrations,
        player_boxes_by_frame=player_boxes,
    )

    people, coverage = [], None
    if people_cache:
        tracks = foot_positions_by_track(people_cache["people"], lambda f: static, skip_frames=skip)
        people = summarise_movement(tracks, fps, static)
        coverage = coverage_grid(tracks, fps, track_ids=[p.track_id for p in people if p.on_this_court])
        contacts = [(i.frame_idx, i.x, i.y) for i in impacts.impacts if i.kind == "contact" and i.frame_idx not in skip]
        credited, _unattributed = attribute_contacts(contacts, people_cache["people"])
        people = [replace(p, confirmed_contacts=credited.get(p.track_id, 0)) for p in people]

    second = None
    if args.second_opinion:
        with open(args.second_opinion, "rb") as handle:
            other = pickle.load(handle)
        second = {int(round(f / other["fps"] * fps)) for f in other.get("bounces", [])}

    report = build_session_report(
        clip_name=args.name or Path(args.ball_cache).parent.name,
        fps=fps,
        num_frames=num_frames,
        calibration=static,
        ball_tracks=ball_tracks,
        impacts=impacts,
        people=people,
        coverage=coverage,
        cutaways=cutaways,
        second_opinion_bounce_frames=second,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out}")
    for name, metric in report["metrics"].items():
        print(f"  {name:30} {metric['value']!s:>8}  {metric['unit']:38} [{metric['basis']}]")


if __name__ == "__main__":
    main()
