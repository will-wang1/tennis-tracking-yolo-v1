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

from src.analysis.court_calibration import (  # noqa: E402
    CourtCalibration,
    static_calibration_from_frames,
)
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


_OTHER_BOUNCE_COLOR = (255, 0, 255)  # magenta, like the pipeline's own bounce crosses
_OTHER_BOUNCE_HOLD_S = 1.5


def load_other_bounces(path: str, fps: float, frame_scale: float) -> dict[int, tuple[float, float, float]]:
    """Another method's bounces (scripts/tennisproject_bounces.py cache),
    re-expressed in THIS video's frames and pixels: {frame: (x, y, t_s)}.

    That method runs on its own copy of the clip (1280x720, 29.97fps), so
    frames go through seconds and positions are scaled by `frame_scale`
    (1.5 for 720p -> 1080p). The position drawn is the one the bounce model
    itself saw - its own smoothed track - not this project's tracker."""
    import pickle

    with open(path, "rb") as handle:
        other = pickle.load(handle)
    out = {}
    xy = other.get("smoothed_xy") or other["ball_track"]
    for f in other.get("bounces", []):
        x, y = xy[f]
        if x is None:
            continue
        t = f / other["fps"]
        out[int(round(t * fps))] = (float(x) * frame_scale, float(y) * frame_scale, t)
    return out


def _draw_other_bounces(frame, frame_idx, bounces, fps, label):
    """Each bounce as a cross that holds for a moment then fades, labelled
    with the method and time, so a viewer can check it against the ball."""
    hold = int(_OTHER_BOUNCE_HOLD_S * fps)
    for start in range(frame_idx - hold, frame_idx + 1):
        hit = bounces.get(start)
        if hit is None:
            continue
        x, y, t = hit
        age = (frame_idx - start) / max(hold, 1)
        size = 14 if age < 0.2 else 10
        thickness = 3 if age < 0.5 else 2
        cx, cy = int(x), int(y)
        cv2.line(frame, (cx - size, cy - size), (cx + size, cy + size), _OTHER_BOUNCE_COLOR, thickness)
        cv2.line(frame, (cx - size, cy + size), (cx + size, cy - size), _OTHER_BOUNCE_COLOR, thickness)
        if age < 0.2:
            cv2.circle(frame, (cx, cy), 22, _OTHER_BOUNCE_COLOR, 2)
        text = f"{label} bounce {t:.2f}s"
        cv2.putText(frame, text, (cx + 16, cy - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(frame, text, (cx + 16, cy - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, _OTHER_BOUNCE_COLOR, 2)
    return frame


_STROKE_BEFORE, _STROKE_AFTER = 10, 28
_SWING_BEFORE, _SWING_AFTER = 8, 18
_SWING_COLOR = (60, 230, 255)


def _draw_swing(frame, swing: dict, frame_idx: int, box: tuple[int, int, int, int], text: bool = True) -> None:
    """A thick box and "SWING" over the player, brightest at the swing's
    peak frame so it reads as one event rather than a sticky label."""
    x1, y1, x2, y2 = box
    fade = max(0.35, 1.0 - abs(frame_idx - swing["frame"]) / 18.0)
    color = tuple(int(c * fade) for c in _SWING_COLOR)
    cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), color, 4)
    if not text:
        return
    y = max(20, y1 - 30)
    cv2.putText(frame, "SWING", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
    cv2.putText(frame, "SWING", (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
_STROKE_COLORS = {"forehand": (60, 200, 255), "backhand": (255, 170, 60), "serve": (120, 255, 120)}


def _draw_stroke(frame, hit: dict, x: int, y: int) -> None:
    """A stroke label over the hitter: the stroke in colour, or for an
    unsure hit the model's raw guess in grey, so it can be judged too."""
    stroke = hit["stroke"]
    if stroke == "unsure":
        text, color, scale = f"? {hit['label']} {max(hit['probabilities'].values()):.2f}", (170, 170, 170), 0.55
    else:
        text, color, scale = stroke.upper() + (" (net)" if hit.get("at_net") else ""), _STROKE_COLORS[stroke], 0.8
    y = max(20, y)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def _person_color(track_id: int) -> tuple[int, int, int]:
    """A stable colour per person id, so the same person keeps one colour
    for the whole clip and an id switch is visible as a colour change."""
    rng = np.random.default_rng(track_id * 7919)
    return tuple(int(v) for v in rng.integers(80, 256, size=3))
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

    cutaway_frames: set[int] = set()
    if args.report:
        # Use the court the session report was measured on, and its
        # cutaways, so the video and the numbers cannot disagree about where
        # the court is. The report fits the court only from shots that show
        # it; a fit over every frame of a video with long non-court stretches
        # comes out off the lines.
        import json

        report = json.loads(Path(args.report).read_text())
        court = CourtCalibration(homography=np.array(report["court"]["homography"], dtype=float))
        calibrations = {frame: court for frame in range(len(candidates))}
        for a_s, b_s in report["clip"]["cutaways_s"]:
            cutaway_frames.update(range(int(round(a_s * fps)), int(round(b_s * fps))))
        print(f"Court: from {args.report} ({len(report['clip']['cutaways_s'])} cutaway(s))")
    elif args.static_court:
        # A fixed camera's per-frame fits are many noisy readings of one
        # homography; refitting each frame just re-rolls the noise and the
        # wireframe visibly jitters. See static_calibration_from_frames.
        static = static_calibration_from_frames(calibrations)
        calibrations = {frame: static for frame in range(len(candidates))}
        print("Court: one static calibration (fixed camera)")

    tracks = track_ball_paths(candidates, max_pixels_per_frame=args.max_jump)
    if args.on_court_only:
        # Same filter the session report uses, so the video shows exactly the
        # balls the metrics count - not the next court's rally, not a false
        # track on a person by the ball cart.
        from src.session.report import on_court_tracks

        static_for_filter = next(iter(calibrations.values()))
        before = len(tracks)
        tracks = on_court_tracks(tracks, static_for_filter)
        print(f"Ball tracks on this court: {len(tracks)} of {before}")
    # Impacts come from the CURRENT pipeline (flattened, single ball), so
    # the markers describe what ships today rather than a preview of what
    # per-track analysis would produce.
    flattened = track_candidates(candidates, max_pixels_per_frame=args.max_jump)
    player_boxes = cache.get("player_boxes") or None
    if args.people:
        # Same boxes the session report classifies impacts with, so the
        # markers on the video match the report's counts.
        from src.analysis.person_tracks import boxes_by_frame

        with open(args.people, "rb") as handle:
            player_boxes = boxes_by_frame(pickle.load(handle)["people"])
    analysis = analyze_impacts(
        flattened,
        fps,
        calibrations_by_frame=calibrations,
        player_boxes_by_frame=player_boxes,
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

    people = None
    if args.people:
        with open(args.people, "rb") as handle:
            people = pickle.load(handle)["people"]
        print(f"People: {args.people}")

    player_of, not_people = None, set()
    if args.players:
        # Stable players (assign_players.py): one label and one colour per
        # person for the whole clip, and the not-a-person tracklets hidden.
        import json

        players_doc = json.loads(Path(args.players).read_text())
        player_of = {int(t): int(p) for t, p in players_doc["player_of_tracklet"].items()}
        not_people = set(players_doc.get("not_people", []))
        print(f"Players: {len(set(player_of.values()))} from {args.players}")

    strokes_at: dict[int, list[dict]] = {}
    if args.strokes:
        # scripts/classify_strokes_bst.py output: each hit's label is shown
        # over the hitter from a little before contact to a second after.
        import json

        hits = json.loads(Path(args.strokes).read_text())["hits"]
        for hit in hits:
            for f in range(hit["frame"] - _STROKE_BEFORE, hit["frame"] + _STROKE_AFTER):
                strokes_at.setdefault(f, []).append(hit)
        print(f"Strokes: {len(hits)} hits from {args.strokes}")

    swings_at: dict[int, list[dict]] = {}
    swing_frames: dict[int, list[int]] = {}  # player (or tracklet) -> swing frames, for a running count
    if args.swings:
        # scripts/detect_swings.py output: "SWING" over the player around
        # each swing, and each player's running count in their label.
        import json

        swings = json.loads(Path(args.swings).read_text())["swings"]
        for sw in swings:
            for f in range(sw["frame"] - _SWING_BEFORE, sw["frame"] + _SWING_AFTER):
                swings_at.setdefault(f, []).append(sw)
            key = sw["player"] if sw["player"] is not None else -sw["tracklet"]
            swing_frames.setdefault(key, []).append(sw["frame"])
        print(f"Swings: {len(swings)} from {args.swings}")

    other_bounces = None
    if args.bounces_from:
        other_bounces = load_other_bounces(args.bounces_from, fps, width / 1280.0)
        print(f"Bounces: {len(other_bounces)} from {args.bounces_from} (replacing this pipeline's impact markers)")


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

        if args.court and frame_idx not in cutaway_frames:
            frame = court.draw(frame, calibrations.get(frame_idx))

        if args.candidates:
            for detection in candidates[frame_idx]:
                cv2.circle(frame, (int(detection.x), int(detection.y)), 9, _CANDIDATE_COLOR, 1)

        # A cutaway shows some other view, so ball tracks and impact markers
        # - all positioned against the court - mean nothing there, and the
        # bounce map re-projected onto it scatters crosses over the picture.
        # People are still people, so their boxes stay.
        in_cutaway = frame_idx in cutaway_frames
        live = 0
        for i, track in enumerate([] if in_cutaway else tracks):
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
            if detection is not None and args.track_labels:
                cv2.putText(
                    frame, str(i), (int(detection.x) + 17, int(detection.y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                )

        for x1, y1, x2, y2 in (cache.get("player_boxes") or {}).get(frame_idx, []):
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 60), 2)

        if people is not None and frame_idx < len(people):
            for track_id, x1, y1, x2, y2, _conf in people[frame_idx]:
                if track_id in not_people:
                    continue  # the ball cart, detected as a person
                if player_of is not None:
                    player = player_of.get(track_id)
                    color = _person_color(1000 + player) if player is not None else (160, 160, 160)
                else:
                    color = _person_color(track_id) if track_id is not None else (160, 160, 160)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                # The foot point is what gets projected onto the court for
                # distance and coverage, so it is drawn - if it sits off
                # the player's feet, every movement number is off too.
                cv2.circle(frame, (int((x1 + x2) / 2), int(y2)), 5, color, -1)
                if player_of is not None:
                    label = f"Player {player_of[track_id]}" if track_id in player_of else "?"
                else:
                    label = f"P{track_id}" if track_id is not None else "?"
                if swing_frames:
                    key = player_of.get(track_id) if player_of is not None and track_id in player_of else -(track_id or 0)
                    done = sum(1 for f in swing_frames.get(key, []) if f <= frame_idx)
                    label += f"  {done} swing{'s' if done != 1 else ''}"
                for sw in swings_at.get(frame_idx, []):
                    if sw["tracklet"] == track_id:
                        # With stroke labels, the stroke name takes the text slot.
                        _draw_swing(frame, sw, frame_idx, (int(x1), int(y1), int(x2), int(y2)), text=not strokes_at)
                cv2.putText(frame, label, (int(x1), int(y1) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
                cv2.putText(frame, label, (int(x1), int(y1) - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                for hit in strokes_at.get(frame_idx, []):
                    if hit["tracklet"] == track_id:
                        _draw_stroke(frame, hit, int(x1), int(y1) - 30)

        if not in_cutaway and other_bounces is not None:
            frame = _draw_other_bounces(frame, frame_idx, other_bounces, fps, args.bounces_label)
        elif not in_cutaway:
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
    parser.add_argument("--people", help="A scripts/track_people.py cache, to draw person boxes and ids")
    parser.add_argument(
        "--bounces-from",
        help="Draw another method's bounces instead of this pipeline's impacts - a "
        "scripts/tennisproject_bounces.py cache (the CatBoost model)",
    )
    parser.add_argument("--bounces-label", default="CatBoost")
    parser.add_argument("--players", help="scripts/assign_players.py players.json - label people as stable players")
    parser.add_argument("--swings", help="scripts/detect_swings.py output - mark each swing and count them per player")
    parser.add_argument("--strokes", help="scripts/classify_strokes_bst.py output - label each hit over its hitter")
    parser.add_argument(
        "--report",
        help="A session_report.json: draw its court (fitted from court-view shots only) and "
        "skip the court outline on its cutaways. Overrides --per-frame-court.",
    )
    parser.add_argument(
        "--on-court-only", action="store_true",
        help="Draw only ball tracks on this court (the ones the session metrics count)",
    )
    parser.add_argument(
        "--no-track-labels", dest="track_labels", action="store_false",
        help="Hide ball track numbers. They label a continuous stretch of tracking, not a ball, "
        "and are numbered best-path-first rather than in time order - a viewer reads '17' as "
        "'the 17th ball', which it is not.",
    )
    parser.add_argument(
        "--per-frame-court", dest="static_court", action="store_false",
        help="Refit the court every frame (the raw cache). Default is one static "
             "calibration, which is correct for a fixed camera and stops the jitter.",
    )
    parser.add_argument("--no-court", dest="court", action="store_false", help="Skip the court wireframe")
    parser.add_argument(
        "--no-candidates", dest="candidates", action="store_false",
        help="Skip the rejected-candidate dots",
    )
    render(parser.parse_args())


if __name__ == "__main__":
    main()
