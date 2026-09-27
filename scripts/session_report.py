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

def scene_cuts(video: str, cache: Path | None = None) -> list[int]:
    """Scene cuts in `video`, cached in `cache` (JSON) keyed on the video's
    path, size and modification time. Cut detection reads every frame -
    17s for a 108s clip, ~10 minutes for an hour - and the pipeline builds
    the report twice, so the second pass reuses the first's."""
    from src.analysis.scene_cuts import detect_scene_cuts
    from src.video.io import VideoReader

    stat = Path(video).stat()
    key = {"video": str(Path(video).resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if cache is not None and cache.exists():
        try:
            saved = json.loads(cache.read_text())
            if saved.get("key") == key:
                return [int(c) for c in saved["cuts"]]
        except (ValueError, KeyError):
            pass
    cuts = [int(c) for c in detect_scene_cuts(VideoReader(video).frames())]
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"key": key, "cuts": cuts}))
    return cuts


def fit_court(
    video: str | None, per_frame: dict, num_frames: int, fps: float, cuts_cache: Path | None = None
) -> tuple[CourtCalibration, list[tuple[int, int]], list[dict]]:
    """The court calibration for the whole clip, fitted ONLY from camera
    shots that actually show the court - plus those that do not, as
    cutaways, and a per-shot record.

    Each shot (between detected cuts) gets its own median calibration,
    which is checked against that shot's own frames (court_line_contrast).
    Shots that pass are pooled for the final static fit; the rest are
    cutaways. Fitting over every frame at once - the first version - holds
    only while most frames show the court: on the full Dingles video an
    11s intro and a minute of the coach talking at the net (~35% of
    frames) dragged the median off the lines, every shot then failed the
    check, and the report came out empty. A fixed fence camera has one
    shot, so this reduces to the plain static fit.

    Without a video to look at, every frame is trusted and nothing is a
    cutaway - there is no picture to check against."""
    if not video:
        return static_calibration_from_frames(per_frame), [], []

    import cv2

    from src.analysis.court_calibration import MIN_COURT_LINE_CONTRAST, court_line_contrast
    cuts = scene_cuts(video, cuts_cache)
    bounds = [0] + sorted(cuts) + [num_frames]
    capture = cv2.VideoCapture(video)

    def grays(a: int, b: int) -> list:
        out = []
        for f in np.linspace(a, b - 1, 6).astype(int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, image = capture.read()
            if ok:
                out.append(cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (3, 3), 0))
        return out

    def contrast(fit, images) -> float:
        return float(np.median([court_line_contrast(g, fit) for g in images])) if images else 0.0

    court_frames: dict = {}
    winners: list[tuple[CourtCalibration, list]] = []
    cutaways: list[tuple[int, int]] = []
    shots: list[dict] = []
    try:
        for a, b in zip(bounds, bounds[1:]):
            shot_cal = {f: per_frame[f] for f in range(a, b) if f in per_frame}
            score, images, shot_fit = 0.0, [], None
            if shot_cal:
                images = grays(a, b)
                shot_fit, score = best_fit(shot_cal, lambda fit: contrast(fit, images))
            court_view = score >= MIN_COURT_LINE_CONTRAST
            print(f"  shot {a / fps:6.1f}-{b / fps:6.1f}s  court-line contrast {score:5.1f} -> "
                  f"{'court view' if court_view else 'CUTAWAY'}")
            shots.append({"start_s": round(a / fps, 2), "end_s": round(b / fps, 2),
                          "court_line_contrast": round(score, 1), "court_view": court_view})
            if court_view:
                court_frames.update(shot_cal)
                winners.append((shot_fit, images))
            else:
                cutaways.append((a, b))
    finally:
        capture.release()

    if not court_frames:
        raise SystemExit(
            "No camera shot in this video shows the court clearly enough to calibrate "
            f"(best court-line contrast {max((s['court_line_contrast'] for s in shots), default=0):.1f}, "
            f"need {MIN_COURT_LINE_CONTRAST:g}). Check the camera can see the court's lines, then re-run."
        )
    # One calibration for the clip: each court shot's winner, scored on the
    # frames of every court shot, best overall.
    everything = [g for _, images in winners for g in images]
    return max((fit for fit, _ in winners), key=lambda fit: contrast(fit, everything)), cutaways, shots


MAX_CANDIDATES = 24


def best_fit(per_frame: dict, score) -> tuple[CourtCalibration, float]:
    """The calibration for one shot: the median fit, or one of the
    detector's own per-frame fits, whichever lines up best with the image
    (`score`, higher is better). The median alone is not safe - on Dingles
    the near-corner keypoints split into two clusters ~80px apart (about
    700 frames vs 500 in one shot), so the median sits on a knife edge and
    sampling the detector every 25th frame instead of every frame tipped it
    to the wrong side (contrast -6 against 28). Scoring against the picture
    settles it whichever side the median falls."""
    candidates = [static_calibration_from_frames(per_frame)]
    seen = set()
    distinct = []
    for f in sorted(per_frame):
        key = np.asarray(per_frame[f].homography).round(6).tobytes()
        if key not in seen:
            seen.add(key)
            distinct.append(per_frame[f])
    step = max(1, len(distinct) // MAX_CANDIDATES)
    candidates += distinct[::step][:MAX_CANDIDATES]
    scored = [(score(c), i, c) for i, c in enumerate(candidates)]
    s, _, fit = max(scored, key=lambda t: (t[0], -t[1]))
    return fit, s


def add_structure(report: dict, tracks: dict, players_doc: dict, fps: float, num_frames: int, cutaway: set) -> None:
    """Drill/break segments, per-player queue and idle time, and a movement
    heatmap per player (src/session/structure.py)."""
    from src.analysis.person_tracks import coverage_grid, smooth_positions
    from src.session.structure import (
        BREAK_S,
        drill_segments,
        intensity_windows,
        players_gathered,
        queue_and_idle,
        zone_shares,
    )

    analysed = np.ones(num_frames, dtype=bool)
    analysed[list(f for f in cutaway if 0 <= f < num_frames)] = False
    in_play = np.zeros(num_frames, dtype=bool)
    for a_s, b_s in report["raw"]["in_play_runs_s"]:
        in_play[int(round(a_s * fps)):int(round(b_s * fps))] = True
    segments = drill_segments(in_play, analysed, fps)
    drill_mask = np.zeros(num_frames, dtype=bool)
    for seg in segments:
        if seg.kind == "drill":
            drill_mask[seg.start:seg.end] = True
    drill_mask &= analysed

    positions_by_player = {}
    for player, tracklets in players_doc["players"].items():
        merged = {}
        for t in tracklets:
            merged.update(smooth_positions(tracks.get(int(t), {})))
        positions_by_player[int(player)] = merged
    for seg in segments:
        if seg.kind == "break":
            seg.gathered = players_gathered(positions_by_player, range(seg.start, seg.end), fps)
    by_player = {p["player"]: p for p in report.get("players", [])}
    for player, positions in positions_by_player.items():
        if player in by_player:
            by_player[player].update(queue_and_idle(positions, fps, drill_mask))
            by_player[player]["zone_shares"] = zone_shares(positions, drill_mask)
    report["raw"]["intensity"] = intensity_windows(positions_by_player, in_play & analysed, drill_mask, fps)
    report["segments"] = [s.to_dict(fps) for s in segments]
    report["raw"]["player_coverage"] = {
        str(player): coverage_grid(tracks, fps, track_ids=[int(t) for t in tracklets]).to_dict()
        for player, tracklets in players_doc["players"].items()
    }

    drills = [s for s in segments if s.kind == "drill"]
    breaks = [s for s in segments if s.kind == "break"]
    report["metrics"]["drills"] = {
        "value": len(drills),
        "unit": "drill segments",
        "basis": "estimated",
        "method": f"Ball-in-play stretches joined across gaps under {BREAK_S:g}s of analysed time; a longer gap "
                  "is a break. Camera cutaways do not split a drill.",
        "caveats": [
            f"{BREAK_S:g}s is a convention, not measured: no footage with several drills has been checked yet.",
            "Segments are not named: the tool cannot tell which drill is which.",
        ],
        "breaks": len(breaks),
        "breaks_with_players_gathered": sum(1 for s in breaks if s.gathered),
    }
    queue = [p.get("queue_share") for p in report.get("players", []) if p.get("queue_share") is not None]
    report["metrics"]["queue_share"] = {
        "value": round(float(np.mean(queue)), 3) if queue else None,
        "unit": "average share of a player's drill time spent waiting off court",
        "basis": "estimated",
        "method": "Per player: standing still for 3s or more (under 0.6m moved in 2s) outside the court lines, "
                  "during drills only. Averaged over players.",
        "caveats": [
            "Includes the coach, who stands beside the court by design - coach and players are not yet told apart.",
            "A player standing behind the baseline between rallies for 3s+ also counts as waiting.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ball-cache", required=True)
    parser.add_argument("--people", help="scripts/track_people.py cache")
    parser.add_argument("--video", help="Source video, to detect camera cutaways")
    parser.add_argument("--second-opinion", help="scripts/tennisproject_bounces.py cache, to mark agreed bounces")
    parser.add_argument("--name", help="Clip name (default: the cache's folder)")
    parser.add_argument("--players", help="scripts/assign_players.py players.json - per-player section")
    parser.add_argument("--swings", help="scripts/detect_swings.py output - adds fed vs live play")
    parser.add_argument("--targets", help="JSON of coaching targets overriding the defaults (src/session/insights.py Targets)")
    parser.add_argument("--coach-player", type=int, help="Which player is the coach, if the guess is wrong")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.ball_cache, "rb") as handle:
        cache = pickle.load(handle)
    fps, num_frames = cache["fps"], cache["num_frames"]
    per_frame = {
        f: (v if isinstance(v, CourtCalibration) else CourtCalibration(homography=v))
        for f, v in cache["calibrations"].items()
    }
    static, cutaways, shots = fit_court(args.video, per_frame, num_frames, fps,
                                         cuts_cache=Path(args.out).parent / "scene_cuts.json")
    calibrations = {f: static for f in range(num_frames)}
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
        people_tracked=people_cache is not None,
    )
    if args.players:
        from src.session.report import player_summaries

        players_doc = json.loads(Path(args.players).read_text())
        report["players"] = player_summaries(people, players_doc, fps)
        not_people = set(players_doc.get("not_people", []))
        report["people"] = [p for p in report["people"] if p["track_id"] not in not_people]
        report["metrics"]["players_identified"] = {
            "value": len(report["players"]),
            "unit": "players",
            "basis": "estimated",
            "method": "Tracklets joined into players by clothing colour, never merging two people seen on "
                      "screen at the same time in different places - see src/analysis/identity.py.",
            "caveats": [
                "Relies on players wearing different clothing.",
                "Includes the coach - coach and players are not yet told apart.",
            ],
        }

    if args.players and people_cache:
        add_structure(report, tracks, players_doc, fps, num_frames, skip)

    from src.session.insights import Targets, build_summary, load_targets, play_note

    swings = []
    if args.swings:
        swings = [s for s in json.loads(Path(args.swings).read_text())["swings"] if s["frame"] not in skip]
        # Balls hit per player: their swings, per minute they were on camera
        # during drills (so a player who arrived late is not penalised).
        for p in report.get("players") or []:
            n = sum(1 for s in swings if s["player"] == p["player"])
            p["shots"] = n
            minutes = (p.get("on_camera_in_drills_s") or p.get("on_camera_s") or 0) / 60.0
            p["shots_per_minute"] = round(n / minutes, 2) if minutes > 0 else None

    report["summary"] = build_summary(
        report, load_targets(args.targets) if args.targets else Targets(), coach_override=args.coach_player
    )
    if args.swings:
        # Fed or live needs the coach, which the summary has just decided.
        from src.session.feeding import feeding_summary

        feeding = report["feeding"] = feeding_summary(swings, fps, report["summary"]["coach_player"])
        caveats = feeding["caveats"]
        live_share = feeding["live_rallies"] / feeding["rallies"] if feeding["rallies"] else None
        report["metrics"]["live_rally_share"] = {
            "value": round(live_share, 3) if live_share is not None else None, "unit": "share of rallies started by a player",
            "basis": "estimated", "method": feeding["method"], "caveats": caveats}
        report["metrics"]["shots_per_rally"] = {
            "value": feeding["shots_per_rally"]["all"], "unit": "shots per rally (detected swings)",
            "basis": "estimated", "method": feeding["method"], "caveats": caveats}
        report["metrics"]["coach_shot_share"] = {
            "value": feeding["coach_shot_share"] if feeding["coach_player"] is not None else None,
            "unit": "share of all shots hit by the coach", "basis": "estimated", "method": feeding["method"],
            "caveats": caveats}
        note = play_note(report["feeding"])
        if note:
            report["summary"]["notes"].insert(0, note)

    # The court this report was measured on, so everything downstream (the
    # rendered video) uses the same one rather than re-deriving its own.
    report["court"] = {"homography": np.asarray(static.homography).tolist(), "shots": shots}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"Wrote {out}")
    for name, metric in report["metrics"].items():
        print(f"  {name:30} {metric['value']!s:>8}  {metric['unit']:38} [{metric['basis']}]")


if __name__ == "__main__":
    main()
