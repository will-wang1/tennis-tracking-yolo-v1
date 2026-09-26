"""Fold one coaching clip's cached detections into a SESSION REPORT: the
structured record a model reads to write coaching feedback, with the raw
events alongside it for anyone who wants to check.

THE CONTRACT. Every metric is an object, never a bare number:

    {"value": ..., "unit": ..., "basis": ..., "method": ..., "caveats": [...]}

`basis` is one of:
    "measured"     - a direct reading this pipeline has been checked on
    "estimated"    - a real reading with a known, material error source
    "lower_bound"  - the true value is at least this; the gap is stated

A model interpreting this file cannot tell a measurement from a plausible
guess unless the file says which is which, and a confident sentence built
on an estimated number is the failure this whole structure exists to
prevent. So `basis` and `caveats` are part of the value, not decoration,
and the report carries its own reading instructions under `how_to_read`.

WHAT IS DELIBERATELY NOT HERE: any judgement of coaching quality. This
measures activity - balls, movement, time in play. Whether a drill was a
good choice is the reader's call, made with the numbers in hand.
"""

from collections import Counter
from typing import Iterable, Optional

import numpy as np

from src.analysis.court_calibration import CourtCalibration
from src.analysis.court_zones import classify_landing_zone
from src.analysis.impact_pipeline import ImpactAnalysis
from src.analysis.person_tracks import (
    COURT_LENGTH_M,
    COURT_WIDTH_M,
    MOVING_SPEED_MPS,
    CoverageGrid,
    PersonMovement,
)
from src.tracking.candidate_tracker import BallTrack, count_simultaneous_frames

SCHEMA_VERSION = "0.1"

# A ball track must spend at least this share of its positions on or near
# THIS court to count. Measured need: on dingles_serve_volley 12 of 32
# tracks sit mostly over the adjacent court or on a person by the ball
# cart; at 0.6 exactly those drop.
ON_COURT_TRACK_FRACTION = 0.6
BALL_SIDE_MARGIN_M = 2.0
BALL_END_MARGIN_M = 4.0

# A ball lost for under this long - hidden behind a player, blurred - is
# still in play; longer than this, play is taken to have stopped.
IN_PLAY_BRIDGE_S = 1.0


def _metric(value, unit, basis, method, caveats=(), **extra) -> dict:
    out = {"value": value, "unit": unit, "basis": basis, "method": method, "caveats": list(caveats)}
    out.update(extra)
    return out


def ball_on_court(position_m: tuple[float, float]) -> bool:
    x, y = position_m
    return (
        -BALL_SIDE_MARGIN_M <= x <= COURT_WIDTH_M + BALL_SIDE_MARGIN_M
        and -BALL_END_MARGIN_M <= y <= COURT_LENGTH_M + BALL_END_MARGIN_M
    )


def on_court_tracks(tracks: Iterable[BallTrack], calibration: CourtCalibration) -> list[BallTrack]:
    """Ball tracks that belong to THIS court. Judged on the share of a
    track's positions over the court rather than any single one, because a
    genuinely airborne ball always projects somewhat off the ground plane
    and a whole-track test tolerates that where a per-point test cannot."""
    kept = []
    for track in tracks:
        points = [calibration.pixel_to_world(d.x, d.y) for d in track.detections.values()]
        if points and np.mean([ball_on_court(p) for p in points]) >= ON_COURT_TRACK_FRACTION:
            kept.append(track)
    return kept


def in_play_mask(
    tracks: Iterable[BallTrack], num_frames: int, fps: float, bridge_s: float = IN_PLAY_BRIDGE_S
) -> np.ndarray:
    """Frame-wise: is a ball in play? Every frame from a track's first to
    last sighting counts, then gaps shorter than `bridge_s` between two
    in-play stretches are filled. A gap at the very start or end is never
    bridged - there is no play on both sides of it."""
    live = np.zeros(num_frames, dtype=bool)
    for track in tracks:
        live[track.start_frame: track.end_frame + 1] = True
    bridge = int(round(bridge_s * fps))
    i = 0
    while i < num_frames:
        if live[i]:
            i += 1
            continue
        j = i
        while j < num_frames and not live[j]:
            j += 1
        if i > 0 and j < num_frames and (j - i) < bridge:
            live[i:j] = True
        i = j
    return live


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) frame ranges where `mask` is True."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def build_session_report(
    *,
    clip_name: str,
    fps: float,
    num_frames: int,
    calibration: CourtCalibration,
    ball_tracks: list[BallTrack],
    impacts: ImpactAnalysis,
    people: list[PersonMovement],
    coverage: Optional[CoverageGrid],
    cutaways: Iterable[tuple[int, int]] = (),
    second_opinion_bounce_frames: Optional[set[int]] = None,
    people_tracked: Optional[bool] = None,
) -> dict:
    """The report as a JSON-ready dict. `cutaways` are [start, end) frame
    ranges where the video shows something other than the fixed court view;
    they are excluded from every time-based metric rather than counted as
    idle. `second_opinion_bounce_frames` (another method's bounces, in THIS
    clip's frame numbers) marks each bounce as agreed or not.

    `people_tracked` says whether person tracking was RUN (default: whether
    any people were passed). When it was not, the people metrics are
    reported as null with the reason, never as zero - "0 m run" and
    "0 contacts credited" would be confident, wrong readings of a
    measurement that simply was not taken."""
    cut_mask = np.zeros(num_frames, dtype=bool)
    cut_list = [(max(0, a), min(num_frames, b)) for a, b in cutaways]
    for a, b in cut_list:
        cut_mask[a:b] = True
    usable_frames = int((~cut_mask).sum())
    usable_s = usable_frames / fps
    minutes = usable_s / 60.0

    # ---- ball in play
    court_tracks = on_court_tracks(ball_tracks, calibration)
    live = in_play_mask(court_tracks, num_frames, fps) & ~cut_mask
    in_play_fraction = float(live.sum()) / usable_frames if usable_frames else 0.0
    play_runs = runs(live)
    multi = count_simultaneous_frames(court_tracks)
    multi_frames = sum(1 for f in multi if not cut_mask[f])

    # ---- impacts
    usable_impacts = [i for i in impacts.impacts if not cut_mask[min(i.frame_idx, num_frames - 1)]]
    kinds = Counter(i.kind for i in usable_impacts)
    total = len(usable_impacts)
    unknown_share = kinds.get("unknown", 0) / total if total else 0.0

    # ---- bounces
    bounces = []
    for td in impacts.touchdowns:
        f = td.impact.frame_idx
        if td.kind != "bounce" or cut_mask[min(f, num_frames - 1)]:
            continue
        wx, wy = calibration.pixel_to_world(td.impact.x, td.impact.y)
        zone = classify_landing_zone(wx, wy)
        entry = {
            "t_s": round(f / fps, 2),
            "frame": f,
            "position_m": [round(float(wx), 2), round(float(wy), 2)],
            "zone": zone.label(),
            "half": zone.half,
            "side": zone.side,
            "depth": zone.depth,
            "bounds": zone.bounds,
        }
        if second_opinion_bounce_frames is not None:
            entry["agreed_by_second_method"] = any(abs(f - g) <= round(0.25 * fps) for g in second_opinion_bounce_frames)
        bounces.append(entry)
    zone_counts = Counter(b["zone"] for b in bounces)
    empty_quadrants = [
        f"{h}_{s}" for h in ("far", "near") for s in ("left", "right")
        if not any(b["half"] == h and b["side"] == s for b in bounces)
    ]

    # ---- people
    if people_tracked is None:
        people_tracked = bool(people)
    on_court_people = [p for p in people if p.on_this_court]
    measured_people = [p for p in on_court_people if p.distance_confidence == "measured"]
    timed = [p for p in measured_people if p.moving_share is not None]
    moving = (
        sum(p.moving_share * p.tracked_frames for p in timed) / sum(p.tracked_frames for p in timed)
        if timed else None
    )
    attributed = sum(p.confirmed_contacts or 0 for p in on_court_people)
    total_contacts = kinds.get("contact", 0)

    report = {
        "schema_version": SCHEMA_VERSION,
        "how_to_read": (
            "Every metric has a `basis`: 'measured' (trustworthy reading), 'estimated' (real reading "
            "with a known material error), or 'lower_bound' (true value is at least this). Do not "
            "state an estimated or lower-bound value as fact; say what it is. `caveats` list the known "
            "failure modes. This report measures activity only - it contains no judgement of "
            "coaching quality, and none should be attributed to it."
        ),
        "clip": {
            "name": clip_name,
            "fps": fps,
            "frames": num_frames,
            "duration_s": round(num_frames / fps, 2),
            "analysed_s": round(usable_s, 2),
            "cutaways_s": [[round(a / fps, 2), round(b / fps, 2)] for a, b in cut_list],
        },
        "metrics": {
            "ball_in_play": _metric(
                round(in_play_fraction, 3),
                "fraction of analysed time",
                "estimated",
                "Ball tracks kept only if >=60% of their positions are on this court; a frame is in "
                f"play between a kept track's first and last sighting; gaps under {IN_PLAY_BRIDGE_S:g}s "
                "are bridged; camera cutaways excluded from numerator and denominator.",
                [
                    "Cannot tell instruction time from idle time: both are simply no ball in play (no audio by design).",
                    "A continuous drill will read high by nature; compare like with like.",
                ],
                seconds=round(float(live.sum()) / fps, 1),
            ),
            "impacts_per_minute": _metric(
                round(total / minutes, 1) if minutes else None,
                "impacts / minute",
                "estimated",
                "Every sharp change in a ball's path, from the flattened single-ball track.",
                ["Counts ball direction changes of any kind: bounces, racket contacts and unattributed kinks."],
                count=total,
                by_kind=dict(kinds),
            ),
            "confirmed_contacts_per_minute": _metric(
                round(kinds.get("contact", 0) / minutes, 1) if minutes else None,
                "racket contacts / minute",
                "lower_bound",
                "Impacts classified as a racket contact from the ball's motion before and after.",
                [
                    f"{unknown_share:.0%} of impacts could not be classified; the true number of balls hit "
                    "lies between this and impacts_per_minute.",
                ],
            ),
            "two_balls_in_play": _metric(
                round(multi_frames / usable_frames, 3) if usable_frames else None,
                "fraction of analysed time",
                "estimated",
                "Frames where two or more on-court ball tracks are live at once.",
                ["~2.5% of match-footage frames show this with only one ball in play: treat small values as noise."],
            ),
            "bounce_landing": _metric(
                len(bounces),
                "confirmed bounces",
                "estimated",
                "Impacts classified as a bounce, mapped to court metres through one static court calibration.",
                [
                    "Only confirmed bounces are mapped; unclassified impacts are not.",
                    "Far-half positions are ~10x less precise than near-half (fence-height camera).",
                ]
                + ([f"No confirmed bounces in: {', '.join(empty_quadrants)} - check the video before reading this as a pattern."] if empty_quadrants else []),
                zone_counts=dict(zone_counts),
                out_count=sum(1 for b in bounces if b["bounds"] == "out"),
            ),
            "people_on_court": _metric(
                len(on_court_people),
                "tracked identities",
                "estimated",
                "Person detections tracked across frames (YOLOv8s + ByteTrack), kept if their median foot position is on this court or its run-off.",
                [
                    "An identity is one continuous stretch on screen, not a named player: a person hidden too long, or a camera cut, starts a new one.",
                    "Coach and players are not yet told apart.",
                ],
            ),
            "movement_near_court": _metric(
                round(sum(p.distance_m or 0 for p in measured_people), 1),
                "metres, summed over near-court identities",
                "measured",
                "Foot point per frame through the court calibration, 0.5s rolling median, frame-to-frame distance with implausible-speed steps dropped.",
                ["Sums identities, not people. One person split across two identities appears twice, each with part of the distance; the total is unaffected."],
                identities=len(measured_people),
            ),
            "near_court_moving_share": _metric(
                None if moving is None else round(moving, 3),
                "fraction of near-court players' time spent moving",
                "estimated",
                f"Per near-court identity, the share of 1-second windows in which they covered ground at "
                f"{MOVING_SPEED_MPS:g} m/s or more; averaged across identities, weighted by time on camera. "
                "The rest of the time is standing or shuffling: a work:rest split.",
                [
                    f"{MOVING_SPEED_MPS:g} m/s is a convention (roughly where shuffling in a ready position ends), "
                    "not a measured boundary - the data has no natural gap to place it in.",
                    "Near-court players only: far-court speeds are too noisy at this camera height.",
                ],
            ),
            "contacts_attributed": _metric(
                attributed,
                f"of {total_contacts} confirmed contacts credited to a tracked person",
                "lower_bound",
                "Each confirmed racket contact is credited to the tracked person nearest the ball at that "
                "frame, if within 0.6 of their box height - the same reach test the classifier uses.",
                [
                    "Only CONFIRMED contacts are credited, so every per-person count is a minimum.",
                    "Credits go to identities, not named players; the coach can be credited with feeds.",
                ],
            ),
        },
        "people": [p.to_dict(fps) for p in on_court_people],
        "raw": {
            "in_play_runs_s": [[round(a / fps, 2), round(b / fps, 2)] for a, b in play_runs],
            "impacts": [
                {"t_s": round(i.frame_idx / fps, 2), "frame": i.frame_idx, "kind": i.kind}
                for i in usable_impacts
            ],
            "bounces": bounces,
            "coverage": coverage.to_dict() if coverage is not None else None,
            "ball_tracks": {"total": len(ball_tracks), "on_this_court": len(court_tracks)},
        },
    }
    if not people_tracked:
        for name in ("people_on_court", "movement_near_court", "near_court_moving_share", "contacts_attributed"):
            metric = report["metrics"][name]
            metric["value"] = None
            metric["caveats"] = ["Person tracking was not run for this session, so this was not measured."]
    return report


STROKE_TYPES = ("serve", "overhead", "forehand", "backhand", "forehand_volley", "backhand_volley", "slice")


def player_summaries(people: list[PersonMovement], players_doc: dict, strokes_doc: Optional[dict], fps: float) -> list[dict]:
    """One record per stable player (identity.py), folding together every
    tracklet that player was seen in.

    Distance is summed over tracklets and split into its measured and
    estimated parts, because a player who rotates between the near and far
    halves has some of each and a single total would hide which part to
    trust. Moving share is weighted by time on camera. Hits and stroke mix
    count CONFIRMED hits only (ball within reach) - a swing with no ball
    near is not a hit."""
    by_track = {p.track_id: p for p in people}
    strokes_by_track: dict[int, list[dict]] = {}
    for s in (strokes_doc or {}).get("strokes", []):
        strokes_by_track.setdefault(int(s["track_id"]), []).append(s)
    out = []
    for player, tracklets in sorted(players_doc["players"].items(), key=lambda kv: int(kv[0])):
        moves = [by_track[t] for t in tracklets if t in by_track]
        frames = sum(m.tracked_frames for m in moves)
        measured = sum(m.distance_m or 0.0 for m in moves if m.distance_confidence == "measured")
        estimated = sum(m.distance_m or 0.0 for m in moves if m.distance_confidence != "measured")
        timed = [m for m in moves if m.moving_share is not None]
        moving = (
            sum(m.moving_share * m.tracked_frames for m in timed) / sum(m.tracked_frames for m in timed)
            if timed else None
        )
        hits = [s for t in tracklets for s in strokes_by_track.get(t, []) if s["hit_confirmed"]]
        swings = [s for t in tracklets for s in strokes_by_track.get(t, [])]
        hands = [s["racket_hand"] for s in swings if s.get("racket_hand_source") == "racket"]
        mix = {k: sum(1 for s in hits if s["stroke"] == k) for k in STROKE_TYPES}
        starts = [m.start_frame for m in moves] or [0]
        ends = [m.end_frame for m in moves] or [0]
        out.append({
            "player": int(player),
            "tracklets": sorted(tracklets),
            "first_seen_s": round(min(starts) / fps, 1),
            "last_seen_s": round(max(ends) / fps, 1),
            "on_camera_s": round(frames / fps, 1),
            "distance_m": round(measured + estimated, 1),
            "distance_measured_m": round(measured, 1),
            "distance_estimated_m": round(estimated, 1),
            "moving_share": None if moving is None else round(moving, 3),
            "confirmed_hits": len(hits),
            "hits_per_minute_on_camera": round(len(hits) / (frames / fps / 60.0), 1) if frames else None,
            "stroke_mix": mix,
            "swings_without_ball": len(swings) - len(hits),
            "racket_hand": (max(set(hands), key=hands.count) if hands else None),
        })
    return out
