"""Ensemble bounce detection: CatBoost's trajectory-shape model alone (see
`catboost_bounce_detector.py`) misses real bounces on this project's camera
angles no matter how its confidence threshold is tuned - it's trained on
different (broadcast) footage, and lowering the threshold recovers some
bounces but not others. Independently, AggieSportsAnalytics/CourtCheck (a
different tennis-analysis project, MIT licensed) hit the exact same wall on
their own non-broadcast camera and ended up training their own CatBoost
checkpoint rather than trusting the public broadcast-trained weights - not
something we can reuse (their retrained checkpoint isn't published), but
their pipeline's surrounding techniques are directly portable:

1. A permissive GEOMETRIC candidate scan (`find_trajectory_breakpoints` -
   every local-max-in-y point, i.e. every frame the ball reaches its lowest
   point on screen) has much higher recall than CatBoost for real bounces,
   since it isn't tied to any training distribution - it's just geometry.
   Taken alone it's far too trigger-happy (it also flags every racket
   contact and some pure noise), so it's used here as a second candidate
   SOURCE alongside CatBoost's own thresholded frames, unioned together,
   rather than as a replacement - recall from either, precision from the
   filters below.
2. A court-bounds PLAUSIBILITY GATE (`filter_bounces_off_court`): a
   candidate's pixel position, projected through the court homography,
   should land on or very near the actual court - a projected position many
   meters past a baseline or sideline is geometrically impossible as a real
   shot and is almost always the detector having briefly locked onto noise.
3. A +/-N frame BALL-POSITION FALLBACK when resolving a candidate's exact
   (x, y): the ball is often occluded at the exact instant of a bounce
   (motion blur, the player's own body/shadow), so requiring a tracked
   position on the bounce frame itself systematically drops real bounces
   rather than just widening the search a few frames either side.

A fourth candidate source, `find_speed_drop_candidates`, isn't from
CourtCheck - it's a direct physics signal neither the y-reversal scan nor
CatBoost's trajectory-shape features capture: a real bounce loses speed on
impact (the court absorbs energy), which shows up as a sudden drop in the
ball's REAL-WORLD speed (via the court homography, the same projection the
minimap uses) even on a flat, skidding shot whose on-screen y barely
reverses - exactly the case the y-reversal scan is blind to.

A sharp deceleration ALONE doesn't distinguish a bounce from a contact,
though - both a bounce and a racket hit slow the ball down for a frame or
two. What tells them apart is what happens next: a bounce is still the
same ball in free flight, so once the impact's over it resumes travelling
in roughly the SAME direction (gravity/momentum carries it on past the
bounce point) and its speed recovers. A contact hands the ball's momentum
to a racket - the ball goes wherever the player sends it, which is
frequently back toward where it came from, not onward in the same
direction. So the scan requires both: a sharp drop right after the
candidate frame, AND - a few frames further out, once any recovery has had
time to show up - a velocity vector that's both regained speed and still
points in roughly the same direction as the approach (a positive dot
product with the "before" vector). A contact typically fails the second
half: either the ball doesn't speed back up (absorbed/blocked) or it does
but in a different/reversed direction (redirected).

A fifth filter, `filter_bounces_near_net`, applies to every source alike
(not just the speed-drop scan) - measured directly on real footage,
candidates within ~7m of the net line were overwhelmingly false positives,
from CatBoost too, not just the speed-drop scan: the ground-plane
homography is least trustworthy exactly there (see that function's
docstring), so a ball merely passing over the net mid-flight can produce
the same kind of spurious signal a real bounce does, regardless of which
detector is looking at it.

`filter_bounces_near_players` (see catboost_bounce_detector.py) still runs
last, for the same reason it always did: neither the geometric scan nor
CatBoost can tell a court bounce from a racket contact by trajectory shape
alone, since both produce the same kind of sharp reversal.
"""

from typing import Optional

import numpy as np

from src.analysis.bounce_detector import BounceEvent, find_trajectory_breakpoints
from src.analysis.catboost_bounce_detector import CatBoostBounceDetector, filter_bounces_near_players
from src.analysis.court_calibration import CourtCalibration
from src.tracking.ball_tracker import TrackedPosition

# Doubles court width / full length, meters - see court_calibration.py.
_DOUBLES_WIDTH = 10.97
_COURT_LENGTH = 23.77
_NET_LINE_Y = _COURT_LENGTH / 2


def filter_bounces_off_court(bounces: list[BounceEvent], margin_m: float = 2.0) -> list[BounceEvent]:
    """Drop a candidate whose court-projected position is implausibly far
    outside the lines - a real shot, even a badly missed one, lands within
    a meter or two of them; anything further is detector noise (the ball
    briefly re-acquired at a wild pixel position, e.g. after leaving frame),
    not a call a line judge would ever face. A candidate with no world
    position (no calibration was available that frame) passes through
    untouched - there's nothing to judge it against.
    """
    kept = []
    for bounce in bounces:
        if bounce.world_x is None or bounce.world_y is None:
            kept.append(bounce)
            continue
        if -margin_m <= bounce.world_x <= _DOUBLES_WIDTH + margin_m and -margin_m <= bounce.world_y <= _COURT_LENGTH + margin_m:
            kept.append(bounce)
    return kept


def filter_bounces_near_net(bounces: list[BounceEvent], margin_m: float = 8.0) -> list[BounceEvent]:
    """Drop a candidate whose court-projected position lands within
    `margin_m` of the net line, regardless of which source(s) flagged it -
    the court homography is a ground-plane mapping (see
    `CourtCalibration`'s docstring), exact for a point actually on the
    court surface and increasingly wrong the higher above it the ball
    really is. A ball crossing the net is at its most elevated point right
    around there, so the projected position swings unreliably across a
    wide band on either side of the net line - not just exactly at it.

    Measured directly on real footage: candidates within ~7m of the net
    line were essentially all spurious (both from the speed-drop scan and
    from CatBoost), while genuine bounces clustered 8m+ from it, out where
    the ball is much closer to the ground when the trajectory-shape/
    velocity signals fire - hence the 8m default. This does mean a bounce
    that lands unusually close to the net (a drop shot, a ball just
    clearing the net before landing) won't get marked - an accepted
    trade-off given how dominated by false positives that band is.
    """
    kept = []
    for bounce in bounces:
        if bounce.world_y is None:
            kept.append(bounce)
            continue
        if abs(bounce.world_y - _NET_LINE_Y) >= margin_m:
            kept.append(bounce)
    return kept


def _world_point(calibrations_by_frame: dict[int, CourtCalibration], pos: TrackedPosition) -> Optional[np.ndarray]:
    calibration = calibrations_by_frame.get(pos.frame_idx)
    if calibration is None:
        return None
    return np.array(calibration.pixel_to_world(pos.x, pos.y), dtype=np.float64)


def find_speed_drop_candidates(
    positions: list[TrackedPosition],
    calibrations_by_frame: dict[int, CourtCalibration],
    fps: float,
    min_speed_before_kmh: float = 20.0,
    max_drop_ratio: float = 0.5,
    reaccel_frames: int = 4,
    min_reaccel_ratio: float = 0.4,
) -> set[int]:
    """Flags a frame where the ball's real-world velocity (via the court
    homography) has both a sharp drop AND resumes in roughly the same
    direction a few frames later - see this module's docstring for why
    both halves matter (a bare deceleration doesn't tell a bounce from a
    contact). Only considers consecutive REAL (non-interpolated)
    detections - an interpolated position is a straight line by
    construction and can't show a genuine deceleration or redirection.

    Doesn't filter near-net candidates itself - see
    `filter_bounces_near_net`, applied afterward to every source's
    candidates alike, not just this one (CatBoost fires near the net too).

    `min_speed_before_kmh` skips slow-moving stretches (e.g. a soft volley
    exchange near the net) where ordinary frame-to-frame noise can look
    like a large ratio-wise "drop" despite being a tiny absolute change.
    `max_drop_ratio` is how much of the incoming speed survives into the
    very next frame for the initial drop to count. `reaccel_frames` is how
    far past the candidate to look for the ball resuming its travel - the
    immediate next frame or two can still read as near-stationary right at
    a real bounce, so this deliberately looks further out than the initial
    drop check does. `min_reaccel_ratio` is how much of the "before" speed
    (per frame) that later velocity needs to have recovered.
    """
    ordered = sorted((p for p in positions if not p.interpolated), key=lambda p: p.frame_idx)
    candidates: set[int] = set()

    for i in range(1, len(ordered) - 1):
        prev_pos, cur_pos, next_pos = ordered[i - 1], ordered[i], ordered[i + 1]
        if cur_pos.frame_idx - prev_pos.frame_idx != 1 or next_pos.frame_idx - cur_pos.frame_idx != 1:
            continue  # not temporally adjacent - a real tracking gap, not a comparable pair

        # Walk forward from `next_pos` while frames stay consecutive, up to
        # reaccel_frames further, to find a later point to check recovery
        # against - stops early at the first gap, which naturally shrinks
        # (never grows) how far past the candidate this can see.
        after_idx = i + 1
        for j in range(i + 2, min(i + 2 + reaccel_frames, len(ordered))):
            if ordered[j].frame_idx - ordered[j - 1].frame_idx != 1:
                break
            after_idx = j
        after_pos = ordered[after_idx]

        w_prev, w_cur, w_next, w_after = (
            _world_point(calibrations_by_frame, prev_pos),
            _world_point(calibrations_by_frame, cur_pos),
            _world_point(calibrations_by_frame, next_pos),
            _world_point(calibrations_by_frame, after_pos),
        )
        if any(w is None for w in (w_prev, w_cur, w_next, w_after)):
            continue

        before_vec = w_cur - w_prev
        immediate_vec = w_next - w_cur
        after_vec = w_after - w_next

        distance_before = float(np.linalg.norm(before_vec))
        speed_before_kmh = distance_before * fps * 3.6
        if speed_before_kmh < min_speed_before_kmh:
            continue
        if float(np.linalg.norm(immediate_vec)) > distance_before * max_drop_ratio:
            continue  # no sharp initial drop - ordinary flight, not an impact

        elapsed_after = after_pos.frame_idx - next_pos.frame_idx
        if elapsed_after <= 0:
            continue
        after_speed_per_frame = float(np.linalg.norm(after_vec)) / elapsed_after
        if after_speed_per_frame < distance_before * min_reaccel_ratio:
            continue  # never sped back up - looks absorbed/blocked, not a bounce continuing onward
        if float(np.dot(before_vec, after_vec)) <= 0:
            continue  # resumed moving, but not the same way it was going - a contact redirected it

        candidates.add(cur_pos.frame_idx)

    return candidates


def _merge_nearby_frames(sorted_frames: list[int], min_gap: int) -> list[int]:
    """Collapse frames within `min_gap` of each other into one - CatBoost
    and the geometric scan often both flag the same physical bounce a frame
    or two apart, which would otherwise show up as two separate events."""
    if not sorted_frames:
        return []
    merged = [sorted_frames[0]]
    for frame in sorted_frames[1:]:
        if frame - merged[-1] > min_gap:
            merged.append(frame)
    return merged


def _resolve_position(
    x_ball: list, y_ball: list, frame_idx: int, fallback_frames: int
) -> Optional[tuple[float, float]]:
    if x_ball[frame_idx] is not None:
        return x_ball[frame_idx], y_ball[frame_idx]
    for delta in range(1, fallback_frames + 1):
        for candidate_frame in (frame_idx - delta, frame_idx + delta):
            if 0 <= candidate_frame < len(x_ball) and x_ball[candidate_frame] is not None:
                return x_ball[candidate_frame], y_ball[candidate_frame]
    return None


DEFAULT_SOURCES = frozenset({"catboost", "geometric", "speed_drop"})


def detect_bounces_ensemble(
    positions: list[TrackedPosition],
    catboost_model: Optional[CatBoostBounceDetector],
    num_frames: Optional[int] = None,
    calibrations_by_frame: Optional[dict[int, CourtCalibration]] = None,
    player_boxes_by_frame: Optional[dict[int, list[tuple[float, float, float, float]]]] = None,
    fps: Optional[float] = None,
    sources: frozenset[str] = DEFAULT_SOURCES,
    min_y_prominence: float = 15.0,
    min_frame_gap: int = 8,
    position_fallback_frames: int = 5,
    court_margin_m: float = 2.0,
    net_margin_m: float = 8.0,
    player_reach_margin: float = 50.0,
) -> list[BounceEvent]:
    """`sources` picks which candidate source(s) feed the union - a subset
    of {"catboost", "geometric", "speed_drop"} - mainly useful for
    isolating one signal to see how it performs alone. `catboost_model` may
    be None only when "catboost" isn't in `sources`."""
    if num_frames is None:
        num_frames = (max((p.frame_idx for p in positions), default=-1)) + 1

    x_ball: list = [None] * num_frames
    y_ball: list = [None] * num_frames
    for p in positions:
        x_ball[p.frame_idx] = p.x
        y_ball[p.frame_idx] = p.y

    catboost_frames: set[int] = set()
    if "catboost" in sources:
        if catboost_model is None:
            raise ValueError("'catboost' is in sources but no catboost_model was given")
        # predict_frames mutates x_ball/y_ball in place, filling short gaps
        # via cubic-spline extrapolation - reused below by _resolve_position too.
        catboost_frames = catboost_model.predict_frames(x_ball, y_ball)

    geometric_frames: set[int] = set()
    if "geometric" in sources:
        geometric_frames = set(
            find_trajectory_breakpoints(positions, min_y_prominence=min_y_prominence, min_frame_gap=min_frame_gap)
        )

    speed_drop_frames: set[int] = set()
    if "speed_drop" in sources and calibrations_by_frame is not None and fps is not None:
        speed_drop_frames = find_speed_drop_candidates(positions, calibrations_by_frame, fps)

    all_candidates = catboost_frames | geometric_frames | speed_drop_frames
    geometric_only = len(geometric_frames - catboost_frames)
    speed_drop_only = len(speed_drop_frames - catboost_frames - geometric_frames)
    candidate_frames = _merge_nearby_frames(sorted(all_candidates), min_frame_gap)
    print(
        f"Bounce candidates: {len(catboost_frames)} from CatBoost, "
        f"{geometric_only} added by the geometric scan, "
        f"{speed_drop_only} added by the speed-drop scan, {len(candidate_frames)} after merging nearby frames"
    )

    events = []
    for frame_idx in candidate_frames:
        resolved = _resolve_position(x_ball, y_ball, frame_idx, position_fallback_frames)
        if resolved is None:
            continue
        x, y = resolved
        world_x = world_y = None
        if calibrations_by_frame is not None:
            calibration = calibrations_by_frame.get(frame_idx)
            if calibration is not None:
                world_x, world_y = calibration.pixel_to_world(x, y)
        events.append(BounceEvent(frame_idx=frame_idx, x=x, y=y, world_x=world_x, world_y=world_y))

    before_court = len(events)
    events = filter_bounces_off_court(events, margin_m=court_margin_m)
    if len(events) < before_court:
        print(f"Dropped {before_court - len(events)} bounce candidate(s) with an implausible off-court position")

    before_net = len(events)
    events = filter_bounces_near_net(events, margin_m=net_margin_m)
    if len(events) < before_net:
        print(f"Dropped {before_net - len(events)} bounce candidate(s) too close to the net (unreliable ground projection)")

    if player_boxes_by_frame is not None:
        before_player = len(events)
        events = filter_bounces_near_players(events, player_boxes_by_frame, reach_margin=player_reach_margin)
        if len(events) < before_player:
            print(f"Dropped {before_player - len(events)} bounce candidate(s) near a player (likely contact)")

    return events
