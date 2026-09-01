"""Bounce detection anchored on the one signal that's actually tied to
physical ground contact: the ball reaching its lowest point on SCREEN
(a local max in pixel y) and rising again. Unlike `velocity_bounce_detector.py`
and `bounce_ensemble.py`'s speed-drop scan, this never touches the court
homography at all - it works in raw pixel space, so it's immune to the
ground-plane projection errors that make a ball crossing the net (still
airborne, not on the ground) mimic a bounce-like signal in world
coordinates. It only converts to world coordinates afterward, to report
where a confirmed bounce landed.

This wasn't the first thing tried this session - CatBoost (trajectory
shape) and two velocity-based approaches (see `bounce_ensemble.py` and
`velocity_bounce_detector.py`) were tried first and each had real
weaknesses: CatBoost under-recalls on this project's camera, and a "same
direction = bounce, reversed = contact" velocity check turns out not to
discriminate well in practice - a rally shot usually keeps heading toward
the opponent's side whether it just bounced OR was just hit, so direction
alone doesn't reliably tell them apart, and the net-crossing homography
error can still fool a velocity signal built from projected coordinates.
Measured directly: running `find_trajectory_breakpoints` (the geometric
local-max-in-y scan that already existed in this codebase) on the
trajectory THIS PIPELINE ACTUALLY SMOOTHS FOR DISPLAY (window=9) found
almost nothing - the smoothing filter was rounding real bounce corners
away before the scan ever saw them. Re-running it on a much more lightly
smoothed trajectory (window=3, still enough to remove single-pixel
detector jitter) recovered a clean, plausible set of candidates instead,
several of them near the net line - real near-net bounces that both
world-space approaches got tangled up on.

Only `filter_bounces_near_players` (contacts happen at a player, not out
on open court) and a loose `filter_bounces_off_court` sanity check apply
here - no near-net exclusion, since the whole point of working in pixel
space is not needing one.
"""

from typing import Optional

from src.analysis.bounce_detector import BounceEvent, find_trajectory_breakpoints
from src.analysis.catboost_bounce_detector import filter_bounces_near_players
from src.analysis.bounce_ensemble import filter_bounces_off_court
from src.analysis.court_calibration import CourtCalibration
from src.tracking.ball_tracker import TrackedPosition


def detect_bounces_geometric(
    positions: list[TrackedPosition],
    calibrations_by_frame: Optional[dict[int, CourtCalibration]] = None,
    player_boxes_by_frame: Optional[dict[int, list[tuple[float, float, float, float]]]] = None,
    min_y_prominence: float = 3.0,
    min_frame_gap: int = 5,
    court_margin_m: float = 2.0,
    player_reach_margin: float = 50.0,
) -> list[BounceEvent]:
    """`positions` should come from a LIGHTLY smoothed tracking pass
    (e.g. `BallTracker(smoothing_window=3)`) - the window this project uses
    for the visible trail/speed readings (9) smooths away the corner this
    function looks for. `min_y_prominence`/`min_frame_gap` are deliberately
    much lower than the legacy `detect_bounces`' defaults (15.0/5) - this
    project's ball is small on screen, especially far-court, so a real
    bounce's corner is a few pixels of prominence, not fifteen.
    """
    candidate_frames = find_trajectory_breakpoints(
        positions, min_y_prominence=min_y_prominence, min_frame_gap=min_frame_gap
    )
    positions_by_frame = {p.frame_idx: p for p in positions}

    events = []
    for frame_idx in candidate_frames:
        pos = positions_by_frame[frame_idx]
        world_x = world_y = None
        if calibrations_by_frame is not None:
            calibration = calibrations_by_frame.get(frame_idx)
            if calibration is not None:
                world_x, world_y = calibration.pixel_to_world(pos.x, pos.y)
        events.append(BounceEvent(frame_idx=frame_idx, x=pos.x, y=pos.y, world_x=world_x, world_y=world_y))

    if calibrations_by_frame is not None:
        before_court = len(events)
        events = filter_bounces_off_court(events, margin_m=court_margin_m)
        if len(events) < before_court:
            print(f"Dropped {before_court - len(events)} bounce candidate(s) with an implausible off-court position")

    if player_boxes_by_frame is not None:
        before_player = len(events)
        events = filter_bounces_near_players(events, player_boxes_by_frame, reach_margin=player_reach_margin)
        if len(events) < before_player:
            print(f"Dropped {before_player - len(events)} bounce candidate(s) near a player (likely contact)")

    return events
