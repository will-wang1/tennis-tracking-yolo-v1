"""Standalone bounce detector based purely on the ball's minimap (world-
space, via the court homography) velocity - no CatBoost, no pixel-space
geometric heuristics, no spatial exclusion zones. Deliberately independent
of `bounce_ensemble.py`: that module's CatBoost/geometric/off-court/net-
margin machinery is a different, more conservative approach; this is the
simpler physics-only signal on its own, to reason about and tune in
isolation.

The physics: a real bounce loses speed on impact (the court absorbs
energy) but is still the same ball in free flight afterward, so it resumes
travelling in roughly the SAME direction once the impact's over. A racket
CONTACT can produce an identical-looking speed drop, but a human redirects
the ball - it goes wherever they send it, which very often means the
velocity direction reverses or changes rather than continuing onward. So:

    big change in speed, direction UNCHANGED  -> bounce
    big change in speed, direction CHANGED     -> contact (not a bounce)

This trades off recall for lack of any spatial exclusion (see
`bounce_ensemble.filter_bounces_near_net` for the alternative, which
excludes a band around the net to avoid a different problem - a ball
crossing the net is elevated, and the ground-plane homography is least
reliable exactly there; that trade real near-net bounces for fewer
projection artifacts. This module makes no such trade at all: a bounce
close to the net is a real, common shot in tennis, so it counts on the
direction check alone to tell it apart from an elevated ball merely
passing through, rather than papering over the gap with a spatial cut.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.analysis.bounce_detector import BounceEvent
from src.analysis.court_calibration import CourtCalibration
from src.tracking.ball_tracker import TrackedPosition


@dataclass
class _WorldPoint:
    frame_idx: int
    x: float  # pixel
    y: float  # pixel
    world: np.ndarray  # (2,) meters


def _world_points(
    positions: list[TrackedPosition], calibrations_by_frame: dict[int, CourtCalibration]
) -> list[_WorldPoint]:
    """Real (non-interpolated) positions only, projected to world
    coordinates, sorted by frame - an interpolated position is a straight
    line by construction and can't show a genuine deceleration or
    redirection, and a frame with no calibration can't be projected at all."""
    points = []
    for pos in sorted((p for p in positions if not p.interpolated), key=lambda p: p.frame_idx):
        calibration = calibrations_by_frame.get(pos.frame_idx)
        if calibration is None:
            continue
        world = np.array(calibration.pixel_to_world(pos.x, pos.y), dtype=np.float64)
        points.append(_WorldPoint(frame_idx=pos.frame_idx, x=pos.x, y=pos.y, world=world))
    return points


def _merge_nearby_frames(sorted_frames: list[int], min_gap: int) -> list[int]:
    if not sorted_frames:
        return []
    merged = [sorted_frames[0]]
    for frame in sorted_frames[1:]:
        if frame - merged[-1] > min_gap:
            merged.append(frame)
    return merged


def detect_bounces_by_velocity(
    positions: list[TrackedPosition],
    calibrations_by_frame: dict[int, CourtCalibration],
    fps: float,
    min_speed_before_kmh: float = 20.0,
    max_drop_ratio: float = 0.5,
    reaccel_frames: int = 4,
    min_reaccel_ratio: float = 0.4,
    min_frame_gap: int = 8,
) -> list[BounceEvent]:
    """`min_speed_before_kmh` skips slow-moving stretches (e.g. a soft
    volley exchange near the net) where ordinary frame-to-frame noise can
    look like a large ratio-wise "drop" despite being a tiny absolute
    change. `max_drop_ratio` is how much of the incoming speed survives
    into the very next frame for the initial drop to count. `reaccel_frames`
    is how far past the candidate to look for the ball resuming its travel
    - the immediate next frame or two can still read as near-stationary
    right at a real bounce, so this deliberately looks further out than the
    initial drop check does. `min_reaccel_ratio` is how much of the
    "before" speed (per frame) that later velocity needs to have
    recovered. `min_frame_gap` merges nearby flagged frames into one event,
    the same physical bounce caught on consecutive frames.
    """
    points = _world_points(positions, calibrations_by_frame)
    candidate_frames: list[int] = []

    for i in range(1, len(points) - 1):
        prev_point, cur_point, next_point = points[i - 1], points[i], points[i + 1]
        if cur_point.frame_idx - prev_point.frame_idx != 1 or next_point.frame_idx - cur_point.frame_idx != 1:
            continue  # not temporally adjacent - a real tracking gap, not a comparable pair

        # Walk forward from `next_point` while frames stay consecutive, up
        # to reaccel_frames further, to find a later point to check
        # recovery against - stops early at the first gap.
        after_idx = i + 1
        for j in range(i + 2, min(i + 2 + reaccel_frames, len(points))):
            if points[j].frame_idx - points[j - 1].frame_idx != 1:
                break
            after_idx = j
        after_point = points[after_idx]

        before_vec = cur_point.world - prev_point.world
        immediate_vec = next_point.world - cur_point.world
        after_vec = after_point.world - next_point.world

        distance_before = float(np.linalg.norm(before_vec))
        speed_before_kmh = distance_before * fps * 3.6
        if speed_before_kmh < min_speed_before_kmh:
            continue
        if float(np.linalg.norm(immediate_vec)) > distance_before * max_drop_ratio:
            continue  # no sharp initial drop - ordinary flight, not an impact

        elapsed_after = after_point.frame_idx - next_point.frame_idx
        if elapsed_after <= 0:
            continue
        after_speed_per_frame = float(np.linalg.norm(after_vec)) / elapsed_after
        if after_speed_per_frame < distance_before * min_reaccel_ratio:
            continue  # never sped back up - looks absorbed/blocked, not a bounce continuing onward
        if float(np.dot(before_vec, after_vec)) <= 0:
            continue  # resumed moving, but not the same way it was going - a contact redirected it

        candidate_frames.append(cur_point.frame_idx)

    merged_frames = _merge_nearby_frames(sorted(candidate_frames), min_frame_gap)

    points_by_frame = {p.frame_idx: p for p in points}
    events = []
    for frame_idx in merged_frames:
        point = points_by_frame[frame_idx]
        events.append(
            BounceEvent(
                frame_idx=frame_idx,
                x=point.x,
                y=point.y,
                world_x=float(point.world[0]),
                world_y=float(point.world[1]),
            )
        )
    return events
