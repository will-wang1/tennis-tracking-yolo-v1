"""Estimate ball speed per shot from an already-tracked trajectory.

A "shot" is the stretch of trajectory between two "breakpoints" (or between
the start/end of tracking and the nearest one) - see `segment_shots`. A
breakpoint is any trajectory direction-change - typically from
`find_trajectory_breakpoints`, which doesn't try to tell a bounce from a
contact (see `src.analysis.bounce_detector`'s module docstring for why
that's a hard problem on its own): either one reliably marks the end of one
shot and the start of the next, which is all shot segmentation actually
needs. Speed is reported as the PEAK instantaneous speed within a shot,
since that's what "how fast was that shot" means in practice (closest to
contact, before drag slows the ball down), not an average over the whole
flight.

If a `CourtCalibration` is given, speed is reported in real-world km/h by
mapping consecutive ball positions through the calibration's ground-plane
homography. That's exact for points on the court surface and increasingly
approximate the higher the ball is above it (e.g. near a serve toss or
smash) - a known, accepted limitation given there's no calibrated
depth/height model. Without a calibration, speed falls back to px/s so the
caller always gets a number and always knows which kind (`unit` field).
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS, CourtCalibration
from src.tracking.ball_tracker import TrackedPosition

# The fastest tennis serve ever recorded is ~263 km/h. A reading above this
# isn't a fast shot, it's a tracking artifact - most often a single-frame
# pixel jump that individually clears BallTracker's outlier threshold (which
# is a fixed pixel distance, not aware of any particular camera's
# real-world scale) but implies an impossible real-world distance once
# mapped through a coarser calibration (a more zoomed-out camera maps the
# same pixel jump to more real-world meters). Excluded from peak-speed
# selection entirely rather than clamped, since a clamped-but-wrong value
# would misrepresent which frame was actually fastest.
MAX_PLAUSIBLE_KMH = 300.0


@dataclass
class ShotSpeed:
    start_frame: int
    end_frame: int
    peak_frame: int
    peak_speed: float
    unit: str  # "km/h" if a calibration was given, else "px/s"


def net_pixel_y_range(calibration: CourtCalibration) -> tuple[float, float]:
    """Reprojects the net line (the court's length-wise midpoint, by
    construction - see `FULL_COURT_REFERENCE_POINTS`) back to pixels at
    both sidelines, returning (min_y, max_y). Perspective means the net's
    pixel height isn't constant across the frame's width, so this is a
    range, not a single value."""
    net_world_y = max(y for _, y in FULL_COURT_REFERENCE_POINTS.values()) / 2
    doubles_width = max(x for x, _ in FULL_COURT_REFERENCE_POINTS.values())
    inverse_homography = np.linalg.inv(calibration.homography)

    pixel_ys = []
    for world_x in (0.0, doubles_width):
        point = np.array([[[world_x, net_world_y]]], dtype=np.float64)
        _, pixel_y = cv2.perspectiveTransform(point, inverse_homography)[0, 0]
        pixel_ys.append(float(pixel_y))
    return min(pixel_ys), max(pixel_ys)


def estimate_net_crossing_speeds(
    positions: list[TrackedPosition],
    fps: float,
    calibrations_by_frame: dict[int, CourtCalibration],
    net_band: float = 60.0,
    min_motion_px: float = 4.0,
    max_frame_gap: int = 2,
) -> list[ShotSpeed]:
    """Speed computed only where the ball is genuinely moving near the net
    line - a narrower, more robust alternative to `estimate_shot_speeds`
    that sidesteps needing reliable bounce/breakpoint segmentation across
    the whole trajectory (see this module's and `bounce_detector`'s
    docstrings for why that's fragile). A frame-to-frame pair counts only
    if BOTH hold:

    - both positions fall within `net_band` pixels of the net's
      reprojected pixel height (`net_pixel_y_range`)
    - the pair shows real motion (>= `min_motion_px`) - a static
      false-positive detector lock-on (e.g. a net-post highlight) repeats
      almost the same pixel frame to frame and is excluded by this alone,
      no lock-on-radius tuning required

    `calibrations_by_frame` maps frame index -> the `CourtCalibration` valid
    at that frame - one entry per frame for a moving/panning camera, or the
    same `CourtCalibration` repeated for every frame for a static one; a
    missing frame index (e.g. the court detector lost the court that frame)
    excludes that frame's pairs rather than guessing. The later frame
    (`cur_pos`) in each pair picks which calibration to use, since a camera
    move between prev and cur is exactly what that frame's fresh detection
    captures.

    Only DETECTED (non-interpolated) positions are considered - an
    interpolated point is a synthetic straight-line guess, not evidence of
    real motion. Consecutive qualifying frames (frame gap <= `max_frame_gap`)
    are grouped into one `ShotSpeed` each (peak speed within the group) -
    same shape `estimate_shot_speeds` returns, so callers don't need to
    know which method produced it.
    """
    detected = sorted((p for p in positions if not p.interpolated), key=lambda p: p.frame_idx)
    net_range_cache: dict[int, tuple[float, float]] = {}

    def net_range_for(calibration: CourtCalibration) -> tuple[float, float]:
        key = id(calibration)
        if key not in net_range_cache:
            min_y, max_y = net_pixel_y_range(calibration)
            net_range_cache[key] = (min_y - net_band, max_y + net_band)
        return net_range_cache[key]

    crossings: list[tuple[int, float]] = []
    for prev_pos, cur_pos in zip(detected, detected[1:]):
        gap = cur_pos.frame_idx - prev_pos.frame_idx
        if gap <= 0 or gap > max_frame_gap:
            continue

        calibration = calibrations_by_frame.get(cur_pos.frame_idx)
        if calibration is None:
            continue

        min_y, max_y = net_range_for(calibration)
        if not (min_y <= prev_pos.y <= max_y and min_y <= cur_pos.y <= max_y):
            continue

        pixel_distance = float(np.hypot(cur_pos.x - prev_pos.x, cur_pos.y - prev_pos.y))
        if pixel_distance < min_motion_px:
            continue  # too little motion to be a real, moving ball

        elapsed_seconds = gap / fps
        distance_m = calibration.pixel_distance_to_meters(prev_pos.x, prev_pos.y, cur_pos.x, cur_pos.y)
        speed_kmh = (distance_m / elapsed_seconds) * 3.6
        if speed_kmh <= MAX_PLAUSIBLE_KMH:
            crossings.append((cur_pos.frame_idx, speed_kmh))

    shots = []
    group: list[tuple[int, float]] = []
    for frame_idx, speed in crossings:
        if group and frame_idx - group[-1][0] > max_frame_gap:
            shots.append(_shot_from_group(group))
            group = []
        group.append((frame_idx, speed))
    if group:
        shots.append(_shot_from_group(group))
    return shots


def _shot_from_group(group: list[tuple[int, float]]) -> ShotSpeed:
    peak_frame, peak_speed = max(group, key=lambda item: item[1])
    return ShotSpeed(
        start_frame=group[0][0],
        end_frame=group[-1][0],
        peak_frame=peak_frame,
        peak_speed=peak_speed,
        unit="km/h",
    )


def instantaneous_speeds(
    positions: list[TrackedPosition],
    fps: float,
    calibration: Optional[CourtCalibration] = None,
    window: int = 1,
) -> dict[int, float]:
    """Speed at each tracked frame, computed from the displacement over the
    preceding `window` tracked samples (default: the immediately preceding
    one). `positions` need not be contiguous - the elapsed frame gap is
    accounted for in the time delta, so a speed can still be computed
    across a short interpolated gap.

    `window` trades responsiveness for noise robustness. A few pixels of
    detector jitter in the ball's estimated center barely matters in px/s
    or on a tightly-zoomed-in camera, but a real-world calibration on a
    more zoomed-out camera maps the same few pixels to more meters, so the
    same jitter can produce wildly implausible km/h swings frame to frame
    (see MAX_PLAUSIBLE_KMH). Widening the window averages that noise out at
    the cost of underestimating brief true peaks - there's no single value
    that's right for every camera's zoom level, same as `--bounce-min-prominence`.
    """
    ordered = sorted(positions, key=lambda p: p.frame_idx)
    speeds: dict[int, float] = {}

    for i in range(window, len(ordered)):
        prev_pos, cur_pos = ordered[i - window], ordered[i]
        elapsed_frames = cur_pos.frame_idx - prev_pos.frame_idx
        if elapsed_frames <= 0:
            continue
        elapsed_seconds = elapsed_frames / fps

        if calibration is not None:
            distance = calibration.pixel_distance_to_meters(
                prev_pos.x, prev_pos.y, cur_pos.x, cur_pos.y
            )
            speed = (distance / elapsed_seconds) * 3.6  # m/s -> km/h
        else:
            distance = float(np.hypot(cur_pos.x - prev_pos.x, cur_pos.y - prev_pos.y))
            speed = distance / elapsed_seconds  # px/s

        speeds[cur_pos.frame_idx] = speed

    return speeds


def segment_shots(positions: list[TrackedPosition], breakpoint_frames: list[int]) -> list[tuple[int, int]]:
    """Split the tracked trajectory into (start_frame, end_frame) segments
    at each breakpoint frame (see `src.analysis.bounce_detector.find_trajectory_breakpoints`,
    or pass confirmed bounce frame indices directly). A shot is
    start-of-tracking-to-first-breakpoint, between-breakpoints, or
    last-breakpoint-to-end-of-tracking."""
    if not positions:
        return []

    ordered = sorted(positions, key=lambda p: p.frame_idx)
    first_frame, last_frame = ordered[0].frame_idx, ordered[-1].frame_idx

    inner_frames = sorted(f for f in breakpoint_frames if first_frame < f < last_frame)
    boundaries = [first_frame, *inner_frames, last_frame]

    segments = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end > start:
            segments.append((start, end))
    return segments


def estimate_shot_speeds(
    positions: list[TrackedPosition],
    breakpoint_frames: list[int],
    fps: float,
    calibration: Optional[CourtCalibration] = None,
    speed_window: int = 1,
) -> list[ShotSpeed]:
    speeds_by_frame = instantaneous_speeds(positions, fps, calibration, window=speed_window)
    unit = "km/h" if calibration is not None else "px/s"

    shots = []
    for start_frame, end_frame in segment_shots(positions, breakpoint_frames):
        segment_speeds = {
            frame: speed
            for frame, speed in speeds_by_frame.items()
            if start_frame < frame <= end_frame
            and (unit != "km/h" or speed <= MAX_PLAUSIBLE_KMH)
        }
        if not segment_speeds:
            continue
        peak_frame = max(segment_speeds, key=segment_speeds.get)
        shots.append(
            ShotSpeed(
                start_frame=start_frame,
                end_frame=end_frame,
                peak_frame=peak_frame,
                peak_speed=segment_speeds[peak_frame],
                unit=unit,
            )
        )
    return shots
