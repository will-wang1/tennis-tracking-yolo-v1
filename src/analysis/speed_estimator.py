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

import numpy as np

from src.analysis.court_calibration import CourtCalibration
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
