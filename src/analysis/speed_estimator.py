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

`estimate_net_crossing_speeds` is the most accurate reading (ball at
~net height, near the ground plane the homography is exact for) but only
fires for shots actually seen crossing the net band with two consecutive
DETECTED positions - a shot lost to occlusion/motion blur right at the net,
or one that never got tracked that close to the net line, produces no
reading at all. `merge_with_net_crossing_speeds` covers that gap: it takes
the full-coverage, breakpoint-segmented `estimate_shot_speeds` output (every
shot gets a peak reading, just a less precise one - averaged over whichever
part of the flight was tracked, not necessarily at net height) and swaps in
the sharper net-crossing number wherever one overlaps, so every shot gets a
reading and the best available one.
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


def net_pixel_line(calibration: CourtCalibration) -> tuple[float, float, float]:
    """The net line reprojected into the image, as `(a, b, c)` with
    `a*x + b*y + c` zero on the line, negative on the far side and positive
    on the near side.

    A single pixel height cannot describe the net: perspective tilts it
    across the frame, which is why `net_pixel_y_range` has to return a range.
    A line handles the tilt exactly and answers the question that matters
    here - which side of the net is the ball on, right now.
    """
    net_world_y = max(y for _, y in FULL_COURT_REFERENCE_POINTS.values()) / 2
    doubles_width = max(x for x, _ in FULL_COURT_REFERENCE_POINTS.values())
    inverse_homography = np.linalg.inv(calibration.homography)

    ends = []
    for world_x in (0.0, doubles_width):
        point = np.array([[[world_x, net_world_y]]], dtype=np.float64)
        ends.append(cv2.perspectiveTransform(point, inverse_homography)[0, 0])
    (x1, y1), (x2, y2) = ends
    # the image line through both ends, oriented so "down the image" is positive
    a, b = y1 - y2, x2 - x1
    c = -(a * x1 + b * y1)
    if b < 0:
        a, b, c = -a, -b, -c
    return float(a), float(b), float(c)


def estimate_flight_net_speeds(
    segments: list,
    fps: float,
    calibrations_by_frame: dict[int, CourtCalibration],
    half_window_frames: float = 2.0,
    min_half_window_frames: float = 1.0,
    min_flight_frames: int = 12,
    max_speed_kmh: float = MAX_PLAUSIBLE_KMH,
) -> list[ShotSpeed]:
    """Ball speed as each fitted FLIGHT crosses the net.

    `estimate_net_crossing_speeds` asks the same question of the raw
    trajectory, and can only answer it when the detector happened to supply
    several consecutive positions inside a pixel band at the net. That is a
    minority of shots: the ball is small, fast and often occluded exactly
    there. Measured on this project's three clips it produced a reading for
    well under half of them, so most shots fell back to a speed averaged
    over whatever part of the flight was tracked - a different and cruder
    quantity.

    A fitted flight has no such gap. The curve is defined at every instant
    between its endpoints, so the crossing time can be solved for directly
    (`net_pixel_line`, sign change of the ball's side of the net) whether or
    not the ball was detected at that moment, and the speed read off the
    curve either side of it. Every flight that crosses the net gets a
    reading, and it is a reading of the same thing every time.

    Why at the net at all: the calibration is a GROUND-PLANE homography, so
    it is exact for a ball on the court and increasingly wrong the higher
    the ball is (see `court_camera.py`). The net crossing is where a rally
    ball is lowest between the two strikes, so it is where that error is
    smallest - not zero, since the ball is still a metre or so up, but the
    best moment available in a monocular view.

    A flight shorter than `min_flight_frames` is skipped: the crossing
    cannot be placed confidently within it, and the narrow measuring window
    that follows turns the fit's own noise into a headline number.

    Speed is measured over `half_window_frames` either side of the crossing
    rather than at the instant itself: a derivative taken at a point on a
    fitted curve inherits the fit's noise, while a displacement over a few
    frames averages it out, and over that short a window the ball has not
    slowed appreciably.
    """
    shots = []
    for segment in segments:
        calibration = calibrations_by_frame.get(int(round(segment.start_frame)))
        if calibration is None:
            continue
        if segment.duration_frames < min_flight_frames:
            # too little flight to place the crossing confidently, and a
            # narrow window there turns fit noise into a headline number -
            # an 11-frame flight on video_input2 read 287 km/h this way
            continue
        a, b, c = net_pixel_line(calibration)

        crossing = None
        steps = np.arange(segment.start_frame, segment.end_frame + 0.05, 0.05)
        sides = [a * segment.position(t)[0] + b * segment.position(t)[1] + c for t in steps]
        for i in range(len(sides) - 1):
            if np.sign(sides[i]) != np.sign(sides[i + 1]):
                crossing = float(steps[i])
                break
        if crossing is None:
            continue

        # a crossing near an end of the flight gets a narrower window rather
        # than no reading, down to the point where the window is too short
        # for the fit's own noise to average out
        half = min(
            half_window_frames,
            crossing - segment.start_frame,
            segment.end_frame - crossing,
        )
        if half < min_half_window_frames:
            continue
        first, last = crossing - half, crossing + half

        start_world = calibration.pixel_to_world(*segment.position(first))
        end_world = calibration.pixel_to_world(*segment.position(last))
        metres = float(np.hypot(end_world[0] - start_world[0], end_world[1] - start_world[1]))
        seconds = (last - first) / fps
        speed = metres / seconds * 3.6
        if not 0.0 < speed <= max_speed_kmh:
            continue

        shots.append(
            ShotSpeed(
                start_frame=int(segment.start_frame),
                end_frame=int(segment.end_frame),
                peak_frame=int(round(crossing)),
                peak_speed=speed,
                unit="km/h",
            )
        )
    return shots


def estimate_net_crossing_speeds(
    positions: list[TrackedPosition],
    fps: float,
    calibrations_by_frame: dict[int, CourtCalibration],
    net_band: float = 60.0,
    min_motion_px: float = 4.0,
    max_frame_gap: int = 2,
    min_window_frames: int = 4,
) -> list[ShotSpeed]:
    """Speed averaged over a WINDOW of consecutive frames while the ball is
    genuinely moving near the net line - a narrower, more robust
    alternative to `estimate_shot_speeds` that sidesteps needing reliable
    bounce/breakpoint segmentation across the whole trajectory (see this
    module's and `bounce_detector`'s docstrings for why that's fragile).

    Only DETECTED (non-interpolated) positions are considered - an
    interpolated point is a synthetic straight-line guess, not evidence of
    real motion. A position counts as "near the net" if it falls within
    `net_band` pixels of the net's reprojected pixel height
    (`net_pixel_y_range`, using that position's OWN frame's calibration -
    see below). Consecutive qualifying positions (frame gap <= `max_frame_gap`)
    are grouped into one window each; a window shorter than
    `min_window_frames` is dropped rather than reported, since an average
    over too few frames is just as noisy as a single frame-to-frame reading.

    Each surviving window's speed is the AVERAGE over the whole window - net
    displacement (first position to last, not the summed per-step path
    length, so back-and-forth detector jitter along the way cancels out
    instead of inflating the reading) divided by total elapsed time - not a
    peak instantaneous reading. Averaging over more frames trades
    responsiveness (a real, sharp velocity change mid-window gets smoothed
    into the average) for robustness to detector jitter, which matters more
    here since a single noisy pixel jump used to be able to single-handedly
    become "the" reported speed. Widen `min_window_frames` for a shakier
    detector, narrow it if legitimate quick net crossings are being dropped
    for lack of frames.

    A window's overall net motion must still be >= `min_motion_px`,
    filtering out a static false-positive detector lock-on (e.g. a
    net-post highlight) sitting near the net band without ever actually
    moving - unlike the old per-pair check, a lock-on can't slip through
    just by jittering a few pixels frame to frame, since it's the window's
    NET displacement (start to end) that's checked, not any single step.

    `calibrations_by_frame` maps frame index -> the `CourtCalibration` valid
    at that frame - one entry per frame for a moving/panning camera, or the
    same `CourtCalibration` repeated for every frame for a static one; a
    missing frame index (e.g. the court detector lost the court that frame)
    excludes that position rather than guessing. A window's own last frame
    picks which calibration converts its net displacement to meters, since
    a camera move during the window is exactly what that frame's fresh
    detection captures.
    """
    detected = sorted((p for p in positions if not p.interpolated), key=lambda p: p.frame_idx)
    net_range_cache: dict[int, tuple[float, float]] = {}

    def net_range_for(calibration: CourtCalibration) -> tuple[float, float]:
        key = id(calibration)
        if key not in net_range_cache:
            min_y, max_y = net_pixel_y_range(calibration)
            net_range_cache[key] = (min_y - net_band, max_y + net_band)
        return net_range_cache[key]

    def near_net(position: TrackedPosition) -> bool:
        calibration = calibrations_by_frame.get(position.frame_idx)
        if calibration is None:
            return False
        min_y, max_y = net_range_for(calibration)
        return min_y <= position.y <= max_y

    windows: list[list[TrackedPosition]] = []
    current: list[TrackedPosition] = []
    for position in detected:
        qualifies = near_net(position)
        if qualifies and current and position.frame_idx - current[-1].frame_idx > max_frame_gap:
            windows.append(current)
            current = []
        if qualifies:
            current.append(position)
        elif current:
            windows.append(current)
            current = []
    if current:
        windows.append(current)

    shots = []
    for window in windows:
        if len(window) < min_window_frames:
            continue

        first, last = window[0], window[-1]
        elapsed_seconds = (last.frame_idx - first.frame_idx) / fps
        if elapsed_seconds <= 0:
            continue

        calibration = calibrations_by_frame.get(last.frame_idx)
        if calibration is None:
            continue

        net_pixel_distance = float(np.hypot(last.x - first.x, last.y - first.y))
        if net_pixel_distance < min_motion_px:
            continue  # too little net motion to be a real, moving ball

        distance_m = calibration.pixel_distance_to_meters(first.x, first.y, last.x, last.y)
        speed_kmh = (distance_m / elapsed_seconds) * 3.6
        if speed_kmh <= MAX_PLAUSIBLE_KMH:
            shots.append(
                ShotSpeed(
                    start_frame=first.frame_idx,
                    end_frame=last.frame_idx,
                    peak_frame=last.frame_idx,
                    peak_speed=speed_kmh,
                    unit="km/h",
                )
            )
    return shots


def merge_with_net_crossing_speeds(
    fallback_shots: list[ShotSpeed], net_shots: list[ShotSpeed]
) -> list[ShotSpeed]:
    """Fill in every `fallback_shots` entry (typically `estimate_shot_speeds`'
    breakpoint-segmented, full-trajectory-coverage output) with a more
    accurate `net_shots` (`estimate_net_crossing_speeds`) reading wherever
    one overlaps - net-crossing speed is measured near the ground plane the
    calibration homography is exact for, so it's preferred whenever the
    ball was actually seen crossing the net during that shot. A fallback
    shot with no overlapping net crossing (ball not tracked that close to
    the net that shot - occlusion, motion blur, or it just never got that
    close in frame) keeps its own (less precise, but present) reading
    instead of being dropped, so every shot still gets a number.

    Each fallback shot's own (start_frame, end_frame) window is always kept
    - a net_shot's window is only the narrow crossing pair, not the whole
    shot - only peak_speed/peak_frame/unit are swapped in from whichever
    overlapping net_shot has the highest peak_speed.
    """
    merged = []
    for shot in fallback_shots:
        overlapping = [
            net_shot
            for net_shot in net_shots
            if net_shot.start_frame <= shot.end_frame and net_shot.end_frame >= shot.start_frame
        ]
        if overlapping:
            best = max(overlapping, key=lambda s: s.peak_speed)
            merged.append(
                ShotSpeed(
                    start_frame=shot.start_frame,
                    end_frame=shot.end_frame,
                    peak_frame=best.peak_frame,
                    peak_speed=best.peak_speed,
                    unit=best.unit,
                )
            )
        else:
            merged.append(shot)
    return merged


def instantaneous_speeds(
    positions: list[TrackedPosition],
    fps: float,
    calibration: Optional[CourtCalibration] = None,
    window: int = 1,
    calibrations_by_frame: Optional[dict[int, CourtCalibration]] = None,
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

    `calibrations_by_frame`, if given, looks up `cur_pos`'s own frame for a
    per-frame calibration (a moving/panning camera - see
    `estimate_net_crossing_speeds`'s docstring) and takes priority over the
    single, static `calibration`; a frame missing from it is treated as
    having no calibration for that sample, same as `calibration=None`.
    """
    ordered = sorted(positions, key=lambda p: p.frame_idx)
    speeds: dict[int, float] = {}

    for i in range(window, len(ordered)):
        prev_pos, cur_pos = ordered[i - window], ordered[i]
        elapsed_frames = cur_pos.frame_idx - prev_pos.frame_idx
        if elapsed_frames <= 0:
            continue
        elapsed_seconds = elapsed_frames / fps

        active_calibration = (
            calibrations_by_frame.get(cur_pos.frame_idx) if calibrations_by_frame is not None else calibration
        )
        if active_calibration is not None:
            distance = active_calibration.pixel_distance_to_meters(
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
    calibrations_by_frame: Optional[dict[int, CourtCalibration]] = None,
) -> list[ShotSpeed]:
    speeds_by_frame = instantaneous_speeds(
        positions, fps, calibration, window=speed_window, calibrations_by_frame=calibrations_by_frame
    )
    unit = "km/h" if (calibration is not None or calibrations_by_frame) else "px/s"

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
