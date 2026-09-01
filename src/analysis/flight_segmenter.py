"""Find impacts by fitting whole FLIGHT SEGMENTS and intersecting them,
rather than by scanning frame by frame for a junction.

`parabolic_bounce_detector.find_impacts` walks the trajectory a frame at a
time and asks, of each frame, whether two arcs meet there. That needs real
samples immediately either side of the impact, so a dropout AT the impact
hides it - and the impact frame is the single most likely one to be missed,
because the ball is blurriest exactly when it is struck. Measured on this
project's clips, that scan finds nothing at all through stretches where a
bounce is known to be: video_input2 has one confirmed bounce around 7.1s
where detections are too sparse for it to fit arcs either side.

This inverts the approach. Instead of looking for the impact, it looks for
the FLIGHTS - long stretches that are cleanly parabolic - and takes the
impacts to be wherever consecutive flights meet. The intersection is
computed from the two fitted curves, so it needs no data at the impact at
all: a twenty-frame hole where the ball landed costs nothing as long as
there is clean flight on either side of it.

Segments are found by RANSAC rather than by least squares over a fixed
window, because least squares has no way to say "these points belong to a
different flight" - a window straddling an impact is fitted to a curve
matching neither side (the failure that had to be guarded against explicitly
in the frame-by-frame scan). RANSAC instead grows a segment from whichever
points agree with one parabola and stops where they stop agreeing, which is
precisely where the impact is.

Screen y is quadratic in time under gravity and screen x close to linear
over a single flight, the same model `parabolic_bounce_detector` fits, and
for the same reason: it is measured in pixels, so it is unaffected by the
court homography's errors on an airborne ball.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.analysis.parabolic_bounce_detector import drop_duplicate_frames
from src.tracking.ball_tracker import TrackedPosition


@dataclass(frozen=True)
class SegmentImpact:
    """Where two consecutive flights meet, with the flights themselves - so
    a caller can read the velocities on either side without re-deriving
    which segments were paired."""

    t: float
    x: float
    y: float
    before: "FlightSegment"
    after: "FlightSegment"


@dataclass(frozen=True)
class FlightSegment:
    """One stretch of uninterrupted flight, as fitted curves in time."""

    start_frame: int
    end_frame: int
    x_coeffs: np.ndarray  # linear in time
    y_coeffs: np.ndarray  # quadratic in time
    t_ref: float
    sample_frames: list[int]
    rmse: float

    def position(self, t: float) -> tuple[float, float]:
        shifted = t - self.t_ref
        return (
            float(np.polyval(self.x_coeffs, shifted)),
            float(np.polyval(self.y_coeffs, shifted)),
        )

    def velocity(self, t: float) -> tuple[float, float]:
        shifted = t - self.t_ref
        return (
            float(np.polyval(np.polyder(self.x_coeffs), shifted)),
            float(np.polyval(np.polyder(self.y_coeffs), shifted)),
        )

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


def _fit(frames: list[int], xs: list[float], ys: list[float], t_ref: float):
    times = np.asarray(frames, dtype=np.float64) - t_ref
    x_coeffs = np.polyfit(times, xs, 1)
    y_coeffs = np.polyfit(times, ys, 2)
    residuals = np.hypot(
        np.polyval(x_coeffs, times) - np.asarray(xs),
        np.polyval(y_coeffs, times) - np.asarray(ys),
    )
    return x_coeffs, y_coeffs, residuals


def _grow_segment(
    ordered: list[TrackedPosition],
    start_index: int,
    tolerance: float,
    max_gap: int,
    seed_size: int,
) -> Optional[FlightSegment]:
    """Extend one flight forward from `start_index` for as long as the
    samples keep agreeing with a single parabola.

    The fit is re-done as the segment grows, so an arc that curves away
    later is still followed; what ends the segment is a sample that the
    curve genuinely cannot explain, which is the impact.
    """
    if start_index + seed_size > len(ordered):
        return None

    t_ref = float(ordered[start_index].frame_idx)
    kept = list(range(start_index, start_index + seed_size))
    frames = [ordered[i].frame_idx for i in kept]
    if frames[-1] - frames[0] > max_gap * seed_size:
        return None
    xs = [ordered[i].x for i in kept]
    ys = [ordered[i].y for i in kept]
    x_coeffs, y_coeffs, residuals = _fit(frames, xs, ys, t_ref)
    if float(np.sqrt(np.mean(residuals**2))) > tolerance:
        return None  # the seed itself isn't clean flight

    index = start_index + seed_size
    while index < len(ordered):
        candidate = ordered[index]
        if candidate.frame_idx - frames[-1] > max_gap:
            break  # too long a dropout to keep calling this one flight
        shifted = candidate.frame_idx - t_ref
        error = float(
            np.hypot(
                np.polyval(x_coeffs, shifted) - candidate.x,
                np.polyval(y_coeffs, shifted) - candidate.y,
            )
        )
        if error > tolerance:
            break  # this sample belongs to a different flight - the impact
        frames.append(candidate.frame_idx)
        xs.append(candidate.x)
        ys.append(candidate.y)
        x_coeffs, y_coeffs, residuals = _fit(frames, xs, ys, t_ref)
        index += 1

    return FlightSegment(
        start_frame=frames[0],
        end_frame=frames[-1],
        x_coeffs=x_coeffs,
        y_coeffs=y_coeffs,
        t_ref=t_ref,
        sample_frames=frames,
        rmse=float(np.sqrt(np.mean(residuals**2))),
    )


def find_flight_segments(
    positions: list[TrackedPosition],
    tolerance: float = 24.0,
    max_gap: int = 12,
    seed_size: int = 7,
    min_samples: int = 9,
    duplicate_ratio: float = 0.25,
    duplicate_window: int = 9,
) -> list[FlightSegment]:
    """Split the trajectory into stretches of clean parabolic flight.

    `tolerance` is in pixels: how far a sample may sit from the fitted curve
    before it is judged to belong to a different flight. It sits well above
    detector noise (about a pixel) because it also has to absorb the model's
    own error - a fast ball's screen path is not exactly parabolic under
    perspective - while staying under the deflection an impact produces.
    Tuned against hand labels on both clips: at 24 every confirmed bounce on
    both is covered (8/8 and 5/5) with segments of a sensible length, while
    tighter values shatter the 60fps clip into fragments and lose several.

    `min_samples` is what a segment must contain to count as a flight at
    all. Nine keeps the fit honest and, with `seed_size` at seven, stops
    short noisy runs being promoted to flights and manufacturing impacts
    between them.

    `max_gap` allows a segment to span a dropout - the whole point of this
    approach - while still refusing to join two flights either side of a
    hole so long that anything could have happened in between.

    Repeated frames are dropped first, for the same reason the frame-by-frame
    scan drops them: on footage encoded above its true frame rate the ball
    stands still every few frames, and a stalled sample fits no parabola.
    Left in, they shatter the trajectory into fragments - measured on
    video_input2, 72 segments of median 5 frames rather than a couple of
    dozen real flights.
    """
    ordered = drop_duplicate_frames(
        sorted((p for p in positions if not p.interpolated), key=lambda p: p.frame_idx),
        duplicate_ratio,
        duplicate_window,
    )
    segments: list[FlightSegment] = []

    index = 0
    while index < len(ordered):
        segment = _grow_segment(ordered, index, tolerance, max_gap, seed_size)
        if segment is None:
            index += 1
            continue
        if len(segment.sample_frames) >= min_samples:
            segments.append(segment)
            index += len(segment.sample_frames)
        else:
            index += 1
    return segments


def find_segment_impacts(
    segments: list[FlightSegment],
    max_separation_frames: int = 25,
    search_margin: float = 4.0,
) -> list[SegmentImpact]:
    """Where consecutive flights meet, one `SegmentImpact` each.

    The instant is found by intersecting the two fitted curves in y, so it
    is recovered even when nothing at all was detected around the impact -
    which is the whole reason for fitting segments rather than scanning
    frames. Searched between the end of one flight and the start of the
    next, widened by `search_margin` because the true impact can sit just
    outside the samples on either side.

    `max_separation_frames` refuses to join two flights separated by so long
    a dropout that there may well have been another impact inside it, where
    a single intersection would be a fiction.
    """
    impacts = []
    for before, after in zip(segments, segments[1:]):
        separation = after.start_frame - before.end_frame
        if separation <= 0 or separation > max_separation_frames:
            continue

        low = before.end_frame - search_margin
        high = after.start_frame + search_margin
        times = np.arange(low, high + 0.05, 0.05)
        gap = np.polyval(after.y_coeffs, times - after.t_ref) - np.polyval(
            before.y_coeffs, times - before.t_ref
        )
        crossings = np.nonzero(np.sign(gap[:-1]) != np.sign(gap[1:]))[0]
        if len(crossings) == 0:
            # Parallel-ish curves that never meet: fall back to the midpoint
            # of the gap, which is still the best estimate of when the ball
            # changed course, just a less precise one.
            impact_t = (before.end_frame + after.start_frame) / 2.0
        else:
            middle = (before.end_frame + after.start_frame) / 2.0
            impact_t = min(
                (float(times[i]) for i in crossings), key=lambda t: abs(t - middle)
            )

        # Averaged from both curves: each is exact at its own end and
        # extrapolated at the other, so neither alone is best at the meeting.
        bx, by = before.position(impact_t)
        ax, ay = after.position(impact_t)
        impacts.append(
            SegmentImpact(
                t=impact_t,
                x=(bx + ax) / 2.0,
                y=(by + ay) / 2.0,
                before=before,
                after=after,
            )
        )
    return impacts


def segment_impacts_as_candidates(
    positions: list[TrackedPosition], **kwargs
) -> list["BounceCandidate"]:
    """Segment intersections packaged as `BounceCandidate`s, so they can be
    judged by `touchdown_detector.classify_touchdowns` exactly like the
    impacts the frame-by-frame scan produces.

    The velocity ratios are read from the two fitted flights at the moment
    they meet - the same measurements `find_impacts` takes, but recovered
    from whole segments, which is what lets them exist at all when the ball
    was never detected at the impact itself.

    They come back as kind "unknown" on purpose. Intersecting two flights
    proves that SOMETHING interrupted the ball; it says nothing about what,
    because none of the scan's court-vs-racket checks (restitution range,
    energy gain, horizontal reversal) have been applied. Attribution is
    `classify_touchdowns`' job, and until it runs these carry no verdict -
    defaulting them to "bounce" would put a confirmed landing marker on
    every recovered impact with no evidence behind it.
    """
    from src.analysis.parabolic_bounce_detector import BounceCandidate

    segment_keys = {"tolerance", "max_gap", "seed_size", "min_samples"}
    segments = find_flight_segments(
        positions, **{k: v for k, v in kwargs.items() if k in segment_keys}
    )
    impacts = find_segment_impacts(
        segments, **{k: v for k, v in kwargs.items() if k not in segment_keys}
    )

    candidates = []
    for impact in impacts:
        vx_before, vy_before = impact.before.velocity(impact.t)
        vx_after, vy_after = impact.after.velocity(impact.t)
        speed_before = float(np.hypot(vx_before, vy_before))
        speed_after = float(np.hypot(vx_after, vy_after))
        candidates.append(
            BounceCandidate(
                frame_idx=int(round(impact.t)),
                t=impact.t,
                x=impact.x,
                y=impact.y,
                restitution=-vy_after / vy_before if abs(vy_before) > 1e-9 else 0.0,
                horizontal_ratio=abs(vx_after) / max(abs(vx_before), 1e-9),
                speed_ratio=speed_after / max(speed_before, 1e-9),
                rmse=max(impact.before.rmse, impact.after.rmse),
                is_bounce=False,
                kind="unknown",
                reason="recovered by intersecting flights - not yet attributed",
                before_bound_frame=impact.before.start_frame,
            )
        )
    return candidates
