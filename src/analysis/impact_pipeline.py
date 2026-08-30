"""Everything between "here are the ball's detections" and "here is what hit
it, and what did the hitting".

This exists so the renderer (main.py) and the offline harness
(scripts/replay_impacts.py) run the SAME chain rather than two copies of it.
The harness is only useful if a score it prints is a statement about what a
render would draw, and a second implementation - even one that starts as a
faithful copy - stops being that the first time one side is tuned.

The chain has four steps, and the order matters:

1. Track the detections into a path, with NO smoothing. Fitted arcs are
   their own noise filter, so smoothing first would flatten the very corner
   being measured (see parabolic_bounce_detector.py). This is a separate
   tracking pass from the smoothed one the visible trail and speed readings
   use.
2. Scan frame by frame for impacts - precise, but needs detections on both
   sides of the impact.
3. Intersect fitted flight segments to recover impacts the scan could not
   see because the detector dropped the ball right where it landed.
4. Re-judge every impact as bounce/contact/unknown from the ball's
   projected court position (touchdown_detector.py), which needs a court
   calibration; without one the scan's own screen-space verdict stands.
"""

from dataclasses import dataclass, field, replace
from typing import Optional

from src.analysis.court_calibration import CourtCalibration
from src.analysis.flight_segmenter import segment_impacts_as_candidates
from src.analysis.parabolic_bounce_detector import BounceCandidate, find_impacts
from src.analysis.touchdown_detector import Touchdown, classify_touchdowns
from src.detection.ball_detector import Detection
from src.tracking.ball_tracker import BallTracker, TrackedPosition


def merge_impacts(
    scanned: list[BounceCandidate],
    from_segments: list[BounceCandidate],
    min_frame_gap: int,
) -> list[BounceCandidate]:
    """One impact list from the two searches, keeping the better-fitting of
    any pair that describe the same event.

    The frame-by-frame scan is precise when it has data either side of the
    impact; the segment intersection still finds the impact when it doesn't.
    They agree on most events, so the union is deduplicated by proximity.
    """
    merged: list[BounceCandidate] = []
    for impact in sorted(list(scanned) + list(from_segments), key=lambda c: c.t):
        if merged and impact.t - merged[-1].t <= min_frame_gap:
            if impact.rmse < merged[-1].rmse:
                merged[-1] = impact
        else:
            merged.append(impact)
    return merged


@dataclass(frozen=True)
class ImpactAnalysis:
    """`impacts` already carries the final verdict of each impact, so most
    callers want only that. `positions` and `touchdowns` are kept because
    the measurements behind a verdict are what make a disagreement with a
    human label diagnosable rather than just wrong."""

    positions: list[TrackedPosition]
    impacts: list[BounceCandidate]
    touchdowns: list[Touchdown] = field(default_factory=list)

    @property
    def bounces(self) -> list[BounceCandidate]:
        return [impact for impact in self.impacts if impact.kind == "bounce"]

    @property
    def contacts(self) -> list[BounceCandidate]:
        return [impact for impact in self.impacts if impact.kind == "contact"]

    @property
    def unattributed(self) -> list[BounceCandidate]:
        return [impact for impact in self.impacts if impact.kind == "unknown"]


def analyze_impacts(
    detections: list[Optional[Detection]],
    fps: float,
    calibrations_by_frame: Optional[dict[int, CourtCalibration]] = None,
    player_boxes_by_frame: Optional[dict[int, list[tuple[float, float, float, float]]]] = None,
    max_pixels_per_frame: float = 150.0,
    max_interpolation_gap: int = 8,
    static_lockon_frames: int = 10,
    static_lockon_radius: float = 20.0,
    use_flight_segments: bool = True,
    merge_window_seconds: float = 0.2,
    **classifier_kwargs,
) -> ImpactAnalysis:
    """Find the ball's impacts and attribute each to the court or a racket.

    `calibrations_by_frame` is what turns "something hit the ball" into
    "the court hit the ball": without it there is no projected court
    position to measure and `touchdowns` comes back empty, leaving the
    scan's own screen-space guess as the verdict. `player_boxes_by_frame`
    is optional even then - it only ever WITHHOLDS a contact verdict, never
    creates one (see touchdown_detector.classify_touchdowns).

    `**classifier_kwargs` are passed through to `classify_touchdowns`.
    """
    tracker = BallTracker(
        max_pixels_per_frame=max_pixels_per_frame,
        max_interpolation_gap=max_interpolation_gap,
        static_lockon_frames=static_lockon_frames,
        static_lockon_radius=static_lockon_radius,
        smoothing_window=0,
    )
    positions = tracker.track(detections)

    impacts = find_impacts(positions)
    if use_flight_segments:
        impacts = merge_impacts(
            impacts,
            segment_impacts_as_candidates(positions),
            int(merge_window_seconds * fps),
        )

    if not calibrations_by_frame:
        return ImpactAnalysis(positions=positions, impacts=impacts)

    touchdowns = classify_touchdowns(
        impacts,
        positions,
        calibrations_by_frame,
        fps,
        player_boxes_by_frame=player_boxes_by_frame,
        **classifier_kwargs,
    )
    return ImpactAnalysis(
        positions=positions,
        impacts=[
            replace(td.impact, is_bounce=td.is_bounce, kind=td.kind, reason=td.reason)
            for td in touchdowns
        ],
        touchdowns=touchdowns,
    )
