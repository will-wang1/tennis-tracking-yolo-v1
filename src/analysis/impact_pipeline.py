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
from src.analysis.rally_clusters import filter_excess_bounces_between_contacts, filter_non_rally_clusters
from src.analysis.touchdown_detector import Touchdown, classify_touchdowns
from src.detection.ball_detector import Detection
from src.tracking.ball_tracker import BallTracker, TrackedPosition


def merge_impacts(
    scanned: list[BounceCandidate],
    from_segments: list[BounceCandidate],
    min_frame_gap: int,
) -> list[BounceCandidate]:
    """One impact list from the two searches, deduplicated by proximity.

    The frame-by-frame scan is precise when it has data either side of the
    impact; the segment intersection still finds the impact when it doesn't.
    They agree on most events, so the union is deduplicated by proximity.

    Where they describe the same event but disagree about WHEN, the segment
    intersection wins. It is computed from two complete fitted flights -
    thirty-odd samples each - while a scan candidate comes from two eight
    frame windows either side of a guessed frame, so the intersection rests
    on far more data. RMSE cannot arbitrate between them, because each is
    measured over its own window length and the two are not comparable;
    using it to choose was comparing two different quantities.

    Measured on video_input2, where the ball lands deep in the far court at
    7.14s: the segmenter puts the impact there, 3.4m inside the baseline,
    while the scan reports two events either side of it, at 6.95s and
    7.28s, both of which a human confirmed were nothing at all. All three
    fall inside one merge window, and preferring the lower RMSE kept the
    7.28s one - a marker on a non-event, and a real bounce missed. Taking
    the segment estimate instead fixes both, and changes nothing on the US
    Open clip.
    """
    tagged = [(impact, False) for impact in scanned]
    tagged += [(impact, True) for impact in from_segments]

    merged: list[tuple[BounceCandidate, bool]] = []
    for impact, from_segment in sorted(tagged, key=lambda pair: pair[0].t):
        if merged and impact.t - merged[-1][0].t <= min_frame_gap:
            previous, previous_from_segment = merged[-1]
            if from_segment and not previous_from_segment:
                merged[-1] = (impact, True)
            elif from_segment == previous_from_segment and impact.rmse < previous.rmse:
                merged[-1] = (impact, from_segment)
        else:
            merged.append((impact, from_segment))
    return [impact for impact, _ in merged]


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
    filter_clusters: bool = True,
    **classifier_kwargs,
) -> ImpactAnalysis:
    """Find the ball's impacts and attribute each to the court or a racket.

    `calibrations_by_frame` is what turns "something hit the ball" into
    "the court hit the ball": without it there is no projected court
    position to measure and `touchdowns` comes back empty, leaving the
    scan's own screen-space guess as the verdict. `player_boxes_by_frame`
    is optional even then - it only ever WITHHOLDS a contact verdict, never
    creates one (see touchdown_detector.classify_touchdowns).

    `filter_clusters` runs `rally_clusters.filter_non_rally_clusters` over
    the result (only when a calibration is available, same requirement as
    `touchdowns` itself) - catches a player bouncing the ball before
    serving, which `classify_touchdowns` has no way to see since the
    signal only shows up ACROSS several impacts, not in any one impact's
    own motion. Off by default only makes sense for comparing against a
    run from before this existed; leave it on otherwise.

    (rally_clusters.py also has `filter_excess_bounces_between_contacts`,
    which is NOT run here - it regressed a real bounce on video_input2, see
    its own docstring. Not wired in until it has a spatial-plausibility
    check to go with the count-based one it has now.)

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
    reclassified = [
        replace(td.impact, is_bounce=td.is_bounce, kind=td.kind, reason=td.reason)
        for td in touchdowns
    ]
    if filter_clusters:
        # Two passes classify_touchdowns structurally can't do itself - see
        # rally_clusters.py. Run after, not folded into that function,
        # because the signal in both cases is only visible ACROSS several
        # impacts (a same-spot run; more than one bounce between two
        # contacts), not in any one impact's own before/after motion,
        # which is all classify_touchdowns ever looks at.
        #
        # Cluster-filtering runs FIRST: a bounce absorbed into a same-spot
        # ball-handling cluster must not still compete as a candidate in
        # the excess-bounce rule that follows - it was never a real bounce
        # in the point to begin with, so it shouldn't cost a REAL bounce
        # its "best fit" comparison there.
        reclassified = filter_non_rally_clusters(reclassified, calibrations_by_frame)
        # filter_excess_bounces_between_contacts is NOT wired in here - see
        # its own docstring's "REGRESSION FOUND" note. It broke a real,
        # validated video_input2 bounce (measured ~15m from the OTHER
        # "extra" bounce it was compared against, at opposite ends of the
        # court - unmistakably two real events, not a duplicate detection
        # of one). The rule needs a spatial-plausibility check before it's
        # safe to run by default; it doesn't have one yet.
    return ImpactAnalysis(
        positions=positions,
        impacts=reclassified,
        touchdowns=touchdowns,
    )
