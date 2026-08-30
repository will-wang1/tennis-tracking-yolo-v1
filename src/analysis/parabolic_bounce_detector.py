"""Bounce detection by fitting the ball's actual flight physics, rather than
peak-detecting a single frame's trajectory shape.

Every earlier approach in this project asks a question about one frame or
one pair of frames: is this frame a local max in pixel y
(`geometric_bounce_detector.py`), did the world-space speed drop between
these two frames (`velocity_bounce_detector.py`, `bounce_ensemble.py`'s
speed-drop scan), does this frame's feature vector look like a bounce
(`catboost_bounce_detector.py`). All of them share two weaknesses that
showed up repeatedly on this footage:

- A single mis-detected ball position fabricates a convincing "corner", so
  noise alone produces false bounces.
- A racket CONTACT produces the same one-frame signature as a bounce - the
  ball arrives, changes direction, leaves - so no amount of threshold
  tuning on a one-frame signal separates the two.

This module instead models a segment of flight the way the physics actually
works. A ball in free flight follows a parabola: constant horizontal
velocity, constant downward acceleration. So it fits a whole ARC on each
side of a candidate impact - x linear in time, y quadratic - by least
squares over several frames, and then asks whether the transition between
those two arcs is one that a COURT could have produced:

- The arcs must actually be good fits. A lone bad detection can perturb a
  least-squares fit but cannot make both sides fit a parabola well, which
  is what kills the noise-driven false positives that single-frame peak
  detection can't distinguish from real corners.
- The ball must be descending on screen going in and rising coming out.
  A ball in continuous flight passing over the net doesn't do this at all
  (it's one arc, no junction), and a shot's apex does the opposite
  (rising, then falling) - so both are rejected structurally, not by a
  spatial exclusion zone.
- Vertical restitution must be physical: the ball cannot leave the ground
  faster upward than it arrived downward. `max_restitution` caps that at
  ~1. A racket routinely sends the ball off far faster than it came in,
  which no court surface can do.
- Horizontal velocity must be roughly PRESERVED - same direction, and not
  faster than it came in (friction only ever takes horizontal speed away).
  A racket contact is exactly the event that adds horizontal speed or
  changes its direction.
- Total speed cannot increase across the impact. A court is passive; it
  removes energy. This is the single most reliable bounce-vs-contact
  discriminator available from monocular video, and it's only measurable
  once you have a velocity estimate stable enough to compare across the
  impact - which is what the arc fits provide and single-frame differences
  do not.

Everything is measured in PIXEL space. That's deliberate: the court
homography is a ground-plane mapping, exact only for points ON the court
(see `court_calibration.py`), so an airborne ball - the ball crossing the
net especially - projects to a badly distorted world position, and every
world-space approach tried here produced phantom bounces around the net as
a result. The homography is used only at the end, to report WHERE a
confirmed bounce landed, which is the one moment the ball genuinely is on
the ground and the mapping is exact.

KNOWN LIMIT - precision is good, recall is not. On video_input2 this finds
two bounces and both were confirmed correct by eye, and every rejection
inspected by hand was also right (the candidate at 13.89s reads restitution
1.08 and reverses its horizontal direction, -5.7px/frame to +3.3 - a racket
contact, not a near-miss bounce). But it almost certainly misses bounces in
the FAR court, and the reason is structural rather than a threshold:

"Descending on screen, then rising" is the check that anchors everything
else, and it assumes screen y tracks the ball's height. Down the far end it
mostly tracks DEPTH instead - the ball moving away from the camera - so a
far-court bounce shows up as screen y merely flattening, never reversing,
and the check cannot fire. Measured on this clip: a bounce has to exist
between the near player's contact at 0.73s and the far player's at 1.97s,
and across that stretch screen y sits flat around 307px while x keeps
moving; no candidate is even evaluable there.

Widening thresholds does not reach these. Sweeping arc length, fit
tolerance, gap tolerance, minimum vertical speed and the crossing search -
54 combinations - produced exactly the same two bounces every time. Fixing
it needs the ball's real height (`court_camera.py`, which does not yet work
on this footage - see its docstring), not looser limits here.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.analysis.bounce_detector import BounceEvent
from src.analysis.court_calibration import CourtCalibration
from src.tracking.ball_tracker import TrackedPosition


@dataclass(frozen=True)
class _Arc:
    """A least-squares free-flight model over one stretch of samples: x
    linear in time, y quadratic (constant gravity).

    Times are given to every method on the same clock as
    `TrackedPosition.frame_idx`, so both arcs around an impact can be
    evaluated at one shared instant, but the fit itself is done about
    `t_ref` - a quadratic fitted against raw frame numbers in the hundreds
    or thousands has a badly conditioned Vandermonde matrix (t^2 in the
    millions against a y in the hundreds), and the resulting coefficients
    lose most of their precision.
    """

    x_coeffs: np.ndarray  # highest power first, as np.polyfit returns
    y_coeffs: np.ndarray
    t_ref: float
    rmse: float

    def y_at(self, t):
        """Accepts an array of times as well as a scalar, for the crossing
        search - `np.polyval` maps over either."""
        return np.polyval(self.y_coeffs, np.asarray(t, dtype=np.float64) - self.t_ref)

    def position(self, t: float) -> tuple[float, float]:
        shifted = t - self.t_ref
        return float(np.polyval(self.x_coeffs, shifted)), float(np.polyval(self.y_coeffs, shifted))

    def velocity(self, t: float) -> tuple[float, float]:
        """Pixels per frame at `t`, analytically from the fitted arc rather
        than differenced between two samples - a difference of two noisy
        detections is itself noisy, which is what made the earlier
        frame-to-frame speed checks unreliable."""
        shifted = t - self.t_ref
        return (
            float(np.polyval(np.polyder(self.x_coeffs), shifted)),
            float(np.polyval(np.polyder(self.y_coeffs), shifted)),
        )


@dataclass(frozen=True)
class BounceCandidate:
    """One impact, kept alongside the measurements that classified it so a
    run can be inspected without re-deriving them.

    `is_bounce` is the verdict: True when every check below says a court
    surface could have produced this, False when something added energy or
    redirected the ball, which only a racket does. Both are reported so a
    run can be checked as a whole - a rally alternates contact and bounce,
    so seeing the contacts is how you tell a MISSED bounce from a stretch
    where the ball genuinely never landed.

    `kind` is the same verdict widened to three values, because "not a
    bounce" and "a racket contact" are NOT the same claim. This scan only
    ever produces "bounce" or "contact"; `touchdown_detector` re-judges
    these and can also return "unknown", for an impact it can neither
    confirm nor attribute. Keeping that separate matters at the drawing
    end: an event with no evidence behind it should get no marker, rather
    than being shown as a contact by default.
    """

    frame_idx: int
    t: float  # sub-frame impact time
    x: float
    y: float
    restitution: float
    horizontal_ratio: float
    speed_ratio: float
    rmse: float
    is_bounce: bool = True
    kind: str = "bounce"  # "bounce" | "contact" | "unknown"
    reason: str = "bounce"  # why it was classified the way it was


def _fit_arc(
    samples: list[TrackedPosition], t_ref: float, min_samples: int = 6
) -> Optional[_Arc]:
    """None if the samples can't properly over-determine the fit.

    Three points fix a quadratic exactly, so its residual is 0 however
    unlike a real flight they are, and four leave it barely constrained -
    which is enough to invent a junction out of noise. Measured on
    video_input2: an impact reported at 0.34s, hand-checked as a false
    positive, sat immediately after an eight-frame dropout where only four
    samples were available, and requiring six removes it while keeping
    every confirmed bounce.
    """
    if len(samples) < min_samples:
        return None
    times = np.array([p.frame_idx for p in samples], dtype=np.float64) - t_ref
    xs = np.array([p.x for p in samples], dtype=np.float64)
    ys = np.array([p.y for p in samples], dtype=np.float64)

    x_coeffs = np.polyfit(times, xs, 1)
    y_coeffs = np.polyfit(times, ys, 2)
    residuals = np.hypot(np.polyval(x_coeffs, times) - xs, np.polyval(y_coeffs, times) - ys)
    return _Arc(
        x_coeffs=x_coeffs,
        y_coeffs=y_coeffs,
        t_ref=t_ref,
        rmse=float(np.sqrt(np.mean(residuals**2))),
    )


def _impact_time(before: _Arc, after: _Arc, around: float, search_frames: float) -> Optional[float]:
    """The instant the two arcs actually meet, searched within
    `search_frames` either side of `around`, or None if they never do.

    A real impact is a single continuous trajectory that changes slope, so
    the two arcs have to cross. What this rules out is a junction where the
    arcs are describing different places entirely - a sign the window spans
    a tracking failure rather than one impact. Solved by scanning for a
    sign change rather than by the quadratic formula, which degenerates
    when the two arcs' curvatures are nearly equal (the common case here).
    """
    step = 0.05
    times = np.arange(around - search_frames, around + search_frames + step, step)
    gaps = after.y_at(times) - before.y_at(times)
    sign_changes = np.nonzero(np.sign(gaps[:-1]) != np.sign(gaps[1:]))[0]
    if len(sign_changes) == 0:
        return None
    # Whichever crossing sits closest to the candidate frame: a pair of
    # near-identical parabolas can cross twice, far apart, and only a
    # crossing near the candidate is the impact being tested.
    crossings = [
        float(times[i] + step * abs(gaps[i]) / max(abs(gaps[i]) + abs(gaps[i + 1]), 1e-9))
        for i in sign_changes
    ]
    return min(crossings, key=lambda t: abs(t - around))


def drop_duplicate_frames(
    ordered: list[TrackedPosition], duplicate_ratio: float, local_window: int
) -> list[TrackedPosition]:
    """Drop samples that repeat the previous frame's position - the ball
    standing still for exactly one frame and then resuming.

    video_input2 is 50fps content encoded at 60fps, so one frame in every
    six is a duplicate of the one before it and the ball appears frozen on
    it. That is a systematic violation of the free-flight model - a stalled
    sample every sixth frame pulls every arc away from the parabola it is
    supposed to be measuring, inflating the fit error and dragging the
    fitted velocities toward zero right where they matter. Removing the
    repeats leaves genuinely distinct samples whose frame indices are still
    very nearly their true times (an output frame index divided by the
    output rate is the real timestamp either way), so the fit needs no other
    adjustment.

    A sample counts as a repeat only if it moved far less than the ball is
    moving locally (`duplicate_ratio` of the median step over
    `local_window` samples), rather than by a fixed pixel threshold: the
    ball genuinely crawls across a handful of pixels per frame in the far
    court and races across dozens in the near court, so no single distance
    separates "duplicated" from "slow" everywhere in the frame.
    """
    if len(ordered) < 3:
        return list(ordered)

    steps = [0.0] + [
        float(np.hypot(b.x - a.x, b.y - a.y)) for a, b in zip(ordered, ordered[1:])
    ]
    half = max(local_window // 2, 1)

    kept = [ordered[0]]
    for i in range(1, len(ordered)):
        sample = ordered[i]
        if sample.frame_idx - kept[-1].frame_idx == 1:
            neighborhood = steps[max(i - half, 1) : i + half + 1]
            local_step = float(np.median(neighborhood)) if neighborhood else 0.0
            moved = float(np.hypot(sample.x - kept[-1].x, sample.y - kept[-1].y))
            if moved < duplicate_ratio * local_step:
                continue
        kept.append(sample)
    return kept


def _window(
    ordered: list[TrackedPosition],
    start_frame: int,
    end_frame: int,
    max_internal_gap: int,
    from_end: bool,
) -> list[TrackedPosition]:
    """Real samples with frame_idx in [start_frame, end_frame], truncated at
    the first internal gap longer than `max_internal_gap` - fitting a flight
    arc across a long dropout would be fitting across an event that may well
    include another impact.

    Truncation walks outward from the candidate, so the samples kept are
    always the ones adjacent to the impact being tested: `from_end` for a
    "before" window (the candidate sits just past `end_frame`), from the
    start for an "after" window.
    """
    in_range = [p for p in ordered if start_frame <= p.frame_idx <= end_frame and not p.interpolated]
    if not in_range:
        return []

    ordered_from_candidate = list(reversed(in_range)) if from_end else in_range
    kept = [ordered_from_candidate[0]]
    for sample in ordered_from_candidate[1:]:
        if abs(sample.frame_idx - kept[-1].frame_idx) > max_internal_gap:
            break
        kept.append(sample)
    return list(reversed(kept)) if from_end else kept


def find_impacts(
    positions: list[TrackedPosition],
    arc_frames: int = 8,
    max_arc_rmse: float = 4.0,
    max_impact_gap_frames: int = 3,
    max_internal_gap: int = 2,
    min_vertical_speed: float = 1.0,
    min_restitution: float = 0.15,
    max_restitution: float = 1.0,
    min_horizontal_speed: float = 0.5,
    max_horizontal_gain: float = 1.15,
    max_speed_gain: float = 1.05,
    min_arc_samples: int = 6,
    min_impact_speed: float = 2.0,
    min_velocity_change: float = 0.35,
    min_frame_gap: int = 8,
    impact_search_frames: float = 1.5,
    duplicate_ratio: float = 0.25,
    duplicate_window: int = 9,
) -> list[BounceCandidate]:
    """Every impact found in the trajectory - each junction between two
    fitted free-flight arcs where the velocity genuinely jumps - classified
    as a court bounce or a racket contact. See the module docstring for what
    each check rules out.

    `min_impact_speed` and `min_velocity_change` are what decide something
    happened at all, before any bounce-vs-contact question: a ball in free
    flight changes velocity only under gravity, so a jump of more than
    `min_velocity_change` of the incoming speed means it was struck by
    something. Reporting contacts as well as bounces is what makes a run
    checkable by eye - a rally alternates the two, so a contact with no
    bounce between it and the next contact is either a volley or a bounce
    that was missed.

    `positions` should come from an UNSMOOTHED tracking pass
    (`BallTracker(smoothing_window=0)`). Unlike the earlier peak-detection
    approaches, this one wants the raw samples: the least-squares fit is
    itself the noise filter, and pre-smoothing both flattens the corner
    being measured and makes `max_arc_rmse` meaningless by construction, so
    a noisy stretch would then look like a perfect fit.

    `arc_frames` is how far each arc reaches from the candidate; long
    enough that noise averages out, short enough to stay within one flight
    (roughly a third of a second at 30fps). `max_arc_rmse` is in pixels -
    how far the samples may sit from an ideal parabola before the stretch
    is judged not to be clean flight at all. `max_impact_gap_frames` allows
    the ball to go undetected right at the impact, which is common: the
    contact frame is the blurriest one in the whole flight.
    """
    ordered = sorted((p for p in positions if not p.interpolated), key=lambda p: p.frame_idx)
    ordered = drop_duplicate_frames(ordered, duplicate_ratio, duplicate_window)
    if not ordered:
        return []
    candidates: list[BounceCandidate] = []

    # Every frame in range, not only frames that have a detection: the
    # impact frames are the blurriest of a flight and are exactly the ones
    # the detector tends to miss, so restricting candidates to detected
    # frames would systematically skip the clearest bounces.
    for frame in range(ordered[0].frame_idx, ordered[-1].frame_idx + 1):
        before_samples = _window(ordered, frame - arc_frames, frame - 1, max_internal_gap, from_end=True)
        after_samples = _window(ordered, frame + 1, frame + arc_frames, max_internal_gap, from_end=False)
        if not before_samples or not after_samples:
            continue
        if frame - before_samples[-1].frame_idx > max_impact_gap_frames:
            continue
        if after_samples[0].frame_idx - frame > max_impact_gap_frames:
            continue

        before = _fit_arc(before_samples, t_ref=float(frame), min_samples=min_arc_samples)
        after = _fit_arc(after_samples, t_ref=float(frame), min_samples=min_arc_samples)
        if before is None or after is None:
            continue
        if before.rmse > max_arc_rmse or after.rmse > max_arc_rmse:
            continue  # not clean flight on both sides - can't trust either velocity

        impact_t = _impact_time(before, after, around=float(frame), search_frames=impact_search_frames)
        if impact_t is None:
            continue
        # The impact has to fall in the gap BETWEEN the two arcs' samples.
        # Each arc is a model of pure flight on one side of the impact, so
        # an impact time landing inside either arc's own data means that arc
        # was fitted across the impact - and a fit straddling an impact
        # returns velocities that are averages of the two sides rather than
        # measurements of either, which can land inside the physical bands
        # below and pass off a racket contact as a bounce.
        if not before_samples[-1].frame_idx <= impact_t <= after_samples[0].frame_idx:
            continue

        # Both velocities are read at the SAME instant, which is also the
        # same point on screen - so perspective scales them both by the same
        # factor and every ratio below is free of it. Comparing each arc's
        # average speed over its own window instead would not be: a ball
        # bouncing away in the far court and returning toward the camera
        # covers visibly more pixels per frame afterwards for no physical
        # reason at all.
        vx_before, vy_before = before.velocity(impact_t)
        vx_after, vy_after = after.velocity(impact_t)

        speed_before = float(np.hypot(vx_before, vy_before))
        speed_after = float(np.hypot(vx_after, vy_after))
        restitution = -vy_after / vy_before if abs(vy_before) > 1e-9 else 0.0
        horizontal_ratio = abs(vx_after) / max(abs(vx_before), 1e-9)

        # Is this an impact at all? In free flight the velocity only drifts
        # under gravity, so across one junction it barely changes; something
        # STRUCK the ball if the velocity jumps. This is what separates a
        # real event from an ordinary mid-flight frame, and it's deliberately
        # asked before any bounce-specific question so that contacts are
        # found too, not silently dropped.
        velocity_change = float(np.hypot(vx_after - vx_before, vy_after - vy_before))
        if speed_before < min_impact_speed:
            continue
        if velocity_change < min_velocity_change * speed_before:
            continue

        # Now: could a COURT have done this, or only a racket?
        moving_horizontally = (
            abs(vx_before) >= min_horizontal_speed and abs(vx_after) >= min_horizontal_speed
        )
        if vy_before < min_vertical_speed:
            # Not falling on the way in - a ball struck while still rising
            # or travelling flat, which the court cannot be responsible for.
            reason = "struck while not descending"
        elif vy_after > -min_vertical_speed:
            reason = "did not rebound upward"
        elif not (min_restitution <= restitution <= max_restitution):
            # A passive surface can't return more upward speed than it
            # received. Loosening this cap past 1 was tried and bought
            # nothing: the two confirmed bounces on this project's footage
            # measure 0.27 and 0.60, well clear of it, while the candidates
            # sitting just above 1 turned out to be racket contacts caught
            # by the horizontal-reversal check below anyway.
            reason = f"restitution {restitution:.2f}"
        elif moving_horizontally and np.sign(vx_after) != np.sign(vx_before):
            reason = "turned the ball around"  # only a racket does this
        elif abs(vx_after) > abs(vx_before) * max_horizontal_gain + min_horizontal_speed:
            reason = "gained horizontal speed"  # friction only ever removes it
        elif speed_after > speed_before * max_speed_gain:
            reason = "gained speed"  # energy went IN, so not the court
        else:
            reason = "bounce"

        x, y = before.position(impact_t)
        candidates.append(
            BounceCandidate(
                frame_idx=int(round(impact_t)),
                t=impact_t,
                x=x,
                y=y,
                restitution=restitution,
                horizontal_ratio=horizontal_ratio,
                speed_ratio=speed_after / max(speed_before, 1e-9),
                rmse=max(before.rmse, after.rmse),
                is_bounce=reason == "bounce",
                kind="bounce" if reason == "bounce" else "contact",
                reason=reason,
            )
        )

    return _suppress_neighbors(candidates, min_frame_gap)


def _suppress_neighbors(
    candidates: list[BounceCandidate], min_frame_gap: int
) -> list[BounceCandidate]:
    """One impact fires on several neighbouring candidate frames (each
    offset window still straddles the same real junction); collapse each
    cluster to a single event.

    Within a cluster a BOUNCE verdict wins over a contact one, and only then
    does fit quality choose between equals. That asymmetry is deliberate. On
    a frame either side of the true junction, one arc is fitted partly
    across the impact, so its velocity is a blend of both sides - which
    inflates the apparent restitution and speed and makes a real bounce read
    as a contact. Both confirmed bounces on this project's footage do
    exactly that: frame 739 measures restitution 0.60, and its neighbours
    740 and 741 measure 1.26 and 1.46. The reverse essentially cannot
    happen, because calling something a bounce needs every check to pass at
    once - descending in, rebounding up, restitution at or under 1,
    horizontal direction kept, and no speed gained - and blending two
    velocities does not conjure all five.
    """
    kept: list[BounceCandidate] = []
    for candidate in sorted(candidates, key=lambda c: c.t):
        if kept and candidate.frame_idx - kept[-1].frame_idx <= min_frame_gap:
            incumbent = kept[-1]
            if candidate.is_bounce and not incumbent.is_bounce:
                kept[-1] = candidate
            elif candidate.is_bounce == incumbent.is_bounce and candidate.rmse < incumbent.rmse:
                kept[-1] = candidate
        else:
            kept.append(candidate)
    return kept


def detect_bounces_parabolic(
    positions: list[TrackedPosition],
    calibrations_by_frame: Optional[dict[int, CourtCalibration]] = None,
    **kwargs,
) -> list[BounceEvent]:
    """`find_bounce_candidates` as `BounceEvent`s, with the landing spot
    projected to court coordinates where a calibration is available.

    No spatial filtering is applied - no off-court cut, no near-net
    exclusion, no near-player rejection. Those were all compensations for
    signals that couldn't tell a bounce from a contact on their own; the
    physical checks here are meant to do that job directly, and layering
    spatial guards on top would hide whether they actually do.
    """
    events = []
    for candidate in find_bounce_candidates(positions, **kwargs):
        world_x = world_y = None
        if calibrations_by_frame is not None:
            calibration = calibrations_by_frame.get(candidate.frame_idx)
            if calibration is not None:
                world_x, world_y = calibration.pixel_to_world(candidate.x, candidate.y)
        events.append(
            BounceEvent(
                frame_idx=candidate.frame_idx,
                x=candidate.x,
                y=candidate.y,
                world_x=world_x,
                world_y=world_y,
            )
        )
    return events


def find_bounce_candidates(positions: list[TrackedPosition], **kwargs) -> list[BounceCandidate]:
    """Just the impacts that a court surface could have produced."""
    return [impact for impact in find_impacts(positions, **kwargs) if impact.is_bounce]
