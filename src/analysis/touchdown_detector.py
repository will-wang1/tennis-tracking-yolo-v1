"""Decide which impacts are court bounces, using the ball's PROJECTED court
position rather than its screen position.

`parabolic_bounce_detector.find_impacts` finds where the ball was struck by
something - that part works, and it works in pixel space for good reasons
(see its docstring). What it cannot do reliably is say whether the court or
a racket did the striking, because every check it uses reads screen y as if
it were the ball's height, and screen y is really a mixture of height and
how far down the court the ball is. Measured against hand-labelled truth on
video_input2, that misfiled three of five real bounces as contacts: at
0.50s the ball was travelling toward the camera so screen y never reversed
at all, and at 1.97s the same effect squashed the incoming vertical speed to
2.5px/frame and inflated the apparent restitution to 3.40.

This module uses the projection error as the signal instead of fighting it.
The court homography maps the ground plane, so it is exact for a ball ON the
court and wrong for one above it - the camera ray through an airborne ball
meets the ground somewhere BEYOND the ball, pushed directly away from the
camera. That inflation grows with height and vanishes at touchdown. So for a
ball approaching the camera, its APPARENT court-position speed is inflated
while it descends and drops sharply the moment it lands:

    same direction along the court, and that rate abruptly drops -> touchdown

which is `looks_like_touchdown` below. Unlike the screen-space checks this
needs no assumption that height dominates screen y - it is measuring the
height effect directly, in the one place the homography's error is
informative rather than a nuisance.

A second, independent signal decides what a REVERSAL means: whether anybody
could have reached the ball. A return needs a racket, so an impact out of
every player's reach cannot be one whatever its projected direction did.
That gate is necessary evidence only, never sufficient - see
`classify_touchdowns` - and it is what separates a dropshot landing from a
return, the two events the direction rule alone cannot tell apart.

Verdicts are therefore THREE-valued. "bounce" and "contact" are positive
claims backed by a measurement; "unknown" means the impact is real but
unattributable here. Collapsing "unknown" into "contact" is not a harmless
default - measured on the US Open clip it put a marker on the server's ball
toss (1.32s) and on a stray blob over the net (6.89s), both hand-confirmed
as neither, plus four artifacts where the ball left frame.

HONESTLY SCOPED: these rules were derived from, and tuned against, hand
labels on two short clips. On the US Open clip the current combination
draws 15 markers and matches every one of the nine hand-labelled events -
five bounces including the 12.75s dropshot the reach gate exists to rescue,
and four contacts - with no marker on either confirmed-spurious event. That
is the clean result; video_input2 is weaker, and honestly so:

  - 1.99s is a confirmed bounce still read as a return. It sits 0.47 box
    heights from a player, inside the reach gate, so the gate cannot save
    it. A 0.4 threshold would - and costs nothing measurable on either clip
    - but 0.4 is less than a racket's own length against a 1.75m player, so
    it is label-fitting rather than physics, and was deliberately NOT taken.
  - 7.28s is drawn as a bounce and is hand-labelled as neither. It predates
    the reach gate and is not caused by it.

The thresholds here should be expected to need revisiting on a different
camera. None of this recovers impacts that `find_impacts` and the flight
segmenter never detected at all, which is a separate problem upstream.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.analysis.court_calibration import CourtCalibration
from src.analysis.parabolic_bounce_detector import BounceCandidate
from src.tracking.ball_tracker import TrackedPosition

# The ball leaving or re-entering the frame looks exactly like an impact -
# the trajectory stops and restarts - and on this footage it leaves through
# the TOP of the frame, heading down-court. Impacts up there are that
# artifact rather than events: hand-labelled, the three highest impacts on
# video_input2 (screen y of 70, 102 and 141) were all confirmed spurious.
_DEFAULT_FRAME_EDGE_MARGIN = 150.0

# How far from a player's box, in units of that box's HEIGHT, still counts
# as within racket reach. Box heights rather than pixels because a far-court
# player is half the size of a near one and a pixel threshold would mean two
# different distances at the two ends of the court. An outstretched arm plus
# racket is roughly 1.4m against a 1.75m player, so 0.6 box heights is about
# a metre beyond the box - and the box usually already contains the arm.
# Measured on the US Open clip, the four hand-labelled racket contacts sit at
# 0.00, 0.09, 0.20 and 0.33 box heights, while the dropshot bounce this
# threshold exists to rescue sits at 1.10. The gap is wide enough that the
# exact value between them does not matter much.
_DEFAULT_MAX_REACH_RATIO = 0.6


@dataclass(frozen=True)
class Touchdown:
    """One impact re-judged against the projection signal.

    `kind` is three-valued on purpose. "bounce" and "contact" are both
    positive claims backed by a measurement; "unknown" means the impact is
    real but nothing here can attribute it - the trajectory kinked and the
    evidence ran out. Collapsing "unknown" into "contact" is what put a
    marker on the server's ball toss and on a stray blob over the net on
    the US Open clip: neither was a racket contact, both were simply not
    bounces. `is_bounce` stays as the bounce/not-bounce boolean.
    """

    impact: BounceCandidate
    approach_before: float  # court-metres per second, toward the camera
    approach_after: float
    is_bounce: bool
    reason: str
    kind: str = "bounce"  # "bounce" | "contact" | "unknown"
    player_reach: Optional[float] = None  # distance to nearest player, in box heights


def _approach_rate(
    court_y_by_frame: dict[int, float],
    start_frame: int,
    end_frame: int,
    min_samples: int,
    fps: float,
) -> Optional[float]:
    """How fast the ball's projected court position is moving toward the
    camera over a span, in court-metres per SECOND.

    Per second, not per frame, so the thresholds below mean the same thing
    on any footage - this project's clips run at both 30 and 60fps, and a
    per-frame rate would silently halve or double between them.

    Fitted over the whole span rather than differenced across two frames:
    the projected position of a fast ball is noisy, and the quantity that
    matters here is a trend, not an instant.
    """
    frames = [f for f in range(start_frame, end_frame) if f in court_y_by_frame]
    if len(frames) < min_samples:
        return None
    values = [court_y_by_frame[f] for f in frames]
    return float(np.polyfit(frames, values, 1)[0]) * fps


def player_reach_ratio(
    x: float,
    y: float,
    boxes: list[tuple[float, float, float, float]],
) -> Optional[float]:
    """Distance from (x, y) to the nearest player box, measured in units of
    that box's own height so it means the same thing at both ends of the
    court.

    None when no boxes were recorded for the frame. That is absence of
    proof, not proof of absence - a missed player detection must never be
    read as "nobody was there", or a genuine racket contact turns into a
    bounce. Callers treat None as "unknown" and fall back to the behaviour
    they would have had without player boxes at all.
    """
    if not boxes:
        return None
    best = None
    for x1, y1, x2, y2 in boxes:
        dx = max(x1 - x, 0.0, x - x2)
        dy = max(y1 - y, 0.0, y - y2)
        ratio = float(np.hypot(dx, dy)) / max(y2 - y1, 1.0)
        if best is None or ratio < best:
            best = ratio
    return best


def looks_like_touchdown(
    approach_before: Optional[float],
    approach_after: Optional[float],
    min_approach: float,
    min_slowdown: float,
    allow_reversal: bool = False,
) -> bool:
    """True when the ball kept travelling the same way down the court while
    its apparent speed along the court dropped - the projected-position
    signature of the ball reaching the ground.

    Two conditions, and both are physical:

    - The DIRECTION along the court must be unchanged. A court bounce sends
      the ball onward the way it was already going; a player RETURNS it,
      which reverses it down-court. Hand-labelled, this alone separates
      every bounce from every contact on the US Open clip - its four
      contacts all show the projected approach flipping from about +8 to
      about -30 metres per second, while all five of its bounces keep their
      sign.
    - The signed rate must DROP. That covers both directions at once,
      because the projection inflation is symmetric: an approaching ball's
      closing speed is inflated while it falls and collapses at touchdown,
      and a receding ball's recession is suppressed while it falls and
      accelerates at touchdown. Either way the signed value falls.

    `min_approach` applies to the MAGNITUDE - below it the ball is barely
    moving along the court and the sign is noise rather than direction.

    `allow_reversal` waives the direction check. Its one legitimate use is
    an impact that happened out of every player's reach: the reversal test
    is really asking "did a racket send this back?", and if no racket could
    have been there the answer is no whatever the projected direction did.
    That matters because the direction CAN flip on its own - a bounce whose
    height change outweighs its real court motion reverses in projection
    while the ball itself carries straight on. A dropshot is the worst case,
    since it lands with almost no court speed left for the height term to
    beat, and the US Open clip's dropshot bounce at 12.75s is exactly this:
    projected approach flipping +8.8 to -13.3 with the nearest player 1.10
    box heights away, far outside any racket's reach.
    """
    if approach_before is None or approach_after is None:
        return False
    if abs(approach_before) < min_approach:
        return False
    if not allow_reversal and np.sign(approach_after) != np.sign(approach_before):
        return False  # the ball was sent back down the court - a return, not a bounce
    return approach_after < approach_before - min_slowdown


def classify_touchdowns(
    impacts: list[BounceCandidate],
    positions: list[TrackedPosition],
    calibrations_by_frame: dict[int, CourtCalibration],
    fps: float,
    window: int = 9,
    min_samples: int = 4,
    min_approach: float = 3.0,
    min_slowdown: float = 3.0,
    frame_edge_margin: float = _DEFAULT_FRAME_EDGE_MARGIN,
    player_boxes_by_frame: Optional[dict[int, list[tuple[float, float, float, float]]]] = None,
    max_reach_ratio: float = _DEFAULT_MAX_REACH_RATIO,
) -> list[Touchdown]:
    """Re-judge each impact from `find_impacts` as bounce, contact or unknown.

    `window` is how many frames either side the approach rate is measured
    over, skipping the two frames adjacent to the impact itself - those
    straddle it and blend the two sides together. `min_approach` and
    `min_slowdown` are in court-metres per second.

    `player_boxes_by_frame` is optional and used in ONE direction only: to
    withhold the "contact" verdict from an impact nobody could have reached.
    It never creates a contact - a bounce landing at a player's feet is
    ordinary tennis, so proximity is necessary evidence for a racket contact
    and nowhere near sufficient. On the US Open clip a hand-labelled bounce
    sits 0.02 box heights from a player. Pass None and the reach test simply
    never fires, leaving the direction rule as the sole arbiter.
    """
    court_y_by_frame = {}
    for position in positions:
        if position.interpolated:
            continue  # a fabricated straight line carries no projection signal
        calibration = calibrations_by_frame.get(position.frame_idx)
        if calibration is not None:
            court_y_by_frame[position.frame_idx] = calibration.pixel_to_world(
                position.x, position.y
            )[1]

    results = []
    for impact in impacts:
        before = _approach_rate(
            court_y_by_frame, impact.frame_idx - window - 2, impact.frame_idx - 1, min_samples, fps
        )
        after = _approach_rate(
            court_y_by_frame, impact.frame_idx + 2, impact.frame_idx + window + 3, min_samples, fps
        )

        reach = player_reach_ratio(
            impact.x, impact.y, (player_boxes_by_frame or {}).get(impact.frame_idx, [])
        )
        out_of_reach = reach is not None and reach > max_reach_ratio

        # Ordered so that every verdict below is a positive finding and the
        # leftovers fall through to "unknown", rather than the other way
        # round. The frame-edge case is first because an impact up there is
        # an artifact of the ball leaving view, and its approach rates
        # describe two unrelated pieces of trajectory.
        if impact.y < frame_edge_margin:
            kind, reason = "unknown", "ball at the frame edge (entering or leaving view)"
        elif looks_like_touchdown(before, after, min_approach, min_slowdown, out_of_reach):
            kind = "bounce"
            reason = (
                "bounce (reversed in projection, but out of every player's reach)"
                if out_of_reach and np.sign(after) != np.sign(before)
                else "bounce"
            )
        elif before is None or after is None:
            kind, reason = "unknown", "not enough tracked positions either side"
        elif abs(before) < min_approach:
            kind, reason = "unknown", "barely moving along the court"
        elif np.sign(after) != np.sign(before) and not out_of_reach:
            kind, reason = "contact", "ball sent back down the court (a return)"
        elif np.sign(after) != np.sign(before):
            # Reversed, out of reach, and it failed the slowdown test above -
            # so it is neither a return (no racket could have been there) nor
            # a touchdown. Reachable when the projected approach flips from
            # receding to closing, which a landing never does.
            kind, reason = "unknown", "direction reversed with no player in reach"
        else:
            kind, reason = "unknown", "approach did not slow - nothing touched down"

        results.append(
            Touchdown(
                impact=impact,
                approach_before=before if before is not None else float("nan"),
                approach_after=after if after is not None else float("nan"),
                is_bounce=kind == "bounce",
                reason=reason,
                kind=kind,
                player_reach=reach,
            )
        )
    return results
