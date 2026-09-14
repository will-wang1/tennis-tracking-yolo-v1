"""Classify each serve's outcome from the STRUCTURE of what happens after
it, not from where the ball landed.

"Was that serve in" needs the ball's real landing spot measured against the
service box, and this project has deliberately not built that: court_zones.
py's `bounds` field only checks the sideline axis, never depth against the
baseline, and every module here that touches an airborne ball's projected
position documents the same limit (parabolic_bounce_detector.py,
touchdown_detector.py) - the ground-plane homography is exact for a point
ON the court and increasingly wrong the higher the ball is, which is
exactly where a serve needs judging. Reading a serve call from that
projection would be measuring noise and calling it a fact.

Structure gives a different, honest signal instead. In real tennis a FAULT
is followed by the SAME player serving again within a few seconds - the
ball toss resets, nothing else happens in between. An IN serve is followed
by either a rally (more contacts, sides alternating) or the point simply
ending, with a comparatively long pause before the NEXT point's serve. So
"does the same server serve again soon" and "does a rally actually
develop" are both readable from impact TIMING and SIDE/IDENTITY alone -
no landing spot needed. That is the whole method: every "serve candidate"
below is the first contact after a gap long enough to mean a new attempt
started, and its outcome is read off what happens between it and the next
one.

This needs `identity_by_frame` (see player_identity.py) to tell "the same
player served again" apart from "the other player is now serving" -
without a persistent identity, near/far alone would misread a receiver
becoming the server after a change of ends as the same player double-
faulting.

HONESTLY SCOPED, the same way this project's rally-gap grouping already is
(see match_stats.DEFAULT_RALLY_GAP_SECONDS's docstring): the gap that
separates "same server, serving again = a fault" from "a genuinely new
point's different serve" is a reasoned estimate - a real second serve
typically follows a fault within a handful of seconds, a real between-point
pause runs longer, players resetting - not a value measured against
labelled multi-point footage, because no clip available to this project
has more than one point in it. Treat the resulting fault/unreturned/rally
labels as a plausible reading of the structure, not a verified call.

WHAT THIS DELIBERATELY DOES NOT DO: say who WON a point. That needs
knowing whether the last shot of a point was a winner or an error, which is
exactly the in/out judgment this module exists to avoid making from
pixels. A point here just ends when the impact sequence stops, with no
claim about why.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

from src.analysis.court_calibration import CourtCalibration
from src.analysis.court_zones import classify_court_half
from src.analysis.parabolic_bounce_detector import BounceCandidate

# How long a gap must be before the NEXT contact counts as starting a new
# attempt (a serve candidate) rather than being mid-rally. Short on
# purpose - real rally shots land well under this apart, so it only needs
# to clear ordinary shot-to-shot cadence, not distinguish a fault gap from
# a between-point one (that distinction is FAULT_GAP_SECONDS's job below).
DEFAULT_MIN_SERVE_GAP_SECONDS = 1.5

# How long a gap may be, with the SAME server on both sides of it, before
# it stops looking like "the same toss-and-serve routine repeating" and
# starts looking like a coincidence (or two separate points the same
# player happened to serve, e.g. back-to-back service games are never
# adjacent, but a bad merge elsewhere could still produce this). See the
# module docstring's "HONESTLY SCOPED" note - not measured, estimated.
DEFAULT_FAULT_GAP_SECONDS = 8.0


@dataclass(frozen=True)
class ServeAttempt:
    """One serve candidate and how the point structure around it read.

    `outcome`:
      - "fault": the same server served again within `fault_gap_seconds`,
        with nothing but this serve happening in between.
      - "unreturned": nobody served again soon, and this serve was the
        ONLY contact before the next long gap - an ace, an unreturned
        serve, or a very short exchange the impact scan didn't resolve
        into separate contacts. Structure can't tell those apart, so it
        isn't asked to.
      - "rally_developed": more than one contact happened after this serve
        before the next long gap - the point was actually played out.
      - "unknown": there wasn't enough evidence (e.g. no identity known
        for the server, or this is the last serve candidate in the clip
        with nothing after it to read a pattern from).
    """

    frame_idx: int
    t_s: float
    side: str  # "near" | "far" - which half the serve was hit from
    identity: Optional[str]  # from identity_by_frame, if known
    outcome: str
    shots_before_next_serve: int  # contacts strictly after this one, before the next serve candidate (0 for a clean unreturned serve)


def _side_and_identity(
    impact: BounceCandidate,
    calibrations_by_frame: dict[int, CourtCalibration],
    identity_by_frame: dict[int, dict[str, str]],
) -> tuple[Optional[str], Optional[str]]:
    calibration = calibrations_by_frame.get(impact.frame_idx)
    if calibration is None:
        return None, None
    world_y = calibration.pixel_to_world(impact.x, impact.y)[1]
    side = classify_court_half(world_y)
    identity = identity_by_frame.get(impact.frame_idx, {}).get(side)
    return side, identity


def classify_serve_sequences(
    impacts: Sequence[BounceCandidate],
    calibrations_by_frame: dict[int, CourtCalibration],
    identity_by_frame: dict[int, dict[str, str]],
    fps: float,
    min_serve_gap_seconds: float = DEFAULT_MIN_SERVE_GAP_SECONDS,
    fault_gap_seconds: float = DEFAULT_FAULT_GAP_SECONDS,
) -> list[ServeAttempt]:
    """One `ServeAttempt` per contact that looks like the start of a new
    attempt (a real serve, or a fault's retry), in chronological order.

    `impacts` should be the full impact list (bounces, contacts, and
    unknowns alike) - a bounce or an unattributed impact can't itself BE a
    serve, but it counts toward "did a rally develop" and toward what the
    next real gap is measured from, so leaving them out would make a point
    that ends in an unattributed kink look like a longer gap than it was.
    """
    ordered = sorted(impacts, key=lambda impact: impact.t)
    contacts = [impact for impact in ordered if impact.kind == "contact"]
    if not contacts:
        return []

    # Which contacts are "serve candidates" - the first contact after a gap
    # of at least min_serve_gap_seconds since the PREVIOUS impact of any
    # kind (a bounce right before a contact means the rally was still
    # going, whatever kind of impact it was).
    candidate_indices = []
    for idx, impact in enumerate(contacts):
        previous_any = None
        for other in ordered:
            if other.t < impact.t:
                previous_any = other
            else:
                break
        if previous_any is None or (impact.t - previous_any.t) / fps >= min_serve_gap_seconds:
            candidate_indices.append(idx)

    attempts = []
    for pos, idx in enumerate(candidate_indices):
        impact = contacts[idx]
        side, identity = _side_and_identity(impact, calibrations_by_frame, identity_by_frame)
        next_idx = candidate_indices[pos + 1] if pos + 1 < len(candidate_indices) else None
        shots_before_next = (next_idx - idx - 1) if next_idx is not None else (len(contacts) - idx - 1)

        if next_idx is None:
            outcome = "unknown"
        else:
            next_impact = contacts[next_idx]
            next_side, next_identity = _side_and_identity(next_impact, calibrations_by_frame, identity_by_frame)
            gap_to_next = (next_impact.t - impact.t) / fps
            same_server = identity is not None and identity == next_identity
            if shots_before_next == 0 and same_server and gap_to_next <= fault_gap_seconds:
                outcome = "fault"
            elif shots_before_next == 0:
                outcome = "unreturned"
            else:
                outcome = "rally_developed"

        attempts.append(
            ServeAttempt(
                frame_idx=impact.frame_idx,
                t_s=impact.t / fps,
                side=side or "unknown",
                identity=identity,
                outcome=outcome,
                shots_before_next_serve=shots_before_next,
            )
        )
    return attempts
