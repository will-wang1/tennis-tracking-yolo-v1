"""Cross-impact structural corrections `classify_touchdowns` cannot make on
its own, because it only ever judges one impact from its own before/after
motion. Two rules live here, both facts about how a live point can behave
that no single impact's kinematics reveals by itself:

`filter_non_rally_clusters` catches impacts that were never part of the
point at all - most often a player bouncing the ball in place before
serving, but the same signature covers any other same-spot ball-handling
(adjusting a grip, a practice bounce) - and reclassifies them from
"unknown".

FOUND ON REAL FOOTAGE, not hypothesised: on the Alcaraz-Djokovic clip,
`classify_touchdowns` reads six consecutive events at 32.86s-37.43s
(three "contact", i.e. counted as RETURNS, plus real ground bounces
between them) as if they were rally play. Their world positions cluster
within about 0.3m in x and 4m in y of each other - the server standing in
one spot, bouncing the ball - and the very next event at 38.82s (the real
serve) sits roughly 8m away, a completely different position. Every one of
those six got folded into shot counts, contact-side attribution, peak
speed, and - worst for match_log.py - the serve/point structure, which
reads "another contact from the same server soon after" as fault
evidence.

WHY THIS ISN'T `touchdown_detector.py`'s job to catch: that module judges
each impact from its OWN before/after ball motion, which a real bounce-in-
place event fits by the physics it checks (the ball genuinely does
descend and rebound at each beat) - there's nothing locally wrong with any
ONE of these events. What's wrong only shows up ACROSS several of them:
tennis's own rule that a live point cannot have the ball touch the ground
or a racket twice on the SAME side without crossing the net rules out any
run of 3+ same-side impacts clustered in one spot - that has to be
something other than rally play, whatever each individual impact's own
physics said. That is a STRUCTURAL fact this module adds, not a
measurement any one impact's kinematics could reveal by itself - which is
also why this runs as a separate pass afterward rather than folded into
classify_touchdowns's own per-impact judgment.

HONEST RISK: a real serve can, in principle, land inside a cluster's
radius if the server barely moves between their last bounce and the
serve itself - nothing here can rule that out from position alone, only
make it unlikely at the tolerance used. `min_cluster_size` (3) is the
main guard: a real point-starting event has no way to already be part of
a 3-long same-spot run, since nothing precedes it. What it can't defend
against is a real serve added AS A NEW member onto an already-3+ cluster,
which would need the serve itself to land within `max_cluster_distance_m`
of the last practice bounce - measured unlikely on the one real example
available (~8m away), not proven impossible in general.
"""

from dataclasses import replace
from typing import Sequence

from src.analysis.court_calibration import CourtCalibration
from src.analysis.parabolic_bounce_detector import BounceCandidate

# Metres, measured against consecutive hops (not total cluster spread) in
# the one real cluster this module was built from: mostly under 1.2m, but
# one hop (a player shifting stance mid-bounce, or projection jitter) runs
# 3.5m - so a tighter threshold would have split one real cluster into two
# and left the smaller piece under min_cluster_size, missing it entirely.
# The real serve that follows the same cluster sits ~8m from its last
# member - more than double this threshold, a wide margin.
DEFAULT_MAX_CLUSTER_DISTANCE_M = 4.5
# How many same-spot, same-side impacts in a row before this concludes
# "not rally play" rather than "coincidence". A lone or even a pair of
# close-together impacts is well within what a real short exchange (two
# volleys near the net) could produce; three in a row with no crossing to
# the other side is not something a live point's own rules allow.
DEFAULT_MIN_CLUSTER_SIZE = 3


def filter_non_rally_clusters(
    impacts: Sequence[BounceCandidate],
    calibrations_by_frame: dict[int, CourtCalibration],
    max_cluster_distance_m: float = DEFAULT_MAX_CLUSTER_DISTANCE_M,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
) -> list[BounceCandidate]:
    """Every impact whose kind is "bounce" or "contact", reclassified to
    "unknown" if it's part of a run of `min_cluster_size` or more
    consecutive impacts (bounces and contacts both - a practice bounce
    genuinely touches the ground, which is a real "bounce" by
    `find_impacts`'s own physics, so leaving bounce-kind impacts out would
    miss half of what a bounce ROUTINE actually produces) that never
    leaves a `max_cluster_distance_m` radius. Impacts with no calibration
    for their frame are left exactly as they were - this can only ever
    make a verdict LESS specific, never invent one from nothing.

    Consecutive here means adjacent among impacts that HAVE a world
    position, not adjacent in the original list - an "unknown" impact with
    no usable position doesn't get to interrupt a run that clearly
    continues around it (the same reasoning `classify_serve_sequences`
    already uses for gaps: an event with nothing to measure isn't evidence
    the pattern broke).
    """
    ordered = sorted(impacts, key=lambda impact: impact.t)

    positioned: list[tuple[BounceCandidate, tuple[float, float]]] = []
    for impact in ordered:
        if impact.kind not in ("bounce", "contact"):
            continue
        calibration = calibrations_by_frame.get(impact.frame_idx)
        if calibration is None:
            continue
        positioned.append((impact, calibration.pixel_to_world(impact.x, impact.y)))

    groups: list[list[tuple[BounceCandidate, tuple[float, float]]]] = []
    for entry in positioned:
        _, world_xy = entry
        if groups:
            _, last_xy = groups[-1][-1]
            distance = ((world_xy[0] - last_xy[0]) ** 2 + (world_xy[1] - last_xy[1]) ** 2) ** 0.5
            if distance <= max_cluster_distance_m:
                groups[-1].append(entry)
                continue
        groups.append([entry])

    downgrade_frames = {
        impact.frame_idx
        for group in groups
        if len(group) >= min_cluster_size
        for impact, _ in group
    }
    if not downgrade_frames:
        return list(impacts)

    return [
        replace(
            impact,
            is_bounce=False,
            kind="unknown",
            reason=(
                f"part of a {min_cluster_size}+ same-spot impact cluster - looks like ball-"
                "handling before/after the point, not a shot (see rally_clusters.py)"
            ),
        )
        if impact.frame_idx in downgrade_frames
        else impact
        for impact in impacts
    ]


def filter_excess_bounces_between_contacts(
    impacts: Sequence[BounceCandidate],
) -> list[BounceCandidate]:
    """A live point bounces at most ONCE between one contact and the next -
    a second bounce on the same side before the ball is returned means the
    point is already over, so it can't be real rally play either. Found on
    the same real footage `filter_non_rally_clusters` was built from: the
    4.57s event in the Alcaraz-Djokovic clip's opening rally is a spurious
    near-net blob sitting between two real contacts that already have a
    real bounce (the serve's actual landing) between them too - three
    "bounce" verdicts where tennis allows one.

    NOT WIRED INTO `impact_pipeline.analyze_impacts` BY DEFAULT -
    REGRESSION FOUND, not just hypothesised: run against video_input2 (one
    of this project's two hand-labelled clips), it reclassified a real,
    correctly-labelled bounce at 4.30s, because a SECOND real bounce at
    6.03s happened to fall in the same "gap between two contacts" and had
    a lower RMSE. The two are ~15m apart - opposite ends of the court -
    unmistakably two separate real events, not a duplicate detection of
    one. The premise ("at most one real bounce per gap") is still true;
    what's missing is verifying the competing bounces are even spatially
    PLAUSIBLE as the same event before picking one - `filter_non_rally_
    clusters` already does exactly that kind of check for its own rule and
    this doesn't yet. Read the "count-based" logic below as a documented,
    tested building block, not a validated fix - it needs a spatial gate
    added (and re-validating against real position data for whatever
    specific case motivated using it) before it is.

    Where more than one "bounce" impact falls in the SAME gap between two
    consecutive contacts (or before the first contact, or after the last -
    a live rally shouldn't produce two real bounces there either), every
    one but the best-fitting is reclassified to "unknown". "Best" is the
    lowest `BounceCandidate.rmse` - how well its two flanking arcs actually
    fit a parabola - the same tie-breaker `parabolic_bounce_detector.
    _suppress_neighbors` already uses to choose between competing
    candidates for the same physical event, applied here across a wider
    window instead of a handful of neighbouring frames.

    Contact-kind impacts, and impacts already reclassified "unknown"
    (including by `filter_non_rally_clusters`, if that ran first - it
    should: a bounce absorbed into a same-spot cluster no longer competes
    here at all), are never touched.
    """
    ordered = sorted(impacts, key=lambda impact: impact.t)

    groups: list[list[BounceCandidate]] = [[]]
    for impact in ordered:
        groups[-1].append(impact)
        if impact.kind == "contact":
            groups.append([])

    downgrade_frames = set()
    for group in groups:
        bounces = [impact for impact in group if impact.kind == "bounce"]
        if len(bounces) < 2:
            continue
        best = min(bounces, key=lambda impact: impact.rmse)
        downgrade_frames.update(impact.frame_idx for impact in bounces if impact is not best)

    if not downgrade_frames:
        return list(impacts)

    return [
        replace(
            impact,
            is_bounce=False,
            kind="unknown",
            reason=(
                "extra bounce between two contacts - a live point bounces at most once there; "
                "kept the better-fitting candidate (see rally_clusters.py)"
            ),
        )
        if impact.frame_idx in downgrade_frames
        else impact
        for impact in impacts
    ]
