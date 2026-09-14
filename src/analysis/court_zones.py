"""Classify a court-projected point (e.g. a bounce location) into
descriptive zones - which half, which side, how deep, and whether it was
in the singles court, the doubles alley, or out.

All of this is derived from `FULL_COURT_REFERENCE_POINTS`
(`court_calibration.py`), the same reference points a calibration was
fitted against, so it stays correct if that layout ever changes.

DEUCE/AD IS A BEST-EFFORT LABEL, not a measurement. It assumes a standard
behind-baseline broadcast camera, not mirrored left-right - the same
assumption `FULL_COURT_REFERENCE_POINTS`'s own docstring makes ("x
increasing toward the right doubles sideline ... for a typical
behind-baseline broadcast angle"). Under that assumption the mapping is
real court geometry, not a guess: a player facing the net from the NEAR
baseline faces the same way the camera looks, so their right hand (the
deuce court) points toward increasing x, matching the viewer's right: a
player facing the net from the FAR baseline faces the camera, so their
right hand points the other way - deuce is toward DECREASING x on that
side. That is why `side` flips between `half`s below rather than using one
fixed x threshold for the whole court. If a clip's camera is mirrored from
that convention, every `side` label here comes out swapped; `half`,
`depth` and `bounds` do not depend on the assumption and stay correct
regardless.
"""

from dataclasses import dataclass

from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS

_COURT_LENGTH = FULL_COURT_REFERENCE_POINTS["baseline_near_left"][1]
_NET_Y = _COURT_LENGTH / 2
_SINGLES_LEFT_X = FULL_COURT_REFERENCE_POINTS["singles_far_left"][0]
_SINGLES_RIGHT_X = FULL_COURT_REFERENCE_POINTS["singles_far_right"][0]
_DOUBLES_LEFT_X = FULL_COURT_REFERENCE_POINTS["baseline_far_left"][0]
_DOUBLES_RIGHT_X = FULL_COURT_REFERENCE_POINTS["baseline_far_right"][0]
_CENTER_X = (_SINGLES_LEFT_X + _SINGLES_RIGHT_X) / 2
_SERVICE_LINE_FAR_Y = FULL_COURT_REFERENCE_POINTS["service_far_left"][1]
_SERVICE_LINE_NEAR_Y = FULL_COURT_REFERENCE_POINTS["service_near_left"][1]


@dataclass(frozen=True)
class LandingZone:
    half: str  # "far" | "near" - which baseline's side of the net
    side: str  # "deuce" | "ad" - see module docstring for the camera assumption
    depth: str  # "short" (net-to-service-line) | "deep" (service-line-to-baseline)
    bounds: str  # "singles" | "doubles_alley" | "out"

    def label(self) -> str:
        return f"{self.half}_{self.side}_{self.depth}"


def classify_court_half(world_y: float) -> str:
    """'far' | 'near' - which baseline's side of the net a world Y position
    is on. Split out of `classify_landing_zone` because it's meaningful for
    a point that isn't necessarily a bounce - e.g. attributing a racket
    CONTACT to whichever player's half it happened on, which is exactly
    where a player must be standing to have hit it (see
    match_stats.attribute_contacts_to_court_half)."""
    return "far" if world_y < _NET_Y else "near"


def classify_landing_zone(world_x: float, world_y: float) -> LandingZone:
    half = classify_court_half(world_y)

    if half == "far":
        side = "deuce" if world_x < _CENTER_X else "ad"
        depth = "deep" if world_y <= _SERVICE_LINE_FAR_Y else "short"
    else:
        side = "deuce" if world_x >= _CENTER_X else "ad"
        depth = "deep" if world_y >= _SERVICE_LINE_NEAR_Y else "short"

    if _SINGLES_LEFT_X <= world_x <= _SINGLES_RIGHT_X:
        bounds = "singles"
    elif _DOUBLES_LEFT_X <= world_x <= _DOUBLES_RIGHT_X:
        bounds = "doubles_alley"
    else:
        bounds = "out"

    return LandingZone(half=half, side=side, depth=depth, bounds=bounds)
