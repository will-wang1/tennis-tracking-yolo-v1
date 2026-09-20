"""Classify a court-projected point (e.g. a bounce location) into
descriptive zones - which half, which side, how deep, and whether it was
in the singles court, the doubles alley, or out.

All of this is derived from `FULL_COURT_REFERENCE_POINTS`
(`court_calibration.py`), the same reference points a calibration was
fitted against, so it stays correct if that layout ever changes.

`side` is LEFT/RIGHT AS THE CAMERA SEES IT, not tennis's deuce/ad. Those
are different questions: deuce/ad is defined relative to the player FACING
the net, so it flips between the two halves and only survives if the
camera is not mirrored from the standard behind-baseline convention.
Camera-left/right needs no such assumption - one fixed x threshold for the
whole court - and it is the question coaching actually asks ("is the feed
going to both wings, or only one"). A deuce/ad reading can still be
recovered downstream by flipping `side` on the far half, if a caller ever
needs it.
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
    side: str  # "left" | "right" - as the camera sees it, see module docstring
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

    # One threshold for the whole court, unlike the deuce/ad reading this
    # replaced - camera-left is camera-left on both halves.
    side = "left" if world_x < _CENTER_X else "right"
    if half == "far":
        depth = "deep" if world_y <= _SERVICE_LINE_FAR_Y else "short"
    else:
        depth = "deep" if world_y >= _SERVICE_LINE_NEAR_Y else "short"

    if _SINGLES_LEFT_X <= world_x <= _SINGLES_RIGHT_X:
        bounds = "singles"
    elif _DOUBLES_LEFT_X <= world_x <= _DOUBLES_RIGHT_X:
        bounds = "doubles_alley"
    else:
        bounds = "out"

    return LandingZone(half=half, side=side, depth=depth, bounds=bounds)
