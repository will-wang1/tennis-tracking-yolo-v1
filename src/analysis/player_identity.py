"""Persistent identity ("player_a"/"player_b") for the two players, robust
to whatever breaks pure positional continuity: `PlayerDetector.split_top_bottom`
assigns "near"/"far" from world position every frame independently, which
is trustworthy WITHIN one continuous camera shot (the ball and players
can't teleport across the net between two frames) but not ACROSS one - a
scene cut (see scene_cuts.py) can show either player from either side, and
a change of ends mid-match genuinely swaps which baseline each player
stands at. This project has no score/game logic yet to predict when an
end-change happens (see match_stats.py - that comes later, built once this
module exists), so identity here is re-derived at any KNOWN discontinuity
rather than at a predicted one; a caller passes `force_reassign=True` on
the frame a cut lands on (from `scene_cuts.detect_scene_cuts`) or after any
other break in tracking continuity it knows about.

METHOD: jersey colour. This project's footage shows two players in
visually distinct kit for the whole match (the same assumption a human
viewer relies on too), so a coarse colour histogram of each player's
detection box - the same technique scene_cuts.py already uses on whole
frames, just applied to a player crop - is a workable proxy for "is this
the same person" without a dedicated re-identification model, which this
project has no pretrained weights for. Identity is carried forward
unchanged frame to frame within a shot (cheap, and positionally reliable
there); at a discontinuity, each of that frame's near/far boxes is matched
against each known identity's stored (exponential-moving-average)
appearance, keeping whichever of the two possible pairings costs less in
total.

A first pass at this (histogramming the WHOLE detection box) failed its
own real-footage check outright: across a real cut in the Alcaraz-Djokovic
clip - a full-court wide shot to a tight closeup - the same player scored
FARTHER apart (0.63) than two different players did in the wide shot alone
(0.55), because the background each box happened to contain (blue court
vs. a dark closeup backdrop) swamped the small patch of actual jersey
either crop held. Narrowing to a torso-only region fixed it - see
`_TORSO_REGION`'s docstring for the numbers - which is the one piece of
this module actually checked against a real discontinuity rather than
only synthetic fixtures.

HONEST LIMITS: two players in similarly-coloured kit (a shared accent
colour, an all-white dress code) would defeat this outright, and there is
no fallback signal behind it - no face, jersey number, or gait cue. Also
UNVALIDATED across a genuine change of ends: no clip available to this
project contains one, and every real cut inspected so far happened to be a
wide-shot-to-single-player-closeup, never a wide-to-wide cut with both
players visible on each side - so "does re-matching work when both players
ARE visible after a cut" is still resting on the synthetic tests alone,
not a real one.
"""

from typing import Optional

import cv2
import numpy as np

PLAYER_A = "player_a"
PLAYER_B = "player_b"

DEFAULT_BINS = 24
# How much a player's running appearance signature favours history over
# any single frame - high, because one frame's crop can be a partial
# occlusion, motion blur, or a shadowed patch of court behind them; the
# signature should drift slowly, not jump on one bad read.
DEFAULT_APPEARANCE_MOMENTUM = 0.85

# The fraction of a detection box actually used, as (x0, x1, y0, y1) in
# [0, 1] box-relative coordinates - narrowed to the TORSO: centred
# horizontally (a person detector's box has slack on both sides, more of it
# whichever way the player is leaning) and the upper-body vertically
# (avoids legs, which swing through a much wider colour range - socks,
# shoes, skin - and avoids the head). Measured on two real frames either
# side of the real cut scene_cuts.py found in the Alcaraz-Djokovic clip
# (a full-court wide shot and a tight closeup, worst case for how much a
# box's background can differ): the naive full-box histogram scored the
# SAME player 0.63 apart and a DIFFERENT player only 0.55 apart - worse
# than useless, since the background (court blue vs. a dark closeup
# backdrop) swamped the small patch of actual jersey either crop contained.
# Restricting to this torso region flips that: same player 0.56, different
# player 0.83-0.97 - a clean margin between them on the one real
# before/after pair this project has to test with.
_TORSO_REGION = (0.25, 0.75, 0.15, 0.55)


def _box_histogram(
    frame: np.ndarray, bbox: tuple[float, float, float, float], bins: int = DEFAULT_BINS
) -> Optional[np.ndarray]:
    """A normalized hue/saturation histogram of a detection box's TORSO
    region (see `_TORSO_REGION`), clipped to the frame - None if the
    resulting crop doesn't overlap the frame at all (a stale or malformed
    box, not something to build an identity signature from)."""
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    tx0, tx1, ty0, ty1 = _TORSO_REGION
    crop_x1, crop_x2 = x1 + w * tx0, x1 + w * tx1
    crop_y1, crop_y2 = y1 + h * ty0, y1 + h * ty1

    cx1, cy1 = int(round(crop_x1)), int(round(crop_y1))
    cx2, cy2 = int(round(crop_x2)), int(round(crop_y2))
    cx1, cy1 = max(cx1, 0), max(cy1, 0)
    cx2, cy2 = min(cx2, frame_w), min(cy2, frame_h)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def _histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


class PlayerIdentityTracker:
    """Stateful - call `.update()` once per frame, the same pattern every
    drawer in src/visualize/draw.py already follows."""

    def __init__(
        self,
        appearance_momentum: float = DEFAULT_APPEARANCE_MOMENTUM,
        bins: int = DEFAULT_BINS,
    ):
        self.appearance_momentum = appearance_momentum
        self.bins = bins
        self._appearance: dict[str, Optional[np.ndarray]] = {PLAYER_A: None, PLAYER_B: None}
        # "near"/"far" -> whichever identity currently holds that role;
        # empty until the first frame both players are seen at once, since
        # there is nothing to assign the (arbitrary) labels from before that
        self._identity_for_role: dict[str, str] = {}

    def _remember(self, identity: str, hist: Optional[np.ndarray]) -> None:
        if hist is None:
            return
        current = self._appearance[identity]
        self._appearance[identity] = (
            hist if current is None else self.appearance_momentum * current + (1 - self.appearance_momentum) * hist
        )

    def _best_assignment(self, near_hist: np.ndarray, far_hist: np.ndarray) -> dict[str, str]:
        a_app, b_app = self._appearance[PLAYER_A], self._appearance[PLAYER_B]
        keep = _histogram_distance(near_hist, a_app) + _histogram_distance(far_hist, b_app)
        swap = _histogram_distance(near_hist, b_app) + _histogram_distance(far_hist, a_app)
        return {"near": PLAYER_A, "far": PLAYER_B} if keep <= swap else {"near": PLAYER_B, "far": PLAYER_A}

    def update(
        self,
        frame: np.ndarray,
        near_bbox: Optional[tuple[float, float, float, float]],
        far_bbox: Optional[tuple[float, float, float, float]],
        force_reassign: bool = False,
    ) -> dict[str, str]:
        """Returns `{"near": identity, "far": identity}`, including only
        the roles that had a box THIS frame - a role with no detection
        this frame gets no entry (nothing to assign or update from) rather
        than a guessed carry-forward value.

        `force_reassign=True` re-derives the near/far -> identity mapping
        by appearance instead of trusting the previous frame's mapping -
        pass it on the frame a scene cut lands on, or after any other
        known break in tracking continuity. It has no effect before both
        identities have an appearance signature yet (nothing to match
        against - falls back to carrying forward, or bootstrapping if this
        is the first double-sighting).
        """
        near_hist = _box_histogram(frame, near_bbox, self.bins) if near_bbox is not None else None
        far_hist = _box_histogram(frame, far_bbox, self.bins) if far_bbox is not None else None

        both_appearances_known = all(v is not None for v in self._appearance.values())
        if not self._identity_for_role and near_hist is not None and far_hist is not None:
            # Bootstrap: the first frame both players are visible at once
            # fixes the (arbitrary) labels - there is no prior signature to
            # match against yet, so position is all there is to go on.
            self._identity_for_role = {"near": PLAYER_A, "far": PLAYER_B}
        elif force_reassign and near_hist is not None and far_hist is not None and both_appearances_known:
            self._identity_for_role = self._best_assignment(near_hist, far_hist)

        result = {}
        if near_hist is not None and "near" in self._identity_for_role:
            identity = self._identity_for_role["near"]
            self._remember(identity, near_hist)
            result["near"] = identity
        if far_hist is not None and "far" in self._identity_for_role:
            identity = self._identity_for_role["far"]
            self._remember(identity, far_hist)
            result["far"] = identity
        return result
