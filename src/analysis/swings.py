"""Swings found from the players' bodies, not the ball.

The ball-based hit detector finds a fraction of the shots in a squad drill:
the ball is small, often hidden by a body at contact, and lost at the far
end. A swing is visible on the player whatever the ball tracker did. The
idea is CourtCheck's (github.com/AggieSportsAnalytics/CourtCheck,
backend/vision/swing_detector.py): a swing is a peak in wrist speed, kept
if the ball is near the player around it. Two changes, both measured on
dingles_serve_volley against hits checked by eye:

1. SCALE. CourtCheck's threshold is 15 px/frame, for broadcast players
   ~300px tall. Our far players are ~80px tall. Speeds here are in
   PLAYER HEIGHTS (the person box), so near and far are judged alike.

2. STEADINESS. Single-frame wrist speed on an 80px player is mostly
   keypoint jitter: 5% of all frames beat 0.155 heights/frame while real
   hits ranged 0.09-0.48, and pre-serve ball bounces reached 0.37 - the
   two overlapped badly. What separates them is the wrist travelling a
   long way relative to the SHOULDERS (so running does not count) over a
   few frames, after a 5-frame median on each wrist coordinate: measured
   as the displacement from f-SPAN to f+SPAN.

MEASURED on dingles_serve_volley (108s, 7 people), against hits and
non-hits checked by eye - one clip, so provisional: at THRESHOLD 0.30 and
MIN_GAP 25, 90 swings; 15 of 17 known hits found, 1 of 16 known non-hits
(a pre-serve ball bounce). Looking at every detection, roughly four in five
of the players' swings are real strokes; the rest are ready-position
fidgets and walking to the ball cart. 11 of the 90 are the COACH moving
the racket while feeding and talking, which coach identification removes.
The ball-based detector found 15 confirmed hits in the same clip.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

SPAN = 6
MEDIAN = 5
# Wrist travel over 2*SPAN frames, in player heights. 0.33 found 13/17
# known hits and 72 swings; 0.30 found 15/17 and 90, about half of the 18
# extra being real far-end swings and net volleys, a third the coach.
THRESHOLD = 0.30
# One player cannot play two shots within a second. At CourtCheck's 15
# frames, 23 pairs of one player's "swings" on Dingles fell 15-24 frames
# apart: a backswing and its follow-through counted twice.
MIN_GAP = 25
MIN_WRIST_CONF = 0.3
# The ball must come within BALL_NEAR_HEIGHTS player heights of the
# player's box within BALL_WINDOW frames of the swing, and must be SEEN.
# The first version (2 heights, 12 frames, unseen allowed) let every
# swing through. Measured on the Dingles swings sorted by eye: at 1 height
# and 8 frames, 92% of real swings pass against 3 of 7 false ones and 5 of
# 11 of the coach's; half a height kept only 86% of the real ones.
BALL_NEAR_HEIGHTS = 1.0
BALL_WINDOW = 8
WRISTS = (9, 10)
SHOULDERS = (5, 6)


@dataclass
class Swing:
    frame: int
    tracklet: int
    strength: float  # peak wrist travel, player heights
    ball_near: Optional[bool]  # None: the ball was not seen around it


def _median(x: np.ndarray, size: int) -> np.ndarray:
    """Median filter over the finite values of x, in place of gaps kept."""
    from scipy.ndimage import median_filter

    out = x.copy()
    ok = np.isfinite(x)
    if ok.sum() > size:
        out[ok] = median_filter(x[ok], size, mode="nearest")
    return out


# Poses may be sampled every other frame to save time (track_poses.py
# --every 2); gaps up to this long are filled by straight-line
# interpolation before smoothing, so the signal is computed as if every
# frame had a pose.
MAX_FILL = 2


def _fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    out = x.copy()
    ok = np.flatnonzero(np.isfinite(x))
    for a, b in zip(ok, ok[1:]):
        if 1 < b - a <= max_gap + 1:
            out[a + 1:b] = np.interp(np.arange(a + 1, b), [a, b], [x[a], x[b]])
    return out


def wrist_travel(
    poses: dict[int, tuple[np.ndarray, np.ndarray]], heights: dict[int, float], n: int,
    span: int = SPAN, median: int = MEDIAN,
) -> np.ndarray:
    """Per frame: how far either wrist moved relative to the shoulders
    between f-span and f+span, in player heights. NaN where unknown.
    `poses` is frame -> (keypoints (17,2), confidence (17,)) for ONE person."""
    rel = np.full((n, 2, 2), np.nan)
    for f, (xy, conf) in poses.items():
        if not 0 <= f < n or conf[SHOULDERS[0]] < MIN_WRIST_CONF or conf[SHOULDERS[1]] < MIN_WRIST_CONF:
            continue
        shoulders = (xy[SHOULDERS[0]] + xy[SHOULDERS[1]]) / 2.0
        for i, w in enumerate(WRISTS):
            if conf[w] >= MIN_WRIST_CONF:
                rel[f, i] = xy[w] - shoulders
    for i in range(2):
        for d in range(2):
            rel[:, i, d] = _median(_fill_short_gaps(rel[:, i, d], MAX_FILL), median)
    out = np.full(n, np.nan)
    for f in range(span, n - span):
        h = heights.get(f)
        if not h:
            continue
        moved = np.linalg.norm(rel[f + span] - rel[f - span], axis=1) / h
        if np.isfinite(moved).any():
            out[f] = np.nanmax(moved)
    return out


def peaks(signal: np.ndarray, threshold: float = THRESHOLD, min_gap: int = MIN_GAP) -> list[int]:
    """Local maxima over `threshold`, strongest first when two are closer
    than `min_gap` (CourtCheck keeps the first; the strongest is the swing,
    the first is often its backswing)."""
    s = np.where(np.isfinite(signal), signal, -np.inf)
    candidates = [
        i for i in range(1, len(s) - 1)
        if s[i] >= threshold and s[i] >= s[i - 1] and s[i] >= s[i + 1]
    ]
    kept: list[int] = []
    for i in sorted(candidates, key=lambda i: -s[i]):
        if all(abs(i - k) >= min_gap for k in kept):
            kept.append(i)
    return sorted(kept)


def ball_near(
    frame: int, box_at: dict[int, tuple], ball_at: dict[int, list[tuple[float, float]]],
    heights: float = BALL_NEAR_HEIGHTS, window: int = BALL_WINDOW,
) -> Optional[bool]:
    """Did the ball come within `heights` player heights of the player's box
    within `window` frames of the swing? None if no ball was seen at all
    in that window (a tracking gap is not evidence against a swing)."""
    seen = False
    for f in range(frame - window, frame + window + 1):
        if f not in ball_at or f not in box_at:
            continue
        x1, y1, x2, y2 = box_at[f][:4]
        for x, y in ball_at[f]:
            seen = True
            dx, dy = max(x1 - x, 0.0, x - x2), max(y1 - y, 0.0, y - y2)
            if np.hypot(dx, dy) <= heights * max(y2 - y1, 1.0):
                return True
    return False if seen else None


def detect_swings(
    people_rows: list[list[tuple]],
    poses_by_frame: list[dict[int, tuple[np.ndarray, np.ndarray]]],
    ball_at: dict[int, list[tuple[float, float]]],
    skip_tracklets: set[int] = frozenset(),
    skip_frames: set[int] = frozenset(),
    threshold: float = THRESHOLD,
    min_gap: int = MIN_GAP,
) -> list[Swing]:
    """Every swing of every tracked person that the ball came close to.
    A swing with no ball near - a shadow swing, waving, the coach
    gesturing - is dropped, and so is one where the ball was not seen at
    all: a hit needs a ball, and a ball-tracking gap costs a few real
    swings at the far end rather than letting every gesture through."""
    n = len(people_rows)
    boxes: dict[int, dict[int, tuple]] = {}
    for f, row in enumerate(people_rows):
        for t, x1, y1, x2, y2, _c in row:
            if t is not None and t not in skip_tracklets:
                boxes.setdefault(int(t), {})[f] = (x1, y1, x2, y2)
    swings = []
    for t, box_at in boxes.items():
        person = {f: poses_by_frame[f][t] for f in box_at if f < len(poses_by_frame) and t in poses_by_frame[f]}
        heights = {f: b[3] - b[1] for f, b in box_at.items()}
        signal = wrist_travel(person, heights, n)
        for f in peaks(signal, threshold, min_gap):
            if f in skip_frames:
                continue
            near = ball_near(f, box_at, ball_at)
            if not near:
                continue
            swings.append(Swing(frame=f, tracklet=t, strength=float(signal[f]), ball_near=near))
    return sorted(swings, key=lambda s: s.frame)


# ---- contact: the ball turning round beside the player -------------------
#
# A hit sends the ball back the way it came. Beside a near-end player the
# ball comes DOWN the picture and leaves UP it; beside a far-end player the
# reverse. Where that turn is seen, its frame is the CONTACT frame - on the
# hand-labelled Dingles swings the true contact sat anywhere from 8 frames
# before to 10 after the wrist-speed peak, so the peak alone is a rough
# time. Measured there as a filter it did not pay: near the camera it
# confirmed 17 of 21 real swings (and 1 of 2 false ones), but at the far end,
# where the ball is 2-3px and often lost, only 18 of 34 real ones while still
# passing 6 of 16 false ones - and serves (nothing coming in) often fail it.
# So it CONFIRMS a swing and times it; it never drops one.

REVERSAL_BEFORE = 10  # frames before the swing peak to look at the ball
REVERSAL_AFTER = 12
REVERSAL_REACH = 2.0  # ball within this many player heights of the box
REVERSAL_SPAN = 6  # frames either side of the turn to measure the ball's direction over
MIN_SPEED_PX = 0.5  # vertical pixels a frame, each way, to count as moving


def ball_reversal(
    frame: int, end: str, box_at: dict[int, tuple], ball_at: dict[int, list[tuple[float, float]]],
) -> Optional[int]:
    """The frame the ball turned round beside this player, or None."""
    if frame not in box_at:
        return None
    b0 = box_at[frame]
    height = max(b0[3] - b0[1], 1.0)
    points: dict[int, tuple[float, float]] = {}
    for g in range(frame - REVERSAL_BEFORE, frame + REVERSAL_AFTER + 1):
        x1, y1, x2, y2 = box_at.get(g, b0)[:4]
        best = None
        for x, y in ball_at.get(g, []):
            d = np.hypot(max(x1 - x, 0.0, x - x2), max(y1 - y, 0.0, y - y2)) / height
            if d <= REVERSAL_REACH and (best is None or d < best[0]):
                best = (d, (x, y))
        if best:
            points[g] = best[1]
    frames = sorted(points)
    if len(frames) < 5:
        return None
    toward = 1 if end == "near" else -1  # image-y direction of a ball coming at this player
    turns = []
    for c in frames:
        pre = [g for g in frames if c - REVERSAL_SPAN <= g < c]
        post = [g for g in frames if c < g <= c + REVERSAL_SPAN]
        if len(pre) < 2 or len(post) < 2:
            continue
        v_in = (points[pre[-1]][1] - points[pre[0]][1]) / max(1, pre[-1] - pre[0])
        v_out = (points[post[-1]][1] - points[post[0]][1]) / max(1, post[-1] - post[0])
        if v_in * toward > MIN_SPEED_PX and v_out * toward < -MIN_SPEED_PX:
            turns.append(c)
    return min(turns, key=lambda c: abs(c - frame)) if turns else None
