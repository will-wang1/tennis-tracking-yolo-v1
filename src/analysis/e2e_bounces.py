"""Bounces from E2E-Spot (scripts/e2e_spot_events.py), placed on the court.

E2E-Spot says WHEN a bounce happened and at WHICH END, from the whole
picture - watching both side by side on dingles_serve_volley, its bounces
were better than our own trajectory-based ones. It does not say where. The
place comes from the tracked ball: at a bounce the ball is at its lowest
point in the picture (it comes down, touches, goes back up), so the ball
detection that bottoms out closest to the call's frame, in the called half,
is the bounce, mapped onto the court through the calibration. With no ball
seen there the bounce still counts, without a position.

The low point is only the nearest FRAME, though: at 25fps a ball at 20 m/s
moves ~0.8m between frames, and near the camera a ball that has just
bounced is still coming towards it, so it keeps moving DOWN the picture for
a frame or two while rising - the lowest point lands late. So where the
ball was tracked either side, the path just before and just after the
bounce are each fitted and the
bounce is put where the two meet, between frames. Each bounce says which
way it was placed.

Positions are good at the near end and rough at the far end (a pixel there
is ~10x more court than near the camera), so in/out and depth are most
trustworthy for near-end bounces.
"""

from typing import Iterable, Optional

import numpy as np

from src.analysis.court_calibration import CourtCalibration
from src.analysis.court_zones import classify_landing_zone

LOOK = 3  # frames either side of the call to find the ball's low point
BREAK_WINDOW = 2  # the break is looked for this many frames either side of E2E-Spot's call
PATH = 14  # frames either side of the call to take the ball's path from


FIT_SIDE = 6  # frames of path each side of the bounce to fit
# Fit tolerances scale with how far the ball moves a frame: near the camera
# it moves 40-50px a frame and perspective curves its path in the picture,
# so a straight fit is 4-8px off on Dingles - a fixed 3px rejected nearly
# every near-end bounce. The floors are for slow, far balls.
MAX_FIT_RMS_PX = 3.0
FIT_RMS_PER_STEP = 0.15
MEET_PX = 3.0
MEET_PER_STEP = 0.1


def _fit_side(t: np.ndarray, x: np.ndarray, y: np.ndarray, degree: int = 1):
    px = np.polyfit(t, x, 1)
    py = np.polyfit(t, y, degree if len(t) > degree + 1 else 1)
    rms = float(np.sqrt(np.mean((np.polyval(px, t) - x) ** 2 + (np.polyval(py, t) - y) ** 2)))
    return px, py, rms


def _meet(pxa, pya, pxb, pyb, a: int, b: int) -> tuple[float, float]:
    """(time, gap): where in [a-0.5, b+0.5] the two fitted paths come
    closest; the gap is infinite if that is at the edge (they do not meet)."""
    ts = np.linspace(a - 0.5, b + 0.5, 50 * (b - a + 1) + 1)
    gap = np.hypot(np.polyval(pxa, ts) - np.polyval(pxb, ts), np.polyval(pya, ts) - np.polyval(pyb, ts))
    i = int(np.argmin(gap))
    return float(ts[i]), float(gap[i]) if 0 < i < len(ts) - 1 else float("inf")


MAX_GAP = 10  # frames the ball may be lost across the bounce and still be bridged


def bounce_between_frames(path: dict[int, tuple[float, float]], near_frame: int) -> Optional[tuple[float, float, float, str]]:
    """(x, y, time in frames, how) for the bounce, or None.

    Near the camera the ball tracker usually LOSES the ball around the
    bounce - fast, blurred, beside a player - for 5-8 frames on Dingles, and
    often finds it again only after the player has hit it back. So the path
    is split at a gap (or between two frames) near E2E-Spot's call, and:
      - if the path after looks like the same ball bounced - still moving
        the same way across, vertically reversed - the bounce is where the
        paths before and after meet ("path fit");
      - if it has turned back across the court (hit, not bounced), only the
        path before is used, carried forward to E2E-Spot's bounce time
        ("incoming path at the call").
    Measured on synthetic bounces at every tenth of a frame: the path fit
    is ~1-4px off where the old lowest-point method was ~20-35px."""
    frames = sorted(path)
    if len(frames) < 3:
        return None
    steps = [np.hypot(path[b][0] - path[a][0], path[b][1] - path[a][1]) for a, b in zip(frames, frames[1:]) if b - a == 1]
    step = float(np.median(steps)) if steps else 0.0

    rms_limit = max(MAX_FIT_RMS_PX, FIT_RMS_PER_STEP * step)
    meet_limit = max(MEET_PX, MEET_PER_STEP * step)

    def fit(side, degree):
        t = np.array(side, float)
        return _fit_side(t, np.array([path[g][0] for g in side]), np.array([path[g][1] for g in side]), degree)

    best = None
    for a, b in zip(frames, frames[1:]):
        if b - a > MAX_GAP or not (near_frame - BREAK_WINDOW - (b - a) <= a and b <= near_frame + BREAK_WINDOW + (b - a)):
            continue
        pre = [g for g in frames if a - FIT_SIDE < g <= a]
        post = [g for g in frames if b <= g < b + FIT_SIDE]
        if len(pre) < 3:
            continue
        (pxa, pya, ra) = fit(pre, 1)
        if len(post) >= 3:
            (pxb, pyb, rb) = fit(post, 1)
            same_way = np.sign(pxa[0]) == np.sign(pxb[0]) or min(abs(pxa[0]), abs(pxb[0])) < 0.5
            reversed_v = np.sign(pya[0]) != np.sign(pyb[0])
            t, gap = _meet(pxa, pya, pxb, pyb, a, b)
            if same_way and reversed_v and gap <= meet_limit + 0.5 * (b - a):
                score = ra + rb
                if best is None or score < best[0]:
                    best = (score, "path fit", a, b, pre, post)
                continue
        # Not a clean bounce after the gap: the incoming path alone, if the call falls in the gap.
        if a <= near_frame <= b + 1 and ra <= rms_limit and best is None:
            best = (10 * rms_limit, "incoming path at the call", a, b, pre, None)
    if best is None or best[0] > 2 * rms_limit and best[1] == "path fit":
        return None
    _, how, a, b, pre, post = best
    if how == "path fit":
        sides = [fit(pre, 2), fit(post, 2)]
        t_star, gap = _meet(sides[0][0], sides[0][1], sides[1][0], sides[1][1], a, b)
        if not np.isfinite(gap):
            sides = [fit(pre, 1), fit(post, 1)]
            t_star, _ = _meet(sides[0][0], sides[0][1], sides[1][0], sides[1][1], a, b)
        (pxa, pya, _), (pxb, pyb, _) = sides
        x = (np.polyval(pxa, t_star) + np.polyval(pxb, t_star)) / 2
        y = (np.polyval(pya, t_star) + np.polyval(pyb, t_star)) / 2
    else:
        pxa, pya, _ = fit(pre, 1)
        t_star = float(near_frame)
        x, y = np.polyval(pxa, t_star), np.polyval(pya, t_star)
    # Guard: no further from the last and first seen ball than the frames
    # in between allow (plus one frame's travel).
    reach = (min(abs(t_star - a), abs(t_star - b)) + 1.2) * step + 2.0
    if min(np.hypot(path[g][0] - x, path[g][1] - y) for g in (a, b)) > reach:
        return None
    return float(x), float(y), float(t_star), how


def _ball_at_bounce(frame: int, far: bool, tracks, court: CourtCalibration) -> Optional[tuple[float, float, float, str]]:
    best = None
    for t in tracks:
        path = {g: t.detections[g] for g in range(frame - PATH, frame + PATH + 1) if g in t.detections}
        near_call = [g for g in path if abs(g - frame) <= LOOK]
        if not near_call:
            continue
        low = max(near_call, key=lambda g: path[g].y)  # lowest point in the picture near the call
        _, wy = court.pixel_to_world(path[low].x, path[low].y)
        if (wy < 23.77 / 2) != far:
            continue  # the other half: a different ball
        is_low = all(path[g].y <= path[low].y for g in path)
        key = (is_low, -abs(low - frame))
        if best is None or key > best[0]:
            # Searched around E2E-Spot's own call: the low point can land late.
            fitted = bounce_between_frames({g: (d.x, d.y) for g, d in path.items()}, frame)
            place = fitted if fitted else (path[low].x, path[low].y, float(low), "lowest point")
            best = (key, place)
    return best[1] if best else None


def e2e_bounces(events: Iterable[dict], tracks, court: CourtCalibration, fps: float,
                skip_frames: set[int] = frozenset()) -> list[dict]:
    """Bounce entries in the session report's format, plus source and
    whether the ball was seen; entries without a seen ball have no position."""
    out = []
    for e in events:
        if not e["label"].endswith("_bounce") or e["frame"] in skip_frames:
            continue
        far = e["label"].startswith("far")
        entry = {"t_s": round(e["frame"] / fps, 2), "frame": e["frame"], "source": "e2e",
                 "score": e.get("score"), "half": "far" if far else "near"}
        ball = _ball_at_bounce(e["frame"], far, tracks, court)
        if ball is not None:
            wx, wy = court.pixel_to_world(ball[0], ball[1])
            zone = classify_landing_zone(wx, wy)
            entry.update(position_m=[round(float(wx), 2), round(float(wy), 2)], zone=zone.label(),
                         side=zone.side, depth=zone.depth, bounds=zone.bounds, ball_px=[round(ball[0], 1), round(ball[1], 1)],
                         bounce_frame=round(ball[2], 2), placed_by=ball[3])
        else:
            entry.update(position_m=None, zone=None, side=None, depth=None, bounds=None)
        out.append(entry)
    return out
