"""Bounces from E2E-Spot (scripts/e2e_spot_events.py), placed on the court.

E2E-Spot says WHEN a bounce happened and at WHICH END, from the whole
picture - watching both side by side on dingles_serve_volley, its bounces
were better than our own trajectory-based ones. It does not say where. The
place comes from the tracked ball: at a bounce the ball is at its lowest
point in the picture (it comes down, touches, goes back up), so the ball
detection that bottoms out closest to the call's frame, in the called half,
is the bounce, mapped onto the court through the calibration. With no ball
seen there the bounce still counts, without a position.

Positions are good at the near end and rough at the far end (a pixel there
is ~10x more court than near the camera), so in/out and depth are most
trustworthy for near-end bounces.
"""

from typing import Iterable, Optional

from src.analysis.court_calibration import CourtCalibration
from src.analysis.court_zones import classify_landing_zone

LOOK = 3  # frames either side of the call to find the ball's low point
PATH = 8  # frames either side to judge "lowest"


def _ball_at_bounce(frame: int, far: bool, tracks, court: CourtCalibration) -> Optional[tuple[float, float, int]]:
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
            best = (key, (path[low].x, path[low].y, low))
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
                         side=zone.side, depth=zone.depth, bounds=zone.bounds, ball_px=[round(ball[0], 1), round(ball[1], 1)])
        else:
            entry.update(position_m=None, zone=None, side=None, depth=None, bounds=None)
        out.append(entry)
    return out
