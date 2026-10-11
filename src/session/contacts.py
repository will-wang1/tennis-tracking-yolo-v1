"""Where players hit the ball, and where their shots land.

A CONTACT is a swing the ball confirmed: the ball was seen turning round
beside the player (src/analysis/swings.py ball_reversal), at the contact
frame. For each one:

  where they stood   the player's feet on the court at contact (the ball
                     itself is in the air, so the court mapping does not
                     place it; the feet are on the ground and it does)
  side               ball to the player's right or left as THEY face the
                     net - for a right-hander, forehand or backhand side.
                     Far-end players face the camera, so their image is
                     mirrored for this.

Each placed bounce is then credited to the shot that caused it: the last
contact before it, from the OTHER end, within SHOT_WINDOW_S - so every
player's shots get a depth and in/out record.

NOT CONTACT HEIGHT. It was tried - the ball against the player's body box -
and checked against the hand-labelled Dingles swings: near-end forehands,
really around the waist, read anywhere from waist to above the shoulder,
and far-end volleys overlapped serves. The ball at contact is usually a
little in front of the player, so from a fence-height camera it appears
higher (near end) or lower (far end) in the picture than it is, and the
court mapping cannot correct it because the ball is in the air. Height
needs the ball's 3D position; until then it is left out rather than shown
wrong.
"""

from collections import Counter
from typing import Optional

import numpy as np

from src.analysis.court_calibration import CourtCalibration

NET_Y, COURT_LENGTH = 23.77 / 2, 23.77
SERVICE_FROM_NET = 6.4
SHOT_WINDOW_S = 3.0  # a bounce this long after a contact can be that shot's
WHERE = ("at the net", "mid-court", "around the baseline", "deep behind the baseline")


def where_from(y_m: float) -> str:
    """Where a player stood, by distance from their own baseline."""
    from_net = abs(y_m - NET_Y)
    if from_net <= SERVICE_FROM_NET:
        return WHERE[0]
    if from_net <= NET_Y - 2.0:
        return WHERE[1]
    if from_net <= NET_Y + 1.5:
        return WHERE[2]
    return WHERE[3]


def contact_points(swings: list[dict], people_rows: list, ball_at: dict[int, list[tuple[float, float]]],
                   court: CourtCalibration, fps: float) -> list[dict]:
    """Every ball-confirmed contact, located (see module docstring)."""
    out = []
    for sw in swings:
        if "ball" not in (sw.get("confirmed_by") or []) or "contact_frame" not in sw:
            continue
        f = sw["contact_frame"]
        box = next((r[1:5] for r in (people_rows[f] if 0 <= f < len(people_rows) else []) if r[0] == sw["tracklet"]), None)
        balls = ball_at.get(f, [])
        if box is None or not balls:
            continue
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bx, by = min(balls, key=lambda b: np.hypot(b[0] - cx, b[1] - cy))
        feet = court.pixel_to_world(cx, y2)
        far = feet[1] < NET_Y
        # Side as the player faces the net: a near player is seen from behind
        # (picture right = their right); a far one faces the camera (mirrored).
        right = (bx > cx) != far
        out.append({
            "t_s": round(f / fps, 2), "frame": f, "player": sw.get("player"), "tracklet": sw["tracklet"],
            "end": "far" if far else "near", "stood_m": [round(feet[0], 2), round(feet[1], 2)],
            "stood": where_from(feet[1]),
            "side": "right" if right else "left", "ball_px": [round(bx, 1), round(by, 1)],
        })
    return out


def credit_bounces(contacts: list[dict], bounces: list[dict], fps: float) -> None:
    """Give each contact the bounce its shot made, if any (in place):
    the first placed bounce on the OTHER half after it, within SHOT_WINDOW_S,
    not already credited to a later contact."""
    ordered = sorted(contacts, key=lambda c: c["frame"])
    for i, c in enumerate(ordered):
        nxt = ordered[i + 1]["frame"] if i + 1 < len(ordered) else None
        for b in sorted(bounces, key=lambda b: b.get("bounce_frame", b["frame"])):
            bf = b.get("bounce_frame", b["frame"])
            if bf <= c["frame"] or bf - c["frame"] > SHOT_WINDOW_S * fps:
                continue
            if nxt is not None and bf > nxt and ordered[i + 1]["end"] == c["end"]:
                break  # the same end hit again first: that later shot owns what follows
            if b.get("half") != c["end"] and b.get("depth"):
                c["landed"] = {"depth": b["depth"], "bounds": b["bounds"], "position_m": b["position_m"]}
                break


def per_player(contacts: list[dict]) -> dict:
    """Each player's contacts: where they stood, their side (near end), and where
    the shots that landed went."""
    out = {}
    for player in sorted({c["player"] for c in contacts if c["player"] is not None}):
        mine = [c for c in contacts if c["player"] == player]
        n = len(mine)
        landed = [c["landed"] for c in mine if c.get("landed")]
        inside = [l for l in landed if l["bounds"] != "out"]
        share = lambda counter, keys: {k: round(counter.get(k, 0) / n, 3) for k in keys}  # noqa: E731
        out[str(player)] = {
            "contacts": n,
            "stood": share(Counter(c["stood"] for c in mine), WHERE),
            # Near end only: checked against labelled strokes there, 4/4
            # backhands left and 10/13 forehands right; at the far end it was a
            # coin toss (4/8) - the ball is tiny and the player faces the camera.
            "side_near_end": ({k: round(v / len(near), 3) for k, v in Counter(c["side"] for c in near).items()}
                              if (near := [c for c in mine if c["end"] == "near"]) else None),
            "near_end_contacts": len(near),
            "side_near_end_counts": dict(Counter(c["side"] for c in near)),
            "shots_landed": len(landed),
            "shots_in": len(inside),
            "shots_deep": sum(l["depth"] == "deep" for l in inside),
            "shots_out": sum(l["bounds"] == "out" for l in landed),
            "deep_share": round(sum(l["depth"] == "deep" for l in inside) / len(inside), 3) if inside else None,
            "out_share": round(sum(l["bounds"] == "out" for l in landed) / len(landed), 3) if landed else None,
        }
    return out


def contacts_section(swings, people_rows, ball_at, court, fps, bounces, coach: Optional[int]) -> dict:
    points = contact_points(swings, people_rows, ball_at, court, fps)
    credit_bounces(points, bounces, fps)
    players = per_player([c for c in points if c["player"] != coach])
    return {
        "points": points,
        "per_player": players,
        "coach_player": coach,
        "method": "Contacts are swings where the ball was seen turning round beside the player. Where they stood is "
                  "their feet on the court; side is the ball against their body box; each bounce is "
                  f"credited to the last contact from the other end within {SHOT_WINDOW_S:g}s before it.",
        "caveats": [
            "Only contacts where the ball was tracked through the hit are counted - far-end contacts are often missed.",
            "Contact height is not given: from one fence camera the ball's height cannot be told from its "
            "distance in front of the player.",
            "Left/right is the side as the player faces the net, from near-end contacts only (far-end readings "
            "were no better than chance); forehand or backhand depends on their hand.",
        ],
    }
