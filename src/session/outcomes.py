"""How each rally ended, and who made the errors.

A rally (src/session/feeding.py) ends with its last shot. What the ball did
after that shot - the first bounce E2E-Spot called after the contact -
says how it ended:

  out           the shot bounced out                    error by the hitter
  net           it bounced on the hitter's own half:    error by the hitter
                it never cleared the net
  not returned  it bounced in, and no reply was seen    winner (or a ball the
                                                         other player did not reach)
  unclear       no bounce seen after it, or in/out
                not known (the bounce was not placed)

An error is tagged with its BALL NUMBER in the rally: 1 = the serve or feed
that started it, 2 = the return, and so on - so "errors on the first two
balls" is what consistency coaching is about.

Caveats (stated in the report): the swing detector misses about one swing
in six, so a missed reply turns a rally into "not returned" and makes its
last shot look like a winner; with two balls in play, a bounce from the
other pair's ball can be taken for this one - which is why a "net" bounce
must land near the hitter, and an "out" or "in" one is the bounce already
credited to this shot (src/session/contacts.py) rather than just the next
one anywhere.
"""

from collections import Counter
from typing import Optional

import numpy as np

SHOT_WINDOW_S = 3.0
NET_NEAR_HITTER_M = 8.0  # a ball into the net bounces on the hitter's side, near them
OUTCOMES = ("out", "net", "not returned", "unclear")


def rally_outcomes(rally_list: list[dict], swings: list[dict], contacts: list[dict], bounce_calls: list[dict],
                   fps: float) -> list[dict]:
    from src.session.contacts import credit_bounces

    contact_of = {c["frame"]: c for c in contacts}  # ball-confirmed contacts, by contact frame

    def hit_frame(s):
        return s.get("contact_frame", s["frame"]) if "ball" in (s.get("confirmed_by") or []) else s["frame"]

    # Every swing's shot gets the bounce it made - not only the ones whose
    # contact the ball confirmed, which at the far end are few: each placed
    # bounce on the other half goes to the last swing from the other end
    # before it (the same crediting as the contact panel).
    shots = [{"frame": hit_frame(s), "end": s.get("end"), "swing": s["frame"]} for s in swings if s.get("end")]
    credit_bounces(shots, [b for b in bounce_calls if b.get("position_m")], fps)
    shot_of = {sh["swing"]: sh for sh in shots}
    by_frame = {s["frame"]: s for s in swings}

    out = []
    for r in rally_list:
        frames = r.get("swing_frames") or []
        if not frames or frames[-1] not in by_frame:
            continue
        last = by_frame[frames[-1]]
        hit = hit_frame(last)
        end, player = last.get("end"), last.get("player")
        contact = contact_of.get(hit)
        shot = shot_of.get(last["frame"], {})
        after = sorted((b for b in bounce_calls if hit < b.get("bounce_frame", b["frame"]) <= hit + SHOT_WINDOW_S * fps),
                       key=lambda b: b.get("bounce_frame", b["frame"]))
        own = [b for b in after if b.get("half") == end]
        outcome, where = "unclear", None
        if shot.get("landed"):
            landed = shot["landed"]
            outcome = "out" if landed["bounds"] == "out" else "not returned"
            where = landed["position_m"]
        elif own and after[0] is own[0]:
            # The first bounce after the shot was on the hitter's own side: it did
            # not clear the net - if it was near them (another pair's ball lands
            # anywhere); without a position, only when it came straight after.
            b = own[0]
            stood = contact["stood_m"] if contact else None
            if b.get("position_m") and stood:
                if np.hypot(*(np.array(b["position_m"]) - stood)) <= NET_NEAR_HITTER_M:
                    outcome, where = "net", b["position_m"]
            elif b.get("bounce_frame", b["frame"]) - hit <= 1.5 * fps:
                outcome, where = "net", b.get("position_m")
        out.append({
            "start_s": r["start_s"], "end_s": round(hit / fps, 2), "frame": hit, "shots": len(frames),
            "fed": r.get("fed", False), "last_hitter": player, "last_hitter_end": end, "outcome": outcome,
            "error_by": player if outcome in ("out", "net") else None,
            "winner_by": player if outcome == "not returned" else None,
            "ball_number": len(frames), "where_m": where,
        })
    return out


def per_player_errors(outcomes: list[dict], swings: list[dict], coach: Optional[int]) -> dict:
    shots = Counter(s.get("player") for s in swings if s.get("player") is not None)
    out = {}
    for player in sorted(p for p in shots if p != coach):
        mine = [o for o in outcomes if o["last_hitter"] == player]
        errors = [o for o in mine if o["error_by"] == player]
        out[str(player)] = {
            "shots": shots[player],
            "errors": len(errors),
            "errors_out": sum(o["outcome"] == "out" for o in errors),
            "errors_net": sum(o["outcome"] == "net" for o in errors),
            "errors_first_two_balls": sum(o["ball_number"] <= 2 for o in errors),
            "not_returned": sum(o["outcome"] == "not returned" for o in mine),
            "error_rate": round(len(errors) / shots[player], 3) if shots[player] else None,
        }
    return out


def outcomes_section(rally_list, swings, contacts, bounce_calls, fps, coach) -> dict:
    outs = rally_outcomes(rally_list, swings, contacts, bounce_calls, fps)
    counts = Counter(o["outcome"] for o in outs)
    return {
        "rallies": outs,
        "endings": {k: counts.get(k, 0) for k in OUTCOMES},
        "per_player": per_player_errors(outs, swings, coach),
        "method": "Each rally's last shot is followed to the first bounce after it: out or into the net (a bounce on "
                  "the hitter's own side, near them) is an error by the hitter; in with no reply seen is 'not "
                  "returned'. Ball number 1 is the serve or feed that started the rally.",
        "caveats": [
            "A missed reply (the swing detector misses about one in six) makes a rally look 'not returned'.",
            "'Not returned' covers winners and balls the other player could not reach or chose to leave.",
            "Errors are only counted where the bounce after the last shot was seen; many rallies end 'unclear'.",
        ],
    }
