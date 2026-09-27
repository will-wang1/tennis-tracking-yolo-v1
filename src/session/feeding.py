"""Fed or live: who puts the ball in play, and how much the coach hits.

A coaching session mixes two kinds of play. In FED play the coach starts
each ball - from a basket, by racket or by hand - and the player plays one
or a few shots. In LIVE play the players start the points themselves
(serves, drop feeds) and rally. Neither is better; the mix is what a
coach wants to see ("I fed for 20 minutes and they barely rallied").

HOW. Everything comes from the swings (src/analysis/swings.py), each one a
person moving their arm fast with the ball close:
  - A RALLY is a run of swings with no gap over RALLY_GAP_S between them.
    Not split where the ball tracker lost the ball: on Dingles that cut 13
    of 23 rallies down to a single shot while play went on. With several
    balls in play at once (two groups rallying side by side) rallies from
    different groups can merge, so rally counts in a multi-ball drill are
    approximate; the fed/live split and the coach's share hold up better.
  - A rally is FED when its first swing is the coach's, LIVE when it is a
    player's.
  - The coach's share of all swings says how much of the hitting the
    coach did, whether feeding or rallying with players.
Rally length is counted in swings, so it is a count of shots.

NOT VALIDATED ON FEEDING. The only footage so far (dingles_serve_volley)
is a serve-and-volley drill in which the players start every point; the
coach barely hits. It confirms the live side. Whether a basket or hand
feed registers as a swing - a gentle underarm toss may be too slow - is
untested until there is footage with feeding in it, and the report says
so.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Swings further apart than this are separate rallies. On Dingles 3s merged
# the two serve-and-volley groups into 8 rallies averaging 9 shots; 2s gave
# 16. A few swings are missed (a live rally's shots are ~1.5s apart), so
# the gap must allow for one.
RALLY_GAP_S = 2.5


@dataclass
class Rally:
    start: int  # frame of the first swing
    end: int  # frame of the last swing
    starter: Optional[int]  # player number of the first swing
    shots: int
    coach_shots: int

    def fed(self, coach: Optional[int]) -> bool:
        return coach is not None and self.starter == coach


def rallies(swings: list[dict], fps: float, coach: Optional[int], gap_s: float = RALLY_GAP_S) -> list[Rally]:
    """Swings (dicts with frame and player) grouped into rallies."""
    out: list[Rally] = []
    current: list[dict] = []

    def close():
        if current:
            out.append(Rally(
                start=current[0]["frame"], end=current[-1]["frame"], starter=current[0]["player"],
                shots=len(current), coach_shots=sum(1 for s in current if coach is not None and s["player"] == coach),
            ))

    for s in sorted(swings, key=lambda s: s["frame"]):
        if current and s["frame"] - current[-1]["frame"] > gap_s * fps:
            close()
            current = []
        current.append(s)
    close()
    return out


def feeding_summary(swings: list[dict], fps: float, coach: Optional[int]) -> dict:
    """The fed/live section of the session report."""
    rs = rallies(swings, fps, coach)
    fed = [r for r in rs if r.fed(coach)]
    live = [r for r in rs if not r.fed(coach)]
    all_shots = sum(r.shots for r in rs)
    coach_shots = sum(r.coach_shots for r in rs)

    def mean_shots(group):
        return round(float(np.mean([r.shots for r in group])), 1) if group else None

    return {
        "coach_player": coach,
        "rallies": len(rs),
        "fed_rallies": len(fed),
        "live_rallies": len(live),
        "fed_share": round(len(fed) / len(rs), 3) if rs else None,
        "shots": all_shots,
        "coach_shots": coach_shots,
        "coach_shot_share": round(coach_shots / all_shots, 3) if all_shots else None,
        "shots_per_rally": {"fed": mean_shots(fed), "live": mean_shots(live), "all": mean_shots(rs)},
        "longest_rally_shots": max((r.shots for r in rs), default=0),
        "rally_list": [
            {"start_s": round(r.start / fps, 2), "end_s": round(r.end / fps, 2), "started_by": r.starter,
             "fed": r.fed(coach), "shots": r.shots}
            for r in rs
        ],
        "method": f"Rallies are runs of detected swings under {RALLY_GAP_S:g}s apart; a rally is fed when "
                  "the coach's swing starts it. Shots are swings (arm moving fast with the ball close).",
        "caveats": [
            "Checked only on a drill where players start every point; recognising a coach's feed "
            "(especially a gentle hand feed) is not yet tested on footage with feeding.",
            "With several balls in play at once, rallies of different groups can merge: rally counts "
            "and lengths are approximate.",
        ] + ([] if coach is not None else ["No coach identified, so every rally counts as live."]),
    }
