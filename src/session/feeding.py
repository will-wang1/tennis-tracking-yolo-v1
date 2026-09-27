"""Fed or live: who puts the ball in play, and how much the coach hits.

A coaching session mixes two kinds of play. In FED play the coach starts
each ball - from a basket, by racket or by hand - and the player plays one
or a few shots. In LIVE play the players start the points themselves
(serves, drop feeds) and rally. Neither is better; the mix is what a
coach wants to see ("I fed for 20 minutes and they barely rallied").

HOW. Everything comes from the swings (src/analysis/swings.py), each one a
person moving their arm fast with the ball close:
  - A RALLY is a chain of REPLIES: a swing's reply is the next swing from
    the OTHER end within RALLY_GAP_S. Two pairs often rally side by side
    (two players a side is the usual squad setup), so when two swings could
    be the reply, the one by the hitter's usual partner - whoever they
    exchanged with most within PARTNER_WINDOW_S - wins. A swing nobody
    replies to ends its rally. Grouping by time alone merged side-by-side
    rallies into one; following the ball does not work either, because the
    ball tracker hops between two balls in play. Not split where the ball
    was lost: on Dingles that cut 13 of 23 rallies down to a single shot.
    Swings without a court end (an older swings file) fall back to time
    alone.
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


MIN_REPLY_S = 0.4  # a reply cannot come sooner than the ball takes to cross
PARTNER_WINDOW_S = 30.0


def rallies(swings: list[dict], fps: float, coach: Optional[int], gap_s: float = RALLY_GAP_S) -> list[Rally]:
    """Swings (dicts with frame, player and end) grouped into rallies."""
    ordered = sorted(swings, key=lambda s: s["frame"])
    if any("end" not in s for s in ordered):
        return _rallies_by_time(ordered, fps, coach, gap_s)
    lo, hi = MIN_REPLY_S * fps, gap_s * fps

    def replies_to(i):
        a = ordered[i]
        return [j for j in range(i + 1, len(ordered))
                if lo <= ordered[j]["frame"] - a["frame"] <= hi and ordered[j]["end"] != a["end"]]

    # Who exchanges with whom: every possible reply, by time.
    exchanges = [(ordered[i]["frame"], ordered[i]["player"], ordered[j]["player"])
                 for i in range(len(ordered)) for j in replies_to(i)]

    def partner_score(i, j):
        f, p, q = ordered[i]["frame"], ordered[i]["player"], ordered[j]["player"]
        if p is None or q is None:
            return 0
        return sum(1 for g, a, b in exchanges if abs(g - f) <= PARTNER_WINDOW_S * fps and {a, b} == {p, q})

    reply_of: dict[int, int] = {}
    claimed: set[int] = set()
    for i in range(len(ordered)):
        options = [j for j in replies_to(i) if j not in claimed]
        if options:
            j = max(options, key=lambda j: (partner_score(i, j), -ordered[j]["frame"]))
            reply_of[i] = j
            claimed.add(j)

    out = []
    for i in range(len(ordered)):
        if i in claimed:
            continue  # someone's reply: inside a rally, not its start
        chain = [i]
        while chain[-1] in reply_of:
            chain.append(reply_of[chain[-1]])
        members = [ordered[k] for k in chain]
        out.append(Rally(
            start=members[0]["frame"], end=members[-1]["frame"], starter=members[0]["player"], shots=len(members),
            coach_shots=sum(1 for m in members if coach is not None and m["player"] == coach),
        ))
    return sorted(out, key=lambda r: r.start)


def _rallies_by_time(ordered: list[dict], fps: float, coach: Optional[int], gap_s: float) -> list[Rally]:
    out: list[Rally] = []
    current: list[dict] = []

    def close():
        if current:
            out.append(Rally(
                start=current[0]["frame"], end=current[-1]["frame"], starter=current[0]["player"],
                shots=len(current), coach_shots=sum(1 for s in current if coach is not None and s["player"] == coach),
            ))

    for s in ordered:
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
        "method": f"A rally is a chain of replies - the next swing from the other end within {RALLY_GAP_S:g}s, "
                  "preferring the hitter's usual partner when two pairs rally side by side. A rally is fed when "
                  "the coach's swing starts it. Shots are swings (arm moving fast with the ball close).",
        "caveats": [
            "Checked only on a drill where players start every point; recognising a coach's feed "
            "(especially a gentle hand feed) is not yet tested on footage with feeding.",
            "A missed swing (about one in six) ends a rally early, so rallies read a little shorter than "
            "they were.",
        ] + ([] if coach is not None else ["No coach identified, so every rally counts as live."]),
    }
