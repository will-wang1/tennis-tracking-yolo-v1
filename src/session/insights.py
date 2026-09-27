"""Turn a session report into what a coach reads: what went well, what to
look at, and one line per player - each finding a plain sentence with the
number that backs it, never a bare number.

Numbers only mean something against a reference, and there is no published
standard for, say, "good ball-in-play time" that this can honestly cite. So
two kinds of finding, and the report says which is which:

  against a TARGET   the session measured against a stated default (Targets
                     below) that a coach can change for their own programme;
                     the page shows every target used
  within the GROUP   a player compared with the rest of the group - "waited
                     4x as long as everyone else" needs no outside standard,
                     and is often the most useful thing on the page

The coach is found first and left out of every group comparison: a coach
standing at the sideline feeding balls would otherwise be "the least active
player" in every session. LIKELY COACH = the person who is BOTH clearly the
least mobile (time moving at most COACH_MOBILITY_RATIO of the next lowest)
AND the most often off court. On Dingles that is player 1 on both counts
(moving 38% vs 58-70% for the others; off court 74%, the highest). It is a
guess and is labelled as one; a coach can override it.

Deliberately NOT in the summary: hits per player (0-3 each on Dingles - too
few confirmed contacts to compare anyone) and impacts per minute (a count
of ball direction changes, which means nothing to a coach). Both stay in
the report's details.
"""

from dataclasses import asdict, dataclass
from statistics import median
from typing import Optional

import numpy as np

COACH_MOBILITY_RATIO = 0.75


@dataclass
class Targets:
    """Defaults a coach can override (scripts/session_report.py --targets).
    Stated on the page as defaults, not as standards."""

    ball_in_play_good: float = 0.70  # at or above: very little dead time
    ball_in_play_poor: float = 0.50  # below: a lot of dead time
    wait_good: float = 0.10  # every player below this share of drill time waiting: nobody waited long
    wait_poor: float = 0.20  # any player above: too long waiting
    wait_unfair_ratio: float = 2.5  # a player waiting this many times the group's median...
    wait_unfair_min: float = 0.10  # ...and at least this share, is flagged
    moving_good: float = 0.60  # group median time on the move at or above: a busy group
    ground_low_ratio: float = 0.80  # a player covering under this share of the group's metres/minute is flagged
    net_drill_min: float = 0.10  # group median time at the net at or above: a drill that involves the net
    net_low_ratio: float = 0.40  # ...then a player at the net under this share of the group's median is flagged
    intensity_drop: float = 0.20  # last third this much below the first (movement or ball in play): a drop


def _off_court_share(coverage: dict) -> float:
    s = np.array(coverage["seconds"], dtype=float)
    if s.sum() <= 0:
        return 0.0
    x0, y0 = coverage["origin_m"]
    c = coverage["cell_m"]
    rows, cols = s.shape
    xs = x0 + (np.arange(cols) + 0.5) * c
    ys = y0 + (np.arange(rows) + 0.5) * c
    outside = (xs[None, :] < -0.5) | (xs[None, :] > 11.47) | (ys[:, None] < -0.5) | (ys[:, None] > 24.27)
    return float(s[outside].sum() / s.sum())


def likely_coach(players: list[dict], coverage: dict) -> Optional[int]:
    rated = [p for p in players if p.get("moving_share") is not None and str(p["player"]) in coverage]
    if len(rated) < 3:
        return None
    by_moving = sorted(rated, key=lambda p: p["moving_share"])
    least, next_least = by_moving[0], by_moving[1]
    if least["moving_share"] > COACH_MOBILITY_RATIO * next_least["moving_share"]:
        return None
    off = {p["player"]: _off_court_share(coverage[str(p["player"])]) for p in rated}
    return least["player"] if off[least["player"]] == max(off.values()) else None


def load_targets(path) -> Targets:
    """A coach's own targets from a JSON file of Targets field names; any
    field left out keeps its default."""
    import json
    from pathlib import Path

    return Targets(**json.loads(Path(path).read_text()))


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def _times(ratio: float) -> str:
    return f"about {round(ratio)}×" if ratio >= 1.95 else f"{ratio:.1f}×"


def build_summary(report: dict, targets: Optional[Targets] = None, coach_override: Optional[int] = None) -> dict:
    t = targets or Targets()
    metrics = report["metrics"]
    players = report.get("players") or []
    coverage = report.get("raw", {}).get("player_coverage") or {}
    well, improve, notes = [], [], []

    # ---- coach
    coach = coach_override if coach_override is not None else likely_coach(players, coverage)
    group = [p for p in players if p["player"] != coach]
    if coach is not None and coach_override is None:
        notes.append({
            "title": f"Player {coach} looks like the coach",
            "detail": "They moved far less than anyone else and spent the most time off court, so they are left "
                      "out of the player comparisons. If that is wrong, set the coach when running the report.",
        })

    # ---- ball in play (target)
    bip = (metrics.get("ball_in_play") or {}).get("value")
    if bip is not None:
        in_five = round(bip * 5)
        if bip >= t.ball_in_play_good:
            well.append({"title": "Very little dead time",
                         "detail": f"A ball was in play {_pct(bip)} of the time - about {in_five} minutes in every 5 were live play.",
                         "basis": "target", "target": f"{_pct(t.ball_in_play_good)} or more"})
        elif bip < t.ball_in_play_poor:
            improve.append({"title": "A lot of dead time", "priority": 2,
                            "short": f"the ball was in play only {_pct(bip)} of the time",
                            "detail": f"A ball was in play only {_pct(bip)} of the time. Look at what filled the rest - "
                                      "ball collection, waiting, long resets.",
                            "basis": "target", "target": f"{_pct(t.ball_in_play_good)} or more"})
        else:
            notes.append({"title": "Some dead time",
                          "detail": f"A ball was in play {_pct(bip)} of the time.", "basis": "target",
                          "target": f"{_pct(t.ball_in_play_good)} or more"})

    # ---- movement (target + group)
    moving = [p["moving_share"] for p in group if p.get("moving_share") is not None]
    if moving:
        m = median(moving)
        if m >= t.moving_good:
            well.append({"title": "Players kept moving",
                         "detail": f"The typical player was on the move {_pct(m)} of the time they were on court.",
                         "basis": "target", "target": f"{_pct(t.moving_good)} or more"})
        else:
            improve.append({"title": "A lot of standing around", "priority": 5,
                            "short": f"the typical player was moving only {_pct(m)} of the time",
                            "detail": f"The typical player was on the move only {_pct(m)} of the time.",
                            "basis": "target", "target": f"{_pct(t.moving_good)} or more"})

    pace = {p["player"]: p["distance_m"] / (p["on_camera_s"] / 60.0) for p in group if p.get("on_camera_s")}
    if len(pace) >= 3:
        pm = median(pace.values())
        slow = [pl for pl, v in pace.items() if v < t.ground_low_ratio * pm]
        for pl in slow:
            share_less = 1 - pace[pl] / pm
            improve.append({"title": f"Player {pl} covered less ground", "priority": 3,
                            "short": f"covered about {_pct(share_less)} less ground",
                            "detail": f"Player {pl} covered about {_pct(share_less)} less ground per minute than the "
                                      f"group ({round(pace[pl])} m vs a typical {round(pm)} m a minute).",
                            "basis": "group", "player": pl})
        if not slow:
            lo, hi = min(pace.values()), max(pace.values())
            well.append({"title": "Work shared evenly",
                         "detail": f"Everyone covered a similar amount of ground: {round(lo)}-{round(hi)} m a minute.",
                         "basis": "group"})

    # ---- waiting (target + group)
    waits = {p["player"]: p["queue_share"] for p in group if p.get("queue_share") is not None}
    if waits:
        worst = max(waits, key=waits.get)
        others = [v for pl, v in waits.items() if pl != worst]
        typical = median(others) if others else 0.0
        longest = max((p.get("longest_queue_s") or 0) for p in group)
        if max(waits.values()) < t.wait_good:
            well.append({"title": "Nobody waited long for a turn",
                         "detail": f"Every player spent under {_pct(t.wait_good)} of the drill waiting off court; "
                                   f"the longest single wait was {round(longest)}s.",
                         "basis": "target", "target": f"under {_pct(t.wait_good)} each"})
        unfair = waits[worst] >= t.wait_unfair_min and (
            typical == 0 or waits[worst] / max(typical, 1e-9) >= t.wait_unfair_ratio
        )
        if waits[worst] > t.wait_poor or unfair:
            ratio = waits[worst] / typical if typical > 0 else None
            compare = f"{_times(ratio)} the rest of the group" if ratio else "while most of the group barely waited"
            improve.append({"title": f"Player {worst} waited much longer than the others", "priority": 0,
                            "short": f"waited {compare}",
                            "detail": f"Player {worst} spent {_pct(waits[worst])} of the drill waiting off court - "
                                      f"{compare}. Check the rotation brings them in as often as everyone else.",
                            "basis": "group", "player": worst})

    # ---- net vs baseline (group)
    nets = {p["player"]: p["zone_shares"]["net"] for p in group if p.get("zone_shares")}
    if len(nets) >= 3:
        nm = median(nets.values())
        if nm >= t.net_drill_min:
            low = [pl for pl, v in nets.items() if v < t.net_low_ratio * nm]
            for pl in low:
                improve.append({"title": f"Player {pl} barely got to the net", "priority": 1, "player": pl,
                                "short": f"was at the net only {_pct(nets[pl])} of the time",
                                "detail": f"Player {pl} spent {_pct(nets[pl])} of the drill at the net, against a typical "
                                          f"{_pct(nm)} - worth checking they get their turns at the net position.",
                                "basis": "group"})
            if not low:
                well.append({"title": "Everyone got time at the net",
                             "detail": f"Every player spent at least {_pct(min(nets.values()))} of the drill at the net "
                                       f"(typical {_pct(nm)}).", "basis": "group"})
        else:
            notes.append({"title": "A baseline session",
                          "detail": f"The typical player was at the net only {_pct(nm)} of the time.", "basis": "group"})

    # ---- intensity over the session (first vs last window)
    windows = report.get("raw", {}).get("intensity") or []
    if len(windows) >= 2:
        def group_pace(w):
            vals = [v for pl, v in w["metres_per_minute"].items() if int(pl) != coach]
            return median(vals) if vals else None
        first, last = windows[0], windows[-1]
        drops = []
        p0, p1 = group_pace(first), group_pace(last)
        if p0 and p1 and p1 < (1 - t.intensity_drop) * p0:
            drops.append(f"players moved {_pct(1 - p1 / p0)} less")
        b0, b1 = first["ball_in_play_share"], last["ball_in_play_share"]
        if b0 and b1 < (1 - t.intensity_drop) * b0:
            drops.append(f"the ball was in play {_pct(1 - b1 / b0)} less")
        if drops:
            improve.append({"title": "Intensity dropped towards the end", "priority": 4,
                            "short": "intensity dropped towards the end",
                            "detail": "In the last part of the session " + " and ".join(drops) + " than at the start - "
                                      "fatigue, or the drill running out of steam.", "basis": "session"})
        elif p0 and p1:
            well.append({"title": "Intensity held up to the end",
                         "detail": f"Players were moving about {round(p1)} m a minute in the last part of the session, "
                                   f"against {round(p0)} at the start.", "basis": "session"})

    # ---- structure
    drills = metrics.get("drills") or {}
    gathered = drills.get("breaks_with_players_gathered") or 0
    if drills.get("breaks"):
        notes.append({"title": f"{drills['breaks']} break{'s' if drills['breaks'] > 1 else ''} in play",
                      "detail": f"{gathered} with the players gathered together - what an instruction huddle looks like."})

    # ---- headline: the best of what went well, then the single most
    # actionable thing to improve - with a player's findings merged into
    # one clause ("waited 5x as long AND covered less ground") because a
    # coach reads them as one problem with one person.
    improve.sort(key=lambda f: f.get("priority", 9))
    positives = " and ".join(f["title"].lower() for f in well[:2])
    positives = positives[:1].upper() + positives[1:] if positives else ""
    if improve:
        top = improve[0]
        if top.get("player") is not None:
            same = [f["short"] for f in improve if f.get("player") == top["player"]]
            joined = same[0] if len(same) == 1 else ", ".join(same[:-1]) + (", and " if len(same) > 2 else " and ") + same[-1]
            fix = f"Player {top['player']} " + joined
        else:
            fix = top["short"]
        headline = f"{positives} - but {fix}." if positives else fix[:1].upper() + fix[1:] + "."
    elif well:
        headline = f"{positives}, and nothing stands out to fix."
    else:
        headline = "Not enough was measured to judge this session."

    # ---- player cards
    cards = []
    pm = median(pace.values()) if pace else None
    wm = median(waits.values()) if waits else None
    for p in players:
        role = "coach" if p["player"] == coach else "player"
        flags = [f["title"] for f in improve if f.get("player") == p["player"]]
        if role == "coach":
            line = "Likely the coach - stood mostly beside the court; not compared with the players."
        else:
            bits = [f"on the move {_pct(p['moving_share'])} of the time" if p.get("moving_share") is not None else None]
            if p["player"] in pace and pm:
                rel = pace[p["player"]] / pm
                bits.append("covered " + ("about the typical ground" if 0.9 <= rel <= 1.1 else
                                          f"{_pct(abs(1 - rel))} {'more' if rel > 1 else 'less'} ground than typical"))
            if p.get("queue_share") is not None:
                bits.append("barely waited" if p["queue_share"] < 0.02 else f"waited {_pct(p['queue_share'])} of the drill")
            if p.get("zone_shares"):
                z = p["zone_shares"]
                bits.append(f"at the net {_pct(z['net'])}, baseline {_pct(z['baseline'] + z['deep'])}")
            line = "; ".join(b for b in bits if b).capitalize() + "."
        cards.append({
            "player": p["player"], "role": role, "line": line, "flags": flags,
            "moving_share": p.get("moving_share"),
            "metres_per_minute": round(pace[p["player"]], 1) if p["player"] in pace else None,
            "wait_share": p.get("queue_share"),
        })

    return {
        "headline": headline,
        "went_well": well,
        "to_improve": improve,
        "notes": notes,
        "players": cards,
        "coach_player": coach,
        "coach_source": "set" if coach_override is not None else ("guessed" if coach is not None else None),
        "targets": {k: v for k, v in asdict(t).items()},
        "group_reference": {
            "metres_per_minute": round(pm, 1) if pm else None,
            "wait_share": round(wm, 3) if wm is not None else None,
        },
    }


# Fed vs live is a coaching choice with no right answer, so it is a note -
# what kind of session it was - never a verdict.
MOSTLY = 0.7


def play_note(feeding: dict) -> Optional[dict]:
    """One note saying whether the session was fed or live, from the
    report's feeding section (src/session/feeding.py)."""
    n = feeding.get("rallies") or 0
    if not n:
        return None
    fed, live = feeding["fed_rallies"], feeding["live_rallies"]
    per = feeding["shots_per_rally"]
    coach_share = feeding.get("coach_shot_share")
    hitting = f" The coach did {_pct(coach_share)} of the hitting." if coach_share is not None else ""
    if feeding.get("coach_player") is None:
        return {"title": "Fed or live: unknown",
                "detail": f"{n} rallies, averaging {per['all']} shots, but with no coach identified a feed cannot "
                          "be told from a player starting the point. Set the coach when running the report."}
    if live / n >= MOSTLY:
        title = "Mostly live play"
        detail = f"Players started {live} of {n} rallies themselves, averaging {per['live']} shots a rally."
    elif fed / n >= MOSTLY:
        title = "Mostly fed"
        detail = f"The coach started {fed} of {n} rallies, with about {per['fed']} shots from each feed."
    else:
        title = "A mix of fed and live play"
        detail = (f"The coach fed {fed} of {n} rallies (about {per['fed']} shots each); players started "
                  f"{live} (about {per['live']} shots each).")
    return {"title": title, "detail": detail + hitting, "basis": "session"}
