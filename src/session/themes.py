"""Lesson themes: did the session do what it set out to?

A coach runs a session ON something - getting to the net, consistency - and
the question is whether the players did more of it as the session went on.
Each theme compares the EARLY part of the session with the LATE part,
because that is what one session can show; comparing against previous
sessions needs several recorded sessions of the same squad.

  netplay        time at the net (first third vs last third of drill time,
                 per player and for the group) and the share of shots
                 played from the net (first half vs second half)
  consistency    rallies over within two shots, and shots per rally
                 (first half vs second half)
  adjustment     slices - needs the stroke model trained on labelled
                 swings; reported as not measured until then
  around_backhand  forehands from the backhand side - same

A theme's verdict is improved / held / dropped against the thresholds
below, or "not enough" when there is too little to compare. They are
conventions, stated on the page, not validated standards.
"""

from statistics import median
from typing import Optional

THEMES = {
    "netplay": "Net play",
    "consistency": "Consistency",
    "adjustment": "Adjustment (slice)",
    "around_backhand": "Getting around the backhand",
}
NEEDS_STROKES = {"adjustment", "around_backhand"}

NET_TIME_CHANGE = 0.05  # 5 points of drill time at the net, first third to last
SHORT_RALLY_SHOTS = 2  # a rally over within this many shots
SHORT_SHARE_CHANGE = 0.10  # 10 points fewer short rallies, first half to second
MIN_RALLIES_PER_HALF = 6
MIN_SHOTS_PER_HALF = 8


def _pct(x: float) -> str:
    return f"{round(100 * x)}%"


def _pts(x: float) -> str:
    return f"{'+' if x >= 0 else '-'}{round(100 * abs(x))} points"


def _verdict(change: float, threshold: float) -> str:
    return "improved" if change >= threshold else ("dropped" if change <= -threshold else "held")


def netplay(report: dict, swings: list[dict], coach: Optional[int]) -> dict:
    out = {"theme": "netplay", "name": THEMES["netplay"], "measures": [], "per_player": {}, "caveats": []}
    windows = report.get("raw", {}).get("net_windows") or []
    change = None
    if len(windows) >= 2:
        first, last = windows[0]["net_share"], windows[-1]["net_share"]
        both = [p for p in first if p in last and int(p) != coach]
        if both:
            early, late = median(first[p] for p in both), median(last[p] for p in both)
            change = late - early
            out["measures"].append({"name": "time at the net", "early": early, "late": late,
                                    "detail": f"The typical player spent {_pct(early)} of the first third of drill "
                                              f"time at the net and {_pct(late)} of the last third."})
            out["per_player"] = {p: {"early": first[p], "late": last[p]} for p in both}
    played = sorted((s for s in swings if s.get("player") not in (None, coach) and "at_net" in s),
                    key=lambda s: s["frame"])
    if len(played) >= 2 * MIN_SHOTS_PER_HALF:
        half = len(played) // 2
        a, b = played[:half], played[half:]
        sa, sb = sum(s["at_net"] for s in a) / len(a), sum(s["at_net"] for s in b) / len(b)
        out["measures"].append({"name": "shots from the net", "early": round(sa, 3), "late": round(sb, 3),
                                "detail": f"{_pct(sa)} of shots in the first half were played from inside the "
                                          f"service line, {_pct(sb)} in the second."})
    if change is None:
        out.update(verdict="not enough", headline="Too little drill time to compare the start with the end.")
        return out
    out["verdict"] = _verdict(change, NET_TIME_CHANGE)
    lead = {"improved": "Players got to the net more as the session went on",
            "dropped": "Players spent less time at the net as the session went on",
            "held": "Time at the net stayed about the same through the session"}[out["verdict"]]
    shown = round(100 * out["measures"][0]["late"]) - round(100 * out["measures"][0]["early"])
    out["headline"] = f"{lead}: {_pct(out['measures'][0]['early'])} of the time early, " \
                      f"{_pct(out['measures'][0]['late'])} late ({_pts(shown / 100)})."
    movers = sorted(out["per_player"].items(), key=lambda kv: kv[1]["late"] - kv[1]["early"])
    if len(movers) >= 3:
        low, high = movers[0], movers[-1]
        out["players_line"] = (f"Most change: Player {high[0]} ({_pct(high[1]['early'])} to {_pct(high[1]['late'])}); "
                               f"least: Player {low[0]} ({_pct(low[1]['early'])} to {_pct(low[1]['late'])}).")
    out["caveats"].append(f"'At the net' is inside the service line. Verdict: a change of {round(100 * NET_TIME_CHANGE)} "
                          "points or more either way; a convention, not a standard.")
    return out


def consistency(feeding: Optional[dict]) -> dict:
    out = {"theme": "consistency", "name": THEMES["consistency"], "measures": [], "caveats": [
        "A rally ending is not always an error: a winner, or the coach stopping play, ends one too. "
        "The tool cannot tell which.",
        "A missed swing (about one in six) ends a rally early.",
        f"Verdict: {round(100 * SHORT_SHARE_CHANGE)} points fewer (or more) rallies over within "
        f"{SHORT_RALLY_SHOTS} shots; a convention, not a standard.",
    ]}
    rallies = sorted((feeding or {}).get("rally_list") or [], key=lambda r: r["start_s"])
    half = len(rallies) // 2
    if half < MIN_RALLIES_PER_HALF:
        out.update(verdict="not enough", headline=f"Too few rallies ({len(rallies)}) to compare the start with the end.")
        return out
    a, b = rallies[:half], rallies[half:]
    short = lambda rs: sum(1 for r in rs if r["shots"] <= SHORT_RALLY_SHOTS) / len(rs)  # noqa: E731
    mean = lambda rs: sum(r["shots"] for r in rs) / len(rs)  # noqa: E731
    sa, sb = short(a), short(b)
    out["measures"] = [
        {"name": f"rallies over within {SHORT_RALLY_SHOTS} shots", "early": round(sa, 3), "late": round(sb, 3),
         "detail": f"{_pct(sa)} of rallies in the first half were over within {SHORT_RALLY_SHOTS} shots, "
                   f"{_pct(sb)} in the second."},
        {"name": "shots per rally", "early": round(mean(a), 1), "late": round(mean(b), 1),
         "detail": f"Rallies averaged {mean(a):.1f} shots in the first half and {mean(b):.1f} in the second."},
    ]
    out["verdict"] = _verdict(sa - sb, SHORT_SHARE_CHANGE)  # fewer short rallies is better
    lead = {"improved": "Rallies lasted longer as the session went on",
            "dropped": "Rallies got shorter as the session went on",
            "held": "Rally length stayed about the same through the session"}[out["verdict"]]
    out["headline"] = f"{lead}: {_pct(sa)} of rallies were over within {SHORT_RALLY_SHOTS} shots early, {_pct(sb)} late."
    return out


def not_measured(key: str) -> dict:
    what = {"adjustment": "slices", "around_backhand": "forehands played from the backhand side"}[key]
    return {"theme": key, "name": THEMES[key], "verdict": "not measured", "measures": [], "caveats": [],
            "headline": f"Needs {what}, which only a stroke model trained on labelled swings from your sessions "
                        "can recognise - not available yet."}


def evaluate(themes: list[str], report: dict, swings: list[dict], coach: Optional[int]) -> list[dict]:
    out = []
    for key in themes:
        if key not in THEMES:
            raise ValueError(f"Unknown theme {key!r}; choose from {', '.join(THEMES)}")
        if key == "netplay":
            out.append(netplay(report, swings, coach))
        elif key == "consistency":
            out.append(consistency(report.get("feeding")))
        else:
            out.append(not_measured(key))
    return out
