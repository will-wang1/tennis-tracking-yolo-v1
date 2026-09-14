"""Fold a chronological `ServeAttempt` list (serve_sequences.py) into
POINTS and match-level serve statistics - fault/double-fault counts,
first-serve-in rate, rally length, who served each point. Everything the
impact structure can honestly support, and nothing more.

WHAT THIS DELIBERATELY DOES NOT DO: say who WON a point. A point's winner
is decided by whether its last shot was a winner or an error, which needs
exactly the ball in/out judgment `serve_sequences.py` was built to avoid
making from pixels - see that module's docstring for why (the ground-plane
homography is least trustworthy exactly where a serve/line call would need
it). Guessing a winner from "who hit last" would be wrong close to at
random - in real tennis the last shot before a point ends is a winner
roughly as often as it is an error, and this pipeline has no signal to
tell which happened. So there is no SCORE here - no 0/15/30/40, no game or
set tally, no "who's ahead" - only a log of what structurally happened.
That is a deliberate scope decision, not a missing feature: a fabricated
score built by guessing point winners would be actively misleading input
for anything reading this data downstream (an AI agent especially, which
has no way to tell a real score from a plausible-looking guess).

A POINT here is at most TWO `ServeAttempt`s - a first serve, and (only if
the first one was a fault) a second - because the rules cap it there,
which is knowledge `serve_sequences.py` itself doesn't have (it reads pure
timing/identity structure, not tennis's own serve-count rule). That cap is
what this module adds: if the SECOND serve also looks like "same server,
soon again" by the same structural test, it can only be a DOUBLE FAULT,
not a third serve attempt - the point ends right there, and whatever
attempt comes next belongs to a new point, not to this one. An earlier
version of this module didn't enforce that cap and would misattribute the
NEXT point's outcome to the double-faulted one; the two-serve limit is
what fixes it, and it comes from the rules of tennis, not from anything
measured.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

from src.analysis.serve_sequences import ServeAttempt


@dataclass(frozen=True)
class Point:
    server_identity: Optional[str]
    server_side: str
    start_frame: int
    start_t_s: float
    fault_count: int  # 0, 1, or 2 - capped at 2 by the rules, see module docstring
    double_fault: bool  # fault_count == 2
    rally_shot_count: int  # shots_before_next_serve of the DECIDING attempt (0 for a double fault)
    outcome: str  # "unreturned" | "rally_developed" | "unknown" | "double_fault"

    def to_dict(self) -> dict:
        return {
            "server_identity": self.server_identity,
            "server_side": self.server_side,
            "start_frame": self.start_frame,
            "start_t_s": round(self.start_t_s, 2),
            "fault_count": self.fault_count,
            "double_fault": self.double_fault,
            "rally_shot_count": self.rally_shot_count,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class MatchLog:
    points: list[Point] = field(default_factory=list)

    @property
    def points_served_by(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for point in self.points:
            if point.server_identity is not None:
                counts[point.server_identity] = counts.get(point.server_identity, 0) + 1
        return counts

    @property
    def fault_counts_by(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for point in self.points:
            if point.server_identity is not None and point.fault_count:
                counts[point.server_identity] = counts.get(point.server_identity, 0) + point.fault_count
        return counts

    @property
    def double_fault_counts_by(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for point in self.points:
            if point.server_identity is not None and point.double_fault:
                counts[point.server_identity] = counts.get(point.server_identity, 0) + 1
        return counts

    @property
    def first_serve_in_rate_by(self) -> dict[str, Optional[float]]:
        """Fraction of a server's points where the FIRST serve attempt
        wasn't a fault - i.e. `fault_count == 0`. None for a server with no
        logged points, rather than a division by zero pretending to be 0%."""
        served = self.points_served_by
        first_in: dict[str, int] = {}
        for point in self.points:
            if point.server_identity is not None and point.fault_count == 0:
                first_in[point.server_identity] = first_in.get(point.server_identity, 0) + 1
        return {
            identity: (first_in.get(identity, 0) / total if total else None)
            for identity, total in served.items()
        }

    @property
    def average_rally_shot_count(self) -> Optional[float]:
        """Mean `rally_shot_count` over points that actually became a
        RALLY (outcome == "rally_developed") - None if there are none.
        Deliberately excludes "unreturned" and "double_fault" points, both
        0 shots by definition, which would just drag a "rally length"
        average toward zero under a name that implies something happened."""
        rallies = [p.rally_shot_count for p in self.points if p.outcome == "rally_developed"]
        return sum(rallies) / len(rallies) if rallies else None

    def to_dict(self) -> dict:
        return {
            "points": [p.to_dict() for p in self.points],
            "point_count": len(self.points),
            "points_served_by": self.points_served_by,
            "fault_counts_by": self.fault_counts_by,
            "double_fault_counts_by": self.double_fault_counts_by,
            "first_serve_in_rate_by": {
                identity: (round(rate, 3) if rate is not None else None)
                for identity, rate in self.first_serve_in_rate_by.items()
            },
            "average_rally_shot_count": (
                round(self.average_rally_shot_count, 2) if self.average_rally_shot_count is not None else None
            ),
        }


def build_match_log(serve_attempts: Sequence[ServeAttempt]) -> MatchLog:
    """`serve_attempts` must be in chronological order (as
    `classify_serve_sequences` already returns them)."""
    points: list[Point] = []
    first_serve: Optional[ServeAttempt] = None  # set while waiting for a second serve

    for attempt in serve_attempts:
        if first_serve is None:
            if attempt.outcome == "fault":
                first_serve = attempt  # wait for the second serve before closing this point
                continue
            points.append(
                Point(
                    server_identity=attempt.identity,
                    server_side=attempt.side,
                    start_frame=attempt.frame_idx,
                    start_t_s=attempt.t_s,
                    fault_count=0,
                    double_fault=False,
                    rally_shot_count=attempt.shots_before_next_serve,
                    outcome=attempt.outcome,
                )
            )
            continue

        # This attempt MUST be the point's second serve - tennis allows no
        # third, so whatever structural label classify_serve_sequences put
        # on it is interpreted through that rule, not taken at face value.
        if attempt.outcome == "fault":
            # A second serve that ALSO looks like "same server, soon
            # again" can only mean a double fault: the point ended right
            # here, with nothing of ITS OWN to report - see this module's
            # docstring for why this is the fix over just reusing whatever
            # attempt happens to come next (that belongs to a new point).
            points.append(
                Point(
                    server_identity=first_serve.identity,
                    server_side=first_serve.side,
                    start_frame=first_serve.frame_idx,
                    start_t_s=first_serve.t_s,
                    fault_count=2,
                    double_fault=True,
                    rally_shot_count=0,
                    outcome="double_fault",
                )
            )
        else:
            points.append(
                Point(
                    server_identity=first_serve.identity,
                    server_side=first_serve.side,
                    start_frame=first_serve.frame_idx,
                    start_t_s=first_serve.t_s,
                    fault_count=1,
                    double_fault=False,
                    rally_shot_count=attempt.shots_before_next_serve,
                    outcome=attempt.outcome,
                )
            )
        first_serve = None

    # classify_serve_sequences only ever labels an attempt "fault" when a
    # LATER attempt exists to judge it against (see that module), so
    # first_serve can never be left dangling once every attempt has been
    # processed - nothing trailing to flush here.
    return MatchLog(points=points)
