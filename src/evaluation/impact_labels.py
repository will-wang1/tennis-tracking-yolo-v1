"""Score detected impacts against hand-labelled ones (data/labels/*.csv).

This is small enough to look trivial and is not: the matching rule decides
what every accuracy number about this project means, and a wrong one
misreports a working detector as broken. Two rules that both sound right
have already been wrong here.

    First match within the window. Rejected: it pairs a label with
    whichever marker it meets first rather than the closest, so a marker
    0.16s away is credited to a label that has an exact marker of its own.

    Nearest marker, per label, independently. Rejected: nothing stops two
    labels from claiming the SAME marker. On video_input2 a confirmed
    bounce at 0.50s sits 0.16s from a confirmed non-event at 0.34s, so the
    one correct bounce marker was scored as both a hit and a false
    positive.

What is used instead is a global assignment: every label/marker pair within
tolerance is a candidate, the closest pairs are taken first, and both sides
are then retired. One marker answers at most one label, and no label is
answered by a marker that another label wanted more.

Only DRAWN impacts - bounces and contacts - are scored. An "unknown"
verdict makes no claim, so it can neither satisfy a label nor violate a
"none" one. That is deliberate: the classifier is allowed to say "I don't
know" at no cost, and pays only for the claims it actually makes.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

KINDS = ("bounce", "contact", "none")
DRAWN_KINDS = ("bounce", "contact")
_DEFAULT_TOLERANCE_S = 0.25


@dataclass(frozen=True)
class Label:
    """One event a human confirmed by watching the clip.

    `tolerance_s` is per label because the labels are not equally precise:
    most were read off a render that printed timestamps, a few were
    volunteered as descriptions ("a bounce very shortly after 6.95") and
    pin down only a window.
    """

    seconds: float
    kind: str
    tolerance_s: float = _DEFAULT_TOLERANCE_S
    note: str = ""


@dataclass(frozen=True)
class ScoredLabel:
    label: Label
    match: Optional[object]  # the BounceCandidate that answered it, if any
    outcome: str  # "ok" | "wrong_kind" | "missed" | "false_positive"


@dataclass(frozen=True)
class Score:
    scored: list[ScoredLabel]
    unclaimed: list[object] = field(default_factory=list)

    def _count(self, outcome: str) -> int:
        return sum(1 for s in self.scored if s.outcome == outcome)

    @property
    def correct(self) -> int:
        return self._count("ok")

    @property
    def wrong_kind(self) -> int:
        return self._count("wrong_kind")

    @property
    def missed(self) -> int:
        return self._count("missed")

    @property
    def false_positives(self) -> int:
        return self._count("false_positive")


def read_labels(path: Path) -> list[Label]:
    labels = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            kind = row["kind"].strip()
            if kind not in KINDS:
                raise ValueError(f"{path}: unknown kind {kind!r}, expected one of {KINDS}")
            labels.append(
                Label(
                    seconds=float(row["seconds"]),
                    kind=kind,
                    tolerance_s=float(row.get("tolerance_s") or _DEFAULT_TOLERANCE_S),
                    note=(row.get("note") or "").strip(),
                )
            )
    return sorted(labels, key=lambda label: label.seconds)


def match_labels(labels: Sequence[Label], drawn: Sequence, fps: float) -> dict[int, object]:
    """Assign at most one marker to each label, closest pairs first.

    Returns `{label index: marker}`, leaving unmatched labels out. See the
    module docstring for why this is not a per-label nearest search.
    """
    candidates = []
    for label_idx, label in enumerate(labels):
        for marker_idx, impact in enumerate(drawn):
            distance = abs(impact.t / fps - label.seconds)
            if distance <= label.tolerance_s:
                candidates.append((distance, label_idx, marker_idx))

    matched: dict[int, object] = {}
    claimed: set[int] = set()
    for _, label_idx, marker_idx in sorted(candidates):
        if label_idx in matched or marker_idx in claimed:
            continue
        matched[label_idx] = drawn[marker_idx]
        claimed.add(marker_idx)
    return matched


def score_impacts(impacts: Sequence, labels: Sequence[Label], fps: float) -> Score:
    drawn = [impact for impact in impacts if impact.kind in DRAWN_KINDS]
    matched = match_labels(labels, drawn, fps)
    claimed = {id(marker) for marker in matched.values()}

    scored = []
    for label_idx, label in enumerate(labels):
        match = matched.get(label_idx)
        if label.kind == "none":
            outcome = "false_positive" if match is not None else "ok"
        elif match is None:
            outcome = "missed"
        elif match.kind == label.kind:
            outcome = "ok"
        else:
            outcome = "wrong_kind"
        scored.append(ScoredLabel(label=label, match=match, outcome=outcome))

    return Score(
        scored=scored,
        unclaimed=[impact for impact in drawn if id(impact) not in claimed],
    )


@dataclass(frozen=True)
class RallyFlaw:
    """One place where the sequence of verdicts is not a rally.

    A rally alternates: somebody hits the ball, it lands, somebody hits it
    back. So between two contacts there should be exactly one bounce, and
    the two contacts should be far enough apart for the ball to have crossed
    the court and back. Neither says which verdict is wrong, only that one of
    them is - which is why this reports rather than corrects.
    """

    first: float  # seconds
    second: float
    bounces_between: int
    problem: str


def rally_flaws(impacts, fps: float, min_exchange_seconds: float = 0.6) -> list[RallyFlaw]:
    """Consecutive contacts that cannot both be right.

    `min_exchange_seconds` is how quickly two players could conceivably hit
    the ball in turn. Measured over 35 exchanges on this project's three
    clips the shortest real one is 0.74s, and the two that break the rule
    sit at 0.19s and 0.28s, so the gap between plausible and impossible is
    wide and the exact threshold in it does not matter much. It has to be
    generous anyway: a volley is played from mid-court, which legitimately
    shortens the exchange.

    A pair with no bounce between them is reported whatever the gap - the
    ball has to land somewhere between two strikes, so a missing bounce is
    a missing detection even when the timing is fine. Both checks locate a
    fault without attributing it: the pair is inconsistent, and which half
    is wrong needs evidence this cannot supply.
    """
    contacts = [impact for impact in impacts if impact.kind == "contact"]
    flaws = []
    for first, second in zip(contacts, contacts[1:]):
        between = sum(
            1
            for impact in impacts
            if first.t < impact.t < second.t and impact.kind == "bounce"
        )
        gap = (second.t - first.t) / fps
        if gap < min_exchange_seconds:
            problem = f"only {gap:.2f}s apart - too quick for the ball to have crossed and come back"
        elif between == 0:
            problem = f"{gap:.2f}s apart with no bounce between them"
        else:
            continue
        flaws.append(
            RallyFlaw(
                first=first.t / fps,
                second=second.t / fps,
                bounces_between=between,
                problem=problem,
            )
        )
    return flaws
