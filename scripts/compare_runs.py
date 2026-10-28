"""Did a change to how the pipeline runs (16-bit models, a faster encoder)
change what it finds? Compares two runs of run_session.py on the same clip:

    TENNIS_FP32=1 python scripts/run_session.py --input clip.mp4 --name before --no-feedback --no-video
    python scripts/run_session.py --input clip.mp4 --name after --no-feedback --no-video
    python scripts/compare_runs.py outputs/before outputs/after --labels data/labels/swing_labels_dingles_serve_volley.json

Each line is one thing the pipeline finds - ball positions, people, E2E-Spot
events, swings, bounces and contacts - counted in both runs and matched
event to event (same frame, give or take a few). --labels adds how many of
the hand-labelled swings each run finds. Prints "check" next to anything that
moved more than a little; the exit code is 1 if anything did.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Tolerances: a change this small is noise, not a different answer.
FRAMES = 3  # events at most this many frames apart are the same event
BALL_PX = 6.0  # the same ball, give or take
CLOSE = 0.95  # share that must match / ratio counts must stay within


def matched(a: list[int], b: list[int], tolerance: int = FRAMES) -> int:
    """Events in `a` with a partner in `b` within `tolerance` frames, each
    partner used once (greedy, in time order)."""
    b = sorted(b)
    used = [False] * len(b)
    found = 0
    for frame in sorted(a):
        best = None
        for j, other in enumerate(b):
            if not used[j] and abs(other - frame) <= tolerance and (best is None or abs(other - frame) < abs(b[best] - frame)):
                best = j
        if best is not None:
            used[best] = True
            found += 1
    return found


def load(run: Path) -> dict:
    def pkl(name):
        with open(run / name, "rb") as handle:
            return pickle.load(handle)

    report = json.loads((run / "session_report.json").read_text())
    return {
        "cache": pkl("replay_cache.pkl"),
        "people": pkl("person_tracks.pkl")["people"],
        "poses": pkl("poses.pkl")["poses"],
        "e2e": json.loads((run / "e2e_spot.json").read_text())["events"],
        "swings": json.loads((run / "swings.json").read_text())["swings"],
        "players": json.loads((run / "players.json").read_text())["players"],
        "report": report,
    }


def compare(before: dict, after: dict, labels: list[dict] | None) -> list[tuple[str, str, str, bool]]:
    rows: list[tuple[str, str, str, bool]] = []

    def events(name: str, a: list[int], b: list[int]) -> None:
        same = matched(a, b)
        ok = same >= CLOSE * max(len(a), len(b)) if (a or b) else True
        rows.append((name, str(len(a)), f"{len(b)}  ({same} the same)", ok))

    def count(name: str, a: float, b: float, fmt: str = "{:.0f}") -> None:
        ok = (min(a, b) >= CLOSE * max(a, b)) if max(a, b) > 0 else True
        rows.append((name, fmt.format(a), fmt.format(b), ok))

    count("frames", before["cache"]["num_frames"], after["cache"]["num_frames"])
    # Ball: the strongest candidate per frame, where both runs saw one.
    a, b = before["cache"]["candidates"], after["cache"]["candidates"]
    seen_a = sum(1 for c in a if c)
    seen_b = sum(1 for c in b if c)
    both = [(x[0], y[0]) for x, y in zip(a, b) if x and y]
    close = sum(1 for x, y in both if np.hypot(x.x - y.x, x.y - y.y) <= BALL_PX)
    count("frames with a ball", seen_a, seen_b)
    rows.append(("  same ball position", "", f"{close} of {len(both)}", close >= CLOSE * max(1, len(both))))

    count("people per frame (mean)", np.mean([len(f) for f in before["people"]]),
          np.mean([len(f) for f in after["people"]]), "{:.2f}")
    count("poses", sum(len(f) for f in before["poses"]), sum(len(f) for f in after["poses"]))
    count("players", len(before["players"]), len(after["players"]))

    for label in sorted({e["label"] for e in before["e2e"] + after["e2e"]}):
        events(f"E2E-Spot {label.replace('_', ' ')}", [e["frame"] for e in before["e2e"] if e["label"] == label],
               [e["frame"] for e in after["e2e"] if e["label"] == label])

    events("swings", [s["frame"] for s in before["swings"]], [s["frame"] for s in after["swings"]])
    for kind in ("bounce", "contact"):
        events(f"{kind}s", [i["frame"] for i in before["report"]["raw"]["impacts"] if i["kind"] == kind],
               [i["frame"] for i in after["report"]["raw"]["impacts"] if i["kind"] == kind])
    events("contacts given a player", [p["frame"] for p in before["report"]["contacts"]["points"]],
           [p["frame"] for p in after["report"]["contacts"]["points"]])

    if labels:
        real = [label["frame"] for label in labels if label["stroke"] not in ("not a swing", "skip")]
        fake = [label["frame"] for label in labels if label["stroke"] == "not a swing"]
        found_a = matched(real, [s["frame"] for s in before["swings"]])
        found_b = matched(real, [s["frame"] for s in after["swings"]])
        rows.append((f"labelled swings found (of {len(real)})", str(found_a), str(found_b), found_b >= found_a - 1))
        wrong_a = matched(fake, [s["frame"] for s in before["swings"]])
        wrong_b = matched(fake, [s["frame"] for s in after["swings"]])
        rows.append((f"labelled non-swings kept (of {len(fake)})", str(wrong_a), str(wrong_b), wrong_b <= wrong_a + 1))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", type=Path, help="outputs/<name> of the reference run")
    parser.add_argument("after", type=Path, help="outputs/<name> of the changed run")
    parser.add_argument("--labels", type=Path, help="data/labels/swing_labels_<session>.json for this clip")
    args = parser.parse_args()

    labels = json.loads(args.labels.read_text())["labels"] if args.labels else None
    rows = compare(load(args.before), load(args.after), labels)
    width = max(len(r[0]) for r in rows)
    print(f"{'':{width}}  {'before':>8}  after")
    for name, a, b, ok in rows:
        print(f"{name:{width}}  {a:>8}  {b}{'' if ok else '   <- check'}")
    changed = [r[0].strip() for r in rows if not r[3]]
    print("\nThe same, within noise." if not changed else f"\nChanged: {', '.join(changed)}")
    sys.exit(1 if changed else 0)


if __name__ == "__main__":
    main()
