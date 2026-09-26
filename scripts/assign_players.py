"""Join a clip's person tracklets into stable players (src/analysis/identity.py).

    python scripts/assign_players.py \\
        --video data/videos/dingles_serve_volley.mp4 \\
        --people outputs/dingles_serve_volley/person_tracks.pkl \\
        --poses outputs/dingles_serve_volley/poses.pkl \\
        --report outputs/dingles_serve_volley/session_report.json \\
        --out outputs/dingles_serve_volley/players.json \\
        [--truth data/labels/dingles_serve_volley_identities.json]

With --truth, also sweeps the merge distance and prints pairwise
precision/recall against the labels, so the threshold is chosen by
measurement. Writes {tracklet: player} plus which tracklets were dropped
as not a person.
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.identity import (  # noqa: E402
    DEFAULT_MERGE_DISTANCE,
    MIN_POSED_SHARE,
    cluster_tracklets,
    colour_fingerprint,
    conflicts,
    pairwise_scores,
)

MIN_TRACKLET_FRAMES = 25  # a second at 25fps; shorter stretches are too few samples to fingerprint
SAMPLES_PER_TRACKLET = 12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--people", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--report", help="session_report.json - cutaway frames are ignored")
    parser.add_argument("--out", required=True)
    parser.add_argument("--truth", help="identities json to measure against")
    parser.add_argument("--merge-distance", type=float, default=DEFAULT_MERGE_DISTANCE)
    args = parser.parse_args()

    import cv2

    people = pickle.load(open(args.people, "rb"))
    poses = pickle.load(open(args.poses, "rb"))
    fps = people["fps"]
    cutaway = set()
    if args.report:
        for a, b in json.loads(Path(args.report).read_text())["clip"]["cutaways_s"]:
            cutaway.update(range(int(round(a * fps)), int(round(b * fps))))

    boxes = defaultdict(dict)
    posed = defaultdict(int)
    for f, row in enumerate(people["people"]):
        if f in cutaway:
            continue
        for t, x1, y1, x2, y2, c in row:
            if t is None:
                continue
            boxes[t][f] = (x1, y1, x2, y2, c)
            p = poses["poses"][f].get(t) if f < len(poses["poses"]) else None
            if p is not None and np.mean(p[1][[5, 6, 11, 12, 13, 14]] > 0.5) >= 0.8:
                posed[t] += 1
    long_enough = {t for t, b in boxes.items() if len(b) >= MIN_TRACKLET_FRAMES}
    not_people = sorted(t for t in long_enough if posed[t] / len(boxes[t]) < MIN_POSED_SHARE)
    tracklets = sorted(long_enough - set(not_people))
    print(f"{len(tracklets)} person tracklets, {len(not_people)} dropped as not a person: {not_people}")

    want = defaultdict(list)
    for t in tracklets:
        fs = sorted(boxes[t])
        for i in np.linspace(0, len(fs) - 1, min(SAMPLES_PER_TRACKLET, len(fs))):
            want[fs[int(i)]].append(t)
    samples = defaultdict(list)
    capture = cv2.VideoCapture(args.video)
    for f in range(people["num_frames"]):
        ok, frame = capture.read()
        if not ok:
            break
        for t in want.get(f, []):
            x1, y1, x2, y2, _ = boxes[t][f]
            crop = frame[int(max(0, y1)):int(y2), int(max(0, x1)):int(x2)]
            if crop.shape[0] >= 16 and crop.shape[1] >= 8:
                samples[t].append(colour_fingerprint(crop))
    capture.release()
    fingerprints = {t: np.mean(v, axis=0) for t, v in samples.items() if v}
    track_boxes = {t: boxes[t] for t in fingerprints}
    cannot = conflicts(track_boxes)

    if args.truth:
        truth_doc = json.loads(Path(args.truth).read_text())
        truth = {t: p for p, ts in truth_doc["people"].items() for t in ts}
        print("\nmerge distance   precision  recall  players found (truth: %d)" % len(truth_doc["people"]))
        for d in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6):
            s = pairwise_scores(cluster_tracklets(fingerprints, track_boxes, d, cannot), truth)
            print(f"   {d:5.2f}         {s['precision']:6.1%}  {s['recall']:6.1%}   {s['clusters']}")

    assignment = cluster_tracklets(fingerprints, track_boxes, args.merge_distance, cannot)
    players = {str(t): k + 1 for t, k in assignment.items()}
    members = defaultdict(list)
    for t, p in players.items():
        members[p].append(int(t))
    out = {
        "merge_distance": args.merge_distance,
        "method": "colour fingerprints + co-occurrence-constrained clustering - see src/analysis/identity.py",
        "player_of_tracklet": players,
        "players": {str(p): sorted(ts) for p, ts in sorted(members.items())},
        "not_people": not_people,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\n{len(members)} players at merge distance {args.merge_distance}")
    for p, ts in sorted(members.items()):
        print(f"  player {p}: tracklets {sorted(ts)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
