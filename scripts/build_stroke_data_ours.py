"""Stroke data from our own hand-labelled swings (the labelling page,
scripts/make_swing_labeller.py), in the same form as
scripts/build_stroke_data_e2e.py writes: each labelled swing's player,
their pose over the window around the swing (wrist-speed peak), and the
label as a class of src/analysis/stroke_tcn.py.

    python scripts/build_stroke_data_ours.py --session outputs/dingles_serve_volley \\
        --labels data/labels/swing_labels_dingles_serve_volley.json --out data/strokes/ours/dingles_serve_volley.npz
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.stroke_tcn import CLASSES, FPS, SEQ_LEN, class_of_label, window_times  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="outputs/<name> with swings.json, poses.pkl, person_tracks.pkl")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    session = Path(args.session)
    swings = {s["frame"]: s for s in json.loads((session / "swings.json").read_text())["swings"]}
    poses = pickle.load(open(session / "poses.pkl", "rb"))
    people = pickle.load(open(session / "person_tracks.pkl", "rb"))
    fps = people["fps"]
    if poses.get("every", 1) != 1:
        raise SystemExit("needs a pose on every frame (track_poses.py --every 1)")
    rows = people["people"]
    out = {"kp": [], "conf": [], "box": [], "far": [], "label": [], "meta": []}
    for lab in json.loads(Path(args.labels).read_text())["labels"]:
        cls = class_of_label(lab)
        sw = swings.get(lab["frame"])
        if cls is None or sw is None or "end" not in sw:
            continue
        t = lab["tracklet"]
        kp = np.full((SEQ_LEN, 17, 2), np.nan, np.float32)
        conf = np.zeros((SEQ_LEN, 17), np.float32)
        box = np.full((SEQ_LEN, 4), np.nan, np.float32)
        for k, ts in enumerate(window_times(sw["frame"] / fps)):
            f = int(round(ts * fps))
            if not 0 <= f < len(rows):
                continue
            b = next((r[1:5] for r in rows[f] if r[0] == t), None)
            if b is None:
                continue
            box[k] = b
            p = poses["poses"][f].get(t) if f < len(poses["poses"]) else None
            if p is not None:
                kp[k], conf[k] = p
        out["kp"].append(kp); out["conf"].append(conf); out["box"].append(box)
        out["far"].append(sw["end"] == "far"); out["label"].append(CLASSES.index(cls))
        out["meta"].append(f"{lab['session']}@{lab['t_s']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, kp=np.stack(out["kp"]), conf=np.stack(out["conf"]), box=np.stack(out["box"]),
                        far=np.array(out["far"]), label=np.array(out["label"]), meta=np.array(out["meta"]),
                        source=np.array("ours"), fps=np.array(FPS))
    counts = np.bincount(out["label"], minlength=len(CLASSES))
    print(f"{len(out['label'])} labelled swings -> {args.out}: " + ", ".join(f"{c} {n}" for c, n in zip(CLASSES, counts) if n))


if __name__ == "__main__":
    main()
