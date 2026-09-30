"""Stroke type for every detected swing, with the trained stroke classifier
(src/analysis/stroke_tcn.py; trained by scripts/train_stroke_tcn.py).

    python scripts/classify_strokes_tcn.py --session outputs/dingles_serve_volley \\
        --weights weights/stroke_tcn.pt --out outputs/dingles_serve_volley/strokes_tcn.json

With --labels (the labelling page's export) it also scores itself on them,
and with --compare (classify_strokes_bst.py --swings output) scores BST on
the same swings, in the classes both can say - forehand, backhand, serve,
not a swing - so the two models can be put side by side:

    python scripts/classify_strokes_tcn.py --session outputs/dingles_serve_volley \\
        --weights weights/stroke_tcn.pt --out outputs/dingles_serve_volley/strokes_tcn.json \\
        --labels data/labels/swing_labels_dingles_serve_volley.json \\
        --compare outputs/dingles_serve_volley/strokes_swings.json
"""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.stroke_tcn import CLASSES, build_model, class_of_label, features, window_for_swing  # noqa: E402

MIN_CONFIDENCE = 0.5  # below this the swing is "unsure", as with BST


def coarse(cls: str) -> str:
    """forehand | backhand | serve | overhead | not_a_swing - what BST can also say
    (slices and volleys fold into their side)."""
    return cls.split("_")[0] if cls.startswith(("forehand", "backhand")) else cls


def bst_coarse(hit: dict) -> str:
    stroke = hit["stroke"]
    if stroke == "net":
        return "forehand" if hit["label"] in ("HNR", "HFL") else "backhand"
    return stroke  # forehand | backhand | serve | unsure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="outputs/<name> with swings.json, poses.pkl, person_tracks.pkl")
    parser.add_argument("--weights", default=str(REPO_ROOT / "weights" / "stroke_tcn.pt"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels", help="Labelling-page export to score against")
    parser.add_argument("--compare", help="classify_strokes_bst.py --swings output, scored the same way")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import torch

    session = Path(args.session)
    swings = json.loads((session / "swings.json").read_text())["swings"]
    poses = pickle.load(open(session / "poses.pkl", "rb"))
    people = pickle.load(open(session / "person_tracks.pkl", "rb"))
    if poses.get("every", 1) != 1:
        raise SystemExit("needs a pose on every frame (track_poses.py --every 1)")
    fps = people["fps"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval()

    usable = [s for s in swings if "end" in s]
    X = np.stack([features(*window_for_swing(s["frame"], s["tracklet"], people["people"], poses["poses"], fps),
                           far=s["end"] == "far") for s in usable])
    far = np.array([s["end"] == "far" for s in usable])
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(X).to(device), torch.from_numpy(far).to(device)), dim=1).cpu().numpy()

    out = []
    for s, p in zip(usable, probs):
        k = int(np.argmax(p))
        out.append({"t_s": s["t_s"], "frame": s["frame"], "tracklet": s["tracklet"], "player": s.get("player"),
                    "end": s["end"], "class": CLASSES[k] if p[k] >= MIN_CONFIDENCE else "unsure",
                    "confidence": round(float(p[k]), 3),
                    "probabilities": {c: round(float(v), 3) for c, v in zip(CLASSES, p)}})
    counts = Counter(o["class"] for o in out)
    doc = {"model": str(args.weights), "min_confidence": MIN_CONFIDENCE, "counts": dict(counts), "swings": out}

    if args.labels:
        labels = {l["frame"]: class_of_label(l) for l in json.loads(Path(args.labels).read_text())["labels"]}
        bst = {h.get("swing_frame", h["frame"]): h for h in json.loads(Path(args.compare).read_text())["hits"]} if args.compare else {}

        def score(guess_of):
            fine = coarse_right = unsure = n = 0
            for o in out:
                want = labels.get(o["frame"])
                if want is None:
                    continue
                n += 1
                got = guess_of(o)
                if got in (None, "unsure"):
                    unsure += 1
                    continue
                coarse_right += coarse(got) == coarse(want)
                fine += got == want
            return {"labelled": n, "unsure": unsure, "coarse_right": coarse_right, "fine_right": fine}

        doc["score"] = {"tcn": score(lambda o: o["class"])}
        print(f"new model on {doc['score']['tcn']['labelled']} labelled swings: "
              f"{doc['score']['tcn']['coarse_right']} right on forehand/backhand/serve/overhead/not-a-swing, "
              f"{doc['score']['tcn']['fine_right']} right including slice and volley, "
              f"{doc['score']['tcn']['unsure']} unsure")
        if bst:
            doc["score"]["bst"] = score(lambda o: bst_coarse(bst[o["frame"]]) if o["frame"] in bst else None)
            b = doc["score"]["bst"]
            print(f"BST on the same swings:                {b['coarse_right']} right on the same classes, "
                  f"{b['unsure']} unsure (BST has no slice, volley or not-a-swing)")
    Path(args.out).write_text(json.dumps(doc, indent=1))
    print(", ".join(f"{c}: {n}" for c, n in counts.most_common()))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
