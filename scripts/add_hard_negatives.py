"""Stage frames from a confirmed false-positive location as training negatives.

Once analyze_detections.py + manual review has confirmed a screen region is
a fixed false-positive source (nothing there is actually the ball), this
pulls the flagged frames out of the source video and writes them straight
into a dataset split with an empty label file each - YOLO's convention for
"this image has no objects of the target class." Training on these teaches
the model to stop firing on that specific distractor.

    python scripts/add_hard_negatives.py --input match.mp4 --csv outputs/detections.csv \
        --x-min 870 --x-max 900 --y-min 210 --y-max 240 --out data/raw/train --prefix net_marker

Spot-check the written images before retraining - if a real ball happens to
pass through that same screen region in a couple of frames (plausible for
shots hit straight down the middle), delete those specific images/labels or
replace the empty label with a real box, since a wrong negative teaches the
model to ignore the ball exactly where it might legitimately be.
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.video.io import VideoReader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source video the CSV was generated from")
    parser.add_argument("--csv", required=True, help="detections.csv from analyze_detections.py")
    parser.add_argument("--x-min", type=float, required=True)
    parser.add_argument("--x-max", type=float, required=True)
    parser.add_argument("--y-min", type=float, required=True)
    parser.add_argument("--y-max", type=float, required=True)
    parser.add_argument("--out", required=True, help="Dataset split dir, e.g. data/raw/train")
    parser.add_argument("--prefix", default="hard_negative", help="Filename prefix")
    parser.add_argument("--limit", type=int, default=30, help="Max frames to pull")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    hits = df[
        df.x.between(args.x_min, args.x_max) & df.y.between(args.y_min, args.y_max)
    ].head(args.limit)
    if hits.empty:
        raise SystemExit("No detections match that region - check the CSV and bounds.")

    images_dir = Path(args.out) / "images"
    labels_dir = Path(args.out) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    reader = VideoReader(args.input)
    written = 0
    for _, row in hits.iterrows():
        reader.cap.set(cv2.CAP_PROP_POS_FRAMES, row.frame)
        ok, frame = reader.cap.read()
        if not ok:
            continue
        stem = f"{args.prefix}_{int(row.frame):06d}"
        cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame)
        (labels_dir / f"{stem}.txt").write_text("")  # empty = no objects
        written += 1

    print(f"Wrote {written} negative frames to {images_dir} (+ empty labels in {labels_dir})")
    print("Spot-check the images before retraining - see this script's docstring.")


if __name__ == "__main__":
    main()
