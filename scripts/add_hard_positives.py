"""Stage manually-relabeled detector MISSES as training positives.

Consumes a CSV produced by scripts/extract_ball_miss_candidates.py and then
hand-labeled (see that script's docstring) - rows with `label == "visible"`
have a human-confirmed `true_x`/`true_y` for a frame the detector missed
entirely. Writes those frames into a dataset split with a real YOLO label
box at the confirmed position, same fixed box size convention as
scripts/prepare_tracknet_ball_dataset.py (16x19px, copied from the existing
Roboflow labels' pixel dimensions rather than re-derived).

This is the targeted counterpart to scripts/prepare_tracknet_ball_dataset.py's
bulk approach: every frame here is one this exact detector already failed
on, not just more generic footage - see scripts/extract_ball_miss_candidates.py's
module docstring for why that matters more than volume alone.

    python scripts/add_hard_positives.py --input hardcourt1.mp4 \
        --labels outputs/hardcourt/ball_misses_labeled.csv \
        --out data/raw/train --prefix hardcourt_miss
"""

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.video.io import VideoReader  # noqa: E402

BOX_W_PX = 16
BOX_H_PX = 19


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source video the CSV was generated from")
    parser.add_argument("--labels", required=True, help="Labeled CSV from extract_ball_miss_candidates.py")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "raw" / "train"))
    parser.add_argument("--prefix", default="hard_positive", help="Filename prefix")
    args = parser.parse_args()

    df = pd.read_csv(args.labels)
    df = df[df["label"] == "visible"]
    if df.empty:
        raise SystemExit("No rows labeled 'visible' with a true_x/true_y - nothing to add")

    reader = VideoReader(args.input)
    frames = list(reader.frames())
    height, width = frames[0].shape[:2]

    images_dir = Path(args.out) / "images"
    labels_dir = Path(args.out) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for _, row in df.iterrows():
        frame_idx = int(row["frame"])
        if frame_idx >= len(frames):
            continue
        stem = f"{args.prefix}_{frame_idx:05d}"
        cv2.imwrite(str(images_dir / f"{stem}.jpg"), frames[frame_idx])

        cx, cy = float(row["true_x"]), float(row["true_y"])
        norm_cx, norm_cy = cx / width, cy / height
        norm_w, norm_h = BOX_W_PX / width, BOX_H_PX / height
        (labels_dir / f"{stem}.txt").write_text(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        written += 1

    print(f"Wrote {written} hard-positive frames to {images_dir}")


if __name__ == "__main__":
    main()
