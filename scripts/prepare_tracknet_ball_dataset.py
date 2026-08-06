"""Convert the TrackNet tennis-ball dataset into this repo's YOLO ball-detection format.

Source: yastrebksv/TrackNet's training data (2017 Summer Universiade match +
other broadcast clips), mirrored on Kaggle as sofuskonglevoll/tracknet-tennis
since the original torrent distribution (hyper.ai) had too few seeders to be
reachable. Downloaded via `kaggle datasets download sofuskonglevoll/tracknet-tennis`.

Layout is `Dataset/game*/Clip*/{NNNN.jpg, Label.csv}`, where Label.csv has
columns `file name,visibility,x-coordinate,y-coordinate,status`. visibility=0
means the ball isn't visible in that frame (x/y are blank) - these become
empty-label negatives, the same convention scripts/add_hard_negatives.py
uses. Any other visibility value means the ball is present at (x, y).

Frames/labels are merged straight into data/raw/train/{images,labels} -
matching how hard negatives were added earlier - so the existing Roboflow
valid/test splits stay untouched as a stable evaluation baseline. Both
datasets are 1280x720, so the fixed ball box size is copied directly from
the existing Roboflow labels' pixel dimensions (16x19px) rather than
re-derived, keeping the two label sources visually consistent.

    python scripts/prepare_tracknet_ball_dataset.py
"""

import argparse
import csv
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BOX_W_PX = 16
BOX_H_PX = 19
IMG_W = 1280
IMG_H = 720


def convert_clip(csv_path: Path, out_images: Path, out_labels: Path, prefix: str) -> tuple[int, int]:
    clip_dir = csv_path.parent
    written = 0
    negatives = 0
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            src_image = clip_dir / row["file name"]
            if not src_image.exists():
                continue
            stem = f"{prefix}_{src_image.stem}"
            dst_image = out_images / f"{stem}.jpg"
            dst_label = out_labels / f"{stem}.txt"
            shutil.copy(src_image, dst_image)

            if row["visibility"] == "0" or not row["x-coordinate"]:
                dst_label.write_text("")
                negatives += 1
            else:
                x, y = float(row["x-coordinate"]), float(row["y-coordinate"])
                cx, cy = x / IMG_W, y / IMG_H
                bw, bh = BOX_W_PX / IMG_W, BOX_H_PX / IMG_H
                dst_label.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            written += 1
    return written, negatives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(REPO_ROOT / "data" / "raw_tracknet_source_kaggle" / "Dataset"),
    )
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "raw" / "train"))
    args = parser.parse_args()

    source = Path(args.source)
    out_images = Path(args.out) / "images"
    out_labels = Path(args.out) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_negatives = 0
    clip_csvs = sorted(source.glob("game*/Clip*/Label.csv"))
    for csv_path in clip_csvs:
        game = csv_path.parent.parent.name
        clip = csv_path.parent.name
        prefix = f"tracknet_{game}_{clip}"
        written, negatives = convert_clip(csv_path, out_images, out_labels, prefix)
        total_written += written
        total_negatives += negatives

    print(f"Converted {len(clip_csvs)} clips -> {total_written} frames "
          f"({total_written - total_negatives} with a ball, {total_negatives} empty) "
          f"in {out_images}")


if __name__ == "__main__":
    main()
