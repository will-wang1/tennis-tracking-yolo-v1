"""Turn the TrackNet dataset's own bounce/hit annotations into training rows
for scripts/train_bounce_classifier.py.

Same source already used for the ball detector retrain (see
scripts/prepare_tracknet_ball_dataset.py) - `Label.csv`'s `status` column,
unused until now, marks each frame 0 (normal flight), 1 (racket hit/contact),
or 2 (ball bounce). Across all 95 clips that's ~535 real, human-labeled
bounces and ~548 real contacts - both directly usable, and the contacts are
exactly the hard negative case (a real y-direction reversal that ISN'T a
bounce) src.analysis.bounce_features is meant to discriminate. Far larger
and more camera/court-diverse than our own two clips or even
scripts/import_reference_bounce_dataset.py's ~150 rows.

Output matches scripts/extract_bounce_candidates.py's CSV schema (feature
columns already computed via src.analysis.bounce_features, video column set
to "tracknet_<game>_<clip>") so scripts/train_bounce_classifier.py can just
concatenate this with our other labeled CSVs.

    python scripts/import_tracknet_bounce_dataset.py \
        --source data/raw_tracknet_source_kaggle/Dataset \
        --out outputs/tracknet_bounce_labeled.csv
"""

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.bounce_features import FEATURE_NAMES, extract_features  # noqa: E402
from src.tracking.ball_tracker import TrackedPosition  # noqa: E402

STATUS_BOUNCE = "2"
STATUS_HIT = "1"


def load_clip_positions(csv_path: Path) -> list[TrackedPosition]:
    """Only VISIBLE rows become a TrackedPosition - visibility=0 rows have
    no real x/y and would corrupt window shape. frame_idx is kept as the
    original row index (informational only - extract_features works purely
    off list order/values, not frame_idx), so a run of invisible frames
    shows up as a gap in frame_idx between consecutive list entries."""
    positions = []
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if row["visibility"] == "0" or not row["x-coordinate"]:
                continue
            positions.append(TrackedPosition(frame_idx=i, x=float(row["x-coordinate"]), y=float(row["y-coordinate"]), interpolated=False))
    return positions


def load_clip_statuses(csv_path: Path) -> dict[int, str]:
    statuses = {}
    with open(csv_path, newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            statuses[i] = row["status"]
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default=str(REPO_ROOT / "data" / "raw_tracknet_source_kaggle" / "Dataset")
    )
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument(
        "--max-frame-span",
        type=int,
        default=None,
        help="Skip a candidate if its window's original frame indices span more than this many "
        "frames (default: 3x --window) - guards against a window quietly stitching together "
        "positions separated by a long invisible-ball gap",
    )
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "tracknet_bounce_labeled.csv"))
    args = parser.parse_args()
    max_frame_span = args.max_frame_span or args.window * 3

    fieldnames = [
        "candidate_id",
        "video",
        "center_frame",
        "start_frame",
        "end_frame",
        "ball_x",
        "ball_y",
        *FEATURE_NAMES,
        "label",
    ]
    out_rows = []
    skipped = 0
    half_window = args.window // 2

    clip_csvs = sorted(Path(args.source).glob("game*/Clip*/Label.csv"))
    print(f"Found {len(clip_csvs)} clips")
    for csv_path in clip_csvs:
        clip_name = f"tracknet_{csv_path.parent.parent.name}_{csv_path.parent.name}"
        positions = load_clip_positions(csv_path)
        statuses = load_clip_statuses(csv_path)
        index_by_frame = {p.frame_idx: i for i, p in enumerate(positions)}

        for frame_idx, status in statuses.items():
            if status == STATUS_BOUNCE:
                label = "bounce"
            elif status == STATUS_HIT:
                label = "not_bounce"
            else:
                continue

            center_idx = index_by_frame.get(frame_idx)
            if center_idx is None:
                skipped += 1  # the labeled frame itself wasn't visible
                continue

            start_idx = max(0, center_idx - half_window)
            end_idx = min(len(positions) - 1, center_idx + half_window)
            window = positions[start_idx : end_idx + 1]
            local_center_idx = center_idx - start_idx

            if local_center_idx <= 0 or local_center_idx >= len(window) - 1:
                skipped += 1
                continue
            if window[-1].frame_idx - window[0].frame_idx > max_frame_span:
                skipped += 1  # too much invisible-ball gap stitched into one window
                continue

            features = extract_features(window, local_center_idx)
            ball_pos = positions[center_idx]
            out_rows.append(
                {
                    "candidate_id": len(out_rows),
                    "video": clip_name,
                    "center_frame": ball_pos.frame_idx,
                    "start_frame": window[0].frame_idx,
                    "end_frame": window[-1].frame_idx,
                    "ball_x": ball_pos.x,
                    "ball_y": ball_pos.y,
                    **dict(zip(FEATURE_NAMES, features)),
                    "label": label,
                }
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    bounce_count = sum(1 for r in out_rows if r["label"] == "bounce")
    print(f"Skipped {skipped} candidate(s) (invisible at the event, at a trajectory edge, or too wide a gap)")
    print(f"Wrote {len(out_rows)} rows ({bounce_count} bounce, {len(out_rows) - bounce_count} not_bounce) to {out_path}")


if __name__ == "__main__":
    main()
