"""Turn artLabss/tennis-tracking's raw labeled bounce trajectory into more
training rows for scripts/train_bounce_classifier.py.

That repo ships two things for bounce detection: a pretrained `clf.pkl`
(NOT used here - unpickling a model file from an untrusted repo can execute
arbitrary code on load) and `bigDF.csv`, their own raw per-frame ball
trajectory (x, y, V, bounce) with ~72 real, hand-labeled bounce events
across ~3674 frames - plain numeric data, safe to consume. Our own two
clips only produced a couple dozen labeled candidates total (see
scripts/extract_bounce_candidates.py), so this is a meaningfully larger,
real-bounce-labeled trajectory to learn from.

Every `bounce == 1` row becomes a positive candidate. Negative candidates
come from the SAME broad local-max-in-y scan
scripts/extract_bounce_candidates.py uses (bounce_detector.find_trajectory_breakpoints,
low threshold) over this trajectory, excluding anything within
--bounce-tolerance frames of a real labeled bounce - these are the
contacts/noise their own labeling implicitly marked as NOT a bounce, the
same kind of hard negative our own candidate extraction targets.

Output matches scripts/extract_bounce_candidates.py's CSV schema exactly
(feature columns already computed via src.analysis.bounce_features, video
column set to "bigDF_reference") so scripts/train_bounce_classifier.py can
just concatenate this with our own labeled CSVs without any special-casing.

    python scripts/import_reference_bounce_dataset.py \
        --source outputs/bigDF_reference.csv --out outputs/bigDF_reference_labeled.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.bounce_detector import _window_around, find_trajectory_breakpoints  # noqa: E402
from src.analysis.bounce_features import FEATURE_NAMES, extract_features, extract_window_sequence  # noqa: E402
from src.tracking.ball_tracker import TrackedPosition  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(REPO_ROOT / "outputs" / "bigDF_reference.csv"))
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--min-y-prominence", type=float, default=1.0)
    parser.add_argument("--min-gap", type=int, default=5)
    parser.add_argument(
        "--bounce-tolerance",
        type=int,
        default=3,
        help="A negative candidate within this many frames of a real labeled bounce is dropped "
        "rather than mislabeled",
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=200,
        help="Cap on negative candidates, so the reference set doesn't swamp our own clips' examples",
    )
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "bigDF_reference_labeled.csv"))
    args = parser.parse_args()

    with open(args.source, newline="") as f:
        rows = list(csv.DictReader(f))

    positions = [
        TrackedPosition(frame_idx=i, x=float(row["x"]), y=float(row["y"]), interpolated=False)
        for i, row in enumerate(rows)
    ]
    bounce_frames = {i for i, row in enumerate(rows) if row["bounce"] == "1"}
    print(f"Loaded {len(positions)} frames, {len(bounce_frames)} labeled real bounces")

    candidate_frames = find_trajectory_breakpoints(
        positions, min_y_prominence=args.min_y_prominence, min_frame_gap=args.min_gap
    )
    negative_frames = [
        f
        for f in candidate_frames
        if f not in bounce_frames and all(abs(f - b) > args.bounce_tolerance for b in bounce_frames)
    ]
    if len(negative_frames) > args.max_negatives:
        step = len(negative_frames) / args.max_negatives
        negative_frames = [negative_frames[int(i * step)] for i in range(args.max_negatives)]
    print(f"{len(candidate_frames)} local-max candidates -> {len(negative_frames)} usable negatives")

    half_window = args.window // 2
    fieldnames = [
        "candidate_id",
        "video",
        "center_frame",
        "start_frame",
        "end_frame",
        "ball_x",
        "ball_y",
        *FEATURE_NAMES,
        "window_xy",
        "label",
    ]
    out_rows = []
    skipped = 0
    labeled_frames = [(f, "bounce") for f in sorted(bounce_frames)] + [
        (f, "not_bounce") for f in negative_frames
    ]
    for candidate_id, (center_frame, label) in enumerate(labeled_frames):
        start = max(0, center_frame - half_window)
        end = min(len(positions) - 1, center_frame + half_window)
        window = positions[start : end + 1]
        local_center_idx = center_frame - start

        if local_center_idx <= 0 or local_center_idx >= len(window) - 1:
            skipped += 1
            continue

        features = extract_features(window, local_center_idx)
        ball_pos = positions[center_frame]
        real_windowed = _window_around(positions, center_frame, args.window)
        window_xy = (
            json.dumps(extract_window_sequence(*real_windowed).tolist()) if real_windowed is not None else ""
        )
        out_rows.append(
            {
                "candidate_id": candidate_id,
                "video": "bigDF_reference",
                "center_frame": center_frame,
                "start_frame": window[0].frame_idx,
                "end_frame": window[-1].frame_idx,
                "ball_x": ball_pos.x,
                "ball_y": ball_pos.y,
                **dict(zip(FEATURE_NAMES, features)),
                "window_xy": window_xy,
                "label": label,
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    if skipped:
        print(f"Skipped {skipped} candidate(s) too close to the trajectory's edge")
    bounce_count = sum(1 for r in out_rows if r["label"] == "bounce")
    print(f"Wrote {len(out_rows)} rows ({bounce_count} bounce, {len(out_rows) - bounce_count} not_bounce) to {out_path}")


if __name__ == "__main__":
    main()
