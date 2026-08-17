"""Find candidate bounce frames for manual labeling.

Runs ball detection+tracking over a video and does a broad, low-threshold
local-max-in-y scan (src.analysis.bounce_detector.find_trajectory_breakpoints,
--min-y-prominence near 0 by default) to catch essentially every
direction-change candidate - real bounces AND contacts AND borderline noise
- since precision doesn't matter here, a human reviews every candidate
next. This deliberately includes contacts/noise as candidates: they make
the best NEGATIVE training examples for scripts/train_bounce_classifier.py,
since they're exactly the cases src.analysis.bounce_detector's fixed
thresholds have the hardest time telling apart from a real bounce.

    python scripts/extract_bounce_candidates.py --input match.mp4 \
        --out outputs/bounce_candidates.csv --thumbnails-out outputs/bounce_candidates/

Writes one CSV row per candidate (feature columns already computed via
src.analysis.bounce_features.extract_features, so
scripts/train_bounce_classifier.py doesn't need to re-run ball detection;
also includes a `window_xy` column - see extract_window_sequence - for
scripts/train_bounce_lstm.py) with an empty `label` column - open the
thumbnails in --thumbnails-out, fill in `label` with "bounce" or
"not_bounce" for each row you can confidently identify, leave ambiguous
rows blank, and save as e.g. outputs/bounce_candidates_labeled.csv.

When labeling, deliberately favor far-court candidates over near-court ones
where you can - see notebooks/train_bounce_lstm_colab.ipynb's docstring for
why (short version: a fixed-pixel bounce dip is much smaller far from the
camera, and the LSTM needs real examples at that scale to recognize it as
a bounce rather than noise).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.bounce_detector import _window_around, find_trajectory_breakpoints  # noqa: E402
from src.analysis.bounce_features import FEATURE_NAMES, extract_features, extract_window_sequence  # noqa: E402
from src.detection.ball_detector import BallDetector  # noqa: E402
from src.tracking.ball_tracker import BallTracker  # noqa: E402
from src.video.io import VideoReader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Match video to scan")
    parser.add_argument("--ball-weights", default=str(REPO_ROOT / "weights" / "ball_detector.pt"))
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--min-y-prominence",
        type=float,
        default=1.0,
        help="Kept deliberately low - see module docstring for why weak candidates matter here",
    )
    parser.add_argument("--min-gap", type=int, default=5)
    parser.add_argument(
        "--window", type=int, default=9, help="Frames of context saved/featurized per candidate (odd number)"
    )
    parser.add_argument("--out", default="outputs/bounce_candidates.csv")
    parser.add_argument("--thumbnails-out", default="outputs/bounce_candidates")
    args = parser.parse_args()

    detector = BallDetector(
        args.ball_weights, confidence=args.confidence, imgsz=args.imgsz, device=args.device
    )
    tracker = BallTracker()

    reader = VideoReader(args.input)
    frames = list(reader.frames())
    if not frames:
        raise SystemExit(f"No frames read from {args.input}")

    print(f"Running ball detection on {len(frames)} frames...")
    detections = [detector.detect(f) for f in frames]
    positions = tracker.track(detections)
    ordered = sorted(positions, key=lambda p: p.frame_idx)
    index_by_frame = {p.frame_idx: i for i, p in enumerate(ordered)}

    candidate_frames = find_trajectory_breakpoints(
        positions, min_y_prominence=args.min_y_prominence, min_frame_gap=args.min_gap
    )
    print(f"Found {len(candidate_frames)} candidate frames")
    if not candidate_frames:
        print("No candidates found - check ball detection is working for this video.")
        return

    half_window = args.window // 2
    thumbnails_dir = Path(args.thumbnails_out)
    thumbnails_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video_name = Path(args.input).name
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
    rows = []
    skipped = 0
    for candidate_id, center_frame in enumerate(candidate_frames):
        center_idx = index_by_frame[center_frame]
        window_start_idx = max(0, center_idx - half_window)
        window_end_idx = min(len(ordered) - 1, center_idx + half_window)
        window = ordered[window_start_idx : window_end_idx + 1]
        local_center_idx = center_idx - window_start_idx

        if local_center_idx <= 0 or local_center_idx >= len(window) - 1:
            skipped += 1  # candidate sits right at the tracked trajectory's edge - no frame on one side
            continue

        features = extract_features(window, local_center_idx)
        ball_pos = ordered[center_idx]

        # window_xy uses REAL detections only (see _window_around), unlike
        # `window` above - a fixed-width slice would hand the LSTM a
        # fabricated straight-line segment across any detection gap, and
        # gaps often sit right at a real bounce (impact motion blur).
        real_windowed = _window_around(ordered, center_idx, args.window)
        window_xy = (
            json.dumps(extract_window_sequence(*real_windowed).tolist()) if real_windowed is not None else ""
        )

        thumb_path = thumbnails_dir / f"candidate_{candidate_id:03d}_frame{center_frame}.jpg"
        cv2.imwrite(str(thumb_path), frames[center_frame])

        rows.append(
            {
                "candidate_id": candidate_id,
                "video": video_name,
                "center_frame": center_frame,
                "start_frame": window[0].frame_idx,
                "end_frame": window[-1].frame_idx,
                "ball_x": ball_pos.x,
                "ball_y": ball_pos.y,
                **dict(zip(FEATURE_NAMES, features)),
                "window_xy": window_xy,
                "label": "",
            }
        )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if skipped:
        print(f"Skipped {skipped} candidate(s) too close to the tracked trajectory's edge")
    print(f"Wrote {len(rows)} candidates to {out_path}")
    print(f"Thumbnails written to {thumbnails_dir}")
    print(
        "Open the thumbnails, fill in 'label' with 'bounce' or 'not_bounce' for each row "
        "you can confidently identify, leave ambiguous rows blank, and save as e.g. "
        "outputs/bounce_candidates_labeled.csv."
    )


if __name__ == "__main__":
    main()
