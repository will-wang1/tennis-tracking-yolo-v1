"""Find candidate stroke-contact frames for manual labeling.

Runs ball detection+tracking and pose detection over a video, picks whoever
is nearest the ball each frame as the striker (src/analysis/striker.py), and
flags frames where the striker's wrist is moving fastest as candidate
stroke-contact moments. This is a crude, high-recall heuristic - it will
flag some non-strokes and miss some real ones - but precision doesn't matter
here since a human reviews every candidate next.

    python scripts/extract_stroke_candidates.py --input match.mp4 \
        --ball-weights weights/ball_detector.pt --pose-weights yolov8n-pose.pt \
        --window 9 --out outputs/stroke_candidates.csv \
        --thumbnails-out outputs/stroke_candidates/

Writes one CSV row per candidate with an empty `label` column - open the
thumbnails in `--thumbnails-out`, fill in `label` for each row you can
confidently identify using one of the values in
src.analysis.stroke_classifier.STROKE_LABELS, leave ambiguous rows blank,
and save as e.g. outputs/stroke_candidates_labeled.csv for
scripts/train_stroke_classifier.py.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.striker import select_striker  # noqa: E402
from src.analysis.stroke_features import LEFT_WRIST, RIGHT_WRIST  # noqa: E402
from src.detection.ball_detector import BallDetector  # noqa: E402
from src.detection.pose_detector import PersonPose, PoseDetector  # noqa: E402
from src.tracking.ball_tracker import BallTracker  # noqa: E402
from src.video.io import VideoReader  # noqa: E402


def find_local_maxima(values: dict[int, float], min_gap: int) -> list[int]:
    """Frames where `values[frame]` is a local peak among consecutive
    frames present in the dict, merging peaks within `min_gap` frames of
    each other and keeping the taller one - same approach as
    bounce_detector's extrema merging."""
    frames = sorted(values)
    candidates = []
    for i in range(1, len(frames) - 1):
        f_prev, f_cur, f_next = frames[i - 1], frames[i], frames[i + 1]
        if f_cur - f_prev != 1 or f_next - f_cur != 1:
            continue  # only compare truly consecutive frames
        if values[f_cur] > values[f_prev] and values[f_cur] > values[f_next]:
            candidates.append(f_cur)

    merged: list[int] = []
    for f in candidates:
        if merged and f - merged[-1] <= min_gap:
            if values[f] > values[merged[-1]]:
                merged[-1] = f
        else:
            merged.append(f)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Match video to scan")
    parser.add_argument("--ball-weights", default=str(REPO_ROOT / "weights" / "ball_detector.pt"))
    parser.add_argument("--pose-weights", default="yolov8n-pose.pt")
    parser.add_argument("--confidence", type=float, default=0.15, help="Ball detector confidence")
    parser.add_argument("--pose-confidence", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--window", type=int, default=9, help="Frames of context saved per candidate (odd number)"
    )
    parser.add_argument("--min-gap", type=int, default=15, help="Minimum frames between two candidates")
    parser.add_argument("--out", default="outputs/stroke_candidates.csv")
    parser.add_argument("--thumbnails-out", default="outputs/stroke_candidates")
    args = parser.parse_args()

    ball_detector = BallDetector(
        args.ball_weights, confidence=args.confidence, imgsz=args.imgsz, device=args.device
    )
    pose_detector = PoseDetector(
        args.pose_weights, confidence=args.pose_confidence, imgsz=args.imgsz, device=args.device
    )
    tracker = BallTracker()

    reader = VideoReader(args.input)
    frames = list(reader.frames())
    if not frames:
        raise SystemExit(f"No frames read from {args.input}")

    print(f"Running ball + pose detection on {len(frames)} frames...")
    detections = []
    poses_by_frame: list[list[PersonPose]] = []
    for frame in tqdm(frames):
        detections.append(ball_detector.detect(frame))
        poses_by_frame.append(pose_detector.detect(frame))

    positions = tracker.track(detections)
    positions_by_frame = {p.frame_idx: p for p in positions}

    striker_by_frame: list[Optional[PersonPose]] = [
        select_striker(poses_by_frame[i], positions_by_frame.get(i)) for i in range(len(frames))
    ]

    wrist_speed_by_frame: dict[int, float] = {}
    for i in range(1, len(frames)):
        prev_pose, cur_pose = striker_by_frame[i - 1], striker_by_frame[i]
        if prev_pose is None or cur_pose is None:
            continue
        left_v = float(np.linalg.norm(cur_pose.keypoints[LEFT_WRIST] - prev_pose.keypoints[LEFT_WRIST]))
        right_v = float(np.linalg.norm(cur_pose.keypoints[RIGHT_WRIST] - prev_pose.keypoints[RIGHT_WRIST]))
        wrist_speed_by_frame[i] = max(left_v, right_v)

    candidate_frames = find_local_maxima(wrist_speed_by_frame, min_gap=args.min_gap)
    print(f"Found {len(candidate_frames)} candidate stroke frames")

    if not candidate_frames:
        print("No candidates found - try a lower --pose-confidence or check pose detection is working.")
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
        "striker_x",
        "striker_y",
        "wrist_speed",
        "label",
    ]
    rows = []
    for candidate_id, center_frame in enumerate(candidate_frames):
        start_frame = max(0, center_frame - half_window)
        end_frame = min(len(frames) - 1, center_frame + half_window)
        pose = striker_by_frame[center_frame]

        thumb_path = thumbnails_dir / f"candidate_{candidate_id:03d}_frame{center_frame}.jpg"
        cv2.imwrite(str(thumb_path), frames[center_frame])

        rows.append(
            {
                "candidate_id": candidate_id,
                "video": video_name,
                "center_frame": center_frame,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "striker_x": pose.center_x if pose else "",
                "striker_y": pose.center_y if pose else "",
                "wrist_speed": wrist_speed_by_frame.get(center_frame, 0.0),
                "label": "",
            }
        )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} candidates to {out_path}")
    print(f"Thumbnails written to {thumbnails_dir}")
    print(
        "Open the thumbnails, fill in 'label' for each row you can confidently "
        "identify (forehand/backhand/serve/volley/other), leave ambiguous rows "
        "blank, and save as e.g. outputs/stroke_candidates_labeled.csv."
    )


if __name__ == "__main__":
    main()
