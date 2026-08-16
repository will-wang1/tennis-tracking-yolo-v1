"""Diagnostic: how many frames get a genuine per-frame court calibration vs.
a carried-forward one, at various confidence settings - to quantify the gap
before deciding how to close it (retrain vs. threshold tuning vs. TTA).

    python scripts/diagnose_court_calibration.py hardcourt1.mp4 grasscourt1.mp4
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.detection.court_keypoint_detector import CourtKeypointDetector
from src.video.io import VideoReader


def diagnose(video_path: str, detector: CourtKeypointDetector, min_kp_conf: float):
    reader = VideoReader(video_path)
    frames = list(reader.frames())

    n_genuine = 0
    n_miss = 0
    per_frame_counts = []
    low_conf_frames = []  # frames with 1-3 confident keypoints (near-miss)
    zero_frames = []  # frames with no detection at all

    for i, frame in enumerate(frames):
        detection = detector.detect(frame)
        if detection is None:
            n_miss += 1
            per_frame_counts.append(0)
            zero_frames.append(i)
            continue
        named = detection.as_named_points(min_confidence=min_kp_conf)
        per_frame_counts.append(len(named))
        if len(named) >= 4:
            n_genuine += 1
        else:
            n_miss += 1
            low_conf_frames.append((i, len(named), round(float(detection.bbox_confidence), 3)))

    total = len(frames)
    print(f"\n=== {video_path} (min_kp_conf={min_kp_conf}) ===")
    print(f"Total frames: {total}")
    print(f"Genuine per-frame calibration (>=4 confident keypoints): {n_genuine}/{total} ({n_genuine/total:.1%})")
    print(f"Carried-forward / gap frames: {n_miss}/{total} ({n_miss/total:.1%})")
    if zero_frames:
        print(f"  - {len(zero_frames)} frames had NO court detection at all: {zero_frames[:20]}{'...' if len(zero_frames) > 20 else ''}")
    near_miss = [f for f in low_conf_frames if f[1] > 0]
    if near_miss:
        print(f"  - {len(near_miss)} frames detected the court but <4 confident keypoints (near-misses, good hard-example candidates):")
        for idx, count, bbox_conf in near_miss[:20]:
            print(f"      frame {idx}: {count} confident keypoints, bbox_conf={bbox_conf}")
        if len(near_miss) > 20:
            print(f"      ... and {len(near_miss) - 20} more")

    # find contiguous miss runs (these are what actually get carried forward)
    runs = []
    run_start = None
    for i, c in enumerate(per_frame_counts):
        genuine = c >= 4
        if not genuine and run_start is None:
            run_start = i
        elif genuine and run_start is not None:
            runs.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, total - 1))
    if runs:
        longest = max(runs, key=lambda r: r[1] - r[0] + 1)
        print(f"  - {len(runs)} contiguous miss-runs; longest: frames {longest[0]}-{longest[1]} ({longest[1]-longest[0]+1} frames)")

    return per_frame_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--court-keypoint-weights", default=str(REPO_ROOT / "weights" / "court_keypoint_detector.pt"))
    parser.add_argument("--confidence", type=float, default=0.25, help="box confidence (matches main.py default)")
    parser.add_argument("--min-kp-conf", type=float, nargs="+", default=[0.5, 0.3, 0.1])
    args = parser.parse_args()

    detector = CourtKeypointDetector(args.court_keypoint_weights, confidence=args.confidence, imgsz=1280)

    for video in args.videos:
        for min_kp_conf in args.min_kp_conf:
            diagnose(video, detector, min_kp_conf)


if __name__ == "__main__":
    main()
