"""Automatically calibrate a court from a trained keypoint detector - no manual pixel-reading.

Replaces the scripts/extract_calibration_frame.py + scripts/calibrate_court.py
manual workflow once you have a trained court-keypoint checkpoint (see
scripts/train_court_keypoints.py). Detects however many of the 14 standard
keypoints are confidently visible in one frame and fits a homography from
all of them (more robust than the manual workflow's exact 4 points).

    python scripts/calibrate_court_auto.py --input match.mp4 \
        --keypoint-weights runs/court_keypoint_detector/weights/best.pt \
        --out configs/court_calibration.json
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.court_calibration import CourtCalibration  # noqa: E402
from src.detection.court_keypoint_detector import CourtKeypointDetector  # noqa: E402
from src.video.io import VideoReader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Match video to calibrate against")
    parser.add_argument("--frame-idx", type=int, default=0, help="Which frame to detect keypoints in")
    parser.add_argument("--keypoint-weights", required=True, help="Trained court-keypoint checkpoint")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--min-keypoint-confidence",
        type=float,
        default=0.5,
        help="Individual keypoints below this are excluded from the homography fit, even if "
        "the overall court detection passed --confidence",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="configs/court_calibration.json")
    args = parser.parse_args()

    reader = VideoReader(args.input)
    frame = None
    for i, candidate in enumerate(reader.frames()):
        if i == args.frame_idx:
            frame = candidate
            break
    if frame is None:
        raise SystemExit(f"Video has fewer than {args.frame_idx + 1} frames")

    detector = CourtKeypointDetector(
        args.keypoint_weights, confidence=args.confidence, imgsz=args.imgsz, device=args.device
    )
    detection = detector.detect(frame)
    if detection is None:
        raise SystemExit(
            "No court detected in this frame - try a different --frame-idx or lower --confidence"
        )

    named_points = detection.as_named_points(min_confidence=args.min_keypoint_confidence)
    print(f"Detected {len(named_points)}/14 keypoints above {args.min_keypoint_confidence} confidence:")
    for name, (x, y) in named_points.items():
        print(f"  {name}: ({x:.0f}, {y:.0f})")

    if len(named_points) < 4:
        raise SystemExit(
            f"Only {len(named_points)} confident keypoints - need at least 4. Try a clearer "
            "frame, lower --min-keypoint-confidence, or fall back to the manual "
            "scripts/calibrate_court.py workflow."
        )

    calibration = CourtCalibration.from_keypoints(named_points)
    calibration.save(args.out)

    center_px = sum(p[0] for p in named_points.values()) / len(named_points), sum(
        p[1] for p in named_points.values()
    ) / len(named_points)
    nearby_px = (center_px[0] + 10, center_px[1])
    meters_per_10px = calibration.pixel_distance_to_meters(*center_px, *nearby_px)
    print(f"Wrote calibration to {args.out}")
    print(f"Sanity check: ~{meters_per_10px / 10:.4f} m/px near the detected keypoints' centroid")


if __name__ == "__main__":
    main()
