"""Visualize an auto-calibration for manual review: detected keypoints plus
the fitted homography's court wireframe reprojected back onto the frame.

Runs scripts/calibrate_court_auto.py's same detect + fit steps, then draws:
  - each detected keypoint as a labeled dot (green = used in the fit, i.e.
    above --min-keypoint-confidence; red = detected but below threshold)
  - the standard 14-point court layout's lines, reprojected from world
    meters back to pixels via the fitted homography's inverse - if the fit
    is good this wireframe should sit exactly on the court's real lines.

    python scripts/review_court_calibration.py --input "tennis point.mp4" \
        --keypoint-weights weights/court_keypoint_detector.pt \
        --out outputs/court_calibration_review.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.court_calibration import CourtCalibration, FULL_COURT_REFERENCE_POINTS  # noqa: E402
from src.detection.court_keypoint_detector import CourtKeypointDetector  # noqa: E402
from src.video.io import VideoReader  # noqa: E402

# Court-line connectivity for the wireframe overlay, by keypoint name.
COURT_EDGES = [
    ("baseline_far_left", "baseline_far_right"),
    ("baseline_near_left", "baseline_near_right"),
    ("baseline_far_left", "baseline_near_left"),
    ("baseline_far_right", "baseline_near_right"),
    ("singles_far_left", "singles_near_left"),
    ("singles_far_right", "singles_near_right"),
    ("service_far_left", "service_far_right"),
    ("service_near_left", "service_near_right"),
    ("center_service_far", "center_service_near"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--frame-idx", type=int, default=0)
    parser.add_argument("--keypoint-weights", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--min-keypoint-confidence", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-calibration", default=None, help="Optionally also write the fitted calibration JSON here")
    parser.add_argument("--out", default="outputs/court_calibration_review.jpg")
    args = parser.parse_args()

    reader = VideoReader(args.input)
    frame = None
    for i, candidate in enumerate(reader.frames()):
        if i == args.frame_idx:
            frame = candidate.copy()
            break
    if frame is None:
        raise SystemExit(f"Video has fewer than {args.frame_idx + 1} frames")

    detector = CourtKeypointDetector(
        args.keypoint_weights, confidence=args.confidence, imgsz=args.imgsz, device=args.device
    )
    detection = detector.detect(frame)
    if detection is None:
        raise SystemExit("No court detected in this frame")

    all_points = detection.as_named_points(min_confidence=0.0)
    confident_points = detection.as_named_points(min_confidence=args.min_keypoint_confidence)
    print(f"Detected {len(all_points)}/14 keypoints, {len(confident_points)} above "
          f"{args.min_keypoint_confidence} confidence")

    for name, (x, y) in all_points.items():
        used = name in confident_points
        color = (0, 220, 0) if used else (0, 0, 220)
        cv2.circle(frame, (int(x), int(y)), 6, color, -1)
        cv2.putText(frame, name, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    calibration = CourtCalibration.from_keypoints(confident_points)
    if args.save_calibration:
        calibration.save(args.save_calibration)
        print(f"Wrote calibration to {args.save_calibration}")

    inverse_homography = np.linalg.inv(calibration.homography)
    projected_px = {}
    for name, (wx, wy) in FULL_COURT_REFERENCE_POINTS.items():
        point = np.array([[[wx, wy]]], dtype=np.float64)
        px, py = cv2.perspectiveTransform(point, inverse_homography)[0, 0]
        projected_px[name] = (px, py)

    for a, b in COURT_EDGES:
        pa, pb = projected_px[a], projected_px[b]
        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 200, 255), 2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"Wrote review image to {out_path}")
    print("Green dots = keypoints used in the fit, red = detected but below "
          "confidence threshold, orange wireframe = fitted court reprojected "
          "back from world coordinates - it should sit exactly on the real lines.")


if __name__ == "__main__":
    main()
