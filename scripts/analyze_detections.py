"""Find where a detector's false positives are coming from.

Runs the ball detector over every frame of a video (bypassing the tracker's
outlier/lock-on filtering entirely) and writes:
  - a CSV of every raw detection (frame, x, y, confidence)
  - a heatmap image showing where detections cluster on screen

A real ball's detections should be spread across the court, roughly tracing
where play happened. A tight, bright hotspot sitting in one fixed screen
location - the net, a sideline, a graphic - is a specific object the model
keeps confusing for the ball. Crop that spot out of a few frames, label it
correctly (bounding box on the real ball if present, otherwise leave
unlabeled as a true negative), add those frames to data/raw, and retrain.
That fixes the actual cause; tracker thresholds can only paper over it.

    python scripts/analyze_detections.py --input path/to/match.mp4
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.detection.ball_detector import BallDetector  # noqa: E402
from src.detection.heatmap import accumulate_heatmap  # noqa: E402
from src.video.io import VideoReader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Video to analyze")
    parser.add_argument(
        "--weights", default=str(REPO_ROOT / "weights" / "ball_detector.pt")
    )
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--csv-out", default="outputs/detections.csv")
    parser.add_argument("--heatmap-out", default="outputs/detection_heatmap.png")
    args = parser.parse_args()

    detector = BallDetector(
        args.weights, confidence=args.confidence, imgsz=args.imgsz, device=args.device
    )
    reader = VideoReader(args.input)
    frames = list(reader.frames())
    if not frames:
        raise SystemExit(f"No frames read from {args.input}")

    print(f"Running detector on {len(frames)} frames (no tracker filtering)...")
    rows = []
    for i, frame in enumerate(tqdm(frames)):
        det = detector.detect(frame)
        if det is not None:
            rows.append((i, det.x, det.y, det.confidence))

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "confidence"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} raw detections to {csv_path}")

    if not rows:
        print("No detections at all - nothing to plot. Try a lower --confidence.")
        return

    points = [(x, y) for _, x, y, _ in rows]
    weights = [conf for _, _, _, conf in rows]
    heat = accumulate_heatmap(points, weights, reader.width, reader.height)

    normalized = (255 * heat / heat.max()).astype(np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frames[0], 0.6, colored, 0.4, 0)

    heatmap_path = Path(args.heatmap_out)
    heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(heatmap_path), overlay)
    print(f"Wrote heatmap to {heatmap_path}")
    print(
        "Bright, tight spots in a fixed location = a specific object the "
        "model keeps confusing for the ball. A diffuse spread following the "
        "court = real ball detections."
    )


if __name__ == "__main__":
    main()
