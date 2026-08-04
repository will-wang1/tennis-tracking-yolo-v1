"""Run the fine-tuned YOLO ball detector + tracker over a video.

    python main.py --input path/to/match.mp4 --output outputs/tracked.mp4
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from src.detection.ball_detector import BallDetector
from src.tracking.ball_tracker import BallTracker
from src.video.io import VideoReader, VideoWriter
from src.visualize.draw import TrailDrawer

REPO_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input match video")
    parser.add_argument("--output", required=True, help="Output annotated video")
    parser.add_argument(
        "--weights",
        default=str(REPO_ROOT / "weights" / "ball_detector.pt"),
        help="Fine-tuned ball detector checkpoint",
    )
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--trail-length", type=int, default=15)
    args = parser.parse_args()

    detector = BallDetector(
        args.weights, confidence=args.confidence, imgsz=args.imgsz, device=args.device
    )
    tracker = BallTracker()

    reader = VideoReader(args.input)
    frames = list(reader.frames())
    if not frames:
        raise SystemExit(f"No frames read from {args.input}")

    print(f"Detecting ball in {len(frames)} frames...")
    detections = [detector.detect(frame) for frame in tqdm(frames)]

    positions = tracker.track(detections)
    positions_by_frame = {p.frame_idx: p for p in positions}
    detected = sum(1 for d in detections if d is not None)
    interpolated = sum(1 for p in positions if p.interpolated)
    print(
        f"Raw detections: {detected}/{len(frames)} frames "
        f"({detected / len(frames):.1%}). "
        f"After interpolation: {len(positions)}/{len(frames)} frames have a "
        f"tracked position ({interpolated} interpolated)."
    )

    writer = VideoWriter(args.output, reader.fps, reader.width, reader.height)
    trail = TrailDrawer(trail_length=args.trail_length)
    for i, frame in enumerate(frames):
        annotated = trail.draw(frame, positions_by_frame.get(i))
        writer.write(annotated)
    writer.close()

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
