"""Find frames where the ball detector missed entirely, for manual re-labeling.

Runs ball detection+tracking over a video and flags every frame with NO
detection (not even a low-confidence one - see BallDetector's module
docstring, confidence is already tuned low) that's still bracketed by real
detections close enough for the tracker to interpolate through (i.e. the
ball plausibly WAS on screen, just not found). A miss with no nearby
tracked position at all (a long dead zone, or before/after the ball ever
appears) is skipped - there's no anchor to even guess where to look.

For each candidate, saves a cropped patch centered on the tracker's
interpolated (x, y) - a linear guess, not a real detection, but close
enough to frame the right neighborhood for a human to spot the actual ball
and read off its true position, which linear interpolation can't recover
on its own (a real bounce/direction-change gets flattened into a straight
line). This is exactly the targeted hard-example mining
scripts/prepare_tracknet_ball_dataset.py's bulk approach can't do: it only
adds MORE frames, not frames where THIS detector specifically fails.

    python scripts/extract_ball_miss_candidates.py --input match.mp4 \
        --out outputs/ball_misses.csv --crops-out outputs/ball_misses/

Writes one CSV row per miss with empty `true_x`/`true_y` columns (in FULL
FRAME pixel coordinates, not crop-relative) - open the crops, read off
where the ball actually is (crop is centered on `interp_x`/`interp_y`, so
`true_x = interp_x + (pixel offset from crop center)`), fill in `true_x`/
`true_y` for every crop where the ball is identifiable, leave rows where
it's genuinely not visible (occluded, off-frame, motion-blurred beyond
recognition) blank, and save as e.g. outputs/ball_misses_labeled.csv for
scripts/add_hard_positives.py.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.detection.ball_detector import BallDetector  # noqa: E402
from src.tracking.ball_tracker import BallTracker  # noqa: E402
from src.video.io import VideoReader  # noqa: E402

CROP_SIZE = 220


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--ball-weights", default=str(REPO_ROOT / "weights" / "ball_detector.pt"))
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--crop-size", type=int, default=CROP_SIZE)
    parser.add_argument("--out", default="outputs/ball_misses.csv")
    parser.add_argument("--crops-out", default="outputs/ball_misses")
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
    positions_by_frame = {p.frame_idx: p for p in positions}

    miss_frames = [i for i, d in enumerate(detections) if d is None]
    print(f"{len(miss_frames)}/{len(frames)} frames had no detection at all")

    crops_dir = Path(args.crops_out)
    crops_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    video_name = Path(args.input).name
    half = args.crop_size // 2
    height, width = frames[0].shape[:2]
    fieldnames = ["candidate_id", "video", "frame", "interp_x", "interp_y", "true_x", "true_y", "label"]
    rows = []
    skipped = 0

    for frame_idx in miss_frames:
        tracked = positions_by_frame.get(frame_idx)
        if tracked is None or not tracked.interpolated:
            skipped += 1  # no anchor to guess where to even look
            continue

        cx, cy = int(tracked.x), int(tracked.y)
        x1, y1 = max(0, cx - half), max(0, cy - half)
        x2, y2 = min(width, cx + half), min(height, cy + half)
        crop = frames[frame_idx][y1:y2, x1:x2]

        crop_path = crops_dir / f"miss_{len(rows):03d}_frame{frame_idx}.jpg"
        cv2.imwrite(str(crop_path), crop)

        rows.append(
            {
                "candidate_id": len(rows),
                "video": video_name,
                "frame": frame_idx,
                "interp_x": tracked.x,
                "interp_y": tracked.y,
                "true_x": "",
                "true_y": "",
                "label": "",
            }
        )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if skipped:
        print(f"Skipped {skipped} miss(es) with no nearby tracked position to anchor a crop on")
    print(f"Wrote {len(rows)} candidates to {out_path}")
    print(f"Crops written to {crops_dir} (each {args.crop_size}x{args.crop_size}, centered on the "
          f"interpolated position)")
    print(
        "Open each crop, read off the ball's true pixel position in the FULL FRAME "
        "(true_x = interp_x + offset-from-crop-center-x, same for y), fill in true_x/true_y, "
        "leave label as 'visible' if you set true_x/true_y or 'not_visible' if the ball truly "
        "isn't identifiable in this crop, and save as outputs/ball_misses_labeled.csv."
    )


if __name__ == "__main__":
    main()
