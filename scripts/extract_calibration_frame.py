"""Save one reference frame from a match video for manual court calibration.

There's no automatic court-line detection in this repo, so calibrating
pixel-to-real-world coordinates for shot speed is a manual, one-time step per
camera angle: this script saves a still frame so you can open it in any
image viewer and read off the pixel (x, y) of four known court corners
(the baseline and service-line corners nearest the camera - most reliably
visible in a broadcast frame), then feed those into `scripts/calibrate_court.py`.

    python scripts/extract_calibration_frame.py --input match.mp4 \
        --frame-idx 0 --out outputs/calibration_frame.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.video.io import VideoReader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Match video to grab a frame from")
    parser.add_argument("--frame-idx", type=int, default=0, help="Which frame to save")
    parser.add_argument("--out", default="outputs/calibration_frame.jpg")
    args = parser.parse_args()

    reader = VideoReader(args.input)
    frame = None
    for i, candidate in enumerate(reader.frames()):
        if i == args.frame_idx:
            frame = candidate
            break
    if frame is None:
        raise SystemExit(f"Video has fewer than {args.frame_idx + 1} frames")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"Wrote {out_path}")
    print(
        "Open it in an image viewer and read off the pixel (x, y) of the four "
        "court corners nearest the camera: baseline-left, baseline-right, "
        "service-left, service-right. Then run scripts/calibrate_court.py "
        "with those coordinates."
    )


if __name__ == "__main__":
    main()
