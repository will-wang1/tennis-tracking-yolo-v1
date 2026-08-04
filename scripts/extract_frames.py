"""Pull frames out of a match video for manual labeling."""

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to source video")
    parser.add_argument("--out", required=True, help="Directory to write frames to")
    parser.add_argument("--every", type=int, default=5, help="Keep 1 out of every N frames")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    stem = Path(args.video).stem
    frame_idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.every == 0:
            cv2.imwrite(str(out_dir / f"{stem}_{frame_idx:06d}.jpg"), frame)
            saved += 1
        frame_idx += 1
    cap.release()

    print(f"Saved {saved} frames (of {frame_idx}) to {out_dir}")


if __name__ == "__main__":
    main()
