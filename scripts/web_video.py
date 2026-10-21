"""Re-encode a video as H.264 so a browser can play it.

The pipeline's older renders are MPEG-4 Part 2 (OpenCV's mp4v), which
browsers will not play. The web app's session page needs H.264:

    python scripts/web_video.py outputs/<name>/annotated_outcomes.mp4

writes outputs/<name>/annotated_outcomes_web.mp4 (or --output).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    out = args.output or args.input.with_name(args.input.stem + "_web.mp4")
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise SystemExit(f"Could not read {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
    if not writer.isOpened():
        raise SystemExit("This OpenCV build cannot write H.264")
    frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        frames += 1
    writer.release()
    print(f"Wrote {out} ({frames} frames, H.264)")


if __name__ == "__main__":
    main()
