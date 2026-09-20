"""Cut a clip out of a source video and normalise it to the frame rate and
frame size every threshold in this pipeline was tuned at.

Two normalisations, both for the same reason - a threshold that is a raw
COUNT means different things on different footage, so the alternative to
normalising the video is retuning a dozen constants per clip and never
being able to compare two clips again.

FRAME RATE -> 25fps. Many thresholds count frames, not seconds:
`static_lockon_frames` (10), `max_interpolation_gap` (8),
`flight_segmenter.max_separation_frames` (25),
`parabolic_bounce_detector.arc_frames` (8),
`speed_estimator.min_flight_frames` (12). Feed 60fps footage in and every
one of them describes a window half as long in real time. This project's
existing clips were all normalised to 25fps for exactly this reason.

FRAME SIZE -> 1920x1080. A smaller set of thresholds are raw PIXEL
distances in SOURCE frame coordinates: `max_pixels_per_frame` (150),
`static_lockon_radius` (20), `touchdown_detector._DEFAULT_FRAME_EDGE_MARGIN`
(150). At 1280x720 the same physical motion covers two thirds as many
pixels, so `max_pixels_per_frame=150` is ~1.5x looser than intended - it
stops rejecting the detector jumps it exists to reject.

Upscaling rather than rescaling the thresholds is deliberate for a FIRST
comparison against this project's match clips: it changes one variable
(the footage) instead of two (the footage and a dozen constants), so a
difference in the diagnostics can actually be attributed. It costs no
detector accuracy either - `WASBBallDetector` resizes internally to
512x288 regardless, so 720->1080->512 and 720->512 hand the model the same
information. Upscaling invents no detail; it only restores the pixel SCALE
the thresholds assume.

    python scripts/normalise_clip.py \\
        --input ~/Downloads/session.mp4 --start 12 --end 120 \\
        --output data/videos/session_drill.mp4

NOTE: audio is dropped - OpenCV cannot carry it. Keep the source file.
Coach talk ratio is one of the more promising coaching-quality signals and
it lives only in the original.
"""

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TARGET_FPS = 25.0
TARGET_SIZE = (1920, 1080)


def normalise(
    input_path: Path,
    output_path: Path,
    start_s: float,
    end_s: float,
    target_fps: float = TARGET_FPS,
    target_size: tuple[int, int] = TARGET_SIZE,
) -> dict:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open {input_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not source_fps or source_fps <= 0:
        raise SystemExit(f"{input_path} reports no frame rate")

    if source_w * target_size[1] != source_h * target_size[0]:
        # Not fatal, but a stretch changes every pixel-distance threshold
        # differently on each axis, which is worse than the scale problem
        # this script exists to fix.
        print(
            f"WARNING: aspect ratio changes {source_w}x{source_h} -> "
            f"{target_size[0]}x{target_size[1]}; distances will distort per axis"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), target_fps, target_size
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open {output_path} for writing")

    # Resample by NEAREST SOURCE FRAME per output timestamp rather than by
    # dropping every Nth frame: 59.94 -> 25 is not an integer ratio, and a
    # fixed drop pattern would make the real interval between kept frames
    # wobble, which is exactly what the frame-counting thresholds above
    # cannot tolerate.
    n_out = int((end_s - start_s) * target_fps)
    if n_out <= 0:
        raise SystemExit(f"Empty range: start={start_s} end={end_s}")

    written = 0
    last_source_idx = -1
    frame = None
    for i in range(n_out):
        source_idx = int(round((start_s + i / target_fps) * source_fps))
        if source_idx >= source_count:
            print(f"Source ran out at output frame {i} (wanted source frame {source_idx})")
            break
        if source_idx != last_source_idx:
            # Seek only when skipping; sequential reads are far faster and
            # a 60->25 resample reads most frames in order anyway.
            if source_idx != last_source_idx + 1:
                capture.set(cv2.CAP_PROP_POS_FRAMES, source_idx)
            ok, frame = capture.read()
            if not ok:
                print(f"Read failed at source frame {source_idx}; stopping")
                break
            last_source_idx = source_idx
        if frame is None:
            continue
        resized = cv2.resize(frame, target_size, interpolation=cv2.INTER_CUBIC)
        writer.write(resized)
        written += 1

    capture.release()
    writer.release()
    return {
        "source": f"{source_w}x{source_h} @ {source_fps:.3f}fps, {source_count} frames",
        "output": f"{target_size[0]}x{target_size[1]} @ {target_fps:g}fps, {written} frames",
        "seconds": written / target_fps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=float, default=0.0, help="seconds into the source")
    parser.add_argument("--end", type=float, required=True, help="seconds into the source")
    parser.add_argument("--fps", type=float, default=TARGET_FPS)
    parser.add_argument("--width", type=int, default=TARGET_SIZE[0])
    parser.add_argument("--height", type=int, default=TARGET_SIZE[1])
    args = parser.parse_args()

    info = normalise(
        Path(args.input).expanduser(),
        Path(args.output),
        args.start,
        args.end,
        target_fps=args.fps,
        target_size=(args.width, args.height),
    )
    print(f"  source: {info['source']}")
    print(f"  output: {info['output']}  ({info['seconds']:.1f}s)")
    print(f"  wrote:  {args.output}")


if __name__ == "__main__":
    main()
