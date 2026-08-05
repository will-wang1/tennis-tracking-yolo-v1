"""Run the fine-tuned YOLO ball detector + tracker over a video.

    python main.py --input path/to/match.mp4 --output outputs/tracked.mp4
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from src.analysis.bounce_detector import detect_bounces, find_trajectory_breakpoints
from src.analysis.court_calibration import CourtCalibration
from src.analysis.speed_estimator import estimate_shot_speeds
from src.analysis.striker import estimate_ground_y, select_striker
from src.analysis.stroke_classifier import StrokeClassifier
from src.detection.ball_detector import BallDetector
from src.detection.pose_detector import PoseDetector
from src.tracking.ball_tracker import BallTracker
from src.video.io import VideoReader, VideoWriter
from src.visualize.draw import BounceMarkerDrawer, PoseDrawer, SidebarDrawer, TrailDrawer

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
    parser.add_argument(
        "--max-jump",
        type=float,
        default=150.0,
        help="Max plausible ball movement per frame, in pixels - larger detector jumps are rejected as outliers",
    )
    parser.add_argument(
        "--interp-gap",
        type=int,
        default=8,
        help="Longest run of missed frames that gets filled by interpolation",
    )
    parser.add_argument(
        "--lockon-frames",
        type=int,
        default=10,
        help="How many consecutive frames confined to --lockon-radius before it's treated as a false lock-on rather than the ball",
    )
    parser.add_argument(
        "--lockon-radius",
        type=float,
        default=20.0,
        help="Pixel radius that counts as 'hasn't moved' for --lockon-frames",
    )
    parser.add_argument(
        "--pose", action="store_true", help="Detect + overlay the striker's pose skeleton"
    )
    parser.add_argument("--pose-weights", default="yolov8n-pose.pt")
    parser.add_argument("--pose-confidence", type=float, default=0.3)
    parser.add_argument(
        "--stroke-classifier",
        default=None,
        help="Path to a trained stroke classifier (weights/stroke_classifier.pkl). Requires --pose.",
    )
    parser.add_argument("--bounce", action="store_true", help="Detect + mark ball landing spots")
    parser.add_argument("--bounce-min-prominence", type=float, default=15.0)
    parser.add_argument("--bounce-min-gap", type=int, default=5)
    parser.add_argument(
        "--bounce-min-x-reversal",
        type=float,
        default=15.0,
        help="Minimum horizontal movement on both sides of a candidate to call it a direction "
        "reversal - a racket contact redirecting the ball, not a bounce off the court",
    )
    parser.add_argument(
        "--bounce-max-height",
        type=float,
        default=160.0,
        help="Max pixels a bounce candidate can sit above the nearest player's feet (only "
        "applied when --pose is also given, which is what makes this cross-check possible) "
        "before it's rejected as too high off the court to be a real bounce. Telling a low "
        "bounce from a contact from pixels alone is inherently approximate - see "
        "src.analysis.bounce_detector's module docstring.",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Path to a court calibration file (configs/court_calibration.json) for real-world "
        "speed. If omitted, --speed falls back to px/s.",
    )
    parser.add_argument(
        "--speed", action="store_true", help="Print peak speed per shot to the console"
    )
    parser.add_argument(
        "--speed-window",
        type=int,
        default=1,
        help="Frames spanned per speed sample - 1 uses raw frame-to-frame displacement. Raise "
        "this on a more zoomed-out camera, where real-world-per-pixel is coarser and a few "
        "pixels of detector jitter can otherwise swing the reported speed by 100+ km/h.",
    )
    parser.add_argument("--sidebar", action="store_true", help="Composite a stroke/speed sidebar panel")
    parser.add_argument("--sidebar-width", type=int, default=250)
    args = parser.parse_args()

    if args.stroke_classifier and not args.pose:
        raise SystemExit("--stroke-classifier requires --pose")

    detector = BallDetector(
        args.weights, confidence=args.confidence, imgsz=args.imgsz, device=args.device
    )
    tracker = BallTracker(
        max_pixels_per_frame=args.max_jump,
        max_interpolation_gap=args.interp_gap,
        static_lockon_frames=args.lockon_frames,
        static_lockon_radius=args.lockon_radius,
    )
    pose_detector = (
        PoseDetector(args.pose_weights, confidence=args.pose_confidence, imgsz=args.imgsz, device=args.device)
        if args.pose
        else None
    )
    calibration = CourtCalibration.load(args.calibration) if args.calibration else None

    reader = VideoReader(args.input)
    frames = list(reader.frames())
    if not frames:
        raise SystemExit(f"No frames read from {args.input}")

    print(f"Detecting ball in {len(frames)} frames...")
    detections = []
    poses_by_frame = [] if args.pose else None
    for frame in tqdm(frames):
        detections.append(detector.detect(frame))
        if args.pose:
            poses_by_frame.append(pose_detector.detect(frame))

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

    striker_by_frame = (
        [select_striker(poses_by_frame[i], positions_by_frame.get(i)) for i in range(len(frames))]
        if args.pose
        else [None] * len(frames)
    )

    strokes_by_frame = {}
    if args.stroke_classifier:
        classifier = StrokeClassifier(args.stroke_classifier)
        for i in range(len(frames)):
            prediction = classifier.predict(i, striker_by_frame[i])
            if prediction is not None:
                strokes_by_frame[prediction.frame_idx] = prediction

    ground_y_by_frame = {}
    if args.bounce and args.pose:
        for i in range(len(frames)):
            ground_y = estimate_ground_y(poses_by_frame[i], positions_by_frame.get(i))
            if ground_y is not None:
                ground_y_by_frame[i] = ground_y

    bounces = (
        detect_bounces(
            positions,
            min_y_prominence=args.bounce_min_prominence,
            min_frame_gap=args.bounce_min_gap,
            min_x_reversal=args.bounce_min_x_reversal,
            calibration=calibration,
            ground_y_by_frame=ground_y_by_frame,
            max_height_above_ground=args.bounce_max_height,
        )
        if args.bounce
        else []
    )
    bounces_by_frame = {b.frame_idx: b for b in bounces}
    if args.bounce:
        print(f"Detected {len(bounces)} bounces")

    # Shot speed is segmented at trajectory "breakpoints" - ANY direction
    # change (bounce OR contact), not just confirmed bounces. Confirming a
    # bounce specifically (for the visual markers above) needs the stronger,
    # pose-cross-referenced signal and is often sparse/absent; a stable
    # per-shot speed reading doesn't need that distinction; see
    # find_trajectory_breakpoints's docstring.
    shot_speed_by_frame = {}
    if args.speed or args.sidebar:
        breakpoint_frames = find_trajectory_breakpoints(
            positions, min_y_prominence=args.bounce_min_prominence, min_frame_gap=args.bounce_min_gap
        )
        shots = estimate_shot_speeds(
            positions, breakpoint_frames, reader.fps, calibration, speed_window=args.speed_window
        )
        if args.speed:
            print(f"Peak speed per shot ({len(shots)} shots):")
            for shot in shots:
                print(f"  frames {shot.start_frame}-{shot.end_frame}: {shot.peak_speed:.0f} {shot.unit}")
        for shot in shots:
            for frame_idx in range(shot.start_frame, shot.end_frame + 1):
                shot_speed_by_frame[frame_idx] = (shot.peak_speed, shot.unit)

    output_width = reader.width + (args.sidebar_width if args.sidebar else 0)
    writer = VideoWriter(args.output, reader.fps, output_width, reader.height)
    trail = TrailDrawer(trail_length=args.trail_length)
    pose_drawer = PoseDrawer() if args.pose else None
    bounce_drawer = BounceMarkerDrawer() if args.bounce else None
    sidebar_drawer = SidebarDrawer(width=args.sidebar_width) if args.sidebar else None

    for i, frame in enumerate(frames):
        annotated = trail.draw(frame, positions_by_frame.get(i))
        if args.pose:
            annotated = pose_drawer.draw(annotated, striker_by_frame[i], strokes_by_frame.get(i))
        if args.bounce:
            annotated = bounce_drawer.draw(annotated, bounces_by_frame.get(i))
        if args.sidebar:
            annotated = sidebar_drawer.draw(annotated, strokes_by_frame.get(i), shot_speed_by_frame.get(i))
        writer.write(annotated)
    writer.close()

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
