"""Run the fine-tuned YOLO ball detector + tracker over a video.

    python main.py --input path/to/match.mp4 --output outputs/tracked.mp4
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from src.analysis.bounce_classifier import BounceClassifier
from src.analysis.bounce_detector import detect_bounces, detect_bounces_ml, find_trajectory_breakpoints
from src.analysis.court_calibration import CourtCalibration
from src.analysis.speed_estimator import (
    estimate_net_crossing_speeds,
    estimate_shot_speeds,
    merge_with_net_crossing_speeds,
)
from src.analysis.striker import estimate_ground_y, select_players, select_striker
from src.analysis.stroke_classifier import StrokeClassifier
from src.detection.ball_detector import BallDetector
from src.detection.court_keypoint_detector import CourtKeypointDetector
from src.detection.pose_detector import PoseDetector
from src.tracking.ball_tracker import BallTracker
from src.video.io import VideoReader, VideoWriter
from src.visualize.draw import BounceMarkerDrawer, CourtOverlayDrawer, PoseDrawer, SidebarDrawer, TrailDrawer

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
        "src.analysis.bounce_detector's module docstring. Ignored if --bounce-classifier is given.",
    )
    parser.add_argument(
        "--bounce-classifier",
        default=None,
        help="Path to a trained bounce classifier (weights/bounce_classifier.pkl, see "
        "scripts/train_bounce_classifier.py) - replaces detect_bounces' fixed thresholds "
        "(--bounce-min-x-reversal, --bounce-max-height) with a learned classifier scored over "
        "a broader candidate scan. Requires --bounce.",
    )
    parser.add_argument(
        "--bounce-classifier-threshold",
        type=float,
        default=0.5,
        help="Minimum bounce probability from --bounce-classifier to accept a candidate. Lower "
        "to catch more real bounces at the cost of more false positives, raise for the opposite.",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Path to a static court calibration file (configs/court_calibration.json) for "
        "real-world speed on a FIXED camera. Ignored if --show-court is given, since that "
        "detects the court fresh every frame instead (handles a panning/zooming camera; a "
        "static file can't). If neither --show-court nor --calibration is given, --speed "
        "falls back to px/s.",
    )
    parser.add_argument(
        "--speed", action="store_true", help="Print peak speed per shot to the console"
    )
    parser.add_argument(
        "--show-court",
        action="store_true",
        help="Detect the court + draw the line wireframe + 4 corner markers EVERY frame (not "
        "just once), so the overlay tracks a moving/zooming camera. Also drives --speed/"
        "--sidebar's real-world numbers instead of --calibration when given. Requires "
        "--court-keypoint-weights.",
    )
    parser.add_argument("--court-keypoint-weights", default=str(REPO_ROOT / "weights" / "court_keypoint_detector.pt"))
    parser.add_argument("--court-keypoint-confidence", type=float, default=0.25)
    parser.add_argument(
        "--court-min-keypoint-confidence",
        type=float,
        default=0.5,
        help="Individual court keypoints below this are excluded from that frame's homography "
        "fit, even if the overall court detection passed --court-keypoint-confidence.",
    )
    parser.add_argument(
        "--speed-window",
        type=int,
        default=1,
        help="Frames spanned per speed sample - 1 uses raw frame-to-frame displacement. Raise "
        "this on a more zoomed-out camera, where real-world-per-pixel is coarser and a few "
        "pixels of detector jitter can otherwise swing the reported speed by 100+ km/h. Only "
        "affects the whole-trajectory fallback reading, not the net-crossing one (see "
        "--net-speed-min-frames).",
    )
    parser.add_argument(
        "--net-speed-min-frames",
        type=int,
        default=4,
        help="Minimum consecutive detected frames the ball must stay within the net band "
        "before estimate_net_crossing_speeds reports a reading - the reading is the AVERAGE "
        "speed over that whole window (net displacement / elapsed time), not a peak "
        "instantaneous reading, so more frames means more jitter-robust but less responsive "
        "to genuine sub-window acceleration. Raise this if speed readings still look noisy/too "
        "high, lower it if legitimate net crossings are being dropped for lack of frames.",
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
    court_keypoint_detector = (
        CourtKeypointDetector(
            args.court_keypoint_weights,
            confidence=args.court_keypoint_confidence,
            imgsz=args.imgsz,
            device=args.device,
        )
        if args.show_court
        else None
    )
    static_calibration = (
        CourtCalibration.load(args.calibration) if args.calibration and not args.show_court else None
    )

    reader = VideoReader(args.input)
    frames = list(reader.frames())
    if not frames:
        raise SystemExit(f"No frames read from {args.input}")

    print(f"Detecting ball in {len(frames)} frames...")
    detections = []
    poses_by_frame = [] if args.pose else None
    # A moving/zooming broadcast camera means the court's pixel position can
    # change frame to frame, so with --show-court the court is re-detected
    # EVERY frame rather than once (contrast a fixed-camera --calibration
    # file, loaded once above). A frame where the detector doesn't find
    # enough confident keypoints (motion blur, court briefly out of frame)
    # carries forward the last good calibration rather than leaving a gap -
    # the camera can't have moved far in one missed frame.
    calibrations_by_frame: dict[int, CourtCalibration] = {}
    last_good_calibration: "CourtCalibration | None" = None
    for i, frame in enumerate(tqdm(frames)):
        detections.append(detector.detect(frame))
        if args.pose:
            poses_by_frame.append(select_players(pose_detector.detect(frame), reader.width))
        if args.show_court:
            court_detection = court_keypoint_detector.detect(frame)
            named_points = (
                court_detection.as_named_points(min_confidence=args.court_min_keypoint_confidence)
                if court_detection is not None
                else {}
            )
            if len(named_points) >= 4:
                last_good_calibration = CourtCalibration.from_keypoints(named_points)
            if last_good_calibration is not None:
                calibrations_by_frame[i] = last_good_calibration
        elif static_calibration is not None:
            calibrations_by_frame[i] = static_calibration

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

    # Both paths take a single calibration for the whole call (they scan the
    # whole trajectory at once); with --show-court's per-frame calibrations,
    # enrich each bounce's world_x/world_y afterward instead, using the
    # calibration valid at that specific bounce's frame.
    if args.bounce and args.bounce_classifier:
        bounces = detect_bounces_ml(
            positions,
            BounceClassifier(args.bounce_classifier),
            min_frame_gap=args.bounce_min_gap,
            threshold=args.bounce_classifier_threshold,
            calibration=static_calibration,
        )
    elif args.bounce:
        bounces = detect_bounces(
            positions,
            min_y_prominence=args.bounce_min_prominence,
            min_frame_gap=args.bounce_min_gap,
            min_x_reversal=args.bounce_min_x_reversal,
            calibration=static_calibration,
            ground_y_by_frame=ground_y_by_frame,
            max_height_above_ground=args.bounce_max_height,
        )
    else:
        bounces = []
    if args.bounce and args.show_court:
        for bounce in bounces:
            bounce_calibration = calibrations_by_frame.get(bounce.frame_idx)
            if bounce_calibration is not None:
                bounce.world_x, bounce.world_y = bounce_calibration.pixel_to_world(bounce.x, bounce.y)
    bounces_by_frame = {b.frame_idx: b for b in bounces}
    if args.bounce:
        print(f"Detected {len(bounces)} bounces")

    # estimate_net_crossing_speeds is the most accurate reading (near the
    # ground plane the calibration homography is exact for) but only fires
    # for shots actually seen crossing the net band with two consecutive
    # DETECTED positions - many shots (occlusion right at the net, a rally
    # ball that stays high, etc.) never qualify and would otherwise get no
    # speed at all. estimate_shot_speeds' breakpoint segmentation covers
    # every shot in the trajectory instead (in km/h when a calibration
    # exists, else px/s), so merge_with_net_crossing_speeds uses that for
    # full coverage and swaps in the sharper net-crossing number wherever
    # one overlaps.
    shot_speed_by_frame = {}
    if args.speed or args.sidebar:
        # Segmenting at every direction-reversal (find_trajectory_breakpoints)
        # over-splits a single real shot whenever detector jitter or a
        # mid-flight wobble looks like a local max - especially at a lower
        # --bounce-min-prominence - and each spurious split then reports its
        # own (often much lower/noisier) speed for what's actually one
        # continuous shot. detect_bounces is far more selective (rejects
        # x-direction reversals, i.e. contacts rather than landings, and -
        # with --pose - anything too far above the nearest player's feet to
        # be a real landing), so when --bounce is on, its output IS the shot
        # boundary: each segment runs bounce-to-bounce, immune to the false
        # splits a bare local-max scan produces. Without --bounce there's no
        # bounce list to segment on, so this falls back to the old scan.
        breakpoint_frames = (
            [b.frame_idx for b in bounces]
            if args.bounce
            else find_trajectory_breakpoints(
                positions, min_y_prominence=args.bounce_min_prominence, min_frame_gap=args.bounce_min_gap
            )
        )
        fallback_shots = estimate_shot_speeds(
            positions,
            breakpoint_frames,
            reader.fps,
            calibration=static_calibration,
            speed_window=args.speed_window,
            calibrations_by_frame=calibrations_by_frame if calibrations_by_frame else None,
        )
        if calibrations_by_frame:
            net_shots = estimate_net_crossing_speeds(
                positions, reader.fps, calibrations_by_frame, min_window_frames=args.net_speed_min_frames
            )
            shots = merge_with_net_crossing_speeds(fallback_shots, net_shots)
        else:
            shots = fallback_shots
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
    court_drawer = CourtOverlayDrawer() if args.show_court else None

    for i, frame in enumerate(frames):
        annotated = frame
        if args.show_court:
            annotated = court_drawer.draw(annotated, calibrations_by_frame.get(i))
        annotated = trail.draw(annotated, positions_by_frame.get(i))
        if args.pose:
            annotated = pose_drawer.draw(annotated, poses_by_frame[i], striker_by_frame[i], strokes_by_frame.get(i))
        if args.bounce:
            annotated = bounce_drawer.draw(annotated, bounces_by_frame.get(i))
        if args.sidebar:
            annotated = sidebar_drawer.draw(annotated, strokes_by_frame.get(i), shot_speed_by_frame.get(i))
        writer.write(annotated)
    writer.close()

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
