"""Run pretrained ball/court/player/bounce detection + tracking over a video.

    python main.py --input path/to/match.mp4 --output outputs/tracked.mp4

Ball detection defaults to the pretrained TrackNet checkpoint
(--detector-backend tracknet) rather than a project-trained YOLO model - see
--detector-backend/--weights if you have your own fine-tuned checkpoint.
Court, player, and bounce detection are likewise pretrained models
(TennisCourtDetector, a COCO Faster R-CNN, and a CatBoost trajectory
regressor respectively - see yastrebksv/TennisProject), not project-trained
ones - there's no pose/stroke classification here, unlike earlier versions
of this pipeline, since none of these pretrained models produce pose data.
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from src.analysis.catboost_bounce_detector import CatBoostBounceDetector, detect_bounces_catboost
from src.analysis.bounce_detector import find_trajectory_breakpoints
from src.analysis.court_calibration import CourtCalibration
from src.analysis.speed_estimator import (
    estimate_net_crossing_speeds,
    estimate_shot_speeds,
    merge_with_net_crossing_speeds,
)
from src.detection.ball_detector import BallDetector
from src.detection.player_detector import PlayerDetector
from src.detection.tennis_court_net import TennisCourtNetDetector
from src.detection.tracknet_ball_detector import TrackNetBallDetector
from src.tracking.ball_tracker import BallTracker
from src.video.io import VideoReader, VideoWriter
from src.visualize.draw import BounceMarkerDrawer, CourtOverlayDrawer, SidebarDrawer, TrailDrawer
from src.visualize.minimap import MinimapDrawer

REPO_ROOT = Path(__file__).resolve().parent

# Unchanged from the old geometric detect_bounces' defaults - only used as
# the shot-speed segmentation fallback now (find_trajectory_breakpoints),
# for when --bounce is off and there's no bounce list to segment shots on.
_FALLBACK_BREAKPOINT_MIN_Y_PROMINENCE = 15.0
_FALLBACK_BREAKPOINT_MIN_FRAME_GAP = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input match video")
    parser.add_argument("--output", required=True, help="Output annotated video")
    parser.add_argument(
        "--detector-backend",
        choices=["yolo", "tracknet"],
        default="tracknet",
        help="'yolo' uses a project-trained --weights checkpoint instead of the pretrained "
        "TrackNet model, if you have one",
    )
    parser.add_argument(
        "--weights",
        default=str(REPO_ROOT / "weights" / "ball_detector.pt"),
        help="Fine-tuned YOLO ball detector checkpoint, only used with --detector-backend yolo",
    )
    parser.add_argument(
        "--tracknet-weights",
        default=str(REPO_ROOT / "weights" / "tracknet_pretrained.pt"),
        help="Pretrained TrackNet checkpoint, only used with --detector-backend tracknet",
    )
    parser.add_argument("--confidence", type=float, default=0.15, help="YOLO backend only")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO backend only")
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
    parser.add_argument("--bounce", action="store_true", help="Detect + mark ball landing spots")
    parser.add_argument(
        "--bounce-weights",
        default=str(REPO_ROOT / "weights" / "bounce_catboost_pretrained.cbm"),
        help="Pretrained CatBoost bounce-detection checkpoint",
    )
    parser.add_argument(
        "--bounce-threshold",
        type=float,
        default=0.45,
        help="Minimum CatBoost bounce probability to call a frame a bounce",
    )
    parser.add_argument(
        "--speed", action="store_true", help="Print peak speed per shot to the console"
    )
    parser.add_argument(
        "--show-court",
        action="store_true",
        help="Detect the court + draw the line wireframe + 4 corner markers EVERY frame, "
        "tracking a panning/zooming camera. Also drives --speed/--sidebar's real-world "
        "numbers. Requires --court-weights.",
    )
    parser.add_argument(
        "--court-weights", default=str(REPO_ROOT / "weights" / "court_net_pretrained.pt")
    )
    parser.add_argument(
        "--minimap",
        action="store_true",
        help="Composite a bird's-eye minimap (ball, bounces, players) in the frame corner. Requires --show-court.",
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
    parser.add_argument("--sidebar", action="store_true", help="Composite a speed sidebar panel")
    parser.add_argument("--sidebar-width", type=int, default=250)
    args = parser.parse_args()

    if args.minimap and not args.show_court:
        raise SystemExit("--minimap requires --show-court")

    if args.detector_backend == "tracknet":
        detector = TrackNetBallDetector(args.tracknet_weights, device=args.device)
    else:
        detector = BallDetector(
            args.weights, confidence=args.confidence, imgsz=args.imgsz, device=args.device
        )
    tracker = BallTracker(
        max_pixels_per_frame=args.max_jump,
        max_interpolation_gap=args.interp_gap,
        static_lockon_frames=args.lockon_frames,
        static_lockon_radius=args.lockon_radius,
    )
    court_detector = TennisCourtNetDetector(args.court_weights, device=args.device) if args.show_court else None
    player_detector = PlayerDetector(device=args.device) if args.minimap else None

    reader = VideoReader(args.input)
    frames = list(reader.frames())
    if not frames:
        raise SystemExit(f"No frames read from {args.input}")

    print(f"Detecting ball in {len(frames)} frames...")
    detections = []
    calibrations_by_frame: dict[int, CourtCalibration] = {}
    last_good_calibration: "CourtCalibration | None" = None
    far_players_by_frame: dict[int, list[tuple[float, float]]] = {}
    near_players_by_frame: dict[int, list[tuple[float, float]]] = {}
    for i, frame in enumerate(tqdm(frames)):
        detections.append(detector.detect(frame))
        if args.show_court:
            # A frame where the detector doesn't find enough confident
            # keypoints (motion blur, court briefly out of frame) carries
            # forward the last good calibration rather than leaving a gap -
            # the camera can't have moved far in one missed frame.
            named_points = court_detector.detect(frame) or {}
            if len(named_points) >= 4:
                last_good_calibration = CourtCalibration.from_keypoints(named_points)
            if last_good_calibration is not None:
                calibrations_by_frame[i] = last_good_calibration
            if args.minimap:
                calibration = calibrations_by_frame.get(i)
                players = player_detector.detect(frame, calibration=calibration)
                far, near = PlayerDetector.split_top_bottom(players)
                far_players_by_frame[i] = [p.world_point for p in far]
                near_players_by_frame[i] = [p.world_point for p in near]

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

    if args.bounce:
        bounce_model = CatBoostBounceDetector(args.bounce_weights, threshold=args.bounce_threshold)
        bounces = detect_bounces_catboost(positions, bounce_model, num_frames=len(frames))
        if args.show_court:
            for bounce in bounces:
                bounce_calibration = calibrations_by_frame.get(bounce.frame_idx)
                if bounce_calibration is not None:
                    bounce.world_x, bounce.world_y = bounce_calibration.pixel_to_world(bounce.x, bounce.y)
        print(f"Detected {len(bounces)} bounces")
    else:
        bounces = []
    bounces_by_frame = {b.frame_idx: b for b in bounces}
    minimap_bounce_points = [(b.world_x, b.world_y) for b in bounces if b.world_x is not None]

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
        # mid-flight wobble looks like a local max. detect_bounces_catboost's
        # output is far more selective, so when --bounce is on, its output
        # IS the shot boundary: each segment runs bounce-to-bounce, immune
        # to the false splits a bare local-max scan produces. Without
        # --bounce there's no bounce list to segment on, so this falls back
        # to the old scan.
        breakpoint_frames = (
            [b.frame_idx for b in bounces]
            if args.bounce
            else find_trajectory_breakpoints(
                positions,
                min_y_prominence=_FALLBACK_BREAKPOINT_MIN_Y_PROMINENCE,
                min_frame_gap=_FALLBACK_BREAKPOINT_MIN_FRAME_GAP,
            )
        )
        fallback_shots = estimate_shot_speeds(
            positions,
            breakpoint_frames,
            reader.fps,
            calibration=None,
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
    bounce_drawer = BounceMarkerDrawer() if args.bounce else None
    sidebar_drawer = SidebarDrawer(width=args.sidebar_width) if args.sidebar else None
    court_drawer = CourtOverlayDrawer() if args.show_court else None
    minimap_drawer = MinimapDrawer() if args.minimap else None

    for i, frame in enumerate(frames):
        annotated = frame
        calibration = calibrations_by_frame.get(i) if args.show_court else None
        if args.show_court:
            annotated = court_drawer.draw(annotated, calibration)
        annotated = trail.draw(annotated, positions_by_frame.get(i))
        if args.bounce:
            annotated = bounce_drawer.draw(annotated, bounces_by_frame.get(i))
        if args.minimap:
            position = positions_by_frame.get(i)
            ball_world = calibration.pixel_to_world(position.x, position.y) if position and calibration else None
            annotated = minimap_drawer.draw(
                annotated,
                ball_world=ball_world,
                bounce_world_points=minimap_bounce_points,
                far_players_world=far_players_by_frame.get(i),
                near_players_world=near_players_by_frame.get(i),
            )
        if args.sidebar:
            annotated = sidebar_drawer.draw(annotated, None, shot_speed_by_frame.get(i))
        writer.write(annotated)
    writer.close()

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
