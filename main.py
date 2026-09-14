"""Run pretrained ball/court/player/bounce detection + tracking over a video.

    python main.py --input path/to/match.mp4 --output outputs/tracked.mp4

Ball detection defaults to the pretrained TrackNet checkpoint
(--detector-backend tracknet) rather than a project-trained YOLO model - see
--detector-backend/--weights if you have your own fine-tuned checkpoint.
Court, player, and bounce detection are likewise pretrained models
(TennisCourtDetector, a COCO Faster R-CNN, and a CatBoost trajectory
regressor respectively - see yastrebksv/TennisProject), not project-trained
ones. Shot classification (--stroke) is pretrained too - MoveNet (Google,
pose) + a small GRU (antoinekeller/tennis_shot_recognition) - unlike the
pose-based stroke classifier this pipeline had before the pivot to
pretrained models, which needed its own training data.
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from src.analysis.bounce_ensemble import detect_bounces_ensemble
from src.analysis.catboost_bounce_detector import CatBoostBounceDetector
from src.analysis.geometric_bounce_detector import detect_bounces_geometric
from src.analysis.flight_segmenter import find_flight_segments
from src.analysis.html_report import clip_label_from_path, write_report_html
from src.analysis.impact_pipeline import analyze_impacts
from src.analysis.match_stats import compute_match_stats
from src.analysis.match_log import build_match_log
from src.analysis.parabolic_bounce_detector import detect_bounces_parabolic
from src.analysis.player_identity import PlayerIdentityTracker
from src.analysis.scene_cuts import detect_scene_cuts
from src.analysis.serve_sequences import classify_serve_sequences
from src.analysis.velocity_bounce_detector import detect_bounces_by_velocity
from src.analysis.bounce_detector import BounceEvent, find_trajectory_breakpoints
from src.analysis.court_calibration import CourtCalibration
from src.analysis.speed_estimator import (
    estimate_flight_speeds,
    estimate_net_crossing_speeds,
    estimate_shot_speeds,
    merge_with_net_crossing_speeds,
)
from src.analysis.shot_classifier import ShotClassifier, ShotEventTracker
from src.detection.ball_detector import BallDetector
from src.detection.movenet_pose_extractor import MoveNetPoseExtractor
from src.detection.player_detector import PlayerDetector
from src.detection.tennis_court_net import TennisCourtNetDetector
from src.detection.tracknet_ball_detector import TrackNetBallDetector
from src.detection.wasb_ball_detector import WASBBallDetector
from src.tracking.ball_tracker import BallTracker
from src.tracking.candidate_tracker import track_candidates
from src.video.io import VideoReader, VideoWriter
from src.visualize.draw import (
    BounceMarkerDrawer,
    CourtOverlayDrawer,
    ImpactMarkerDrawer,
    ShotArcDrawer,
    ShotLabelDrawer,
    SidebarDrawer,
    StatsPanelDrawer,
    TrailDrawer,
)
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
        choices=["yolo", "tracknet", "wasb"],
        default="wasb",
        help="'wasb' (default) is WASB-SBDT's pretrained tennis HRNet - on this project's "
        "footage it drops the ball less often and is noticeably steadier frame-to-frame than "
        "'tracknet' (measured: 83%% vs 76%% raw detection rate, ~4x lower mean frame-to-frame "
        "jitter on the same video). 'yolo' uses a project-trained --weights checkpoint instead, "
        "if you have one.",
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
    parser.add_argument(
        "--wasb-weights",
        default=str(REPO_ROOT / "weights" / "wasb_tennis_pretrained.pth.tar"),
        help="Pretrained WASB-SBDT tennis checkpoint, only used with --detector-backend wasb",
    )
    parser.add_argument("--confidence", type=float, default=0.15, help="YOLO backend only")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO backend only")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--ball-overlay",
        choices=["trail", "arc"],
        default="trail",
        help="How the ball's path is drawn. 'trail' (default) draws a row of dots, one per "
        "tracked frame. 'arc' draws the current shot as the fitted flights the ball actually "
        "flew, joined across the bounce into one path (see ShotArcDrawer) - smoother, and "
        "nothing in it comes from a raw detection, but it shows a model of the path rather "
        "than the tracked positions themselves.",
    )
    parser.add_argument(
        "--trail-length", type=int, default=8, help="Dots to keep in the trail"
    )
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
        "--smoothing-window",
        type=int,
        default=9,
        help="Savitzky-Golay filter window (frames) applied to the tracked trajectory to remove "
        "detector jitter - set below 3 to disable. Both ball detector backends localize the "
        "ball via an intensity-weighted centroid rather than a shape-fitting method, so most of "
        "the jitter this used to need to remove is already gone at the source - this only needs "
        "to mop up what's left. A larger window keeps cutting jitter further but rounds off "
        "real sharp bounce corners more too, which also blunts the bounce detector's own "
        "features.",
    )
    parser.add_argument(
        "--smoothing-polyorder",
        type=int,
        default=2,
        help="Polynomial order for --smoothing-window's filter - 2 tracks a parabolic arc well "
        "without following raw jitter; must be lower than --smoothing-window",
    )
    parser.add_argument(
        "--no-candidate-tracking",
        action="store_true",
        help="Take each frame's single strongest ball blob instead of choosing a path through "
        "several candidates per frame (see src/tracking/candidate_tracker.py). The strongest "
        "blob is often not the ball - a line, the net cord or a shoe can outscore it, and at an "
        "impact the ball is blurriest and weakest - so the default keeps every plausible blob "
        "and picks the trajectory that best explains them. Only affects the 'wasb' backend.",
    )
    parser.add_argument(
        "--no-flight-segments",
        action="store_true",
        help="Skip the flight-segment impact search (src/analysis/flight_segmenter.py), which "
        "finds impacts by intersecting fitted flights instead of scanning frame by frame. The "
        "scan needs detections either side of the impact, so it misses bounces hidden by a "
        "dropout; intersecting segments recovers those. Only used with --bounce-method parabolic.",
    )
    parser.add_argument(
        "--no-scene-cuts",
        action="store_true",
        help="Skip detecting broadcast camera cuts (src/analysis/scene_cuts.py). On by default and "
        "cheap (pure frame histograms, no model) because a cut invalidates the court calibration "
        "carried forward from the frame before it - the whole point of the check is to reset that "
        "rather than smear a stale wireframe/minimap across an unrelated shot. Disable only if it's "
        "misfiring on unusually busy footage.",
    )
    parser.add_argument("--bounce", action="store_true", help="Detect + mark ball landing spots")
    parser.add_argument(
        "--contacts",
        action="store_true",
        help="Also mark every racket CONTACT, not just bounces - each labelled with its "
        "timestamp so a run can be checked by eye. A rally alternates contact and bounce, so a "
        "contact with no bounce before the next contact is either a volley or a bounce that was "
        "missed, which is what makes this the quickest way to see where recall is failing. "
        "Requires --bounce and --bounce-method parabolic (the only method that classifies "
        "impacts rather than only reporting bounces).",
    )
    parser.add_argument(
        "--bounce-method",
        choices=["parabolic", "geometric", "ensemble", "velocity"],
        default="parabolic",
        help="'parabolic' (default, see src/analysis/parabolic_bounce_detector.py) fits a "
        "free-flight arc to each side of a candidate impact and keeps only transitions a court "
        "surface could physically have produced - the ball descending in and rising out, giving "
        "back less vertical speed than it received, keeping its horizontal direction, and never "
        "leaving faster than it arrived. Fitting whole arcs rather than reading one frame's "
        "shape is what separates a real bounce from both detector noise and a racket contact. "
        "The other three all judge a single frame or frame pair and were each less reliable on "
        "this project's footage - kept for comparison, not recommended: 'geometric' takes the "
        "ball's lowest point on SCREEN (see geometric_bounce_detector.py, needs "
        "--bounce-smoothing-window), 'ensemble' unions CatBoost with that scan and a "
        "minimap-velocity scan behind off-court/near-net/near-player filters (bounce_ensemble.py), "
        "and 'velocity' uses the minimap velocity direction alone (velocity_bounce_detector.py).",
    )
    parser.add_argument(
        "--bounce-smoothing-window",
        type=int,
        default=3,
        help="Savitzky-Golay window for the SEPARATE, lightly smoothed trajectory --bounce-method "
        "geometric scans - deliberately much smaller than --smoothing-window (used for the "
        "visible trail), which rounds off real bounce corners before the scan ever sees them. "
        "Only used with --bounce-method geometric.",
    )
    parser.add_argument(
        "--bounce-sources",
        default="catboost,geometric,speed_drop",
        help="Comma-separated subset of {catboost,geometric,speed_drop} - which candidate "
        "source(s) feed detect_bounces_ensemble's union. Mainly for isolating one signal to see "
        "how it performs alone, e.g. --bounce-sources speed_drop.",
    )
    parser.add_argument(
        "--bounce-weights",
        default=str(REPO_ROOT / "weights" / "bounce_catboost_pretrained.cbm"),
        help="Pretrained CatBoost bounce-detection checkpoint",
    )
    parser.add_argument(
        "--bounce-threshold",
        type=float,
        default=0.3,
        help="Minimum CatBoost bounce probability to call a frame a bounce - the model's default "
        "0.45 cutoff (tuned on its own training footage) under-recalls on this project's camera "
        "angle, missing real bounces; --bounce-player-margin is what should be filtering out the "
        "resulting extra racket-contact false positives, not a higher threshold.",
    )
    parser.add_argument(
        "--bounce-player-margin",
        type=float,
        default=50.0,
        help="A bounce candidate within this many pixels of a detected player's box is dropped "
        "as a likely racket contact rather than a real court bounce - see "
        "filter_bounces_near_players. Only takes effect with --minimap, which is what computes "
        "player boxes; set to 0 to disable while keeping --minimap.",
    )
    parser.add_argument(
        "--bounce-court-margin",
        type=float,
        default=2.0,
        help="A bounce candidate whose court-projected position lands more than this many "
        "meters outside the doubles lines is dropped as detector noise - see "
        "filter_bounces_off_court. Only takes effect with --show-court.",
    )
    parser.add_argument(
        "--bounce-net-margin",
        type=float,
        default=8.0,
        help="A bounce candidate whose court-projected position lands within this many meters "
        "of the net line is dropped - the ground-plane homography is least reliable exactly "
        "there (a ball crossing the net is at its most elevated), which measured on real "
        "footage was producing mostly false positives out to about this distance. Only takes "
        "effect with --show-court.",
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
        help="Minimum consecutive DETECTED frames the ball must stay within the net band "
        "before estimate_net_crossing_speeds reports a reading. Only used when no flight "
        "segmentation is available: with one, the net crossing is solved on the fitted curve "
        "instead (estimate_flight_speeds), which needs no detection at the crossing at all "
        "and so does not have a window to size. More frames means more jitter-robust but less "
        "responsive to genuine sub-window acceleration.",
    )
    parser.add_argument("--sidebar", action="store_true", help="Composite a speed sidebar panel")
    parser.add_argument("--sidebar-width", type=int, default=250)
    parser.add_argument(
        "--stats-panel",
        action="store_true",
        help="Composite a running match-stats panel (rally count, shots/bounces so far, near/"
        "far shot split, peak speed so far) - cumulative AS OF each frame, not the video's final "
        "tally, so scrubbing mid-render shows what a live viewer would know at that point. "
        "Requires --bounce. The near/far split needs --show-court; peak speed needs --speed. "
        "Unlike --sidebar (current stroke + instantaneous speed only), this remembers everything "
        "up to now - see src/visualize/draw.py's StatsPanelDrawer. Shot speeds are computed "
        "automatically when this is on, same as --speed, so the peak-so-far line works without "
        "passing --speed separately.",
    )
    parser.add_argument("--stats-panel-width", type=int, default=280)
    parser.add_argument(
        "--stroke",
        action="store_true",
        help="Classify each player's shots as forehand/backhand/serve (MoveNet pose + a "
        "pretrained GRU, see src/analysis/shot_classifier.py). Requires --minimap, which is "
        "what computes the player boxes this points the pose model at.",
    )
    parser.add_argument(
        "--movenet-weights",
        default=str(REPO_ROOT / "weights" / "movenet_singlepose_lightning_int8.tflite"),
        help="Pretrained MoveNet SinglePose Lightning checkpoint, only used with --stroke",
    )
    parser.add_argument(
        "--stroke-weights",
        default=str(REPO_ROOT / "weights" / "shot_classifier_rnn_pretrained.h5"),
        help="Pretrained shot-classifier GRU checkpoint, only used with --stroke",
    )
    parser.add_argument(
        "--stats",
        help="Write a JSON match summary here - rally count/duration, shot and bounce counts "
        "per rally, peak shot speeds, bounce locations, and (with --stroke) forehand/backhand"
        "/serve counts. Requires --bounce; nothing here is detected fresh, it only folds "
        "together what --bounce/--speed/--stroke already computed (see "
        "src/analysis/match_stats.py). Rally segmentation is a genuinely new inference on top "
        "of that, though, and an honestly untested one - see that module's docstring.",
    )
    parser.add_argument(
        "--report",
        help="Write a self-contained HTML match report here (scoreboard, bounce map, shot-speed "
        "chart, court-coverage comparison, stroke mix, serve-speed trend, rally log) - built "
        "from the same MatchStats object --stats writes to JSON, so requires --stats too. "
        "Opens directly in a browser, no server needed (see src/analysis/html_report.py).",
    )
    args = parser.parse_args()
    if args.stats and not args.bounce:
        raise SystemExit("--stats needs --bounce")
    if args.report and not args.stats:
        raise SystemExit("--report needs --stats")
    if args.stats_panel and not args.bounce:
        raise SystemExit("--stats-panel needs --bounce")

    if args.minimap and not args.show_court:
        raise SystemExit("--minimap requires --show-court")
    if args.stroke and not args.minimap:
        raise SystemExit("--stroke requires --minimap")
    if args.contacts and not (args.bounce and args.bounce_method == "parabolic"):
        raise SystemExit("--contacts requires --bounce with --bounce-method parabolic")
    if args.bounce_method == "velocity" and not args.show_court:
        raise SystemExit("--bounce-method velocity requires --show-court")

    if args.detector_backend == "wasb":
        detector = WASBBallDetector(args.wasb_weights, device=args.device)
    elif args.detector_backend == "tracknet":
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
        smoothing_window=args.smoothing_window,
        smoothing_polyorder=args.smoothing_polyorder,
    )
    court_detector = TennisCourtNetDetector(args.court_weights, device=args.device) if args.show_court else None
    player_detector = PlayerDetector(device=args.device) if args.minimap else None
    pose_extractor = MoveNetPoseExtractor(args.movenet_weights) if args.stroke else None
    near_shot_classifier = ShotClassifier(args.stroke_weights) if args.stroke else None
    far_shot_classifier = ShotClassifier(args.stroke_weights) if args.stroke else None
    near_shot_tracker = ShotEventTracker() if args.stroke else None
    far_shot_tracker = ShotEventTracker() if args.stroke else None

    # THREE separate streaming passes over the video (scene cuts, then
    # detection, then rendering below) rather than one shared list of
    # decoded frames - `list(reader.frames())` used to be the approach, and
    # it OOMs outright on anything longer than a short clip (measured: a
    # 3-minute 1080p clip is ~28GB of raw frames, which is what actually
    # crashed trying this against real footage). Each pass opens its own
    # VideoReader and consumes it as a generator, so at most one raw frame
    # is ever alive at a time; everything a later pass needs (detections,
    # calibrations, boxes) is small derived data, not pixels, so re-decoding
    # three times costs far less than holding the video in memory once.
    cut_reader = VideoReader(args.input)
    # Cheap (frame histograms only, no model) and run unconditionally unless
    # opted out, because it feeds a correctness fix below (resetting the
    # carried-forward court calibration at a cut), not just reporting - see
    # scene_cuts.py for what this can and can't be trusted to catch.
    cut_frames = [] if args.no_scene_cuts else detect_scene_cuts(cut_reader.frames())
    if cut_frames:
        print(f"Detected {len(cut_frames)} camera cut(s) at frame(s): {cut_frames}")
    cut_frame_set = set(cut_frames)

    reader = VideoReader(args.input)
    print(f"Detecting ball in {reader.frame_count_hint() or '?'} frames...")
    detections = []
    calibrations_by_frame: dict[int, CourtCalibration] = {}
    last_good_calibration: "CourtCalibration | None" = None
    far_players_by_frame: dict[int, list[tuple[float, float]]] = {}
    near_players_by_frame: dict[int, list[tuple[float, float]]] = {}
    player_boxes_by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    near_shot_display_by_frame: dict[int, str] = {}
    far_shot_display_by_frame: dict[int, str] = {}
    near_player_bbox_by_frame: dict[int, tuple[float, float, float, float]] = {}
    far_player_bbox_by_frame: dict[int, tuple[float, float, float, float]] = {}
    identity_by_frame: dict[int, dict[str, str]] = {}
    identity_tracker = PlayerIdentityTracker() if args.minimap else None
    use_candidates = args.detector_backend == "wasb" and not args.no_candidate_tracking
    candidates_by_frame: list[list] = []
    num_frames = 0
    for i, frame in enumerate(tqdm(reader.frames(), total=reader.frame_count_hint())):
        num_frames = i + 1
        if use_candidates:
            candidates_by_frame.append(detector.detect_candidates(frame))
            detections.append(None)  # filled in below, once the whole path is known
        else:
            detections.append(detector.detect(frame))
        if i in cut_frame_set:
            # The camera itself changed - "the camera can't have moved far
            # in one missed frame" (below) is exactly the assumption a cut
            # breaks, so the carried-forward calibration must NOT survive
            # into the new shot. A fresh one is picked up as soon as this
            # frame's own court detection succeeds, same as any other gap.
            last_good_calibration = None
        if args.show_court:
            # A frame where the detector doesn't find enough confident
            # keypoints (motion blur, court briefly out of frame) carries
            # forward the last good calibration rather than leaving a gap -
            # the camera can't have moved far in one missed frame.
            named_points = court_detector.detect(frame) or {}
            if len(named_points) >= 4:
                try:
                    fitted = CourtCalibration.from_keypoints(named_points)
                except ValueError:
                    # No homography fits these keypoints at all. Same
                    # situation as too few keypoints: carry the last good
                    # calibration forward rather than fail the render.
                    fitted = None
                # Four matched keypoints are enough to FIT a homography but
                # not enough to tell whether it describes a real view - a
                # 4-point fit is exact, so it has no residual to check. A
                # replay or closeup fits the wrong lines just as confidently
                # as a live wide shot fits the right ones, so the result is
                # checked against the court's own geometry before it is
                # trusted (see CourtCalibration.is_plausible_view).
                if fitted is not None and fitted.is_plausible_view(frame.shape[1], frame.shape[0]):
                    last_good_calibration = fitted
            if last_good_calibration is not None:
                calibrations_by_frame[i] = last_good_calibration
            if args.minimap:
                calibration = calibrations_by_frame.get(i)
                players = player_detector.detect(frame, calibration=calibration)
                far, near = PlayerDetector.split_top_bottom(players)
                far_players_by_frame[i] = [p.world_point for p in far]
                near_players_by_frame[i] = [p.world_point for p in near]
                player_boxes_by_frame[i] = [p.bbox for p in players]
                if near:
                    near_player_bbox_by_frame[i] = near[0].bbox
                if far:
                    far_player_bbox_by_frame[i] = far[0].bbox

                # Cheap (histogram compares on boxes already computed above)
                # and independent of --stroke - identity is useful for
                # match_log's serve/point structure even when stroke type
                # is never classified. force_reassign on a cut frame is
                # what lets it recover from a camera change instead of
                # blindly trusting position - see player_identity.py.
                identity_by_frame[i] = identity_tracker.update(
                    frame,
                    near_bbox=near_player_bbox_by_frame.get(i),
                    far_bbox=far_player_bbox_by_frame.get(i),
                    force_reassign=i in cut_frame_set,
                )

                if args.stroke:
                    near_pose = pose_extractor.extract(frame, near[0].bbox) if near else None
                    near_prediction = near_shot_classifier.update(near_pose)
                    near_display_label = near_shot_tracker.update(i, near_prediction)
                    if near_display_label is not None:
                        near_shot_display_by_frame[i] = near_display_label

                    far_pose = pose_extractor.extract(frame, far[0].bbox) if far else None
                    far_prediction = far_shot_classifier.update(far_pose)
                    # far_shot_tracker.counts still accumulates (used by
                    # --stats/far_shot_counts, already documented and shown
                    # as unreliable there - see match_stats.py/report_
                    # template.html's fallback). The DISPLAY label is
                    # deliberately not written to far_shot_display_by_frame:
                    # the pose classifier was trained only on the near
                    # player's behind-baseline view (see shot_classifier.py)
                    # and, on this camera's framing, was measured calling a
                    # wrong type with real confidence rather than the
                    # near-universal "neutral" it gives on other footage -
                    # showing that guess on screen is worse than showing
                    # nothing. Only the structural "serve" override below
                    # (classify_serve_sequences, not pose) ever populates
                    # this dict for the far player.
                    far_shot_tracker.update(i, far_prediction)

    if num_frames == 0:
        raise SystemExit(f"No frames read from {args.input}")

    if args.stroke:
        print(
            f"Near player shots: {near_shot_tracker.counts['forehand']} forehand, "
            f"{near_shot_tracker.counts['backhand']} backhand, {near_shot_tracker.counts['serve']} serve"
        )
        print(
            f"Far player shots: {far_shot_tracker.counts['forehand']} forehand, "
            f"{far_shot_tracker.counts['backhand']} backhand, {far_shot_tracker.counts['serve']} serve"
        )

    if use_candidates:
        # Choose the ball's path through the per-frame candidates before any
        # smoothing: which blob is the ball is a question about the
        # trajectory, and answering it per frame throws the real ball away
        # exactly where it is hardest to see.
        detections = track_candidates(candidates_by_frame, max_pixels_per_frame=args.max_jump)
        kept = sum(1 for d in detections if d is not None)
        offered = sum(1 for row in candidates_by_frame if row)
        print(f"Candidate tracking: {kept}/{num_frames} frames tracked from {offered} frames offering candidates")

    positions = tracker.track(detections)
    positions_by_frame = {p.frame_idx: p for p in positions}
    detected = sum(1 for d in detections if d is not None)
    interpolated = sum(1 for p in positions if p.interpolated)
    print(
        f"Raw detections: {detected}/{num_frames} frames "
        f"({detected / num_frames:.1%}). "
        f"After interpolation: {len(positions)}/{num_frames} frames have a "
        f"tracked position ({interpolated} interpolated)."
    )

    impacts = []
    analysis = None
    if args.bounce and args.bounce_method == "parabolic":
        # Judging bounce-vs-contact needs the ball's PROJECTED court
        # position, which measures the height effect directly, rather than
        # screen y, which confuses height with court depth - so the
        # calibration is what upgrades this from "something hit the ball"
        # to "the court hit the ball". Scored against the hand labels in
        # data/labels/ (scripts/replay_impacts.py), that gets all 13 of the
        # US Open clip's labelled events and 12 of video_input2's 17, where
        # the screen-space checks alone found two of video_input2's bounces.
        # Player boxes are passed only to WITHHOLD the contact verdict
        # from impacts nobody could have reached - see
        # touchdown_detector.classify_touchdowns. They are only collected
        # under --minimap, and the classifier degrades to the direction
        # rule alone without them.
        analysis = analyze_impacts(
            detections,
            reader.fps,
            calibrations_by_frame=calibrations_by_frame if args.show_court else None,
            player_boxes_by_frame=player_boxes_by_frame if args.minimap else None,
            max_pixels_per_frame=args.max_jump,
            max_interpolation_gap=args.interp_gap,
            static_lockon_frames=args.lockon_frames,
            static_lockon_radius=args.lockon_radius,
            use_flight_segments=not args.no_flight_segments,
        )
        impacts = analysis.impacts
        if args.show_court:
            # The bounce's world position is reprojected from the MAIN
            # (smoothed) trajectory's position at that frame - the same
            # position the minimap's live ball dot and the on-screen trail
            # are drawn from - rather than from the impact's own unsmoothed
            # x/y. The two can disagree right at a bounce: smoothing rounds
            # off exactly the sharp corner the impact detector needs raw
            # (see BallTracker/parabolic_bounce_detector), so the unsmoothed
            # position the detector measured is not always where the ball
            # is shown to be on screen at that instant. Using the displayed
            # position instead means the landing mark always sits on top of
            # the visible ball, at the cost of a little precision in
            # exchange for a picture that agrees with itself. Classification
            # (is_bounce, kind, timing) is untouched - only where the result
            # is DRAWN changes.
            def _bounce_event(impact):
                shown = positions_by_frame.get(impact.frame_idx)
                x, y = (shown.x, shown.y) if shown is not None else (impact.x, impact.y)
                calibration = calibrations_by_frame.get(impact.frame_idx)
                world_x, world_y = calibration.pixel_to_world(x, y) if calibration else (None, None)
                return BounceEvent(
                    frame_idx=impact.frame_idx, x=x, y=y, world_x=world_x, world_y=world_y
                )

            bounces = [_bounce_event(impact) for impact in impacts if impact.is_bounce]
        else:
            bounces = detect_bounces_parabolic(analysis.positions)
        print(f"Detected {len(bounces)} bounces")
        if args.contacts:
            contacts = analysis.contacts
            unknown = analysis.unattributed
            print(
                f"Detected {len(contacts)} racket contacts and {len(unknown)} unattributed "
                f"impacts ({len(impacts)} in total). Only bounces and contacts are drawn:"
            )
            for impact in impacts:
                label = {"bounce": "BOUNCE ", "contact": "contact", "unknown": "  -    "}[impact.kind]
                print(f"  {impact.t / reader.fps:6.2f}s  {label}  {impact.reason}")
    elif args.bounce and args.bounce_method == "geometric":
        # A separate, lightly smoothed tracking pass - the window used for
        # the visible trail/speed readings rounds off exactly the corner
        # this method looks for (see geometric_bounce_detector.py).
        bounce_tracker = BallTracker(
            max_pixels_per_frame=args.max_jump,
            max_interpolation_gap=args.interp_gap,
            static_lockon_frames=args.lockon_frames,
            static_lockon_radius=args.lockon_radius,
            smoothing_window=args.bounce_smoothing_window,
        )
        bounce_positions = bounce_tracker.track(detections)
        bounces = detect_bounces_geometric(
            bounce_positions,
            calibrations_by_frame=calibrations_by_frame if args.show_court else None,
            player_boxes_by_frame=player_boxes_by_frame if args.minimap else None,
            court_margin_m=args.bounce_court_margin,
            player_reach_margin=args.bounce_player_margin,
        )
        print(f"Detected {len(bounces)} bounces")
    elif args.bounce and args.bounce_method == "velocity":
        bounces = detect_bounces_by_velocity(positions, calibrations_by_frame, reader.fps)
        print(f"Detected {len(bounces)} bounces")
    elif args.bounce:
        bounce_sources = frozenset(s.strip() for s in args.bounce_sources.split(",") if s.strip())
        bounce_model = (
            CatBoostBounceDetector(args.bounce_weights, threshold=args.bounce_threshold)
            if "catboost" in bounce_sources
            else None
        )
        bounces = detect_bounces_ensemble(
            positions,
            bounce_model,
            num_frames=num_frames,
            calibrations_by_frame=calibrations_by_frame if args.show_court else None,
            player_boxes_by_frame=player_boxes_by_frame if args.minimap else None,
            fps=reader.fps if args.show_court else None,
            sources=bounce_sources,
            court_margin_m=args.bounce_court_margin,
            net_margin_m=args.bounce_net_margin,
            player_reach_margin=args.bounce_player_margin,
        )
        print(f"Detected {len(bounces)} bounces")
    else:
        bounces = []
    bounces_by_frame = {b.frame_idx: b for b in bounces}
    impacts_by_frame = {impact.frame_idx: impact for impact in impacts}
    minimap_bounce_points = [(b.world_x, b.world_y) for b in bounces if b.world_x is not None]

    # Computed here rather than inside the --stats block below: the
    # structural "serve" label override just after needs it whenever
    # --stroke is on too, independent of whether --stats was requested.
    serve_attempts = (
        classify_serve_sequences(impacts, calibrations_by_frame, identity_by_frame, reader.fps)
        if args.show_court and args.minimap
        else []
    )
    if args.stroke and serve_attempts:
        # A serve is always the first contact of a new point - a fact
        # classify_serve_sequences reads from timing/identity structure
        # alone, independent of pose. Overriding the DISPLAYED label with
        # it (for every attempt, fault retries included - a faulted first
        # serve is still a serve) fixes a real, measured pose-classifier
        # miss found by hand-labelling this project's own footage: a
        # genuine serve shown as "FOREHAND". This never invents a serve
        # that didn't structurally happen, only relabels one the pose
        # model already had a swing to react to.
        for attempt in serve_attempts:
            target = {"near": near_shot_display_by_frame, "far": far_shot_display_by_frame}.get(attempt.side)
            if target is None:
                continue
            # 30 frames matches ShotEventTracker's own default
            # display_recency - how long a label stays up after the frame
            # it was actually detected on.
            for frame_idx in range(attempt.frame_idx, attempt.frame_idx + 30):
                target[frame_idx] = "serve"

    # One flight segmentation, used for both the speed readings and the arc
    # overlay, so the number reported and the curve drawn describe the same
    # fitted flight. Needs the UNSMOOTHED trajectory (see
    # parabolic_bounce_detector) - the analysis pass already tracked one.
    flight_positions = analysis.positions if analysis is not None else None
    if flight_positions is None and (args.speed or args.sidebar or args.ball_overlay == "arc"):
        flight_positions = BallTracker(
            max_pixels_per_frame=args.max_jump,
            max_interpolation_gap=args.interp_gap,
            static_lockon_frames=args.lockon_frames,
            static_lockon_radius=args.lockon_radius,
            smoothing_window=0,
        ).track(detections)
    flight_segments = find_flight_segments(flight_positions) if flight_positions else []
    bounce_frames = [impact.frame_idx for impact in impacts if impact.kind == "bounce"]

    # estimate_flight_speeds measures each fitted flight at whichever
    # instant its calibration error is smallest - the net crossing where
    # one exists, else a bounce at either end of the flight, else its own
    # midpoint - covering every flight rather than only the ~half that
    # cross the net. See its docstring for the full reasoning and the
    # three-tier priority. estimate_shot_speeds still provides the floor
    # for a flight neither method could measure at all, and
    # merge_with_net_crossing_speeds swaps in the sharper number by tier.
    shot_speed_by_frame = {}
    shots: list = []
    if args.speed or args.sidebar or args.stats or args.stats_panel:
        # Segmenting at every direction-reversal (find_trajectory_breakpoints)
        # over-splits a single real shot whenever detector jitter or a
        # mid-flight wobble looks like a local max. detect_bounces_ensemble's
        # output is far more selective (multiple candidate sources, then
        # filtered), so when --bounce is on, its output IS the shot
        # boundary: each segment runs bounce-to-bounce, immune to the false
        # splits a bare local-max scan produces. Without --bounce there's no
        # bounce list to segment on, so this falls back to the old scan.
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
            net_shots = (
                estimate_flight_speeds(
                    flight_segments, reader.fps, calibrations_by_frame, bounce_frames=bounce_frames
                )
                if flight_segments
                else estimate_net_crossing_speeds(
                    positions,
                    reader.fps,
                    calibrations_by_frame,
                    min_window_frames=args.net_speed_min_frames,
                )
            )
            shots = merge_with_net_crossing_speeds(fallback_shots, net_shots)
        else:
            shots = fallback_shots
        if args.speed:
            print(f"Peak speed per shot ({len(shots)} shots):")
            for shot in shots:
                print(
                    f"  frames {shot.start_frame}-{shot.end_frame}: "
                    f"{shot.peak_speed:.0f} {shot.unit}  [{shot.method}]"
                )
        for shot in shots:
            for frame_idx in range(shot.start_frame, shot.end_frame + 1):
                shot_speed_by_frame[frame_idx] = (shot.peak_speed, shot.unit)
    # Keyed by end_frame rather than folded into shot_speed_by_frame: a shot
    # spans a whole range of frames, but StatsPanelDrawer wants to fold a
    # shot's peak into the running total exactly ONCE, the frame it becomes
    # known - not on every frame the shot happens to be displayed on.
    shots_by_end_frame = {shot.end_frame: shot for shot in shots}

    if args.stats:
        # world_points -> a single point per frame: the primary (first)
        # detection - split_top_bottom can return more than one box per
        # half (a stray line-judge/ball-kid detection, or doubles), and
        # match_stats.compute_player_movement wants one trajectory, not a
        # list to disambiguate itself.
        near_player_positions = {
            i: points[0] for i, points in near_players_by_frame.items() if points
        }
        far_player_positions = {
            i: points[0] for i, points in far_players_by_frame.items() if points
        }
        # serve_attempts was already computed above (needed there
        # regardless of --stats, for the stroke-label override) - reused
        # rather than recomputed. Needs a court calibration (to attribute
        # each contact to a side) and identity tracking (to tell "the same
        # server again" from "the other player now serving"), which is
        # exactly what gated computing it in the first place.
        match_log = build_match_log(serve_attempts) if serve_attempts else None
        match_stats = compute_match_stats(
            impacts,
            shots,
            bounces,
            reader.fps,
            near_shot_counts=near_shot_tracker.counts if args.stroke else None,
            far_shot_counts=far_shot_tracker.counts if args.stroke else None,
            near_player_positions_by_frame=near_player_positions if args.minimap else None,
            far_player_positions_by_frame=far_player_positions if args.minimap else None,
            near_shot_events=near_shot_tracker.events if args.stroke else None,
            far_shot_events=far_shot_tracker.events if args.stroke else None,
            calibrations_by_frame=calibrations_by_frame if args.show_court else None,
            scene_cuts=cut_frames,
            match_log=match_log,
        )
        match_stats.write_json(args.stats)
        print(
            f"Wrote {args.stats}: {len(match_stats.rallies)} rally/rallies, "
            f"{match_stats.total_bounces} bounces, {match_stats.total_contacts} contacts"
        )
        if match_stats.contact_side_counts:
            counts = match_stats.contact_side_counts
            print(
                f"Contacts by court half (from contact position, not pose - "
                f"covers both players): near={counts.get('near', 0)}, far={counts.get('far', 0)}"
            )
        if match_stats.match_log is not None and match_stats.match_log.points:
            log = match_stats.match_log
            print(
                f"Serve/point structure (structural, no ball landing spot used - see "
                f"serve_sequences.py): {len(log.points)} point(s), "
                f"faults={log.fault_counts_by}, double_faults={log.double_fault_counts_by}, "
                f"first_serve_in_rate={log.first_serve_in_rate_by}"
            )
        if args.report:
            write_report_html(match_stats, clip_label_from_path(args.input), args.report)
            print(f"Wrote {args.report}")

    output_width = (
        reader.width
        + (args.sidebar_width if args.sidebar else 0)
        + (args.stats_panel_width if args.stats_panel else 0)
    )
    writer = VideoWriter(args.output, reader.fps, output_width, reader.height)
    # The arc comes from the same flight segmentation the speed readings
    # use, so what is drawn and what is reported cannot disagree.
    arc_drawer = None
    trail = None
    if args.ball_overlay == "arc":
        arc_drawer = ShotArcDrawer(flight_segments, impacts)
    else:
        trail = TrailDrawer(trail_length=args.trail_length, fps=reader.fps)
    bounce_drawer = BounceMarkerDrawer(fps=reader.fps) if args.bounce and not args.contacts else None
    impact_drawer = ImpactMarkerDrawer(reader.fps) if args.contacts else None
    sidebar_drawer = SidebarDrawer(width=args.sidebar_width) if args.sidebar else None
    stats_panel_drawer = (
        StatsPanelDrawer(fps=reader.fps, width=args.stats_panel_width) if args.stats_panel else None
    )
    court_drawer = CourtOverlayDrawer() if args.show_court else None
    minimap_drawer = MinimapDrawer() if args.minimap else None
    shot_label_drawer = ShotLabelDrawer() if args.stroke else None

    # A fresh reader (re-decoding from the start) rather than reusing
    # `frames` - that list was never kept around in the first place, see
    # this function's earlier comment on why detection is a streaming pass.
    render_reader = VideoReader(args.input)
    for i, frame in enumerate(render_reader.frames()):
        annotated = frame
        calibration = calibrations_by_frame.get(i) if args.show_court else None
        if args.show_court:
            annotated = court_drawer.draw(annotated, calibration)
        if arc_drawer is not None:
            annotated = arc_drawer.draw(annotated, i)
        else:
            annotated = trail.draw(annotated, positions_by_frame.get(i), impacts_by_frame.get(i))
        if args.contacts:
            annotated = impact_drawer.draw(annotated, i, impacts_by_frame, calibration)
        elif args.bounce:
            annotated = bounce_drawer.draw(annotated, bounces_by_frame.get(i), calibration)
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
        if args.stroke:
            annotated = shot_label_drawer.draw(annotated, near_player_bbox_by_frame.get(i), near_shot_display_by_frame.get(i))
            annotated = shot_label_drawer.draw(annotated, far_player_bbox_by_frame.get(i), far_shot_display_by_frame.get(i))
        if args.sidebar:
            annotated = sidebar_drawer.draw(annotated, near_shot_display_by_frame.get(i), shot_speed_by_frame.get(i))
        if args.stats_panel:
            annotated = stats_panel_drawer.draw(
                annotated,
                i,
                impact=impacts_by_frame.get(i),
                calibration=calibration,
                completed_shot=shots_by_end_frame.get(i),
            )
        writer.write(annotated)
    writer.close()

    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
