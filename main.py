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

All of the above is orchestrated by src/pipeline.py::run_pipeline - this
file is just the argparse layer over it, so the same pipeline can be called
directly (e.g. from a web backend) without going through the CLI.
"""

import argparse
from pathlib import Path

from src.pipeline import PipelineOptions, run_pipeline

REPO_ROOT = Path(__file__).resolve().parent


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
        "--calibration",
        default=None,
        help="Path to a static court calibration JSON (see scripts/calibrate_court.py or "
        "CourtCalibration.save) - a single fixed homography applied to every frame, driving "
        "--speed/--sidebar/--bounce's real-world numbers without needing a trained "
        "court-keypoint model. Mutually exclusive with --show-court, which instead re-detects "
        "the court every frame for a panning/zooming camera.",
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
    args = parser.parse_args()

    options = PipelineOptions(**vars(args))
    try:
        run_pipeline(options)
    except ValueError as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
