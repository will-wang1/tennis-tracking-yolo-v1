"""Reusable entry point for the detect/track/analyze/render pipeline.

This is the same logic `main.py` runs from the command line, extracted into
an importable function so a caller other than argparse (a web backend, a
test, a notebook) can drive it with typed options instead of shelling out to
`python main.py`. `main.py` itself is now a thin wrapper: parse args, build
`PipelineOptions`, call `run_pipeline`, print a summary.

`PipelineOptions` field names deliberately mirror `main.py`'s argparse `dest`
names one-for-one, so the CLI wrapper can build one with
`PipelineOptions(**vars(args))` with no renaming.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional

from tqdm import tqdm

from src.analysis.bounce_detector import BounceEvent, find_trajectory_breakpoints
from src.analysis.bounce_ensemble import detect_bounces_ensemble
from src.analysis.catboost_bounce_detector import CatBoostBounceDetector
from src.analysis.court_calibration import CourtCalibration
from src.analysis.flight_segmenter import find_flight_segments
from src.analysis.geometric_bounce_detector import detect_bounces_geometric
from src.analysis.impact_pipeline import analyze_impacts
from src.analysis.match_stats import MatchStats, compute_match_stats
from src.analysis.parabolic_bounce_detector import BounceCandidate, detect_bounces_parabolic
from src.analysis.shot_classifier import ShotClassifier, ShotEventTracker
from src.analysis.speed_estimator import (
    ShotSpeed,
    estimate_flight_speeds,
    estimate_net_crossing_speeds,
    estimate_shot_speeds,
    merge_with_net_crossing_speeds,
)
from src.analysis.velocity_bounce_detector import detect_bounces_by_velocity
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
    TrailDrawer,
)
from src.visualize.minimap import MinimapDrawer

REPO_ROOT = Path(__file__).resolve().parent.parent

# Unchanged from the old geometric detect_bounces' defaults - only used as
# the shot-speed segmentation fallback now (find_trajectory_breakpoints),
# for when bounce detection is off and there's no bounce list to segment on.
_FALLBACK_BREAKPOINT_MIN_Y_PROMINENCE = 15.0
_FALLBACK_BREAKPOINT_MIN_FRAME_GAP = 5

ProgressCallback = Callable[[str, int, int], None]


@dataclass
class PipelineOptions:
    input: str
    output: str
    detector_backend: str = "wasb"
    weights: str = str(REPO_ROOT / "weights" / "ball_detector.pt")
    tracknet_weights: str = str(REPO_ROOT / "weights" / "tracknet_pretrained.pt")
    wasb_weights: str = str(REPO_ROOT / "weights" / "wasb_tennis_pretrained.pth.tar")
    confidence: float = 0.15
    imgsz: int = 1280
    device: Optional[str] = None
    ball_overlay: str = "trail"
    trail_length: int = 8
    max_jump: float = 150.0
    interp_gap: int = 8
    lockon_frames: int = 10
    lockon_radius: float = 20.0
    smoothing_window: int = 9
    smoothing_polyorder: int = 2
    no_candidate_tracking: bool = False
    no_flight_segments: bool = False
    bounce: bool = False
    contacts: bool = False
    bounce_method: str = "parabolic"
    bounce_smoothing_window: int = 3
    bounce_sources: str = "catboost,geometric,speed_drop"
    bounce_weights: str = str(REPO_ROOT / "weights" / "bounce_catboost_pretrained.cbm")
    bounce_threshold: float = 0.3
    bounce_player_margin: float = 50.0
    bounce_court_margin: float = 2.0
    bounce_net_margin: float = 8.0
    speed: bool = False
    show_court: bool = False
    court_weights: str = str(REPO_ROOT / "weights" / "court_net_pretrained.pt")
    minimap: bool = False
    speed_window: int = 1
    net_speed_min_frames: int = 4
    sidebar: bool = False
    sidebar_width: int = 250
    stroke: bool = False
    movenet_weights: str = str(REPO_ROOT / "weights" / "movenet_singlepose_lightning_int8.tflite")
    stroke_weights: str = str(REPO_ROOT / "weights" / "shot_classifier_rnn_pretrained.h5")
    stats: Optional[str] = None
    # Static, per-video court calibration (see src/analysis/court_calibration.py).
    # Unlike --show-court, this needs no trained keypoint model: it's a single
    # fixed homography (e.g. built from 4 points a user clicked in a browser)
    # applied to every frame. Mutually exclusive with --show-court, which
    # instead re-detects the court every frame for a panning/zooming camera.
    calibration: Optional[str] = None


@dataclass
class PipelineResult:
    output_path: str
    frame_count: int
    fps: float
    detected_frames: int
    interpolated_frames: int
    shots: list[ShotSpeed] = field(default_factory=list)
    bounces: list[BounceEvent] = field(default_factory=list)
    impacts: list[BounceCandidate] = field(default_factory=list)
    match_stats: Optional[MatchStats] = None


def _validate(options: PipelineOptions) -> None:
    if options.stats and not options.bounce:
        raise ValueError("stats requires bounce")
    if options.minimap and not options.show_court:
        raise ValueError("minimap requires show_court")
    if options.stroke and not options.minimap:
        raise ValueError("stroke requires minimap")
    if options.contacts and not (options.bounce and options.bounce_method == "parabolic"):
        raise ValueError("contacts requires bounce with bounce_method parabolic")
    if options.calibration and options.show_court:
        raise ValueError(
            "calibration cannot be combined with show_court - show_court already "
            "auto-calibrates every frame from the trained court-keypoint model"
        )
    if options.bounce_method == "velocity" and not (options.show_court or options.calibration):
        raise ValueError("bounce_method velocity requires show_court or calibration")


def run_pipeline(
    options: PipelineOptions, progress_cb: Optional[ProgressCallback] = None
) -> PipelineResult:
    """Run detection, tracking, analysis, and rendering end to end.

    `progress_cb(stage, done, total)` is called periodically during the two
    frame passes ("detect" and "render") so a caller (e.g. a Celery worker)
    can report progress without polling. Optional - omit for the plain CLI.
    """
    _validate(options)

    if options.detector_backend == "wasb":
        detector = WASBBallDetector(options.wasb_weights, device=options.device)
    elif options.detector_backend == "tracknet":
        detector = TrackNetBallDetector(options.tracknet_weights, device=options.device)
    else:
        detector = BallDetector(
            options.weights,
            confidence=options.confidence,
            imgsz=options.imgsz,
            device=options.device,
        )
    tracker = BallTracker(
        max_pixels_per_frame=options.max_jump,
        max_interpolation_gap=options.interp_gap,
        static_lockon_frames=options.lockon_frames,
        static_lockon_radius=options.lockon_radius,
        smoothing_window=options.smoothing_window,
        smoothing_polyorder=options.smoothing_polyorder,
    )
    court_detector = (
        TennisCourtNetDetector(options.court_weights, device=options.device)
        if options.show_court
        else None
    )
    player_detector = PlayerDetector(device=options.device) if options.minimap else None
    pose_extractor = MoveNetPoseExtractor(options.movenet_weights) if options.stroke else None
    near_shot_classifier = ShotClassifier(options.stroke_weights) if options.stroke else None
    far_shot_classifier = ShotClassifier(options.stroke_weights) if options.stroke else None
    near_shot_tracker = ShotEventTracker() if options.stroke else None
    far_shot_tracker = ShotEventTracker() if options.stroke else None

    reader = VideoReader(options.input)
    frames = list(reader.frames())
    if not frames:
        raise ValueError(f"No frames read from {options.input}")

    static_calibration = CourtCalibration.load(options.calibration) if options.calibration else None
    have_calibration = options.show_court or static_calibration is not None

    print(f"Detecting ball in {len(frames)} frames...")
    detections = []
    calibrations_by_frame: dict[int, CourtCalibration] = (
        {i: static_calibration for i in range(len(frames))} if static_calibration is not None else {}
    )
    last_good_calibration: "CourtCalibration | None" = None
    far_players_by_frame: dict[int, list[tuple[float, float]]] = {}
    near_players_by_frame: dict[int, list[tuple[float, float]]] = {}
    player_boxes_by_frame: dict[int, list[tuple[float, float, float, float]]] = {}
    near_shot_display_by_frame: dict[int, str] = {}
    far_shot_display_by_frame: dict[int, str] = {}
    near_player_bbox_by_frame: dict[int, tuple[float, float, float, float]] = {}
    far_player_bbox_by_frame: dict[int, tuple[float, float, float, float]] = {}
    use_candidates = options.detector_backend == "wasb" and not options.no_candidate_tracking
    candidates_by_frame: list[list] = []
    for i, frame in enumerate(tqdm(frames)):
        if use_candidates:
            candidates_by_frame.append(detector.detect_candidates(frame))
            detections.append(None)  # filled in below, once the whole path is known
        else:
            detections.append(detector.detect(frame))
        if options.show_court:
            # A frame where the detector doesn't find enough confident
            # keypoints (motion blur, court briefly out of frame) carries
            # forward the last good calibration rather than leaving a gap -
            # the camera can't have moved far in one missed frame.
            named_points = court_detector.detect(frame) or {}
            if len(named_points) >= 4:
                last_good_calibration = CourtCalibration.from_keypoints(named_points)
            if last_good_calibration is not None:
                calibrations_by_frame[i] = last_good_calibration
            if options.minimap:
                calibration = calibrations_by_frame.get(i)
                players = player_detector.detect(frame, calibration=calibration)
                far, near = PlayerDetector.split_top_bottom(players)
                far_players_by_frame[i] = [p.world_point for p in far]
                near_players_by_frame[i] = [p.world_point for p in near]
                player_boxes_by_frame[i] = [p.bbox for p in players]

                if options.stroke:
                    if near:
                        near_player_bbox_by_frame[i] = near[0].bbox
                    if far:
                        far_player_bbox_by_frame[i] = far[0].bbox

                    near_pose = pose_extractor.extract(frame, near[0].bbox) if near else None
                    near_prediction = near_shot_classifier.update(near_pose)
                    near_display_label = near_shot_tracker.update(i, near_prediction)
                    if near_display_label is not None:
                        near_shot_display_by_frame[i] = near_display_label

                    far_pose = pose_extractor.extract(frame, far[0].bbox) if far else None
                    far_prediction = far_shot_classifier.update(far_pose)
                    far_display_label = far_shot_tracker.update(i, far_prediction)
                    if far_display_label is not None:
                        far_shot_display_by_frame[i] = far_display_label
        if progress_cb:
            progress_cb("detect", i + 1, len(frames))

    if options.stroke:
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
        detections = track_candidates(candidates_by_frame, max_pixels_per_frame=options.max_jump)
        kept = sum(1 for d in detections if d is not None)
        offered = sum(1 for row in candidates_by_frame if row)
        print(
            f"Candidate tracking: {kept}/{len(frames)} frames tracked from {offered} frames offering candidates"
        )

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

    impacts: list[BounceCandidate] = []
    analysis = None
    if options.bounce and options.bounce_method == "parabolic":
        # Judging bounce-vs-contact needs the ball's PROJECTED court
        # position, which measures the height effect directly, rather than
        # screen y, which confuses height with court depth - so the
        # calibration is what upgrades this from "something hit the ball"
        # to "the court hit the ball". Player boxes are passed only to
        # WITHHOLD the contact verdict from impacts nobody could have
        # reached - see touchdown_detector.classify_touchdowns. They are
        # only collected under --minimap, and the classifier degrades to
        # the direction rule alone without them.
        analysis = analyze_impacts(
            detections,
            reader.fps,
            calibrations_by_frame=calibrations_by_frame if have_calibration else None,
            player_boxes_by_frame=player_boxes_by_frame if options.minimap else None,
            max_pixels_per_frame=options.max_jump,
            max_interpolation_gap=options.interp_gap,
            static_lockon_frames=options.lockon_frames,
            static_lockon_radius=options.lockon_radius,
            use_flight_segments=not options.no_flight_segments,
        )
        impacts = analysis.impacts
        if have_calibration:
            # The bounce's world position is reprojected from the MAIN
            # (smoothed) trajectory's position at that frame - the same
            # position the minimap's live ball dot and the on-screen trail
            # are drawn from - rather than from the impact's own unsmoothed
            # x/y. Using the displayed position instead means the landing
            # mark always sits on top of the visible ball, at the cost of a
            # little precision in exchange for a picture that agrees with
            # itself. Classification (is_bounce, kind, timing) is untouched
            # - only where the result is DRAWN changes.
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
        if options.contacts:
            contacts = analysis.contacts
            unknown = analysis.unattributed
            print(
                f"Detected {len(contacts)} racket contacts and {len(unknown)} unattributed "
                f"impacts ({len(impacts)} in total). Only bounces and contacts are drawn:"
            )
            for impact in impacts:
                label = {"bounce": "BOUNCE ", "contact": "contact", "unknown": "  -    "}[impact.kind]
                print(f"  {impact.t / reader.fps:6.2f}s  {label}  {impact.reason}")
    elif options.bounce and options.bounce_method == "geometric":
        # A separate, lightly smoothed tracking pass - the window used for
        # the visible trail/speed readings rounds off exactly the corner
        # this method looks for (see geometric_bounce_detector.py).
        bounce_tracker = BallTracker(
            max_pixels_per_frame=options.max_jump,
            max_interpolation_gap=options.interp_gap,
            static_lockon_frames=options.lockon_frames,
            static_lockon_radius=options.lockon_radius,
            smoothing_window=options.bounce_smoothing_window,
        )
        bounce_positions = bounce_tracker.track(detections)
        bounces = detect_bounces_geometric(
            bounce_positions,
            calibrations_by_frame=calibrations_by_frame if have_calibration else None,
            player_boxes_by_frame=player_boxes_by_frame if options.minimap else None,
            court_margin_m=options.bounce_court_margin,
            player_reach_margin=options.bounce_player_margin,
        )
        print(f"Detected {len(bounces)} bounces")
    elif options.bounce and options.bounce_method == "velocity":
        bounces = detect_bounces_by_velocity(positions, calibrations_by_frame, reader.fps)
        print(f"Detected {len(bounces)} bounces")
    elif options.bounce:
        bounce_sources = frozenset(s.strip() for s in options.bounce_sources.split(",") if s.strip())
        bounce_model = (
            CatBoostBounceDetector(options.bounce_weights, threshold=options.bounce_threshold)
            if "catboost" in bounce_sources
            else None
        )
        bounces = detect_bounces_ensemble(
            positions,
            bounce_model,
            num_frames=len(frames),
            calibrations_by_frame=calibrations_by_frame if have_calibration else None,
            player_boxes_by_frame=player_boxes_by_frame if options.minimap else None,
            fps=reader.fps if have_calibration else None,
            sources=bounce_sources,
            court_margin_m=options.bounce_court_margin,
            net_margin_m=options.bounce_net_margin,
            player_reach_margin=options.bounce_player_margin,
        )
        print(f"Detected {len(bounces)} bounces")
    else:
        bounces = []
    bounces_by_frame = {b.frame_idx: b for b in bounces}
    impacts_by_frame = {impact.frame_idx: impact for impact in impacts}
    minimap_bounce_points = [(b.world_x, b.world_y) for b in bounces if b.world_x is not None]

    # One flight segmentation, used for both the speed readings and the arc
    # overlay, so the number reported and the curve drawn describe the same
    # fitted flight. Needs the UNSMOOTHED trajectory (see
    # parabolic_bounce_detector) - the analysis pass already tracked one.
    flight_positions = analysis.positions if analysis is not None else None
    if flight_positions is None and (options.speed or options.sidebar or options.ball_overlay == "arc"):
        flight_positions = BallTracker(
            max_pixels_per_frame=options.max_jump,
            max_interpolation_gap=options.interp_gap,
            static_lockon_frames=options.lockon_frames,
            static_lockon_radius=options.lockon_radius,
            smoothing_window=0,
        ).track(detections)
    flight_segments = find_flight_segments(flight_positions) if flight_positions else []
    bounce_frames = [impact.frame_idx for impact in impacts if impact.kind == "bounce"]

    # estimate_flight_speeds measures each fitted flight at whichever
    # instant its calibration error is smallest - the net crossing where
    # one exists, else a bounce at either end of the flight, else its own
    # midpoint - covering every flight rather than only the ~half that
    # cross the net. estimate_shot_speeds still provides the floor for a
    # flight neither method could measure at all, and
    # merge_with_net_crossing_speeds swaps in the sharper number by tier.
    shot_speed_by_frame = {}
    shots: list[ShotSpeed] = []
    if options.speed or options.sidebar or options.stats:
        # Segmenting at every direction-reversal (find_trajectory_breakpoints)
        # over-splits a single real shot whenever detector jitter or a
        # mid-flight wobble looks like a local max. detect_bounces_ensemble's
        # output is far more selective (multiple candidate sources, then
        # filtered), so when bounce detection is on, its output IS the shot
        # boundary: each segment runs bounce-to-bounce, immune to the false
        # splits a bare local-max scan produces. Without bounce detection
        # there's no bounce list to segment on, so this falls back to the
        # old scan.
        breakpoint_frames = (
            [b.frame_idx for b in bounces]
            if options.bounce
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
            speed_window=options.speed_window,
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
                    min_window_frames=options.net_speed_min_frames,
                )
            )
            shots = merge_with_net_crossing_speeds(fallback_shots, net_shots)
        else:
            shots = fallback_shots
        if options.speed:
            print(f"Peak speed per shot ({len(shots)} shots):")
            for shot in shots:
                print(
                    f"  frames {shot.start_frame}-{shot.end_frame}: "
                    f"{shot.peak_speed:.0f} {shot.unit}  [{shot.method}]"
                )
        for shot in shots:
            for frame_idx in range(shot.start_frame, shot.end_frame + 1):
                shot_speed_by_frame[frame_idx] = (shot.peak_speed, shot.unit)

    match_stats: Optional[MatchStats] = None
    if options.bounce:
        # Computed whenever bounce detection ran, not only when a JSON path
        # is given - a caller (e.g. the web backend) may want the structured
        # MatchStats object without writing a file.
        match_stats = compute_match_stats(
            impacts,
            shots,
            bounces,
            reader.fps,
            near_shot_counts=near_shot_tracker.counts if options.stroke else None,
            far_shot_counts=far_shot_tracker.counts if options.stroke else None,
        )
        if options.stats:
            match_stats.write_json(options.stats)
            print(
                f"Wrote {options.stats}: {len(match_stats.rallies)} rally/rallies, "
                f"{match_stats.total_bounces} bounces, {match_stats.total_contacts} contacts"
            )

    output_width = reader.width + (options.sidebar_width if options.sidebar else 0)
    writer = VideoWriter(options.output, reader.fps, output_width, reader.height)
    # The arc comes from the same flight segmentation the speed readings
    # use, so what is drawn and what is reported cannot disagree.
    arc_drawer = None
    trail = None
    if options.ball_overlay == "arc":
        arc_drawer = ShotArcDrawer(flight_segments, impacts)
    else:
        trail = TrailDrawer(trail_length=options.trail_length)
    bounce_drawer = BounceMarkerDrawer() if options.bounce and not options.contacts else None
    impact_drawer = ImpactMarkerDrawer(reader.fps) if options.contacts else None
    sidebar_drawer = SidebarDrawer(width=options.sidebar_width) if options.sidebar else None
    court_drawer = CourtOverlayDrawer() if options.show_court else None
    minimap_drawer = MinimapDrawer() if options.minimap else None
    shot_label_drawer = ShotLabelDrawer() if options.stroke else None

    for i, frame in enumerate(frames):
        annotated = frame
        calibration = calibrations_by_frame.get(i) if have_calibration else None
        if options.show_court:
            annotated = court_drawer.draw(annotated, calibration)
        if arc_drawer is not None:
            annotated = arc_drawer.draw(annotated, i)
        else:
            annotated = trail.draw(annotated, positions_by_frame.get(i))
        if options.contacts:
            annotated = impact_drawer.draw(annotated, i, impacts_by_frame, calibration)
        elif options.bounce:
            annotated = bounce_drawer.draw(annotated, bounces_by_frame.get(i), calibration)
        if options.minimap:
            position = positions_by_frame.get(i)
            ball_world = calibration.pixel_to_world(position.x, position.y) if position and calibration else None
            annotated = minimap_drawer.draw(
                annotated,
                ball_world=ball_world,
                bounce_world_points=minimap_bounce_points,
                far_players_world=far_players_by_frame.get(i),
                near_players_world=near_players_by_frame.get(i),
            )
        if options.stroke:
            annotated = shot_label_drawer.draw(annotated, near_player_bbox_by_frame.get(i), near_shot_display_by_frame.get(i))
            annotated = shot_label_drawer.draw(annotated, far_player_bbox_by_frame.get(i), far_shot_display_by_frame.get(i))
        if options.sidebar:
            annotated = sidebar_drawer.draw(annotated, near_shot_display_by_frame.get(i), shot_speed_by_frame.get(i))
        writer.write(annotated)
        if progress_cb:
            progress_cb("render", i + 1, len(frames))
    writer.close()

    print(f"Wrote {options.output}")

    return PipelineResult(
        output_path=options.output,
        frame_count=len(frames),
        fps=reader.fps,
        detected_frames=detected,
        interpolated_frames=interpolated,
        shots=shots,
        bounces=bounces,
        impacts=impacts,
        match_stats=match_stats,
    )
