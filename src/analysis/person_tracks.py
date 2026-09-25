"""Turn cached person tracks (scripts/track_people.py) into court-space
movement: where each tracked person stood, how far they moved, and how
much of that number can be trusted.

A person's court position is their FOOT POINT - bottom-centre of the
detection box - through the court calibration. That is the one point of a
standing person that is actually on the ground plane the homography is
exact for.

SMOOTHING IS NOT OPTIONAL. Box edges jitter by a few pixels every frame
with nobody moving, and summing frame-to-frame steps turns jitter straight
into distance. Measured on the first continuous shot of
dingles_serve_volley, raw vs 0.5s rolling median:

    near-baseline player     40.2m -> 35.8m   (then flat: real motion)
    far-baseline bystander   41.7m -> 19.5m   (and still falling at 1.5s)

The near curve flattens by ~9-13 frames - the jitter is gone and what is
left is the person. The far curve never flattens, because there one pixel
is ~13cm of court (0.128 m/px vs 0.010 near the camera on this
fence-height view), so jitter and walking are the same size. That is why
every distance here carries a CONFIDENCE derived from the metres-per-pixel
at the person's own position, rather than one global caveat: the
near-court numbers are measurements, the far-court ones are estimates, and
a reader - human or model - needs to know which is which per person.

Identity limits are the tracker's (see scripts/track_people.py): an id is
a continuous stretch of one person on screen, not a named player, and a
person hidden longer than the tracker's buffer returns under a new id.
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS, CourtCalibration
from src.analysis.match_stats import compute_player_movement

COURT_WIDTH_M = FULL_COURT_REFERENCE_POINTS["baseline_far_right"][0]
COURT_LENGTH_M = FULL_COURT_REFERENCE_POINTS["baseline_near_left"][1]

# How far outside the lines still counts as THIS court. A player chasing a
# wide ball or queueing behind the baseline is still in the session; a
# player on the next court is not. Adjacent indoor courts typically sit
# ~3.5-4m apart sideline to sideline, so the side margin stops short of
# that, and the end margin covers a normal run-off behind the baseline.
DEFAULT_SIDE_MARGIN_M = 3.0
DEFAULT_END_MARGIN_M = 6.4

# 0.5s: where the near-court distance curve flattens (see module docstring).
DEFAULT_SMOOTHING_FRAMES = 13

# Above this many court-metres per image pixel, box jitter is the same size
# as a walking step and a distance is an estimate, not a measurement.
# On the measured fence camera: 0.010 near the camera, 0.128 at the far
# baseline; the threshold sits between, nearer the good end.
LOW_CONFIDENCE_METRES_PER_PIXEL = 0.05

COVERAGE_CELL_M = 1.0


@dataclass(frozen=True)
class PersonMovement:
    track_id: int
    start_frame: int
    end_frame: int
    tracked_frames: int
    median_position_m: tuple[float, float]
    on_this_court: bool
    distance_m: Optional[float]
    average_speed_mps: Optional[float]
    max_speed_mps: Optional[float]
    metres_per_pixel: float
    distance_confidence: str  # "measured" | "estimated"

    def to_dict(self, fps: float) -> dict:
        return {
            "track_id": self.track_id,
            "start_s": round(self.start_frame / fps, 2),
            "end_s": round(self.end_frame / fps, 2),
            "tracked_s": round(self.tracked_frames / fps, 2),
            "median_position_m": [round(v, 2) for v in self.median_position_m],
            "on_this_court": self.on_this_court,
            "distance_m": None if self.distance_m is None else round(self.distance_m, 1),
            "average_speed_mps": None if self.average_speed_mps is None else round(self.average_speed_mps, 2),
            "max_speed_mps": None if self.max_speed_mps is None else round(self.max_speed_mps, 2),
            "metres_per_pixel": round(self.metres_per_pixel, 4),
            "distance_confidence": self.distance_confidence,
        }


@dataclass(frozen=True)
class CoverageGrid:
    """Seconds of on-court presence per 1m cell, over the court plus its
    run-off. `origin_m` is the world (x, y) of cell [0][0]'s corner."""

    origin_m: tuple[float, float]
    cell_m: float
    seconds: list[list[float]] = field(default_factory=list)  # [row (y)][col (x)]

    def to_dict(self) -> dict:
        return {
            "origin_m": list(self.origin_m),
            "cell_m": self.cell_m,
            "seconds": [[round(v, 2) for v in row] for row in self.seconds],
        }


def foot_positions_by_track(
    people_by_frame: list[list[tuple]],
    calibration_for_frame: Callable[[int], Optional[CourtCalibration]],
    skip_frames: Iterable[int] = (),
) -> dict[int, dict[int, tuple[float, float]]]:
    """track_id -> {frame: world (x, y)} from a track_people.py cache's
    `people` list. Frames in `skip_frames` (camera cutaways, where the
    calibration describes a different view) and untracked boxes are
    dropped rather than guessed at."""
    skip = set(skip_frames)
    tracks: dict[int, dict[int, tuple[float, float]]] = {}
    for frame, row in enumerate(people_by_frame):
        if frame in skip:
            continue
        calibration = calibration_for_frame(frame)
        if calibration is None:
            continue
        for track_id, x1, _y1, x2, y2, _conf in row:
            if track_id is None:
                continue
            world = calibration.pixel_to_world((x1 + x2) / 2.0, y2)
            tracks.setdefault(int(track_id), {})[frame] = (float(world[0]), float(world[1]))
    return tracks


def smooth_positions(
    positions: dict[int, tuple[float, float]], window: int = DEFAULT_SMOOTHING_FRAMES
) -> dict[int, tuple[float, float]]:
    """Rolling MEDIAN over `window` consecutive samples. Median rather than
    mean so one wildly wrong box (a limb clipped, two people merged) cannot
    drag the path toward itself."""
    if window <= 1 or len(positions) < 2:
        return dict(positions)
    frames = sorted(positions)
    points = np.array([positions[f] for f in frames], dtype=float)
    half = window // 2
    out = {}
    for i, frame in enumerate(frames):
        segment = points[max(0, i - half): i + half + 1]
        median = np.median(segment, axis=0)
        out[frame] = (float(median[0]), float(median[1]))
    return out


def is_on_this_court(
    position_m: tuple[float, float],
    side_margin_m: float = DEFAULT_SIDE_MARGIN_M,
    end_margin_m: float = DEFAULT_END_MARGIN_M,
) -> bool:
    x, y = position_m
    return (
        -side_margin_m <= x <= COURT_WIDTH_M + side_margin_m
        and -end_margin_m <= y <= COURT_LENGTH_M + end_margin_m
    )


def metres_per_pixel_at(calibration: CourtCalibration, position_m: tuple[float, float]) -> float:
    """Court metres covered by one image pixel of vertical movement at this
    court position - the resolution a foot point is measured at there."""
    px, py = calibration.world_to_pixel(*position_m)
    a = calibration.pixel_to_world(px, py)
    b = calibration.pixel_to_world(px, py - 1.0)
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


def summarise_movement(
    tracks: dict[int, dict[int, tuple[float, float]]],
    fps: float,
    calibration: CourtCalibration,
    smoothing_frames: int = DEFAULT_SMOOTHING_FRAMES,
    min_tracked_seconds: float = 1.0,
) -> list[PersonMovement]:
    """One PersonMovement per track long enough to say anything about,
    longest first."""
    out = []
    for track_id, raw in tracks.items():
        if len(raw) < min_tracked_seconds * fps:
            continue
        smoothed = smooth_positions(raw, smoothing_frames)
        pts = np.array(list(raw.values()))
        median = (float(np.median(pts[:, 0])), float(np.median(pts[:, 1])))
        movement = compute_player_movement(smoothed, fps)
        mpp = metres_per_pixel_at(calibration, median)
        out.append(
            PersonMovement(
                track_id=track_id,
                start_frame=min(raw),
                end_frame=max(raw),
                tracked_frames=len(raw),
                median_position_m=median,
                on_this_court=is_on_this_court(median),
                distance_m=movement.distance_m if movement else None,
                average_speed_mps=movement.average_speed_mps if movement else None,
                max_speed_mps=movement.max_speed_mps if movement else None,
                metres_per_pixel=mpp,
                distance_confidence="estimated" if mpp > LOW_CONFIDENCE_METRES_PER_PIXEL else "measured",
            )
        )
    return sorted(out, key=lambda m: -m.tracked_frames)


def coverage_grid(
    tracks: dict[int, dict[int, tuple[float, float]]],
    fps: float,
    track_ids: Optional[Iterable[int]] = None,
    cell_m: float = COVERAGE_CELL_M,
    side_margin_m: float = DEFAULT_SIDE_MARGIN_M,
    end_margin_m: float = DEFAULT_END_MARGIN_M,
) -> CoverageGrid:
    """Seconds spent in each cell by the given tracks (all, by default).
    Positions outside the court-plus-margins are left out rather than
    clamped to the edge, which would paint a false band along it."""
    x0, y0 = -side_margin_m, -end_margin_m
    cols = int(np.ceil((COURT_WIDTH_M + 2 * side_margin_m) / cell_m))
    rows = int(np.ceil((COURT_LENGTH_M + 2 * end_margin_m) / cell_m))
    grid = np.zeros((rows, cols))
    wanted = set(track_ids) if track_ids is not None else None
    for track_id, positions in tracks.items():
        if wanted is not None and track_id not in wanted:
            continue
        for x, y in positions.values():
            c = int((x - x0) // cell_m)
            r = int((y - y0) // cell_m)
            if 0 <= r < rows and 0 <= c < cols:
                grid[r, c] += 1.0 / fps
    return CoverageGrid(origin_m=(x0, y0), cell_m=cell_m, seconds=grid.tolist())


def boxes_by_frame(people_by_frame: list[list[tuple]]) -> dict[int, list[tuple[float, float, float, float]]]:
    """Every person box per frame, tracked or not, in the shape
    `analyze_impacts(player_boxes_by_frame=...)` takes.

    The impact classifier was built to have player boxes - they are what
    lets it withhold a "contact" nobody could have reached, and what stops
    it calling a far player's shot "ball leaving the top of frame". Without
    them, on the fence-height Dingles clip, 27 of 59 unattributed impacts
    were that frame-edge rule firing on the far baseline (y~193px, inside
    the 150px edge band); with these boxes it is 7, and 46.6% of impacts
    come out of every player's reach - inside the 26-68% measured on match
    footage, so the reach gate works on a crowded coaching court."""
    return {
        frame: [(x1, y1, x2, y2) for _id, x1, y1, x2, y2, _conf in row]
        for frame, row in enumerate(people_by_frame)
        if row
    }
