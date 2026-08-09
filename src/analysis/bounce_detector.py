"""Detect ball bounce/landing events from an already-tracked trajectory.

Screen y increases downward, so the ball's lowest point on screen - its
bounce - is a local MAXIMUM in y among consecutive tracked samples, with the
implied vertical motion flipping from falling to rising around it. A shot's
apex is the opposite: a local MINIMUM in y, so it's never confused with a
bounce by the y-check alone.

A local max in y is NOT enough on its own, though: a racket CONTACT reverses
the ball's y-direction too, and for a broadcast camera looking down the
court from behind a baseline, 2D pixel-y conflates two different things -
how HIGH the ball is off the ground, and how FAR down the court it is
(perspective). A shot travelling toward the far player and a ball falling
toward the ground both move y the same way, so contact and bounce can look
identical in y alone. Two supplementary signals help:

- Horizontal direction: a bounce continues rolling the way the ball was
  already travelling; a contact tends to send it back where it came from.
  This is a real but weaker signal for this camera angle specifically,
  since lateral (x) motion isn't where the primary court-crossing action
  is - see `min_x_reversal`.
- Ground proximity (`ground_y_by_frame`, optional): the far stronger signal.
  A real bounce happens at court level, so if a nearby player's foot/ankle
  position is available for that frame, a genuine bounce should land close
  to it - not up near racket/torso height. Pass this whenever pose data is
  available; without it, this check is skipped and only the weaker
  trajectory-only signals apply.

This is a pure post-processing pass over `BallTracker.track()`'s output -
no model, no new detections, just geometry on the existing trajectory (plus
optionally cross-referencing already-computed pose data).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from src.analysis.court_calibration import CourtCalibration
from src.tracking.ball_tracker import TrackedPosition

if TYPE_CHECKING:
    from src.analysis.bounce_classifier import BounceClassifier


@dataclass
class BounceEvent:
    frame_idx: int
    x: float
    y: float
    world_x: Optional[float] = None
    world_y: Optional[float] = None


def find_trajectory_breakpoints(
    positions: list[TrackedPosition],
    min_y_prominence: float = 15.0,
    min_frame_gap: int = 5,
) -> list[int]:
    """Frame indices of every local-max-in-y point in the trajectory,
    regardless of whether it's a bounce or a contact - unlike
    `detect_bounces`, this makes no attempt to tell the two apart.

    Useful as natural boundaries between "shots" for speed segmentation: a
    contact reliably marks the end of one shot and the start of the next
    just as well as a bounce does, and is generally easier to detect
    cleanly (see `detect_bounces`'s module docstring for why confidently
    distinguishing a bounce from a contact is a much harder problem).
    """
    ordered = sorted(positions, key=lambda p: p.frame_idx)

    candidate_indices: list[int] = []
    for i in range(1, len(ordered) - 1):
        prev_pos, cur_pos, next_pos = ordered[i - 1], ordered[i], ordered[i + 1]
        if cur_pos.frame_idx - prev_pos.frame_idx != 1 or next_pos.frame_idx - cur_pos.frame_idx != 1:
            continue
        if cur_pos.y > prev_pos.y and cur_pos.y > next_pos.y:
            prominence = min(cur_pos.y - prev_pos.y, cur_pos.y - next_pos.y)
            if prominence >= min_y_prominence:
                candidate_indices.append(i)

    merged_indices: list[int] = []
    for idx in candidate_indices:
        if (
            merged_indices
            and ordered[idx].frame_idx - ordered[merged_indices[-1]].frame_idx <= min_frame_gap
        ):
            if ordered[idx].y > ordered[merged_indices[-1]].y:
                merged_indices[-1] = idx
        else:
            merged_indices.append(idx)

    return [ordered[idx].frame_idx for idx in merged_indices]


def _is_horizontal_reversal(
    prev_pos: TrackedPosition, cur_pos: TrackedPosition, next_pos: TrackedPosition, min_x_reversal: float
) -> bool:
    """True if x-direction flips across `cur_pos` with enough magnitude on
    both sides to be a real reversal, not noise. A near-vertical shot (dx
    close to 0 on both sides, e.g. a lob) isn't a reversal either way, so a
    small dx never counts as one."""
    dx_before = cur_pos.x - prev_pos.x
    dx_after = next_pos.x - cur_pos.x
    if abs(dx_before) < min_x_reversal or abs(dx_after) < min_x_reversal:
        return False
    return (dx_before > 0) != (dx_after > 0)


def detect_bounces(
    positions: list[TrackedPosition],
    min_y_prominence: float = 15.0,
    min_frame_gap: int = 5,
    min_x_reversal: float = 15.0,
    calibration: Optional[CourtCalibration] = None,
    ground_y_by_frame: Optional[dict[int, float]] = None,
    max_height_above_ground: float = 160.0,
) -> list[BounceEvent]:
    """Find bounce points in `positions` (need not be pre-sorted).

    `min_y_prominence` filters ordinary parabolic-motion sampling noise, not
    just sub-pixel jitter - real footage moves the ball ~9px/frame at the
    median even mid-flight, so this needs to be well above typical
    frame-to-frame motion, not just above zero, or ordinary arcs get flagged
    as bounces. `min_x_reversal` rejects candidates where the ball's
    horizontal direction also flips - a weak supplementary signal, see
    module docstring. `min_frame_gap` merges extrema that detector noise
    splits into multiple nearby candidates around one real bounce, keeping
    the most prominent.

    `ground_y_by_frame` (optional, frame_idx -> a nearby player's foot/ankle
    y for that frame - see `src.analysis.striker.estimate_ground_y`) is the
    strongest available signal: a candidate more than
    `max_height_above_ground` pixels above the given ground reference is
    rejected as too high off the court to be a real bounce - almost
    certainly a contact instead. Frames with no ground reference available
    fall back to the trajectory-only signals above.
    """
    ordered = sorted(positions, key=lambda p: p.frame_idx)

    candidate_indices: list[int] = []
    for i in range(1, len(ordered) - 1):
        prev_pos, cur_pos, next_pos = ordered[i - 1], ordered[i], ordered[i + 1]
        if cur_pos.frame_idx - prev_pos.frame_idx != 1 or next_pos.frame_idx - cur_pos.frame_idx != 1:
            continue  # a real tracking gap isn't a temporal neighbor - don't compare across it
        if cur_pos.y > prev_pos.y and cur_pos.y > next_pos.y:
            if _is_horizontal_reversal(prev_pos, cur_pos, next_pos, min_x_reversal):
                continue  # the ball was redirected by a racket, not the court
            ground_y = ground_y_by_frame.get(cur_pos.frame_idx) if ground_y_by_frame else None
            if ground_y is not None and cur_pos.y < ground_y - max_height_above_ground:
                continue  # too far above the court to be a landing - likely a contact
            prominence = min(cur_pos.y - prev_pos.y, cur_pos.y - next_pos.y)
            if prominence >= min_y_prominence:
                candidate_indices.append(i)

    merged_indices: list[int] = []
    for idx in candidate_indices:
        if (
            merged_indices
            and ordered[idx].frame_idx - ordered[merged_indices[-1]].frame_idx <= min_frame_gap
        ):
            if ordered[idx].y > ordered[merged_indices[-1]].y:
                merged_indices[-1] = idx
        else:
            merged_indices.append(idx)

    events = []
    for idx in merged_indices:
        pos = ordered[idx]
        world_x = world_y = None
        if calibration is not None:
            world_x, world_y = calibration.pixel_to_world(pos.x, pos.y)
        events.append(
            BounceEvent(frame_idx=pos.frame_idx, x=pos.x, y=pos.y, world_x=world_x, world_y=world_y)
        )
    return events


def detect_bounces_ml(
    positions: list[TrackedPosition],
    classifier: "BounceClassifier",
    min_y_prominence: float = 1.0,
    min_frame_gap: int = 5,
    window: int = 9,
    threshold: float = 0.5,
    calibration: Optional[CourtCalibration] = None,
) -> list[BounceEvent]:
    """Bounce detection via a trained classifier
    (src.analysis.bounce_classifier.BounceClassifier, see
    scripts/train_bounce_classifier.py) instead of `detect_bounces`' fixed
    x-reversal/ground-proximity thresholds.

    Candidates come from the same broad, low-threshold local-max-in-y scan
    scripts/extract_bounce_candidates.py uses (`find_trajectory_breakpoints`
    - note the deliberately low `min_y_prominence` default here, NOT
    `detect_bounces`' stricter 15.0: the classifier, not the threshold, is
    what discriminates a real bounce from a contact or noise among them, so
    starving it of borderline candidates up front would defeat the point).
    A `min_y_prominence` this permissive would flag far too much as a real
    bounce under `detect_bounces`' heuristics, but here it only decides
    what's worth asking the classifier about.
    """
    ordered = sorted(positions, key=lambda p: p.frame_idx)
    index_by_frame = {p.frame_idx: i for i, p in enumerate(ordered)}

    candidate_frames = find_trajectory_breakpoints(
        positions, min_y_prominence=min_y_prominence, min_frame_gap=min_frame_gap
    )

    half_window = window // 2
    events = []
    for center_frame in candidate_frames:
        center_idx = index_by_frame[center_frame]
        start_idx = max(0, center_idx - half_window)
        end_idx = min(len(ordered) - 1, center_idx + half_window)
        window_positions = ordered[start_idx : end_idx + 1]
        local_center_idx = center_idx - start_idx

        if local_center_idx <= 0 or local_center_idx >= len(window_positions) - 1:
            continue  # candidate sits right at the tracked trajectory's edge
        if not classifier.is_bounce(window_positions, local_center_idx, threshold=threshold):
            continue

        pos = ordered[center_idx]
        world_x = world_y = None
        if calibration is not None:
            world_x, world_y = calibration.pixel_to_world(pos.x, pos.y)
        events.append(
            BounceEvent(frame_idx=pos.frame_idx, x=pos.x, y=pos.y, world_x=world_x, world_y=world_y)
        )
    return events
