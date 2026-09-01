"""Choose the ball's path through several detector candidates per frame,
instead of trusting the strongest blob in each frame on its own.

`WASBBallDetector.detect_candidates` now hands back every plausible blob in
a frame rather than only the best one, because the strongest peak is often
not the ball: a player's shoe, a court line, the net cord and a bright
background patch all light the heatmap up, and at the moments that matter
most - a bounce or a racket contact - the ball is blurred and its own peak
is at its weakest. Picking per-frame maxima throws the real ball away in
exactly those frames. Measured on this project's clips, per-frame maxima at
the detector's default threshold left 15 multi-frame dropouts in one clip,
several of them straddling a hand-confirmed bounce.

Which candidate is the ball is a question about the TRAJECTORY, though, not
about any single frame. A ball moves smoothly and its velocity changes
slowly except at an impact; a false peak jumps around. So this scores whole
paths rather than points, by dynamic programming over the candidate lattice
(a Viterbi pass): each step pays for how far the ball would have had to move
and how sharply it would have had to turn, and is rewarded for the
detector's own confidence. The best-scoring path through the whole sequence
is the track.

This is the "tracking" half of WASB-SBDT, whose detector this project
already uses - the upstream repo pairs its detector with an online tracker
for exactly this reason, and running the detector alone leaves that on the
table.

Gaps are handled by letting the path SKIP frames at a cost: a run of frames
where the ball genuinely isn't visible (occluded by a player, out of frame)
should not force a bad candidate into the track. `max_skip` bounds how far
it can coast, and the skip penalty is what stops it from skipping
everything.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from src.detection.ball_detector import Detection


@dataclass(frozen=True)
class _Node:
    """One candidate in one frame, with the best path that reaches it."""

    cost: float
    previous: Optional[tuple[int, int]]  # (frame index, candidate index)


def _step_cost(
    previous: Detection,
    current: Detection,
    before_previous: Optional[Detection],
    frames_apart: int,
    max_pixels_per_frame: float,
    turn_weight: float,
    min_speed: float,
    static_penalty: float,
) -> Optional[float]:
    """What it costs to say these two detections are the same ball.

    Distance is measured per frame elapsed, so coasting across a gap is
    judged on the same scale as a single step. Returns None when the move is
    faster than a tennis ball can travel, which prunes the lattice.
    """
    distance = float(np.hypot(current.x - previous.x, current.y - previous.y))
    speed = distance / frames_apart
    if speed > max_pixels_per_frame:
        return None

    cost = speed / max_pixels_per_frame
    if speed < min_speed:
        # A ball in play is never still on screen, so a path that stays put
        # is a static false positive - the net cord, a line, a logo - and
        # those are otherwise the CHEAPEST path available, costing nothing
        # to move and nothing to turn. Without this the search prefers them
        # to the real ball whenever they score higher, which they often do.
        cost += static_penalty * (1.0 - speed / min_speed)
    if before_previous is not None:
        # Penalise a sharp change of direction. Real flight curves gently
        # under gravity; a false peak jumping between two objects does not.
        incoming = np.array([previous.x - before_previous.x, previous.y - before_previous.y])
        outgoing = np.array([current.x - previous.x, current.y - previous.y])
        change = float(np.linalg.norm(outgoing - incoming))
        cost += turn_weight * change / max_pixels_per_frame
    return cost


def _best_path(
    candidates_by_frame: Sequence[Sequence[Detection]],
    available: list[set[int]],
    max_pixels_per_frame: float,
    max_skip: int,
    skip_penalty: float,
    turn_weight: float,
    confidence_weight: float,
    detection_reward: float,
    min_speed: float,
    static_penalty: float,
) -> list[tuple[int, int]]:
    """The single cheapest path through whichever candidates are still
    `available`, as a list of (frame, candidate index)."""
    num_frames = len(candidates_by_frame)
    nodes: list[dict[int, _Node]] = [{} for _ in range(num_frames)]

    for frame in range(num_frames):
        for index in available[frame]:
            detection = candidates_by_frame[frame][index]
            entry = -(detection_reward + confidence_weight * detection.confidence)
            best = _Node(cost=entry, previous=None)  # starting fresh here

            for back in range(1, min(max_skip, frame) + 1):
                previous_frame = frame - back
                for previous_index, previous_node in nodes[previous_frame].items():
                    previous_detection = candidates_by_frame[previous_frame][previous_index]
                    # The turn cost needs the step before the previous one.
                    # Taking it from the previous node's own best path makes
                    # this first-order rather than a true second-order
                    # search - a standard approximation, and the direction
                    # of arrival is stable enough for it to hold.
                    before = None
                    if previous_node.previous is not None:
                        bf, bi = previous_node.previous
                        before = candidates_by_frame[bf][bi]
                    step = _step_cost(
                        previous_detection,
                        detection,
                        before,
                        back,
                        max_pixels_per_frame,
                        turn_weight,
                        min_speed,
                        static_penalty,
                    )
                    if step is None:
                        continue
                    cost = previous_node.cost + step + skip_penalty * (back - 1) + entry
                    if cost < best.cost:
                        best = _Node(cost=cost, previous=(previous_frame, previous_index))
            nodes[frame][index] = best

    end = min(
        ((frame, index, node.cost) for frame, row in enumerate(nodes) for index, node in row.items()),
        key=lambda item: item[2],
        default=None,
    )
    if end is None:
        return []

    path = []
    frame, index = end[0], end[1]
    while True:
        path.append((frame, index))
        previous = nodes[frame][index].previous
        if previous is None:
            break
        frame, index = previous
    path.reverse()
    return path


def track_candidates(
    candidates_by_frame: Sequence[Sequence[Detection]],
    max_pixels_per_frame: float = 150.0,
    max_skip: int = 12,
    skip_penalty: float = 0.35,
    turn_weight: float = 0.5,
    confidence_weight: float = 1.0,
    detection_reward: float = 1.0,
    min_speed: float = 0.3,
    static_penalty: float = 0.8,
    min_path_length: int = 4,
) -> list[Optional[Detection]]:
    """The most plausible path through the per-frame candidates, as one
    detection (or None) per frame - the same shape `BallTracker.track`
    already consumes, so this slots in ahead of it.

    `skip_penalty` is charged per frame skipped: too low and the path
    wanders through nothing, too high and it forces a false peak into a
    stretch where the ball really is invisible. `turn_weight` sets how
    strongly a sudden change of direction is punished relative to raw speed;
    it must stay modest, since a real bounce IS a sudden change of
    direction and the track has to be able to follow one.

    `detection_reward` is what makes the search prefer a track that explains
    MORE frames. Every accepted candidate earns it, so a step that moves
    plausibly is net negative and worth taking, while a step that only fits
    by teleporting is not. Without it the cheapest path would be the
    trivial one - a single high-confidence point and nothing else.

    `min_speed` and `static_penalty` are what keep a stationary false
    positive from winning outright; see `_step_cost`.

    Paths are extracted REPEATEDLY, not once. A rally is not one unbroken
    flight: the ball is occluded, leaves frame, and is missed for stretches
    longer than `max_skip`, and no single path can span those. Taking only
    the best path discards everything either side of the first long break -
    measured on video_input2, that kept 450 of 773 detected frames and lost
    a confirmed bounce with them. So the best path is taken, its candidates
    removed, and the search repeated until what is left is shorter than
    `min_path_length` and no longer describes a flight.
    """
    num_frames = len(candidates_by_frame)
    if num_frames == 0:
        return []

    available = [set(range(len(row))) for row in candidates_by_frame]
    chosen: list[Optional[Detection]] = [None] * num_frames
    while True:
        path = _best_path(
            candidates_by_frame,
            available,
            max_pixels_per_frame,
            max_skip,
            skip_penalty,
            turn_weight,
            confidence_weight,
            detection_reward,
            min_speed,
            static_penalty,
        )
        if len(path) < min_path_length:
            break
        for frame, index in path:
            if chosen[frame] is None:
                chosen[frame] = candidates_by_frame[frame][index]
            available[frame].discard(index)
        # Frames already committed can't host another path.
        for frame, _ in path:
            available[frame].clear()
    return chosen
