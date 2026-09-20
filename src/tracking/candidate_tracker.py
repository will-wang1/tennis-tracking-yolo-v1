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

TWO BALLS AT ONCE (`track_ball_paths`). Paths have always been extracted
repeatedly here, but only ever to stitch ONE ball back together across the
breaks it disappears behind - so each extracted path CLEARED every
candidate in the frames it used, and the results were flattened into one
detection per frame. Both steps encode "there is exactly one ball", which
match footage can assume and coaching footage cannot: measured on
dingles_serve_volley (a two-ball "Dingles" drill), 33.2% of frames hold two
spatially distinct balls in flight, against 2.5-2.6% on two match clips.

`track_ball_paths` keeps the same search and drops only that assumption,
returning the paths themselves instead of a flattened list.
`track_candidates` stays exactly as it was - it calls the same extractor in
its original exclusive-frame mode and flattens - so every existing caller
sees byte-identical output.

What the one-ball assumption was quietly buying is a SHADOW filter, and
removing it without a replacement makes things worse, not better: with
frames left open, a second path simply hugs the first one a few pixels away
and the extra paths explain almost no new frames (2121 -> 2174 on the
coaching clip, while the path count triples). `min_separation` is that
replacement - a path that runs alongside an accepted one is the same ball
seen twice, not a new one.
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


@dataclass(frozen=True)
class BallTrack:
    """One continuous run of a single ball, as `frame -> Detection`.

    A track is not "a ball" for the whole clip: the same physical ball
    produces several tracks when it disappears for longer than `max_skip`.
    Telling those apart from a genuinely SECOND ball is a question about
    time, not appearance - two tracks that overlap in frames cannot be one
    ball, and two that don't may well be - and deliberately is not decided
    here (see `overlaps`, which is the test a caller needs to do it).
    """

    detections: dict[int, Detection]

    @property
    def start_frame(self) -> int:
        return min(self.detections)

    @property
    def end_frame(self) -> int:
        return max(self.detections)

    def __len__(self) -> int:
        return len(self.detections)

    def overlaps(self, other: "BallTrack") -> bool:
        """True when both tracks are present in at least one shared frame -
        i.e. these cannot be the same ball."""
        return bool(self.detections.keys() & other.detections.keys())

    def as_frame_list(self, num_frames: int) -> list[Optional[Detection]]:
        """This track alone in the one-detection-per-frame shape the rest of
        the pipeline consumes, so `analyze_impacts` can be run per track
        without any of it having to learn about multiple balls."""
        out: list[Optional[Detection]] = [None] * num_frames
        for frame, detection in self.detections.items():
            if 0 <= frame < num_frames:
                out[frame] = detection
        return out


def _is_shadow(
    path: dict[int, Detection],
    accepted: Sequence[dict[int, Detection]],
    min_separation: float,
    min_shared_frames: int,
) -> bool:
    """Is this path just an already-accepted one, seen a few pixels off?

    Compared on the MEDIAN separation over the frames the two share, not
    the minimum: two genuinely different balls can pass close to each other
    for a moment (they cross the net in opposite directions), and a
    minimum-distance test would throw the second one away exactly then. A
    shadow, by contrast, stays alongside its original for the whole overlap.

    Paths sharing fewer than `min_shared_frames` frames are left alone -
    there isn't enough evidence to call them the same ball, and the cost of
    being wrong is asymmetric: a discarded real ball is invisible in every
    number downstream, while a surviving shadow shows up as an implausible
    extra track a caller can still filter.
    """
    for other in accepted:
        shared = path.keys() & other.keys()
        if len(shared) < min_shared_frames:
            continue
        separations = [
            float(np.hypot(path[f].x - other[f].x, path[f].y - other[f].y)) for f in shared
        ]
        if float(np.median(separations)) < min_separation:
            return True
    return False


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

    tracks = _extract_paths(
        candidates_by_frame,
        max_pixels_per_frame=max_pixels_per_frame,
        max_skip=max_skip,
        skip_penalty=skip_penalty,
        turn_weight=turn_weight,
        confidence_weight=confidence_weight,
        detection_reward=detection_reward,
        min_speed=min_speed,
        static_penalty=static_penalty,
        min_path_length=min_path_length,
        exclusive_frames=True,
        min_separation=0.0,
        min_shared_frames=0,
        max_tracks=None,
    )
    chosen: list[Optional[Detection]] = [None] * num_frames
    for track in tracks:
        for frame, detection in track.detections.items():
            if chosen[frame] is None:
                chosen[frame] = detection
    return chosen


def _extract_paths(
    candidates_by_frame: Sequence[Sequence[Detection]],
    *,
    max_pixels_per_frame: float,
    max_skip: int,
    skip_penalty: float,
    turn_weight: float,
    confidence_weight: float,
    detection_reward: float,
    min_speed: float,
    static_penalty: float,
    min_path_length: int,
    exclusive_frames: bool,
    min_separation: float,
    min_shared_frames: int,
    max_tracks: Optional[int],
) -> list[BallTrack]:
    """Repeatedly pull the cheapest remaining path out of the lattice.

    `exclusive_frames` is the one-ball assumption, isolated: with it on, a
    path's frames are closed to every other path (the original behaviour,
    kept so `track_candidates` is unchanged); with it off, another ball may
    still be found in those frames and `min_separation` is what stops that
    being the same ball a few pixels over.

    Terminates either way: every iteration removes at least
    `min_path_length` candidates from a finite lattice, including the
    iterations whose path is then rejected as a shadow - a rejected path
    still consumes its candidates, so the search cannot re-find it forever.
    """
    available = [set(range(len(row))) for row in candidates_by_frame]
    accepted: list[dict[int, Detection]] = []
    while max_tracks is None or len(accepted) < max_tracks:
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
            available[frame].discard(index)
        if exclusive_frames:
            # Frames already committed can't host another path.
            for frame, _ in path:
                available[frame].clear()
        found = {frame: candidates_by_frame[frame][index] for frame, index in path}
        if min_separation > 0.0 and _is_shadow(
            found, accepted, min_separation, min_shared_frames
        ):
            continue
        accepted.append(found)
    return [BallTrack(detections=found) for found in accepted]


def track_ball_paths(
    candidates_by_frame: Sequence[Sequence[Detection]],
    max_pixels_per_frame: float = 150.0,
    max_skip: int = 12,
    skip_penalty: float = 0.35,
    turn_weight: float = 0.5,
    confidence_weight: float = 1.0,
    detection_reward: float = 1.0,
    min_speed: float = 0.3,
    static_penalty: float = 0.8,
    min_path_length: int = 6,
    min_separation: float = 60.0,
    min_shared_frames: int = 3,
    max_tracks: Optional[int] = 64,
) -> list[BallTrack]:
    """Every ball the lattice supports, as separate tracks, cheapest first.

    Same search as `track_candidates` with the one-ball assumption removed -
    see this module's docstring for the measurement that motivates it. The
    tracks come back in the order the search found them, which is best-first
    by path cost, NOT chronological: the longest, most confident flight is
    first whether or not it starts first.

    HONEST LIMITS, all three of which need a second coaching clip before any
    of these numbers should be trusted as general:

    `min_separation` (pixels) is fitted to this project's current clips, and
    fitting a threshold to the clips in hand is exactly what has misled this
    project before. At 60px it separates the measured 33.2% two-ball frames
    on a real two-ball drill from a 2.5-2.6% floor on match footage, where
    the true answer is ~0%. So that floor is a real FALSE-POSITIVE rate, not
    a rounding error: a caller must expect occasional spurious tracks rather
    than treat every track as a ball.

    It is also a flat pixel distance, which the court's own geometry argues
    against: a ball at the far baseline covers a fraction of the pixels of
    one near the camera (measured at 0.128 vs 0.010 court-metres per pixel
    on a fence-height camera), so one threshold is simultaneously too tight
    far away and too loose near. Scaling it by depth needs a calibration
    this function deliberately doesn't take, and is left until there is
    footage to check it against.

    `min_path_length` defaults higher here than in `track_candidates` (6 vs
    4): with frames no longer closed after use, short paths are far easier
    to find and are much more often noise than a real second ball.

    `max_tracks` bounds the work rather than the truth. Each extra track
    costs another full pass over the lattice, and a clip that wants more
    tracks than this is telling you something about the footage - the cap
    being hit means the track list is incomplete, not that the clip has
    exactly this many balls.
    """
    if not candidates_by_frame:
        return []
    return _extract_paths(
        candidates_by_frame,
        max_pixels_per_frame=max_pixels_per_frame,
        max_skip=max_skip,
        skip_penalty=skip_penalty,
        turn_weight=turn_weight,
        confidence_weight=confidence_weight,
        detection_reward=detection_reward,
        min_speed=min_speed,
        static_penalty=static_penalty,
        min_path_length=min_path_length,
        exclusive_frames=False,
        min_separation=min_separation,
        min_shared_frames=min_shared_frames,
        max_tracks=max_tracks,
    )


def count_simultaneous_frames(tracks: Sequence[BallTrack]) -> dict[int, int]:
    """How many tracks are present in each frame that has any, for the
    frames where that count is 2 or more.

    This is the measurement that says whether footage is multi-ball at all,
    which is a property of the SESSION (a drill with two balls in play)
    rather than of any one track, so it belongs beside the tracker rather
    than inside it."""
    counts: dict[int, int] = {}
    for track in tracks:
        for frame in track.detections:
            counts[frame] = counts.get(frame, 0) + 1
    return {frame: n for frame, n in counts.items() if n >= 2}
