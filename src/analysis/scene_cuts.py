"""Detect a broadcast CUT - the camera switching to a replay, a crowd
reaction, a different court entirely - within a frame sequence, as opposed
to a pan/zoom/fast-motion moment that stays on the same continuous shot.

Every other module in this pipeline assumes one continuous camera angle for
the whole input (court calibration carries the last good reading forward
frame to frame, ball tracking assumes consecutive frames describe the same
physical motion). That assumption holds for this project's three demo
clips - each is a single pre-cropped point with no cut in it - but not for
real broadcast footage of a match, which cuts away constantly. Nothing here
tries to UNDERSTAND a cut (what it cuts TO, whether it's a replay); it only
says WHERE one happened, so a caller can stop trusting continuity across it
(see main.py, which resets the carried-forward court calibration at a cut)
and so a downstream consumer (match_stats.py's point-boundary logic, or an
AI agent reading the stats JSON) gets a real signal instead of silently
reasoning across a seam that was never one shot.

METHOD: a hard cut changes almost every pixel in the frame at once, which a
whole-frame colour histogram captures cheaply without needing to know what
changed. A pan, zoom, or the ball/players moving fast changes the
histogram too, but gradually and only partially - a cut is a full-frame,
one-frame-wide spike against whatever the SAME clip's own baseline jitter
looks like just before and after it. That "own baseline" qualifier is the
whole design: a fixed distance threshold tuned on one clip is exactly the
kind of label-fitting this project has repeatedly found doesn't
generalize (see touchdown_detector.py's docstring), so the threshold here
is a robust LOCAL z-score - how surprising is this jump against the
jump sizes right around it - rather than an absolute number carried over
from whatever footage it was last eyeballed on.

VALIDATED, ONCE: this project's other clips are all pre-cropped single
points with no cut in them, but `hardcourt1.mp4` (real Australian Open
broadcast footage) turned out to have one - a wide court shot cutting to a
player closeup at frame 332, mid-point. Run cold, with no tuning against
this clip at all, `detect_scene_cuts` finds exactly that frame and nothing
else, at a distance (0.79) roughly 100x the clip's own median frame-to-
frame jitter (0.007). The three synthetic tests in tests/test_scene_cuts.py
(a smooth pan, a static scene, and a hard colour swap standing in for a
cut) all check out too. That is one real clip's worth of evidence, not a
validated detector - it has not been checked against a cut to a REPLAY
(different footage, not just a different angle of the same instant), a
graphic overlay taking over the frame, or a genuine multi-point match
where cuts happen between points as well as within one. Treat it as sound
and reasonably trustworthy, not as proven across the range of cuts real
broadcast footage actually contains.
"""

from typing import Iterable, Sequence

import cv2
import numpy as np

DEFAULT_BINS = 32
DEFAULT_LOCAL_WINDOW = 15
DEFAULT_MIN_Z_SCORE = 6.0
# A floor under the z-score test: in a near-static scene even a tiny
# absolute jump can look like a huge multiple of the surrounding (near
# zero) jitter, which the z-score alone would misread as a cut.
DEFAULT_MIN_DISTANCE = 0.35
# Candidates within this many frames of each other collapse into one - see
# detect_scene_cuts's docstring for the real footage that made this
# necessary. Comfortably shorter than any two GENUINELY separate cuts this
# project has seen close together (nothing under a few seconds).
DEFAULT_MIN_GAP_FRAMES = 10


def _frame_histogram(frame: np.ndarray, bins: int) -> np.ndarray:
    """A normalized hue/saturation histogram - hue+saturation rather than
    full BGR because it stays comparatively stable under the brightness
    changes a broadcast frame has WITHIN one continuous shot (a player
    walking through a shadow, a slight exposure shift), which a raw-colour
    histogram would otherwise read as more change than a cut actually is.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def frame_distances(frames: Iterable[np.ndarray], bins: int = DEFAULT_BINS) -> list[float]:
    """One value per consecutive PAIR of frames - `distances[i]` is how much
    frame i+1 differs from frame i, via histogram Bhattacharyya distance
    (0 = identical distributions, 1 = totally disjoint). Exposed on its own
    because `detect_scene_cuts` and any caller wanting to plot/inspect the
    raw signal both want it without recomputing histograms twice.

    Takes any ITERABLE, deliberately not a `Sequence` - a full match is far
    too much video to hold as a materialized list of raw frames at once
    (measured: a 3-minute 1080p clip is ~28GB decoded, which is what
    actually OOM'd on this project's own machine the first time this ran
    against one - the algorithm itself never needed that). Passing
    `VideoReader.frames()`'s own generator straight through keeps at most
    one raw frame alive at a time: each is reduced to a small histogram (a
    few KB) and immediately eligible for garbage collection before the
    next is decoded. `histograms` below is the only thing this holds for
    the whole video, and it is cheap even for hours of footage.
    """
    histograms = [_frame_histogram(f, bins) for f in frames]
    if len(histograms) < 2:
        return []
    return [
        float(cv2.compareHist(histograms[i], histograms[i + 1], cv2.HISTCMP_BHATTACHARYYA))
        for i in range(len(histograms) - 1)
    ]


def detect_scene_cuts(
    frames: Iterable[np.ndarray],
    bins: int = DEFAULT_BINS,
    local_window: int = DEFAULT_LOCAL_WINDOW,
    min_z_score: float = DEFAULT_MIN_Z_SCORE,
    min_distance: float = DEFAULT_MIN_DISTANCE,
    min_gap_frames: int = DEFAULT_MIN_GAP_FRAMES,
) -> list[int]:
    """Frame indices where the CAMERA changed - each returned index is the
    first frame of the new shot (so a cut between frame 40 and 41 is
    reported as 41, matching how a caller would want to reset state
    starting FROM that frame).

    `local_window` frames on each side of a candidate jump set what
    "surprising" means there - a robust (median/MAD-based) statistic
    rather than mean/stdev, since a real cut's own huge distance would
    otherwise inflate the very baseline it's being judged against. Both
    `min_z_score` and `min_distance` must be cleared: the z-score alone
    misfires in a near-static scene (see `DEFAULT_MIN_DISTANCE`'s
    docstring note above), and the absolute distance alone misfires on a
    clip whose ordinary motion is already visually busy.

    CONFIRMED FALSE-POSITIVE MODE, found on real footage (a 3-minute
    Alcaraz-Djokovic broadcast clip): an animated on-screen graphic - a
    Hawk-Eye "OUT / CLOSE CALL" challenge overlay sliding in - disturbed
    ~15 consecutive frames enough that three of them independently crossed
    the threshold (frames 3987, 3988, 3990), none of them a real camera
    change. The single hard cut earlier in the SAME clip (frame 332,
    court-to-closeup, see this module's top docstring) produced exactly
    one candidate, so the failure mode here is specifically a sustained
    noisy STRETCH rather than one clean spike - `min_gap_frames` merges
    candidates that land within that many frames of each other (same
    reasoning as parabolic_bounce_detector.py's `_suppress_neighbors`,
    which collapses one physical bounce that fires on several neighbouring
    frames), keeping only the strongest one. That turns three false
    positives into one for this case, still real evidence something
    happened worth a caller's attention, but no longer three near-empty
    "shots" a few frames apart.
    """
    distances = frame_distances(frames, bins)
    if len(distances) < 4:
        return []

    arr = np.array(distances)
    candidates: list[tuple[int, float]] = []  # (frame_idx, distance)
    for i, d in enumerate(distances):
        lo = max(0, i - local_window)
        hi = min(len(distances), i + local_window + 1)
        neighborhood = np.delete(arr[lo:hi], i - lo)
        if len(neighborhood) < 4:
            continue
        median = float(np.median(neighborhood))
        mad = float(np.median(np.abs(neighborhood - median)))
        # 1.4826x makes MAD a consistent estimator of stdev under a normal
        # distribution - the standard conversion, not a tuned constant.
        scale = 1.4826 * mad if mad > 1e-9 else 1e-9
        z = (d - median) / scale
        if d >= min_distance and z >= min_z_score:
            candidates.append((i + 1, d))

    kept: list[tuple[int, float]] = []
    for frame_idx, d in candidates:
        if kept and frame_idx - kept[-1][0] <= min_gap_frames:
            if d > kept[-1][1]:
                kept[-1] = (frame_idx, d)
        else:
            kept.append((frame_idx, d))
    return [frame_idx for frame_idx, _ in kept]


def continuous_ranges(cut_frames: Sequence[int], num_frames: int) -> list[tuple[int, int]]:
    """`cut_frames` (from `detect_scene_cuts`) split into the (start, end)
    - inclusive - frame ranges of continuous camera shots between them, so
    a caller can iterate "one shot at a time" instead of re-deriving this
    from the cut list itself."""
    if num_frames <= 0:
        return []
    boundaries = sorted(set(f for f in cut_frames if 0 < f < num_frames))
    starts = [0] + boundaries
    ends = boundaries + [num_frames]
    return [(start, end - 1) for start, end in zip(starts, ends) if end - 1 >= start]
