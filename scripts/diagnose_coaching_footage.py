"""Measure the three things that decide how much of the impact layer has to
be rebuilt for coaching footage, instead of guessing at them.

Each check exists because a specific threshold in this pipeline was fitted
on broadcast MATCH footage, and coaching footage violates the assumption
behind it in a way that is cheap to measure and expensive to guess wrong:

1. CANDIDATE LOAD (`--check candidates`)
   `candidate_tracker.track_candidates` picks one best path through the
   per-frame candidate lattice by dynamic programming. A coaching session
   has dead balls lying on the court that a match never does, and each one
   is a candidate in EVERY frame - inflating the lattice and offering the
   path cheap wrong options wherever a real ball passes near one. This
   reports candidates-per-frame so a coaching clip can be compared against
   a match clip directly.

2. REACH RATIO (`--check reach`)
   `touchdown_detector` withholds a "contact" verdict from any impact
   further than `max_reach_ratio` (0.6) box heights from the NEAREST
   player. That gate is the pipeline's only defence against calling a
   false trajectory reversal a racket hit. With several people on court it
   may never fire at all - not by erroring, but by quietly always finding
   somebody in reach. This reports the distribution of reach ratios at
   real impacts and the fraction that clear the gate, which is the number
   that says whether the gate is alive or dead.

3. IMPACT STRENGTH (`--check strength`)
   `classify_touchdowns` needs an approach rate of `min_approach` (3.0
   m/s) and a slowdown of `min_slowdown` (3.0) to call an impact at all.
   Both were fitted on rally balls. A gentle underarm feed or a drop-feed
   is far slower, and if feeds sit below these the approach/reversal test
   is the wrong signal for them rather than a mistuned one. This reports
   the approach-rate distribution so the answer is a measurement.

Runs off the same cache `replay_impacts.py` builds, so it needs no GPU and
no second pass over the video:

    python scripts/replay_impacts.py --build --input session.mp4 \\
        --cache outputs/session/replay_cache.pkl
    python scripts/replay_impacts.py --cache outputs/session/replay_cache.pkl \\
        --add-player-boxes
    python scripts/diagnose_coaching_footage.py \\
        --cache outputs/session/replay_cache.pkl

Pass several `--cache` paths to print the checks side by side; that is the
intended use, with a known match clip alongside a coaching clip, because
every number here is only meaningful as a COMPARISON against footage the
thresholds are known to work on.

Nothing here judges pass/fail. These are the inputs to that judgment, and
inventing a threshold for "too many candidates" from the same run that
measures them would be the label-fitting this project has repeatedly found
does not generalize.
"""

import argparse
import pickle
import sys

import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.court_calibration import CourtCalibration, static_calibration_from_frames  # noqa: E402
from src.analysis.impact_pipeline import analyze_impacts  # noqa: E402
from src.analysis.touchdown_detector import _DEFAULT_MAX_REACH_RATIO  # noqa: E402
from src.tracking.candidate_tracker import track_candidates  # noqa: E402

_PERCENTILES = (5, 25, 50, 75, 95)


def _percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile on an already-sorted list. Deliberately not
    numpy.percentile's interpolation: these are small samples of real
    measurements and an interpolated value is not one of them."""
    if not values:
        return float("nan")
    rank = max(0, min(len(values) - 1, int(round(pct / 100.0 * (len(values) - 1)))))
    return values[rank]


def _summarise(values: list[float]) -> str:
    """Percentiles over the FINITE values only. A single NaN makes sorting
    undefined and silently scrambles every percentile - it did, on the
    first coaching clip, producing a p75 below the p50 - so non-finite
    values are dropped and counted rather than sorted."""
    finite = [v for v in values if v is not None and np.isfinite(v)]
    dropped = len(values) - len(finite)
    if not finite:
        return "no data" + (f"  ({dropped} non-finite dropped)" if dropped else "")
    ordered = sorted(finite)
    cells = [f"p{p}={_percentile(ordered, p):.2f}" for p in _PERCENTILES]
    return f"n={len(ordered)}  " + "  ".join(cells) + (f"  ({dropped} non-finite dropped)" if dropped else "")


def load_cache(path: Path) -> dict:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def calibrations_from_cache(cache: dict) -> dict[int, CourtCalibration]:
    """Caches store plain 3x3 arrays, not CourtCalibration objects - see
    replay_impacts.build_cache's own note on why."""
    return {
        frame: value if isinstance(value, CourtCalibration) else CourtCalibration(homography=value)
        for frame, value in cache.get("calibrations", {}).items()
    }


def analyse(cache: dict, max_jump: float, static_court: bool = False) -> tuple:
    detections = track_candidates(cache["candidates"], max_pixels_per_frame=max_jump)
    analysis = analyze_impacts(
        detections,
        cache["fps"],
        calibrations_by_frame=_calibrations(cache, static_court),
        player_boxes_by_frame=cache.get("player_boxes") or None,
        max_pixels_per_frame=max_jump,
    )
    return detections, analysis


def _calibrations(cache: dict, static_court: bool) -> dict[int, CourtCalibration]:
    """Per-frame calibrations as cached, or - for a FIXED camera - one
    static fit for every frame. Per-frame fits on a fixed camera jitter
    (corners move a median 35.9px between consecutive frames on the
    Dingles clip), and that jitter turns into approach rate: the
    diagnostics first measured on it reported up to 144 m/s. Opt-in,
    because a panning broadcast camera genuinely needs per-frame fits."""
    per_frame = calibrations_from_cache(cache)
    if not static_court or not per_frame:
        return per_frame
    static = static_calibration_from_frames(per_frame)
    return {frame: static for frame in per_frame}


def check_candidates(cache: dict) -> list[str]:
    """Candidates per frame - the lattice the path search has to choose
    through. Empty frames are reported separately from the load on
    non-empty ones: a clip can have a high mean purely because the ball is
    visible more often, which is a different fact from clutter."""
    per_frame = [len(frame) for frame in cache["candidates"]]
    non_empty = [n for n in per_frame if n > 0]
    total = len(per_frame)
    lines = [
        f"  frames                {total}",
        f"  frames with no candidate  {total - len(non_empty)} "
        f"({100.0 * (total - len(non_empty)) / total:.1f}%)" if total else "  no frames",
        f"  candidates/frame (non-empty frames)  {_summarise([float(n) for n in non_empty])}",
        f"  max candidates in any frame          {max(per_frame) if per_frame else 0}",
    ]
    return lines


def check_reach(cache: dict, analysis) -> list[str]:
    """Reach ratio at each impact, and how often the gate that depends on
    it could actually fire.

    Read off `Touchdown.player_reach` rather than recomputed here, so this
    reports exactly what `classify_touchdowns` itself saw - including the
    frames where no box was cached, which it treats as "unknown" rather
    than "nobody there"."""
    boxes_by_frame = cache.get("player_boxes") or {}
    if not boxes_by_frame:
        return ["  no player boxes in cache - run replay_impacts.py --add-player-boxes first"]
    if not analysis.touchdowns:
        return ["  no touchdowns (needs court calibrations in the cache)"]

    ratios = [float(td.player_reach) for td in analysis.touchdowns if td.player_reach is not None]
    unknown = sum(1 for td in analysis.touchdowns if td.player_reach is None)
    if not ratios:
        return [f"  no impact had a player box ({unknown} unknown of {len(analysis.touchdowns)})"]

    gate = _DEFAULT_MAX_REACH_RATIO
    out_of_reach = sum(1 for r in ratios if r > gate)
    box_counts = [float(len(b)) for b in boxes_by_frame.values()]
    return [
        f"  impacts measured      {len(ratios)}"
        + (f"  ({unknown} had no box cached)" if unknown else ""),
        f"  people per frame      {_summarise(box_counts)}",
        f"  reach ratio           {_summarise(ratios)}",
        f"  out of reach (>{gate})  {out_of_reach}/{len(ratios)} "
        f"({100.0 * out_of_reach / len(ratios):.1f}%)  <- the gate is dead at 0%",
    ]


def check_strength(analysis, fps: float) -> list[str]:
    """Approach rate on each side of every impact, as the pipeline itself
    measured it (`Touchdown.approach_before`/`approach_after`, in court
    metres per second) - including its widened-window retry, so this is
    the real signal the thresholds see, not an idealised one.

    `classify_touchdowns` needs `min_approach` (3.0 m/s by default) to
    call an impact at all, so the share falling below that is the number
    that says whether slow feeds are reachable by the current test."""
    if not analysis.touchdowns:
        return ["  no touchdowns (needs court calibrations in the cache)"]

    before = [abs(float(td.approach_before)) for td in analysis.touchdowns if td.approach_before is not None]
    after = [abs(float(td.approach_after)) for td in analysis.touchdowns if td.approach_after is not None]
    kinds: dict[str, int] = {}
    for td in analysis.touchdowns:
        kinds[td.kind] = kinds.get(td.kind, 0) + 1

    lines = [
        "  verdicts              " + "  ".join(f"{k}={v}" for k, v in sorted(kinds.items())),
        f"  |approach| before     {_summarise(before)}",
        f"  |approach| after      {_summarise(after)}",
    ]
    before = [r for r in before if np.isfinite(r)]
    if before:
        weak = sum(1 for r in before if r < 3.0)
        lines.append(
            f"  below min_approach=3.0  {weak}/{len(before)} "
            f"({100.0 * weak / len(before):.1f}%)  <- feeds landing here need different logic"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cache",
        action="append",
        required=True,
        help="A replay_impacts.py cache. Repeat to compare clips side by side - "
        "ideally a known match clip alongside a coaching one.",
    )
    parser.add_argument(
        "--check",
        action="append",
        choices=["candidates", "reach", "strength"],
        help="Run only these checks (default: all three).",
    )
    parser.add_argument("--max-jump", type=float, default=150.0, help="BallTracker max_pixels_per_frame")
    parser.add_argument(
        "--static-court", action="store_true",
        help="Use one static court fit for every frame - right for a fixed camera, wrong for a panning one",
    )
    args = parser.parse_args()
    checks = args.check or ["candidates", "reach", "strength"]

    for cache_path in args.cache:
        path = Path(cache_path)
        if not path.exists():
            print(f"\n=== {path} ===\n  cache not found")
            continue
        cache = load_cache(path)
        fps = cache["fps"]
        print(f"\n=== {path.parent.name} ===")
        print(f"  {cache['num_frames']} frames @ {fps:g}fps  ({cache['num_frames'] / fps:.0f}s)")

        # Only run the (slightly) expensive path reconstruction if a check
        # that needs it was actually asked for.
        analysis = None
        if "reach" in checks or "strength" in checks:
            _, analysis = analyse(cache, args.max_jump, args.static_court)
            print(f"  {len(analysis.impacts)} impacts detected")

        if "candidates" in checks:
            print("\n  [candidate load]")
            for line in check_candidates(cache):
                print(line)
        if "reach" in checks:
            print("\n  [reach gate]")
            for line in check_reach(cache, analysis):
                print(line)
        if "strength" in checks:
            print("\n  [impact strength]")
            for line in check_strength(analysis, fps):
                print(line)
    print()


if __name__ == "__main__":
    main()
