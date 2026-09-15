"""Build an EVALUATION label skeleton for a new clip - the workflow behind
`data/labels/*.csv`.

Not to be confused with scripts/extract_bounce_candidates.py, which writes
TRAINING rows (feature columns plus a bounce/not_bounce label). This writes
the other schema: `seconds,kind,tolerance_s,note`, the one
src/evaluation/impact_labels.py scores a whole run against, where `kind` is
bounce / contact / none.

    python scripts/extract_label_candidates.py --input match.mp4 \
        --cache outputs/match/replay_cache.pkl \
        --out data/labels/match_impacts.csv

Then open the thumbnail strips in `<out>_thumbs/`, fill in the `kind`
column of every row, and the file is ready for
`scripts/replay_impacts.py --labels`.

WHY BOTH CANDIDATE SOURCES
--------------------------
Candidates come from the union of two scans, and the union is the whole
point:

- a HIGH-RECALL trajectory scan (find_trajectory_breakpoints at a low
  prominence) proposes essentially every direction change in the tracked
  ball path, including ones the pipeline's own logic throws away;
- the PIPELINE's own impacts (analyze_impacts) add whatever it claims that
  the raw scan missed.

Seeding only from the pipeline would make the resulting ground truth blind
to exactly the failure that matters most. On the Alcaraz-Djokovic clip 5 of
29 labelled events were "recall gap" cases - a real serve or landing the
pipeline proposed NOTHING for, despite the ball being tracked cleanly at
0.85-0.95 confidence throughout. An evaluation set built from the
pipeline's own proposals cannot contain those rows by construction, and
would score a blind pipeline as perfect.

WHY THE SKELETON DOES NOT SHOW WHAT THE PIPELINE THOUGHT
--------------------------------------------------------
The `kind` column is left empty and no verdict is written next to it, on
purpose. Telling a labeller "the pipeline called this a bounce" anchors
them toward agreeing, which quietly biases the ground truth toward the
system it is supposed to judge - and this file's whole value is being an
independent check. Provenance is still recorded, in a SEPARATE
`<out>_provenance.csv` written for analysis after labelling is done, so
the two never have to be mixed.

LABELLING GUIDE
---------------
The three kinds are about RALLY PLAY, not about physics. An event that
really happened but was not part of the point is `none`.

- `bounce`  - the ball touched the court AS PART OF THE POINT.
- `contact` - a racket struck it as part of the point (a serve counts).
- `none`    - anything else. Three cases matter, and the first is the one
  that is easy to get wrong:
    * a player BOUNCING THE BALL BEFORE SERVING. The ball genuinely hits
      the court, so it is tempting to call it `bounce` - do not. The
      existing labels mark every one of these `none`
      (data/labels/alcaraz_djokovic_impacts.csv has twelve, all from
      pre-serve bouncing), because excluding them is exactly what
      src/analysis/rally_clusters.py is built to do. Labelling them
      `bounce` scores the pipeline as WRONG precisely where it is right.
      The same goes for any other idle ball-handling between points.
    * the ball hitting the NET or net cord - not a court bounce.
    * nothing at all: a detector artefact, a blob over the net, the ball
      re-entering frame. These rows are the valuable ones (see
      data/labels/README.md) - they are the only thing stopping a detector
      from buying recall with markers drawn on nothing.

- Leave `tolerance_s` at the default 0.25 when the strip pins the instant
  down; widen it (0.35+) and say so in `note` when you can only place the
  event within a range.
- Delete a row ONLY if the strip is too unclear to call. A non-event is
  `none`, not a deletion.
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.bounce_detector import find_trajectory_breakpoints  # noqa: E402
from src.analysis.court_calibration import CourtCalibration  # noqa: E402
from src.analysis.impact_pipeline import analyze_impacts  # noqa: E402
from src.tracking.candidate_tracker import track_candidates  # noqa: E402
from src.video.io import VideoReader  # noqa: E402

_LABELLING_GUIDE = """# How to label these files

One row per candidate event. Fill in `kind` for every row, then score with
`scripts/replay_impacts.py --labels`.

Open the matching strip in `<name>_thumbs/<frame>.jpg`. Each strip is nine
frames; the yellow-bordered panel is the candidate instant. The top row is
the whole frame for context, the bottom row a zoom on the tracked ball.

## The three kinds are about RALLY PLAY, not physics

| kind | meaning |
| --- | --- |
| `bounce` | the ball touched the court AS PART OF THE POINT |
| `contact` | a racket struck it as part of the point (a serve counts) |
| `none` | anything else - see below |

**The one that is easy to get wrong**: a player BOUNCING THE BALL BEFORE
SERVING is `none`, not `bounce`. The ball really does hit the court, so
`bounce` feels right - but these are not part of the point, and excluding
them is exactly what `src/analysis/rally_clusters.py` is built to do.
Labelling them `bounce` scores the pipeline as WRONG precisely where it is
behaving correctly. `data/labels/alcaraz_djokovic_impacts.csv` has twelve
of these, all `none`. The same applies to any idle ball-handling between
points.

Also `none`:
- the ball hitting the NET or net cord (not a court bounce);
- nothing at all - a detector artefact, a blob over the net, the ball
  re-entering frame. These rows are the valuable ones: they are the only
  thing stopping a detector from buying recall with markers on nothing.

## Other columns

- `tolerance_s`: leave at 0.25 when the strip pins the instant down. Widen
  it (0.35+) and say so in `note` if you can only place the event within a
  range.
- `note`: anything you want to record. Free text, never parsed.

## Two rules

1. Delete a row ONLY if the strip is too unclear to call. A non-event is
   `none`, not a deletion.
2. Do not open the `_provenance.csv` while labelling. It records which scan
   proposed each candidate, including what the pipeline thought - and
   knowing that biases the ground truth toward the system it is meant to
   judge. It is there for analysis afterwards.
"""

STRIP_RADIUS = 4  # frames either side of a candidate in its thumbnail strip
STRIP_WIDTH = 320  # px per panel in the strip
CROP_HALF_PX = 110  # half-size of the zoomed crop taken around the tracked ball
# Candidates this close are one physical impact seen by two scans. 3 frames
# is 0.12s at 25fps - far shorter than a ball can travel between two real
# impacts (a bounce and the next racket contact are several tenths apart),
# so merging cannot fuse two genuine events.
MERGE_WITHIN_FRAMES = 3
DEFAULT_TOLERANCE_S = 0.25


def load_cache(path: Path) -> dict:
    with open(path, "rb") as handle:
        cache = pickle.load(handle)
    cache["calibrations"] = {
        frame: value if isinstance(value, CourtCalibration) else CourtCalibration(homography=value)
        for frame, value in cache["calibrations"].items()
    }
    return cache


def candidate_frames(cache: dict, min_prominence: float, min_gap: int) -> dict[int, set[str]]:
    """{frame index: which scans proposed it}, from both sources."""
    detections = cache.get("detections")
    if cache.get("candidates") is not None:
        detections = track_candidates(cache["candidates"], max_pixels_per_frame=150.0)
    if detections is None:
        raise SystemExit("cache holds neither 'candidates' nor 'detections'")

    analysis = analyze_impacts(
        detections,
        cache["fps"],
        calibrations_by_frame=cache["calibrations"],
        player_boxes_by_frame=cache.get("player_boxes") or None,
    )

    sources: dict[int, set[str]] = {}
    positions = [p for p in analysis.positions if p is not None]
    for frame in find_trajectory_breakpoints(
        positions, min_y_prominence=min_prominence, min_frame_gap=min_gap
    ):
        sources.setdefault(frame, set()).add("trajectory_scan")
    for impact in analysis.impacts:
        sources.setdefault(impact.frame_idx, set()).add(f"pipeline:{impact.kind}")

    ball_xy = {p.frame_idx: (p.x, p.y) for p in positions}
    return _merge_near_duplicates(sources, MERGE_WITHIN_FRAMES), ball_xy


def _merge_near_duplicates(sources: dict[int, set[str]], max_gap: int) -> dict[int, set[str]]:
    """Collapse candidates a few frames apart into one row.

    The two scans locate the same physical impact at slightly different
    frames - the pipeline's arc fit and the raw trajectory's local maximum
    rarely agree to the frame. Emitted separately, one event becomes two
    rows. FOUND BY LABELLING, not hypothesised: the first eight skeletons
    carried 120 such pairs, and a labeller reasonably marks one side as the
    event and the other `none`. The scorer then pairs a correct marker with
    the `none` row (a false positive) and leaves the real row unmatched (a
    miss) - one right answer scored as two errors, systematically, on
    exactly the events the pipeline found.

    Groups are anchored to their FIRST frame, so a group can never span
    more than `max_gap` frames; chaining gap-to-gap would let a run of
    close candidates swallow genuinely separate events. The kept frame is
    the pipeline's own where there is one, since that is what gets scored.
    """
    groups: list[list[int]] = []
    for frame in sorted(sources):
        if groups and frame - groups[-1][0] <= max_gap:
            groups[-1].append(frame)
        else:
            groups.append([frame])

    merged: dict[int, set[str]] = {}
    for group in groups:
        from_pipeline = [f for f in group if any(s.startswith("pipeline") for s in sources[f])]
        keep = from_pipeline[0] if from_pipeline else group[0]
        merged[keep] = set().union(*(sources[f] for f in group))
    return merged


def _build_panel(frame, index: int, center: int, ball_xy) -> "cv2.typing.MatLike":
    """One column of a strip: the whole frame on top for context (is the
    ball near a player? the net? the baseline?) and a zoomed crop around
    the tracked ball underneath.

    The crop is what makes the strip labellable at all - at the width a
    9-panel strip allows, a tennis ball in a full 1080p frame is about two
    pixels across, and telling a bounce from a racket contact means seeing
    exactly the thing that is invisible at that scale.
    """
    context = frame.copy()
    ball = ball_xy.get(index)
    if ball is not None:
        x, y = ball
        cv2.circle(context, (int(x), int(y)), 20, (0, 0, 255), 2)
    scale = STRIP_WIDTH / context.shape[1]
    context = cv2.resize(context, (STRIP_WIDTH, int(context.shape[0] * scale)))

    height, width = frame.shape[:2]
    if ball is not None:
        cx, cy = int(ball[0]), int(ball[1])
    else:
        cx, cy = width // 2, height // 2
    half = CROP_HALF_PX
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(width, cx + half), min(height, cy + half)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        crop = frame[: 2 * half, : 2 * half]
    crop = cv2.resize(crop, (STRIP_WIDTH, STRIP_WIDTH))
    if ball is not None:
        cv2.circle(crop, (int((cx - x0) * STRIP_WIDTH / max(1, x1 - x0)),
                          int((cy - y0) * STRIP_WIDTH / max(1, y1 - y0))),
                   14, (0, 0, 255), 2)
    else:
        cv2.putText(crop, "no ball", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    panel = cv2.vconcat([context, crop])
    colour = (0, 255, 255) if index == center else (90, 90, 90)
    thickness = 3 if index == center else 1
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), colour, thickness)
    cv2.putText(panel, str(index), (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return panel


def write_strips(video: Path, frames_wanted: dict[int, list[int]], ball_xy, out_dir: Path) -> None:
    """One contact-sheet image per candidate: a few frames either side with
    the tracked ball ringed, so a bounce can be told from a contact by its
    MOTION. A single still frame is rarely enough to call either one.

    Frames are annotated and DOWNSCALED as they stream past, never held at
    full resolution - a few hundred candidates times nine frames of 1080p
    is several GB if kept whole, the same mistake main.py's three-pass
    rewrite was undoing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted_by_frame: dict[int, list[int]] = {}
    for center, group in frames_wanted.items():
        for frame_index in group:
            wanted_by_frame.setdefault(frame_index, []).append(center)
    if not wanted_by_frame:
        return

    panels: dict[int, dict[int, "cv2.typing.MatLike"]] = {center: {} for center in frames_wanted}
    last_needed = max(wanted_by_frame)

    reader = VideoReader(str(video))
    for index, frame in enumerate(reader.frames()):
        if index in wanted_by_frame:
            for center in wanted_by_frame[index]:
                panels[center][index] = _build_panel(frame, index, center, ball_xy)
        if index >= last_needed:
            break

    for center, group in sorted(frames_wanted.items()):
        ordered = [panels[center][f] for f in group if f in panels[center]]
        if ordered:
            cv2.imwrite(str(out_dir / f"{center:06d}.jpg"), cv2.hconcat(ordered))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="The clip, used for thumbnail strips")
    parser.add_argument(
        "--cache",
        required=True,
        help="A scripts/replay_impacts.py --build cache for this clip - reuses the "
        "expensive detector pass rather than re-running it",
    )
    parser.add_argument("--out", required=True, help="Label skeleton CSV to write")
    parser.add_argument(
        "--min-prominence",
        type=float,
        default=2.0,
        help="Low on purpose: this scan is meant to over-propose, since a human "
        "filters next and a missed candidate can never be labelled (default: 2.0)",
    )
    parser.add_argument("--min-gap", type=int, default=5, help="Minimum frames between candidates")
    parser.add_argument("--no-thumbnails", action="store_true")
    args = parser.parse_args()

    cache = load_cache(Path(args.cache))
    fps = cache["fps"]
    sources, ball_xy = candidate_frames(cache, args.min_prominence, args.min_gap)
    if not sources:
        raise SystemExit("no candidates found - is the cache for this clip?")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # `strip` is navigational only - the strips are named by FRAME while
        # the labels are in SECONDS, and converting between them by hand 900
        # times is exactly the kind of friction that produces mislabelled
        # rows. read_labels() reads seconds/kind/tolerance_s/note by name and
        # ignores anything else, so carrying it costs nothing. It deliberately
        # says nothing about what the pipeline thought.
        writer.writerow(["seconds", "strip", "kind", "tolerance_s", "note"])
        for frame in sorted(sources):
            writer.writerow([f"{frame / fps:.2f}", f"{frame:06d}.jpg", "", DEFAULT_TOLERANCE_S, ""])

    # The guide lives next to the CSV, not only in this file's docstring -
    # labelling happens in a spreadsheet, where a script docstring is
    # invisible, and the pre-serve-bounce rule below is easy to get
    # backwards without it.
    guide = out.parent / "LABELLING_GUIDE.md"
    guide.write_text(_LABELLING_GUIDE, encoding="utf-8")

    provenance = out.with_name(f"{out.stem}_provenance.csv")
    with open(provenance, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seconds", "frame", "proposed_by"])
        for frame in sorted(sources):
            writer.writerow([f"{frame / fps:.2f}", frame, " ".join(sorted(sources[frame]))])

    only_scan = sum(1 for s in sources.values() if not any(x.startswith("pipeline") for x in s))
    only_pipeline = sum(1 for s in sources.values() if "trajectory_scan" not in s)
    print(f"{len(sources)} candidates -> {out}")
    print(f"   {only_scan} the pipeline proposed NOTHING for (its potential recall gaps)")
    print(f"   {only_pipeline} only the pipeline proposed (its potential false positives)")
    print(f"provenance (do not consult while labelling) -> {provenance}")

    if not args.no_thumbnails:
        thumbs = out.with_name(f"{out.stem}_thumbs")
        wanted = {
            frame: [f for f in range(frame - STRIP_RADIUS, frame + STRIP_RADIUS + 1) if f >= 0]
            for frame in sources
        }
        write_strips(Path(args.input), wanted, ball_xy, thumbs)
        print(f"thumbnail strips -> {thumbs}")

    print()
    print("Next: fill in the `kind` column (bounce / contact / none) for every row,")
    print("then score a run with:")
    print(f"   python scripts/replay_impacts.py --cache {args.cache} --labels {out}")


if __name__ == "__main__":
    main()
