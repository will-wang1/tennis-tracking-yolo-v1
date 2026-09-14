# How to label these files

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
