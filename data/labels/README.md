# Hand-labelled impacts

One row per event a human confirmed by watching the clip, for the two demo
videos the impact classifier was developed against. These are the only
ground truth this project has for bounce-vs-contact, and until now they
existed only as prose in a chat log - which is why the same numbers kept
being re-derived by hand, and occasionally re-derived wrong.

`scripts/replay_impacts.py --labels` scores a run against one of these
files without re-running any neural network.

## Columns

| column | meaning |
| --- | --- |
| `seconds` | when the event happens, in video time |
| `kind` | `bounce` (court), `contact` (racket), or `none` |
| `tolerance_s` | how far a predicted marker may sit from `seconds` and still be the same event |
| `note` | anything the labeller said about it |

`kind = none` rows are the valuable ones: they are places a previous render
drew a marker and a human confirmed there was no impact there at all (a
ball toss, a blob over the net, the ball re-entering frame). Scoring
against bounces and contacts alone would let a detector buy recall with
markers on nothing.

`tolerance_s` is per row because the labels are not all equally precise.
Most were read off a render that already printed timestamps, so they are
good to a few frames. A few were volunteered as descriptions instead
("should be a bounce at around 4.5", "a bounce very shortly after 6.95")
and carry a wider window that says exactly how much the label pins down.

## Provenance and what these labels are not

Transcribed from a labelling session over two rendered clips. They are one
person's reading of two short videos from one camera angle each, and the
classifier's thresholds were tuned against them, so a score on these files
is a check that the rules still reproduce their own source - not evidence
that they generalize. A clip nobody has labelled yet is the only thing that
would be.

Timings follow whichever render the labeller was watching, so a label can
sit a few frames from the true impact even where the event is certain;
`9.67` in `video_input2` is flagged that way in its note. Where a marker
was corrected rather than volunteered ("3.73 is actually contact but
labelled bounce"), the time is the marker's, not an independent reading.
