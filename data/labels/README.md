# Hand labels

| File | What | Made with |
|---|---|---|
| `swing_labels_<session>.json` | Every detected swing labelled: stroke (forehand, backhand, serve, overhead, not a swing, skip) plus volley and slice. Used to measure swing detection and stroke classification, and to train the stroke model. | `scripts/make_swing_labeller.py` |
| `<session>_identities.json` | Which person-tracker tracklets are the same person - ground truth for stable player identity. | by hand |
