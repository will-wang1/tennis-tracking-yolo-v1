# Data

- `videos/` - source footage (not in git). `run_session.py` writes each
  session's normalised copy here.
- `labels/` - hand labels: `swing_labels_<session>.json` from the swing
  labelling page (scripts/make_swing_labeller.py), and
  `<session>_identities.json`, which tracklets are which person.
- `external/e2e_spot_tennis/` - E2E-Spot's stroke-labelled broadcast points
  (BSD-3), for pretraining the stroke model.
- `strokes/` - stroke training windows built by
  scripts/build_stroke_data_*.py (not in git; rebuilt from footage).
