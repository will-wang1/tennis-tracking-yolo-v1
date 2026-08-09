"""Train a bounce/not_bounce classifier from labeled candidate CSVs.

Consumes one or more CSVs sharing scripts/extract_bounce_candidates.py's
schema (that script's own output, hand-labeled, and/or
scripts/import_reference_bounce_dataset.py's output) - feature columns are
already computed (src.analysis.bounce_features.FEATURE_NAMES), so this just
concatenates, filters to labeled rows, and fits a small scikit-learn
classifier. Same "train and inference must use identical features" pattern
as scripts/train_stroke_classifier.py, enforced here by both sides reading
straight from FEATURE_NAMES-named columns rather than recomputing anything.

    python scripts/train_bounce_classifier.py \
        --labels outputs/hardcourt/bounce_candidates_labeled.csv \
                 outputs/grasscourt/bounce_candidates_labeled.csv \
                 outputs/bigDF_reference_labeled.csv \
        --out weights/bounce_classifier.pkl

Our own two clips contribute only a couple dozen labeled candidates -
nowhere near enough on their own for a train/test split to mean anything,
which is exactly why scripts/import_reference_bounce_dataset.py's ~150
additional rows matter here. This still reports StratifiedKFold
cross-validation instead of a single holdout accuracy, and the result
should be read as directional, not a certified accuracy figure - same
honesty caveat as train_stroke_classifier.py.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.bounce_features import FEATURE_NAMES  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", nargs="+", required=True, help="One or more labeled candidate CSVs")
    parser.add_argument("--out", default="weights/bounce_classifier.pkl")
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.labels]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].notna() & (df["label"].astype(str).str.strip() != "")]
    if df.empty:
        raise SystemExit("No labeled rows found across the given CSVs")

    print(f"{len(df)} labeled examples from {len(args.labels)} file(s), class counts:")
    print(df["label"].value_counts().to_string())

    X = df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy()

    pipeline = Pipeline(
        [("scale", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, random_state=0))]
    )

    class_counts = pd.Series(y).value_counts()
    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        print(
            f"Class '{class_counts.idxmin()}' has only {min_class_count} example - "
            "cross-validation skipped, fitting on all data with no held-out check."
        )
    else:
        n_splits = max(2, min(5, min_class_count))
        scores = cross_val_score(
            pipeline, X, y, cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        )
        print(
            f"StratifiedKFold({n_splits}) accuracy: {scores.mean():.2f} +/- {scores.std():.2f} "
            f"(scores: {np.round(scores, 2)})"
        )
        print(
            f"Treat this as directional, not a certified accuracy estimate - it's averaged "
            f"over {len(X)} examples total, most from a different camera/court than the two "
            f"clips this'll actually run on."
        )

    pipeline.fit(X, y)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_names": FEATURE_NAMES}, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
