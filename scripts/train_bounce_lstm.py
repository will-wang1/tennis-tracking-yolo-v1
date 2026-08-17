"""Train a windowed-sequence LSTM bounce/not_bounce classifier.

Consumes the SAME labeled candidate CSVs scripts/train_bounce_classifier.py
does (scripts/extract_bounce_candidates.py's own output, hand-labeled,
and/or the reference/TrackNet importers) - this only additionally needs
their `window_xy` column, a JSON-encoded (window_length, 2) array of
[x, y] positions relative to the candidate frame (see
src.analysis.bounce_features.extract_window_sequence).

    python scripts/train_bounce_lstm.py \
        --labels outputs/hardcourt/bounce_candidates_labeled.csv \
                 outputs/grasscourt/bounce_candidates_labeled.csv \
                 outputs/bigDF_reference_labeled.csv \
                 outputs/tracknet_bounce_labeled.csv \
        --out weights/bounce_lstm.keras

Unlike scripts/train_bounce_classifier.py's hand-picked feature vector
(y-prominence, dx, speed ratio...), this trains directly on the raw
per-frame trajectory shape - the model has to learn what distinguishes a
bounce from a contact itself, from examples, rather than being told which
properties to look at. That needs more data and more diversity to
generalize than a small feature vector does, which is exactly why this
script pools every labeled source above rather than training per-video:
our own two clips alone (a few dozen rows) aren't enough on their own for
a sequence model to learn much from.

IMPORTANT - far-court coverage: a real bounce's pixel dip is much smaller
far from the camera (perspective), so if the labeled data is mostly
near-court examples, the model will likely learn "small dip = noise, not a
bounce" and miss real far-court bounces - the exact gap that motivated
integrating this approach (see
https://github.com/s-ganguli/AI-Tennis-Ball-Bounce-Detection, whose own
object-detection stage deliberately curated a 1:2 near:far training image
ratio for the same reason). When labeling scripts/extract_bounce_candidates.py's
thumbnails, favor far-court candidates - there'll naturally be fewer of
them since the detector finds fewer far-court candidates in the first
place, so each one is worth more.

Meant to run on Colab (see notebooks/train_bounce_lstm_colab.ipynb) -
TensorFlow trains this fine on CPU for a dataset this size, but a GPU
runtime makes iterating on window size / architecture much faster.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_dataset(label_paths: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    frames = [pd.read_csv(path) for path in label_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["label"].notna() & (df["label"].astype(str).str.strip() != "")]
    df = df[df["window_xy"].notna() & (df["window_xy"].astype(str).str.strip() != "")]
    if df.empty:
        raise SystemExit(
            "No labeled rows with a window_xy value found - re-run the extraction/import "
            "script(s) that produced these CSVs (older CSVs predate the window_xy column)."
        )

    raw_sequences = [np.array(json.loads(raw), dtype=np.float64) for raw in df["window_xy"]]
    lengths = Counter(len(seq) for seq in raw_sequences)
    window_length = lengths.most_common(1)[0][0]

    # _window_around (src.analysis.bounce_detector) only includes REAL
    # (non-interpolated) detections, so it truncates - not necessarily
    # symmetrically - whenever there aren't enough real neighbors nearby.
    # That's most common right at a genuine bounce (impact motion blur
    # often causes a brief missed detection), so it's expected, not a
    # --window mismatch between CSVs. A truncated window's center frame
    # isn't recorded, so its position within the shorter array can't be
    # trusted enough to pad/align - drop those rows rather than guess.
    keep_mask = [len(seq) == window_length for seq in raw_sequences]
    dropped = len(keep_mask) - sum(keep_mask)
    if dropped:
        other_lengths = sorted(l for l in lengths if l != window_length)
        print(
            f"Dropping {dropped} row(s) with a truncated window_xy (lengths {other_lengths}, "
            f"keeping window_length={window_length}) - truncation happens when a candidate's real "
            "neighbors run out, most often right at a genuine bounce, so this is expected rather "
            "than a labeling error."
        )
    sequences = [seq for seq, keep in zip(raw_sequences, keep_mask) if keep]
    df = df[keep_mask]

    X = np.stack(sequences)
    y = (df["label"].astype(str).str.strip() == "bounce").to_numpy(dtype=np.float64)
    return X, y, window_length


def build_model(window_length: int):
    import tensorflow as tf
    from tensorflow.keras import layers

    model = tf.keras.Sequential(
        [
            layers.Input(shape=(window_length, 2)),
            layers.LSTM(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.AUC(name="auc")]
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", required=True, help="One or more labeled candidate CSVs")
    parser.add_argument("--out", default="weights/bounce_lstm.keras")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="StratifiedKFold folds for a directional held-out AUC estimate before the final "
        "fit-on-everything - set to 0 to skip (faster iteration once you trust the setup)",
    )
    args = parser.parse_args()

    X, y, window_length = load_dataset(args.labels)
    bounce_count = int(y.sum())
    print(
        f"{len(y)} labeled examples ({bounce_count} bounce, {len(y) - bounce_count} not_bounce), "
        f"window_length={window_length}, from {len(args.labels)} file(s)"
    )

    if args.cv_folds and args.cv_folds > 1:
        from sklearn.model_selection import StratifiedKFold

        min_class_count = min(bounce_count, len(y) - bounce_count)
        if min_class_count < args.cv_folds:
            print(
                f"Smallest class has only {min_class_count} example(s), fewer than "
                f"--cv-folds={args.cv_folds} - skipping cross-validation, fitting on all data "
                "with no held-out check."
            )
        else:
            aucs = []
            splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=0)
            for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
                model = build_model(window_length)
                model.fit(
                    X[train_idx], y[train_idx], epochs=args.epochs, batch_size=args.batch_size, verbose=0
                )
                _, auc = model.evaluate(X[val_idx], y[val_idx], verbose=0)
                aucs.append(auc)
                print(f"  fold {fold + 1}/{args.cv_folds}: val AUC = {auc:.3f}")
            print(
                f"StratifiedKFold({args.cv_folds}) AUC: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f} - "
                "treat as directional, not a certified accuracy estimate, same caveat as "
                "scripts/train_bounce_classifier.py."
            )

    print("Fitting final model on all data...")
    model = build_model(window_length)
    model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, verbose=2, validation_split=0.2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    meta_path = out_path.with_suffix(out_path.suffix + ".json")
    meta_path.write_text(json.dumps({"window_length": window_length}))
    print(f"Wrote {out_path} (+ {meta_path.name})")


if __name__ == "__main__":
    main()
