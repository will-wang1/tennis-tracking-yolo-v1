"""Train a stroke-type classifier from hand-labeled candidate frames.

Consumes a CSV produced by scripts/extract_stroke_candidates.py and then
hand-labeled (see that script's docstring). For each labeled row, re-runs
pose detection over its [start_frame, end_frame] window, extracts features
with src.analysis.stroke_features.extract_features - the SAME function used
at inference time in src.analysis.stroke_classifier, so training and
inference never drift apart - and fits a small scikit-learn classifier.

    python scripts/train_stroke_classifier.py \
        --labels outputs/stroke_candidates_labeled.csv \
        --pose-weights yolov8n-pose.pt --window 9 \
        --out weights/stroke_classifier.pkl

With only a couple of short clips to label, expect dozens of examples at
most across 4-5 classes - a handful per class. That's too little for a
train/test split to mean much, so this script reports StratifiedKFold
cross-validation scores instead of a single holdout accuracy, and by default
collapses volley+other into a single "other" class to avoid classes with
only 1-2 examples (pass --keep-all-labels to disable that).

This ships a real, working end-to-end pipeline - the resulting classifier's
accuracy is honestly limited by how few labeled strokes exist right now,
same as weights/ball_detector.pt itself: a working v1 expected to improve as
more footage gets labeled, not a stub.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.stroke_features import extract_features  # noqa: E402
from src.detection.pose_detector import PoseDetector  # noqa: E402
from src.video.io import VideoReader  # noqa: E402

COLLAPSE_TO_OTHER = {"volley", "other"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="Labeled CSV from extract_stroke_candidates.py")
    parser.add_argument("--pose-weights", default="yolov8n-pose.pt")
    parser.add_argument("--pose-confidence", type=float, default=0.3)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default=None)
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--out", default="weights/stroke_classifier.pkl")
    parser.add_argument(
        "--keep-all-labels",
        action="store_true",
        help="Don't collapse volley/other into a single 'other' class",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.labels)
    df = df[df["label"].notna() & (df["label"].astype(str).str.strip() != "")]
    if df.empty:
        raise SystemExit("No labeled rows found - fill in the 'label' column first")

    if not args.keep_all_labels:
        df = df.copy()
        df["label"] = df["label"].apply(lambda label: "other" if label in COLLAPSE_TO_OTHER else label)

    print(f"{len(df)} labeled examples, class counts:")
    print(df["label"].value_counts().to_string())

    pose_detector = PoseDetector(
        args.pose_weights, confidence=args.pose_confidence, imgsz=args.imgsz, device=args.device
    )

    frames_by_video: dict[str, list] = {}
    features, labels = [], []
    half_window = args.window // 2

    for _, row in df.iterrows():
        video = row["video"]
        if video not in frames_by_video:
            frames_by_video[video] = list(VideoReader(video).frames())
        frames = frames_by_video[video]

        center = int(row["center_frame"])
        start = max(0, center - half_window)
        end = min(len(frames) - 1, center + half_window)
        window_frames = frames[start : end + 1]

        striker_x, striker_y = row["striker_x"], row["striker_y"]
        pose_window = []
        for frame in window_frames:
            frame_poses = pose_detector.detect(frame)
            if not frame_poses:
                continue
            nearest = min(
                frame_poses,
                key=lambda p: (p.center_x - striker_x) ** 2 + (p.center_y - striker_y) ** 2,
            )
            pose_window.append(nearest)

        if len(pose_window) < 2:
            print(f"Skipping candidate {row['candidate_id']} - too few pose detections in its window")
            continue

        features.append(extract_features(pose_window))
        labels.append(row["label"])

    if len(features) < 4:
        raise SystemExit(
            f"Only {len(features)} usable examples after pose re-detection - "
            "need more labeled candidates before training."
        )

    X = np.array(features)
    y = np.array(labels)

    pipeline = Pipeline(
        [("scale", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, random_state=0))]
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
            f"Treat this as directional, not a real accuracy estimate - it's averaged "
            f"over {len(X)} examples total across {len(class_counts)} classes."
        )

    pipeline.fit(X, y)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
