"""Forehand/backhand/serve shot classification from a player's pose over
time - port of antoinekeller/tennis_shot_recognition's pretrained RNN
("tennis_rnn.h5", no LICENSE file in that repo - treat as research-use-only
like this project's other unlicensed pretrained sources), fed by
`src.detection.movenet_pose_extractor.MoveNetPoseExtractor` instead of that
repo's own single-player self-bootstrapping tracker (see that module's
docstring for why).

The checkpoint is a tiny 4-layer Keras Sequential (GRU(24) -> Dropout ->
Dense(8, relu) -> Dense(4, softmax)) saved in the legacy Keras-2 HDF5
format. It won't load via `keras.models.load_model` on this project's Keras
3 - GRU's `time_major` constructor argument was removed between versions,
and the saved config still carries it. `_load_pretrained_gru` works around
this by rebuilding the identical architecture fresh (verified against the
checkpoint's own stored `model_config`) and loading the raw HDF5 weight
arrays into it directly via h5py, bypassing Keras's own (incompatible)
config deserialization entirely.

KNOWN LIMIT - it only works on the NEAR player. Measured over 300 frames of
the zverev clip, the far player is detected in every one, produces a pose
the feature extractor accepts in every one, and is classified "neutral" in
every one, at confidence 1.00; the near player over the same stretch gets 27
forehands and 14 backhands. So this is not a detection gap or a confidence
threshold - the model is confidently asserting that no shot is being played,
for a player who is playing them.

The cause is viewpoint. The camera sits behind the near baseline, so it sees
the near player from behind - the view the checkpoint was trained on - and
the far player from the front, at roughly a third the size (median box
height 66px against 180px). Mirroring the far player's keypoints was tried,
swapping left and right joints and flipping x, on the theory that a front
view is a back view reflected: it changes nothing, still 272 of 272
"neutral". Fixing this needs far-court examples in the training set, not a
transformation of the input.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

SHOT_LABELS = ["backhand", "forehand", "neutral", "serve"]

WINDOW_SIZE = 30
_FEATURES_PER_FRAME = 26  # 13 non-face keypoints x (y, x)
# MoveNet's 17 COCO keypoints, indices 1-4 are the face points this
# classifier's training data discarded as uninformative for stroke shape.
_FACE_KEYPOINT_INDICES = (1, 2, 3, 4)
_EXPECTED_VISIBLE_KEYPOINTS = 17 - len(_FACE_KEYPOINT_INDICES)


@dataclass
class ShotPrediction:
    label: str
    confidence: float
    probabilities: np.ndarray


def _load_pretrained_gru(weights_path: str | Path):
    from tensorflow import keras

    with h5py.File(str(weights_path), "r") as f:
        group = f["model_weights"]
        gru_kernel = group["gru/gru/gru_cell/kernel:0"][:]
        gru_recurrent = group["gru/gru/gru_cell/recurrent_kernel:0"][:]
        gru_bias = group["gru/gru/gru_cell/bias:0"][:]
        dense1_kernel = group["dense/dense/kernel:0"][:]
        dense1_bias = group["dense/dense/bias:0"][:]
        dense2_kernel = group["dense_1/dense_1/kernel:0"][:]
        dense2_bias = group["dense_1/dense_1/bias:0"][:]

    model = keras.Sequential(
        [
            keras.layers.Input((WINDOW_SIZE, _FEATURES_PER_FRAME)),
            keras.layers.GRU(24, dropout=0.1, reset_after=True, name="gru"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(8, activation="relu", name="dense"),
            keras.layers.Dense(4, activation="softmax", name="dense_1"),
        ]
    )
    model.get_layer("gru").set_weights([gru_kernel, gru_recurrent, gru_bias])
    model.get_layer("dense").set_weights([dense1_kernel, dense1_bias])
    model.get_layer("dense_1").set_weights([dense2_kernel, dense2_bias])
    return model


def keypoints_to_features(keypoints: np.ndarray) -> Optional[np.ndarray]:
    """`keypoints` is MoveNet's raw (17, 3) (y, x, score) output (see
    MoveNetPoseExtractor). Returns a flattened 26-dim (y, x) feature vector
    over the 13 non-face keypoints, or None if any of them weren't
    confidently detected this frame (score <= 0) - a gap, not a frame to
    feed the window."""
    keypoints = keypoints.copy()
    keypoints[list(_FACE_KEYPOINT_INDICES), 2] = 0
    visible = keypoints[keypoints[:, 2] > 0][:, 0:2]
    if visible.shape[0] != _EXPECTED_VISIBLE_KEYPOINTS:
        return None
    return visible.reshape(_FEATURES_PER_FRAME)


class ShotClassifier:
    """Stateful - call `.update()` once per frame with that frame's pose
    (or None on a tracking gap, which resets the window: a shot's motion
    pattern is a genuinely continuous 1-second sequence, and splicing
    across a gap would hand the model a fabricated one, the same reasoning
    `BallTracker` applies before smoothing across a detection gap)."""

    def __init__(self, weights_path: str | Path, window_size: int = WINDOW_SIZE):
        self.model = _load_pretrained_gru(weights_path)
        self.window_size = window_size
        self.window: deque[np.ndarray] = deque(maxlen=window_size)

    def update(self, keypoints: Optional[np.ndarray]) -> Optional[ShotPrediction]:
        if keypoints is None:
            self.window.clear()
            return None

        features = keypoints_to_features(keypoints)
        if features is None:
            self.window.clear()
            return None

        self.window.append(features)
        if len(self.window) < self.window_size:
            return None

        sequence = np.array(self.window, dtype=np.float32).reshape(1, self.window_size, _FEATURES_PER_FRAME)
        probabilities = np.asarray(self.model(sequence, training=False))[0]
        label = SHOT_LABELS[int(np.argmax(probabilities))]
        return ShotPrediction(label=label, confidence=float(probabilities.max()), probabilities=probabilities)


class ShotEventTracker:
    """Debounces per-frame `ShotClassifier` predictions into discrete shot
    events - a real swing stays above the confidence threshold for many
    consecutive frames, which would otherwise count as many separate shots.
    Mirrors the original demo pipeline's `ShotCounter` (0.98 confidence
    threshold, 60-frame minimum gap between counted shots - roughly 1
    second at a typical broadcast frame rate, shorter than any real shot
    interval in a rally)."""

    def __init__(self, confidence_threshold: float = 0.98, min_frame_gap: int = 60, display_recency: int = 30):
        self.confidence_threshold = confidence_threshold
        self.min_frame_gap = min_frame_gap
        self.display_recency = display_recency
        self.counts: dict[str, int] = {"forehand": 0, "backhand": 0, "serve": 0}
        self.events: list[tuple[int, str]] = []  # (frame_idx, label)
        self._frames_since_last_shot = min_frame_gap
        self._last_label: Optional[str] = None

    def update(self, frame_idx: int, prediction: Optional[ShotPrediction]) -> Optional[str]:
        """Returns the label to display this frame (the most recent counted
        shot, while still within `display_recency` frames of it), or None -
        distinct from what gets counted, so a shot stays visibly announced
        for a beat after the frame it was actually detected on."""
        self._frames_since_last_shot += 1
        if (
            prediction is not None
            and prediction.label != "neutral"
            and prediction.confidence > self.confidence_threshold
            and self._frames_since_last_shot > self.min_frame_gap
        ):
            self.counts[prediction.label] += 1
            self.events.append((frame_idx, prediction.label))
            self._frames_since_last_shot = 0
            self._last_label = prediction.label

        return self._last_label if self._frames_since_last_shot < self.display_recency else None
