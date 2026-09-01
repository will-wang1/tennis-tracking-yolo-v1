"""Wrapper around Google's pretrained MoveNet SinglePose Lightning (TFLite,
Apache 2.0 licensed - https://tfhub.dev/google/movenet/singlepose/lightning)
for player pose, feeding `src.analysis.shot_classifier.ShotClassifier`.

MoveNet is single-person: it assumes whatever's roughly centered in its
192x192 input is THE subject, so it needs pointing at a player rather than
run on the full frame - `extract()` takes an external bbox (this project's
own `PlayerDetector` output) to crop a square region around the player
before resizing down to the model's input size, instead of the self-
bootstrapping "player" tracker the model's original demo pipeline
(antoinekeller/tennis_shot_recognition, no LICENSE file - treat as
research-use-only like this project's other unlicensed pretrained sources)
used: this project already tracks player boxes independently (Faster
R-CNN, more robust than a from-scratch keypoint-cluster tracker) and needs
two players tracked at once, which a single self-bootstrapping RoI can't do.

Output is intentionally left in the MODEL's own normalized [0, 1] space
(coordinates relative to the square crop, NOT transformed to frame pixels)
in MoveNet's native (y, x, score) per-keypoint order - `ShotClassifier` was
pretrained on exactly that representation (a position/scale-invariant crop
of the player, not raw frame pixels), so transforming to frame coordinates
here would feed it a distribution it never saw in training.
"""

from typing import Optional

import cv2
import numpy as np
import tensorflow as tf

MODEL_INPUT_SIZE = 192
# How much wider than the source bbox to crop, so the model sees some
# margin around the player (racket/limb extension) rather than a tight box -
# matches the original demo pipeline's RoI sizing.
_CROP_MARGIN = 1.3
_MIN_CROP_SIZE = 150


class MoveNetPoseExtractor:
    def __init__(self, weights_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=str(weights_path))
        self.interpreter.allocate_tensors()
        self._input_index = self.interpreter.get_input_details()[0]["index"]
        self._output_index = self.interpreter.get_output_details()[0]["index"]

    def extract(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> Optional[np.ndarray]:
        """`bbox` is (x1, y1, x2, y2) in frame pixels - typically a player's
        detection box. Returns a (17, 3) array of (y, x, score), each in
        [0, 1] relative to the square crop around `bbox` - or None if the
        crop would fall entirely outside the frame."""
        crop = _square_crop_bounds(bbox, frame.shape)
        if crop is None:
            return None
        x1, y1, x2, y2 = crop
        subframe = frame[y1:y2, x1:x2]
        if subframe.size == 0:
            return None

        resized = cv2.resize(subframe, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        # No BGR->RGB conversion - the pretrained classifier this feeds was
        # trained on features extracted the same (color-order-inconsistent
        # but internally self-consistent) way; converting now would feed it
        # an input distribution it never saw.
        input_tensor = resized[np.newaxis, ...].astype(np.uint8)

        self.interpreter.set_tensor(self._input_index, input_tensor)
        self.interpreter.invoke()
        keypoints = self.interpreter.get_tensor(self._output_index)
        return keypoints.reshape(17, 3)


def _square_crop_bounds(
    bbox: tuple[float, float, float, float], frame_shape: tuple[int, ...]
) -> Optional[tuple[int, int, int, int]]:
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = bbox
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    size = max((x2 - x1) * _CROP_MARGIN, (y2 - y1) * _CROP_MARGIN, _MIN_CROP_SIZE)
    size = min(size, frame_w, frame_h)
    half = size / 2

    center_x = min(max(center_x, half), frame_w - half)
    center_y = min(max(center_y, half), frame_h - half)
    if center_x < half or center_y < half:
        return None  # frame smaller than the minimum crop in some dimension

    return int(center_x - half), int(center_y - half), int(center_x + half), int(center_y + half)
