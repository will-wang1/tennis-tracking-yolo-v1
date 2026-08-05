"""Thin wrapper around a trained court-keypoint checkpoint.

Detects the 14 standard tennis court keypoints (see
`src.analysis.court_calibration.FULL_COURT_REFERENCE_POINTS` for their
real-world layout) in a single frame. There's exactly one court per image,
so unlike `PoseDetector` this returns a single best detection, not a list.

No pretrained checkpoint ships with this repo (unlike the ball detector) -
train one first with `scripts/train_court_keypoints.py`. See
`scripts/prepare_court_keypoint_dataset.py` and the README for the full
dataset -> training -> inference workflow.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ultralytics import YOLO

# Order matches the yastrebksv/TennisCourtDetector reference dataset's
# annotation order exactly (see that project's court_reference.py) -
# training data prepared by scripts/prepare_court_keypoint_dataset.py
# preserves this order, so a checkpoint trained on it can be indexed
# directly against these names.
COURT_KEYPOINT_NAMES = [
    "baseline_far_left",
    "baseline_far_right",
    "baseline_near_left",
    "baseline_near_right",
    "singles_far_left",
    "singles_near_left",
    "singles_far_right",
    "singles_near_right",
    "service_far_left",
    "service_far_right",
    "service_near_left",
    "service_near_right",
    "center_service_far",
    "center_service_near",
]

# For horizontal-flip augmentation during training: index i's flipped
# counterpart. "far"/"near" (depth) is unaffected by a horizontal flip,
# only "left"/"right" swaps; the two center-service points have no
# left/right counterpart, so they map to themselves.
COURT_KEYPOINT_FLIP_INDEX = [1, 0, 3, 2, 6, 7, 4, 5, 9, 8, 11, 10, 12, 13]


@dataclass
class CourtKeypoints:
    keypoints: np.ndarray  # (14, 2) pixel xy, indexed by COURT_KEYPOINT_NAMES
    keypoint_confidence: np.ndarray  # (14,)
    bbox_confidence: float

    def as_named_points(self, min_confidence: float = 0.5) -> dict[str, tuple[float, float]]:
        """Only the keypoints confidently detected, keyed by name - feed
        straight to `CourtCalibration.from_keypoints`."""
        return {
            name: (float(self.keypoints[i][0]), float(self.keypoints[i][1]))
            for i, name in enumerate(COURT_KEYPOINT_NAMES)
            if self.keypoint_confidence[i] >= min_confidence
        }


class CourtKeypointDetector:
    def __init__(
        self,
        weights_path: str,
        confidence: float = 0.25,
        imgsz: int = 1280,
        device: Optional[str] = None,
    ):
        self.model = YOLO(weights_path)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame: np.ndarray) -> Optional[CourtKeypoints]:
        """Return the single highest-confidence court detection, or None.

        A fixed broadcast camera only needs this run once per video (the
        court doesn't move mid-clip) - see scripts/calibrate_court_auto.py.
        """
        results = self.model.predict(
            frame, conf=self.confidence, imgsz=self.imgsz, device=self.device, verbose=False
        )
        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or keypoints is None or len(boxes) == 0:
            return None

        box_confidences = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(box_confidences))
        all_xy = keypoints.xy.cpu().numpy()
        if keypoints.conf is not None:
            all_conf = keypoints.conf.cpu().numpy()
        else:
            all_conf = np.ones(all_xy.shape[:2])

        return CourtKeypoints(
            keypoints=all_xy[best_idx],
            keypoint_confidence=all_conf[best_idx],
            bbox_confidence=float(box_confidences[best_idx]),
        )
