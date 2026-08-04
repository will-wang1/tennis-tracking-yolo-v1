"""Thin wrapper around a fine-tuned YOLO checkpoint for ball detection."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    x: float  # center x, pixels
    y: float  # center y, pixels
    confidence: float


class BallDetector:
    """Runs the fine-tuned YOLO ball model on individual frames.

    A low confidence threshold is used by default: missed detections are
    recovered later by the tracker's interpolation, but a missed frame that
    was never even a low-confidence candidate can't be recovered at all.
    False positives are cheap to filter downstream; false negatives aren't.
    """

    def __init__(
        self,
        weights_path: str | Path,
        confidence: float = 0.15,
        imgsz: int = 1280,
        device: Optional[str] = None,
    ):
        self.model = YOLO(str(weights_path))
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """Return the highest-confidence ball detection in `frame`, if any."""
        results = self.model.predict(
            frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        confidences = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confidences))
        x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
        return Detection(
            x=float((x1 + x2) / 2),
            y=float((y1 + y2) / 2),
            confidence=float(confidences[best_idx]),
        )
