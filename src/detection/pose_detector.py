"""Thin wrapper around a YOLOv8-pose checkpoint for multi-person keypoint detection.

Unlike `BallDetector`, this is inherently multi-person: both players are
usually visible, so `detect()` returns a list, one entry per detected
person, rather than a single best guess. `src/analysis/striker.py` picks
which one is actually swinging at the ball each frame.

The default weights are the bare pretrained COCO-keypoint checkpoint
(`ultralytics` downloads and caches it on first use) rather than a repo
artifact like `weights/ball_detector.pt` - there's no fine-tuning step for
pose in this pipeline.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ultralytics import YOLO

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


@dataclass
class PersonPose:
    keypoints: np.ndarray  # (17, 2) pixel xy, indexed by COCO_KEYPOINT_NAMES
    keypoint_confidence: np.ndarray  # (17,)
    bbox_confidence: float
    center_x: float
    center_y: float


class PoseDetector:
    def __init__(
        self,
        weights_path: str = "yolov8n-pose.pt",
        confidence: float = 0.3,
        imgsz: int = 1280,
        device: Optional[str] = None,
    ):
        self.model = YOLO(weights_path)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device

    def detect(self, frame: np.ndarray) -> list[PersonPose]:
        """Return one PersonPose per detected person above `confidence`."""
        results = self.model.predict(
            frame, conf=self.confidence, imgsz=self.imgsz, device=self.device, verbose=False
        )
        result = results[0]
        boxes = result.boxes
        keypoints = result.keypoints
        if boxes is None or keypoints is None or len(boxes) == 0:
            return []

        box_confidences = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        all_xy = keypoints.xy.cpu().numpy()  # (N, 17, 2)
        if keypoints.conf is not None:
            all_conf = keypoints.conf.cpu().numpy()  # (N, 17)
        else:
            all_conf = np.ones(all_xy.shape[:2])

        poses = []
        for i in range(len(box_confidences)):
            x1, y1, x2, y2 = xyxy[i]
            poses.append(
                PersonPose(
                    keypoints=all_xy[i],
                    keypoint_confidence=all_conf[i],
                    bbox_confidence=float(box_confidences[i]),
                    center_x=float((x1 + x2) / 2),
                    center_y=float((y1 + y2) / 2),
                )
            )
        return poses
