"""Pretrained (COCO) Faster R-CNN person detector, used to place the two
players on the minimap - port of yastrebksv/TennisProject's
person_detector.py, adapted to use this project's own
`CourtCalibration.pixel_to_world` (meters) for near/far court-half
assignment instead of the original's warped pixel-mask approach - simpler,
and reuses the same calibration `TennisCourtNetDetector` already produces.

No pose/skeleton output here (that's a different, heavier model) - just a
bounding box and a foot point (bottom-center of the box) per player, which
is all a minimap dot needs.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

from ._tracknet_arch import resolve_device
from src.analysis.court_calibration import CourtCalibration

_COCO_PERSON_LABEL = 1
# Full court length is 23.77m (see court_calibration.py's
# FULL_COURT_REFERENCE_POINTS); a player's world_y past the midpoint is on
# the "near" half, the origin side is "far".
_COURT_HALF_LENGTH = 23.77 / 2


@dataclass
class PlayerDetection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    foot_point: tuple[float, float]  # bottom-center of bbox, pixels
    world_point: Optional[tuple[float, float]] = None  # (x, y) meters, if a calibration was available

    @property
    def world_y(self) -> Optional[float]:
        return self.world_point[1] if self.world_point is not None else None


class PlayerDetector:
    def __init__(self, device: Optional[str] = None, min_score: float = 0.85):
        self.device = resolve_device(device)
        self.min_score = min_score
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.model.to(self.device)
        self.model.eval()

    def detect(self, frame: np.ndarray, calibration: Optional[CourtCalibration] = None) -> list[PlayerDetection]:
        tensor = torch.from_numpy(frame.transpose((2, 0, 1)) / 255.0).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            preds = self.model(tensor)[0]

        detections = []
        for box, label, score in zip(preds["boxes"], preds["labels"], preds["scores"]):
            if label.item() != _COCO_PERSON_LABEL or score.item() <= self.min_score:
                continue
            x1, y1, x2, y2 = box.detach().cpu().numpy().tolist()
            foot_point = ((x1 + x2) / 2, y2)
            world_point = calibration.pixel_to_world(*foot_point) if calibration is not None else None
            detections.append(PlayerDetection(bbox=(x1, y1, x2, y2), foot_point=foot_point, world_point=world_point))
        return detections

    @staticmethod
    def split_top_bottom(players: list[PlayerDetection]) -> tuple[list[PlayerDetection], list[PlayerDetection]]:
        """'top'/'far' half (world_y < court midpoint) vs 'bottom'/'near' half.
        Players with no calibration (world_y is None) are dropped from both -
        there's no court-relative half to assign them to."""
        far = [p for p in players if p.world_y is not None and p.world_y < _COURT_HALF_LENGTH]
        near = [p for p in players if p.world_y is not None and p.world_y >= _COURT_HALF_LENGTH]
        return far, near
