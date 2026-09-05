"""Wrapper around the pretrained TennisCourtDetector checkpoint
(https://github.com/yastrebksv/TennisCourtDetector) - locates the 14
standard tennis-court keypoints in a single frame via a per-keypoint heatmap,
refines each to a line-intersection where possible, and returns them using
the SAME names as `src.analysis.court_calibration.FULL_COURT_REFERENCE_POINTS`,
so the result plugs directly into `CourtCalibration.from_keypoints` and
`src.visualize.draw.CourtOverlayDrawer` unchanged.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from ._tracknet_arch import TrackNetArch, resolve_device
from .keypoint_refine import refine_kps

MODEL_WIDTH = 640
MODEL_HEIGHT = 360

# Order matches yastrebksv/TennisCourtDetector's 14-point layout exactly
# (court_reference.py's `key_points`: baseline corners, then singles
# sidelines, then service lines, then the center service line) - see
# court_calibration.py's FULL_COURT_REFERENCE_POINTS docstring for the same
# geometry under these names.
KEYPOINT_NAMES = [
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
# T-junction keypoints (3 lines crossing, not 2) - refine_kps assumes exactly
# 2 detected lines, so these keep the raw heatmap-circle position instead.
_SKIP_REFINE = {8, 9, 12}


class TennisCourtNetDetector:
    """Non-stateful: unlike the ball TrackNet model, court detection uses one
    frame at a time, no temporal stacking."""

    def __init__(self, weights_path: str | Path, device: Optional[str] = None, refine: bool = True):
        self.device = resolve_device(device)
        self.refine = refine

        self.model = TrackNetArch(in_channels=3, out_channels=15)
        state_dict = torch.load(str(weights_path), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def detect(self, frame: np.ndarray) -> Optional[dict[str, tuple[float, float]]]:
        """Named keypoints found this frame (need not be all 14), or None if none were."""
        height, width = frame.shape[:2]
        img = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
        inp = img.astype(np.float32) / 255.0
        inp = torch.from_numpy(np.rollaxis(inp, 2, 0)).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            out = self.model(inp)[0]
        pred = torch.sigmoid(out).cpu().numpy()

        scale_x = width / MODEL_WIDTH
        scale_y = height / MODEL_HEIGHT

        points: dict[str, tuple[float, float]] = {}
        for i, name in enumerate(KEYPOINT_NAMES):
            heatmap = (pred[i].reshape(MODEL_HEIGHT, MODEL_WIDTH) * 255).astype(np.uint8)
            _, heatmap = cv2.threshold(heatmap, 170, 255, cv2.THRESH_BINARY)
            circles = cv2.HoughCircles(
                heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=2, minRadius=10, maxRadius=25
            )
            if circles is None:
                continue

            x_pred = float(circles[0][0][0]) * scale_x
            y_pred = float(circles[0][0][1]) * scale_y
            if self.refine and i not in _SKIP_REFINE:
                x_pred, y_pred = refine_kps(frame, int(y_pred), int(x_pred), crop_size=40)
            points[name] = (x_pred, y_pred)

        return points if points else None
