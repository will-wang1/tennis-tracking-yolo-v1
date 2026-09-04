"""Wrapper around a pretrained TrackNet checkpoint for ball detection.

Standby/backup detector for when `weights/ball_detector.pt` (the fine-tuned
YOLO model) is in a bad state - e.g. mid-retrain on Kaggle. Uses the
pretrained weights from https://github.com/yastrebksv/TrackNet (no license
file in that repo, so treat these weights as research-use-only, not for
redistribution), which is also where this project's own
`data/raw_tracknet_source*` training data originated.

Unlike the YOLO detector, TrackNet takes the current frame plus the two
preceding frames stacked as a 9-channel input and regresses a per-pixel
heatmap (formulated as 256-way pixel classification, per the original
paper) rather than a bounding box - the ball position is the center of the
brightest circular blob in that heatmap. That makes this detector
stateful: `detect()` must be called once per frame, in order, for a single
video - it keeps a rolling buffer of the previous two frames internally and
returns `None` for the first two calls of a video until that buffer fills.
"""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from ._tracknet_arch import TrackNetArch, resolve_device
from .ball_detector import Detection

MODEL_WIDTH = 640
MODEL_HEIGHT = 360


class TrackNetBallDetector:
    """Same `detect(frame) -> Optional[Detection]` interface as `BallDetector`, but stateful across calls."""

    def __init__(self, weights_path: str | Path, device: Optional[str] = None):
        self.device = resolve_device(device)

        self.model = TrackNetArch(in_channels=9, out_channels=256)
        # See wasb_ball_detector.py's comment on this same line - weights_only=True
        # breaks loading legacy-format checkpoints regardless of map_location.
        state_dict = torch.load(str(weights_path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self._prev_frames: deque[np.ndarray] = deque(maxlen=2)

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """Feed the next frame of a video, in order. Returns None until the 2-frame buffer fills."""
        height, width = frame.shape[:2]
        resized = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))

        if len(self._prev_frames) < 2:
            self._prev_frames.append(resized)
            return None

        img_prev, img_preprev = self._prev_frames[1], self._prev_frames[0]
        stacked = np.concatenate((resized, img_prev, img_preprev), axis=2).astype(np.float32) / 255.0
        stacked = np.rollaxis(stacked, 2, 0)
        inp = torch.from_numpy(stacked).unsqueeze(0).float().to(self.device)

        self._prev_frames.append(resized)

        with torch.no_grad():
            out = self.model(inp)
        pred = out.argmax(dim=1).squeeze(0).cpu().numpy()

        xy = self._postprocess(pred, width, height)
        if xy is None:
            return None
        x, y, confidence = xy
        return Detection(x=x, y=y, confidence=confidence)

    # Reject a thresholded blob outside this pixel-area range on the 640x360
    # heatmap - too small is stray single-pixel noise, too large is the
    # model lighting up something that isn't a compact ball-sized dot
    # (a line, a bright patch of court).
    _MIN_BLOB_AREA = 1
    _MAX_BLOB_AREA = 60

    @staticmethod
    def _postprocess(pred: np.ndarray, orig_width: int, orig_height: int) -> Optional[tuple[float, float, float]]:
        """Threshold the predicted heatmap and locate the ball as a small blob.

        Originally used `cv2.HoughCircles` to find the blob center, but that
        fits a circle shape to the thresholded region rather than just
        averaging it - on this heatmap's coarse, blocky blobs (a handful of
        pixels at 640x360, then scaled ~3x back up for 1080p) that shape-fit
        is unstable frame to frame even when the underlying blob barely
        moves, which is what most of the ball trajectory's on-screen jitter
        actually came from (measured: median frame-to-frame curvature ~16px
        with Hough vs ~3.5px with the intensity-weighted centroid below, on
        the same real footage). A weighted centroid - the blob's pixels
        averaged by their heatmap intensity, i.e. its center of mass - is a
        much steadier estimate of "where most of the model's confidence
        actually sits" and doesn't require the blob to look circular at all.

        Scale factors are derived from the actual frame size rather than the
        original repo's hardcoded 2x, since that assumed exactly 1280x720 input.
        """
        heatmap = pred.reshape(MODEL_HEIGHT, MODEL_WIDTH).astype(np.uint8)
        peak_value = float(heatmap.max())
        _, binary = cv2.threshold(heatmap, 127, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        if num_labels <= 1:
            return None
        # label 0 is the background component - only consider real blobs
        areas = stats[1:, cv2.CC_STAT_AREA]
        best_label = 1 + int(np.argmax(areas))
        area = stats[best_label, cv2.CC_STAT_AREA]
        if not (TrackNetBallDetector._MIN_BLOB_AREA <= area <= TrackNetBallDetector._MAX_BLOB_AREA):
            return None

        mask = labels == best_label
        weights = heatmap.astype(np.float64) * mask
        total_weight = weights.sum()
        if total_weight <= 0:
            return None
        rows, cols = np.indices(heatmap.shape)
        cx = float((cols * weights).sum() / total_weight)
        cy = float((rows * weights).sum() / total_weight)

        scale_x = orig_width / MODEL_WIDTH
        scale_y = orig_height / MODEL_HEIGHT
        return cx * scale_x, cy * scale_y, peak_value / 255.0
