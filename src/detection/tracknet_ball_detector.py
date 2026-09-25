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
        state_dict = torch.load(str(weights_path), map_location=self.device, weights_only=True)
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


# --- yastrebksv/TennisProject's exact ball tracking -------------------------
#
# `TrackNetBallDetector` above deliberately departs from the original
# post-processing (a weighted centroid instead of HoughCircles, no
# nearest-to-previous outlier filter) because that tracked this project's
# footage more steadily. That is the right call for ball TRACKING - and the
# wrong one for feeding TennisProject's pretrained CatBoost bounce model,
# which learned its features from tracks made THIS way. So the original is
# reproduced here verbatim, quirks included, for that one purpose.


def tennisproject_postprocess(
    feature_map: np.ndarray,
    prev_pred: list,
    scale: int = 2,
    max_dist: float = 80,
) -> tuple[Optional[float], Optional[float]]:
    """TennisProject ball_detector.BallDetector.postprocess, unchanged.

    Note `feature_map *= 255` on an argmax map already in 0..255: the
    product overflows when cast to uint8 and wraps, so what survives the
    127 threshold is NOT simply the brightest pixels. That is almost
    certainly unintended upstream, and it is kept anyway - the bounce model
    was trained on the tracks this produced, so "fixing" it would feed the
    model a different distribution and stop this being the same method.
    """
    feature_map = feature_map * 255
    feature_map = feature_map.reshape((MODEL_HEIGHT, MODEL_WIDTH))
    feature_map = feature_map.astype(np.uint8)
    _, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(
        heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2, maxRadius=7
    )
    x, y = None, None
    if circles is not None:
        if prev_pred[0]:
            for i in range(len(circles[0])):
                x_temp = circles[0][i][0] * scale
                y_temp = circles[0][i][1] * scale
                dist = float(np.hypot(x_temp - prev_pred[0], y_temp - prev_pred[1]))
                if dist < max_dist:
                    x, y = x_temp, y_temp
                    break
        else:
            x = circles[0][0][0] * scale
            y = circles[0][0][1] * scale
    return x, y


def track_ball_tennisproject(
    frames,
    weights_path: str | Path,
    device: Optional[str] = None,
) -> list[tuple[Optional[float], Optional[float]]]:
    """TennisProject ball_detector.BallDetector.infer_model, streamed.

    Same inputs, stacking order, argmax and post-processing as upstream; the
    only differences are that frames are consumed from an iterator instead
    of a list held in memory, and inference runs under no_grad - neither
    changes a single output value.

    Requires 1280x720 frames. Upstream hardcodes `scale=2` from its 640x360
    heatmap, so any other size returns coordinates in the wrong units - and
    the bounce features are raw pixel differences in exactly those units.
    """
    device = resolve_device(device)
    model = TrackNetArch(in_channels=9, out_channels=256)
    model.load_state_dict(torch.load(str(weights_path), map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    ball_track: list[tuple[Optional[float], Optional[float]]] = [(None, None)] * 2
    prev_pred: list = [None, None]
    window: deque[np.ndarray] = deque(maxlen=3)
    for index, frame in enumerate(frames):
        if frame.shape[:2] != (720, 1280):
            raise ValueError(
                f"TennisProject's method needs 1280x720 frames, got {frame.shape[1]}x{frame.shape[0]} "
                f"at frame {index} - normalise the clip first (scripts/normalise_clip.py)."
            )
        window.append(cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT)))
        if len(window) < 3:
            continue
        img_preprev, img_prev, img = window[0], window[1], window[2]
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2).astype(np.float32) / 255.0
        inp = np.expand_dims(np.rollaxis(imgs, 2, 0), axis=0)
        with torch.no_grad():
            out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = tennisproject_postprocess(output, prev_pred)
        prev_pred = [x_pred, y_pred]
        ball_track.append((x_pred, y_pred))
    return ball_track
