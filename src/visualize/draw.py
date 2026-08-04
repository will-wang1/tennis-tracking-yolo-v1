"""Overlay the ball's tracked position and recent trail on a frame."""

from collections import deque
from typing import Optional

import cv2
import numpy as np

from src.tracking.ball_tracker import TrackedPosition

TRAIL_COLOR_DETECTED = (0, 255, 255)  # yellow
TRAIL_COLOR_INTERPOLATED = (0, 165, 255)  # orange


class TrailDrawer:
    def __init__(self, trail_length: int = 15):
        self.trail: deque[TrackedPosition] = deque(maxlen=trail_length)

    def draw(self, frame: np.ndarray, position: Optional[TrackedPosition]) -> np.ndarray:
        if position is not None:
            self.trail.append(position)

        for i, point in enumerate(self.trail):
            color = TRAIL_COLOR_INTERPOLATED if point.interpolated else TRAIL_COLOR_DETECTED
            fade = (i + 1) / max(len(self.trail), 1)
            radius = max(2, int(6 * fade))
            cv2.circle(frame, (int(point.x), int(point.y)), radius, color, -1)

        return frame
