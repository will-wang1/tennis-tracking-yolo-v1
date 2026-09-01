"""Bird's-eye minimap overlay - ball position, recent bounces, and player
positions projected onto a synthetic top-down court via the same
`CourtCalibration` (meters) used for speed estimation. Same idea as
yastrebksv/TennisProject's minimap (`main.py`'s `get_court_img`/minimap
compositing), but built from this project's own court-reference geometry
(`src.analysis.court_calibration.FULL_COURT_REFERENCE_POINTS`) instead of
porting their separate pixel-based `CourtReference`/`court_reference.py`.
"""

from typing import Optional

import cv2
import numpy as np

from src.analysis.court_calibration import FULL_COURT_REFERENCE_POINTS

_COURT_WIDTH_M = 10.97
_COURT_LENGTH_M = 23.77

_LINE_COLOR = (255, 255, 255)
_COURT_COLOR = (40, 110, 40)
_BALL_COLOR = (0, 255, 0)
_BOUNCE_COLOR = (0, 255, 255)
_FAR_PLAYER_COLOR = (255, 80, 0)
_NEAR_PLAYER_COLOR = (0, 80, 255)

_LINE_EDGES = [
    ("baseline_far_left", "baseline_far_right"),
    ("baseline_near_left", "baseline_near_right"),
    ("baseline_far_left", "baseline_near_left"),
    ("baseline_far_right", "baseline_near_right"),
    ("singles_far_left", "singles_near_left"),
    ("singles_far_right", "singles_near_right"),
    ("service_far_left", "service_far_right"),
    ("service_near_left", "service_near_right"),
    ("center_service_far", "center_service_near"),
]


class MinimapDrawer:
    def __init__(self, width: int = 166, height: int = 350, margin: int = 20, corner_inset: int = 30):
        self.width = width
        self.height = height
        self.margin = margin
        self.corner_inset = corner_inset
        self._base = self._build_base()

    def _world_to_minimap(self, world_x: float, world_y: float) -> tuple[int, int]:
        usable_w = self.width - 2 * self.margin
        usable_h = self.height - 2 * self.margin
        px = self.margin + (world_x / _COURT_WIDTH_M) * usable_w
        py = self.margin + (world_y / _COURT_LENGTH_M) * usable_h
        return int(round(px)), int(round(py))

    def _build_base(self) -> np.ndarray:
        img = np.full((self.height, self.width, 3), _COURT_COLOR, dtype=np.uint8)
        for name_a, name_b in _LINE_EDGES:
            pa = self._world_to_minimap(*FULL_COURT_REFERENCE_POINTS[name_a])
            pb = self._world_to_minimap(*FULL_COURT_REFERENCE_POINTS[name_b])
            cv2.line(img, pa, pb, _LINE_COLOR, 1)
        return img

    def draw(
        self,
        frame: np.ndarray,
        ball_world: Optional[tuple[float, float]] = None,
        bounce_world_points: Optional[list[tuple[float, float]]] = None,
        far_players_world: Optional[list[tuple[float, float]]] = None,
        near_players_world: Optional[list[tuple[float, float]]] = None,
    ) -> np.ndarray:
        minimap = self._base.copy()

        for world_point in bounce_world_points or []:
            cv2.circle(minimap, self._world_to_minimap(*world_point), 3, _BOUNCE_COLOR, -1)
        for world_point in far_players_world or []:
            cv2.circle(minimap, self._world_to_minimap(*world_point), 5, _FAR_PLAYER_COLOR, -1)
        for world_point in near_players_world or []:
            cv2.circle(minimap, self._world_to_minimap(*world_point), 5, _NEAR_PLAYER_COLOR, -1)
        if ball_world is not None:
            cv2.circle(minimap, self._world_to_minimap(*ball_world), 4, _BALL_COLOR, -1)

        frame_height, frame_width = frame.shape[:2]
        y0 = self.corner_inset
        x0 = frame_width - self.corner_inset - self.width
        if y0 >= 0 and x0 >= 0 and y0 + self.height <= frame_height and x0 + self.width <= frame_width:
            frame[y0 : y0 + self.height, x0 : x0 + self.width] = minimap
        return frame
