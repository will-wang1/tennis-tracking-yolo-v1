"""Minimal OpenCV video read/write helpers."""

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    def __init__(self, path: str | Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame
        self.cap.release()

    def frame_count_hint(self) -> "int | None":
        """The container's own frame count metadata, if it reports one -
        COSMETIC only (a progress bar's `total=`), never a substitute for
        actually counting frames as they're decoded. Some containers
        (variable frame rate especially) report this inaccurately or not
        at all, which is fine for a progress bar and would not be fine for
        anything that has to be right."""
        count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return int(count) if count and count > 0 else None


class VideoWriter:
    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()
