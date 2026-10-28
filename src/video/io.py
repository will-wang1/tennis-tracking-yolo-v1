"""Minimal OpenCV video read/write helpers."""

import functools
import shutil
import subprocess
import tempfile
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


# Encoders to try, fastest first: the NVIDIA GPU's, the Mac's, then the CPU.
_H264_ENCODERS = {
    "h264_nvenc": ["-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"],
    "h264_videotoolbox": ["-b:v", "3M"],
    "libx264": ["-preset", "veryfast", "-crf", "23"],
}


@functools.cache
def h264_encoder() -> "str | None":
    """The first H.264 encoder that actually works here, or None without
    ffmpeg. Listed is not enough: NVENC is listed by Ubuntu's ffmpeg whether
    or not the driver can run it, so each one encodes a test frame."""
    if not shutil.which("ffmpeg"):
        return None
    for name, options in _H264_ENCODERS.items():
        test = ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "color=size=256x256:rate=25",
                "-frames:v", "1", "-c:v", name, *options, "-f", "null", "-"]
        if subprocess.run(test, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            return name
    return None


class BrowserVideoWriter:
    """Writes frames straight to an H.264 MP4 a browser can play and seek
    before it has all arrived (yuv420p, index at the front) - one encode,
    instead of OpenCV's mp4v followed by a second pass through ffmpeg.
    Needs ffmpeg (h264_encoder() is not None)."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int):
        encoder = h264_encoder()
        if encoder is None:
            raise RuntimeError("BrowserVideoWriter needs ffmpeg with an H.264 encoder")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._errors = tempfile.TemporaryFile()
        self._process = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
             "-c:v", encoder, *_H264_ENCODERS[encoder],
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(self.path)],
            stdin=subprocess.PIPE, stderr=self._errors,
        )

    def isOpened(self) -> bool:  # noqa: N802 - cv2.VideoWriter's name, so callers can use either
        return self._process.poll() is None

    def write(self, frame: np.ndarray) -> None:
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError:
            self.release()

    def release(self) -> None:
        if self._process.stdin and not self._process.stdin.closed:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
        if self._process.wait() != 0:
            self._errors.seek(0)
            message = self._errors.read().decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg could not write {self.path}: {message}")


def is_browser_ready(path: str | Path) -> bool:
    """H.264 with the index (moov) before the frames (mdat), as
    BrowserVideoWriter writes it - so it needs no second encode."""
    capture = cv2.VideoCapture(str(path))
    codec = int(capture.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode(errors="replace").lower()
    capture.release()
    if codec not in ("avc1", "h264"):
        return False
    with open(path, "rb") as handle:
        while header := handle.read(8):
            if len(header) < 8:
                return False
            size, kind = int.from_bytes(header[:4], "big"), header[4:]
            if kind == b"moov":
                return True
            if kind == b"mdat":
                return False
            if size == 1:
                size = int.from_bytes(handle.read(8), "big") - 8
            elif size == 0:
                return False
            handle.seek(size - 8, 1)
    return False
