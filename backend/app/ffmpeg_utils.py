"""Shell out to ffmpeg/ffprobe for everything that doesn't need the ML
pipeline itself - reading a video's duration/fps/dimensions, grabbing one
frame for the calibration UI, and (in the worker) transcoding the
pipeline's raw OpenCV mp4v output to browser-playable H.264. Keeping this on
ffmpeg rather than importing src.video.io/opencv here means the lightweight
API container doesn't need the project's full ML requirements.txt - only
the worker does.
"""

import json
import subprocess
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def probe_video(path: str) -> dict:
    """Returns {"duration_s": float, "fps": float, "width": int, "height": int}."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,duration",
            "-show_entries", "format=duration",
            "-of", "json",
            path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe failed: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    num, den = (stream.get("r_frame_rate") or "0/1").split("/")
    fps = float(num) / float(den) if float(den) else 0.0
    duration_s = stream.get("duration") or data.get("format", {}).get("duration")
    return {
        "duration_s": float(duration_s) if duration_s is not None else None,
        "fps": fps,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def extract_frame(input_path: str, output_path: str, at_seconds: float = 0.5) -> None:
    """Grabs a single JPEG frame, a bit into the clip rather than frame 0 to
    dodge a black fade-in/logo card that's common at the very start."""
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(at_seconds),
            "-i", input_path,
            "-frames:v", "1",
            "-q:v", "2",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not Path(output_path).exists():
        raise FFmpegError(f"ffmpeg frame extraction failed: {result.stderr.strip()}")


def transcode_to_h264(input_path: str, output_path: str) -> None:
    """The pipeline writes mp4v (see src/video/io.py::VideoWriter) which most
    browsers won't play inline - re-encode to H.264/yuv420p with a faststart
    moov atom so the results page can stream it directly."""
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not Path(output_path).exists():
        raise FFmpegError(f"ffmpeg transcode failed: {result.stderr.strip()}")
