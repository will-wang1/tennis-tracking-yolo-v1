import numpy as np
import pytest

from src.detection._tracknet_arch import half_precision
from src.video.io import BrowserVideoWriter, VideoWriter, h264_encoder, is_browser_ready


def test_half_precision_only_on_cuda(monkeypatch):
    monkeypatch.delenv("TENNIS_FP32", raising=False)
    assert half_precision("cuda") and half_precision("cuda:0")
    assert not half_precision("mps") and not half_precision("cpu")
    monkeypatch.setenv("TENNIS_FP32", "1")
    assert not half_precision("cuda")


def test_mp4v_is_not_browser_ready(tmp_path):
    path = tmp_path / "old.mp4"
    writer = VideoWriter(path, 25, 64, 48)
    for _ in range(5):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.close()
    assert not is_browser_ready(path)


@pytest.mark.skipif(h264_encoder() is None, reason="needs ffmpeg")
def test_browser_writer_needs_no_second_encode(tmp_path):
    import cv2

    path = tmp_path / "new.mp4"
    writer = BrowserVideoWriter(path, 25, 64, 48)
    assert writer.isOpened()
    for i in range(10):
        writer.write(np.full((48, 64, 3), i * 20, np.uint8))
    writer.release()
    assert is_browser_ready(path)
    capture = cv2.VideoCapture(str(path))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
