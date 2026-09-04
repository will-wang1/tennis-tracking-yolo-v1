"""Wrapper around WASB-SBDT's pretrained tennis ball detector - a stronger
alternative to `TrackNetBallDetector` (see that module's docstring), from
https://github.com/nttcom/WASB-SBDT (BMVC 2023, Tarashima et al., "Widely
Applicable Strong Baseline for Sports Ball Detection and Tracking" - MIT
licensed, unlike TrackNet/TennisCourtDetector/TennisProject elsewhere in
this project, so no research-use-only caveat here).

Architecturally it's an HRNet (see `_wasb_hrnet.py`) instead of TrackNet's
plain U-Net: HRNet keeps a full-resolution feature branch alive through
every stage rather than downsampling then upsampling, which the WASB paper
credits for better small/fast-object localization at a FRACTION of
TrackNetV2's parameter count (1.5M vs 11.3M) - and empirically, on this
project's own footage, it drops the ball far less often than TrackNet (see
the comparison run before this was wired in).

Like TrackNetBallDetector this takes 3 stacked frames and outputs a
per-pixel heatmap rather than a box, but differs in three ways that matter
for a drop-in replacement:

- Frame order is ASCENDING (oldest, ..., newest) - the reverse of
  TrackNetBallDetector's stacking - because that's the order WASB-SBDT's
  own training data pipeline uses (see `datasets/tennis.py`'s
  `frame_names[i:i+frames_in]`), and the pretrained weights were fit to
  that convention.
- The model outputs one heatmap PER input frame (3 channels in, 3 out) -
  this project only wants the newest frame's answer each call, so
  `detect()` reads off the last channel and throws the other two away
  rather than trying to use all three (WASB-SBDT's own eval harness
  amortizes across overlapping windows across separate calls; running the
  full model once per frame like this project's frame-by-frame streaming
  loop does is simpler and matches its own `step=1` - the densest, most
  accurate setting - for the tennis benchmark it was scored on).
- Preprocessing is an aspect-ratio-preserving affine crop+resize (see
  `_get_affine_transform`/`_affine_transform`, ported from the same repo)
  rather than a plain stretch resize - a plain resize would subtly distort
  the pretrained model's expected input geometry. For a 16:9 source frame
  (matching the 512x288 = 16:9 input this model was trained at) this
  reduces to an ordinary resize with no cropping or padding; other aspect
  ratios get center-cropped to 16:9 first, same as the model saw in
  training.
"""

from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from ._tracknet_arch import resolve_device
from ._wasb_hrnet import HRNet
from .ball_detector import Detection

MODEL_WIDTH = 512
MODEL_HEIGHT = 288
FRAMES_IN = 3

_HRNET_CONFIG = {
    "frames_in": FRAMES_IN,
    "frames_out": FRAMES_IN,
    "out_scales": [0],
    "MODEL": {
        "EXTRA": {
            "FINAL_CONV_KERNEL": 1,
            "STEM": {"INPLANES": 64, "STRIDES": [1, 1]},
            "STAGE1": {"NUM_MODULES": 1, "NUM_BRANCHES": 1, "BLOCK": "BOTTLENECK", "NUM_BLOCKS": [1], "NUM_CHANNELS": [32], "FUSE_METHOD": "SUM"},
            "STAGE2": {"NUM_MODULES": 1, "NUM_BRANCHES": 2, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2], "NUM_CHANNELS": [16, 32], "FUSE_METHOD": "SUM"},
            "STAGE3": {"NUM_MODULES": 1, "NUM_BRANCHES": 3, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2, 2], "NUM_CHANNELS": [16, 32, 64], "FUSE_METHOD": "SUM"},
            "STAGE4": {"NUM_MODULES": 1, "NUM_BRANCHES": 4, "BLOCK": "BASIC", "NUM_BLOCKS": [2, 2, 2, 2], "NUM_CHANNELS": [16, 32, 64, 128], "FUSE_METHOD": "SUM"},
            "DECONV": {"NUM_DECONVS": 0, "KERNEL_SIZE": []},
        },
    },
}

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_affine_transform(center: np.ndarray, scale: float, output_size: tuple[int, int], inv: bool = False) -> np.ndarray:
    """Ported from WASB-SBDT's `utils/image.py` (itself from the standard
    CenterNet-style affine-crop preprocessing). `scale` is a single scalar
    (the source square's side length, `max(src_h, src_w)`) - the source
    square is centered on `center`, then mapped onto `output_size`, which
    only preserves the source's aspect ratio when `output_size` itself has
    the same aspect ratio the model was trained at (512x288 = 16:9 here);
    otherwise this centers-and-crops the excess rather than distorting it.
    """
    dst_w, dst_h = output_size
    src_dir = np.array([0, scale * -0.5], dtype=np.float32)
    dst_dir = np.array([0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = [dst_w * 0.5, dst_h * 0.5] + dst_dir
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _affine_transform_point(pt: tuple[float, float], trans: np.ndarray) -> tuple[float, float]:
    x, y = pt
    result = trans @ np.array([x, y, 1.0], dtype=np.float32)
    return float(result[0]), float(result[1])


class WASBBallDetector:
    """Same `detect(frame) -> Optional[Detection]` interface as `BallDetector`/
    `TrackNetBallDetector`, but stateful across calls (needs a rolling
    3-frame buffer)."""

    # WASB-SBDT's own value, and lowering it turns out to cost accuracy
    # rather than buy coverage. The threshold does not just decide how MANY
    # blobs survive - it decides their SHAPE, because the blob is the
    # connected region above it, and a lower threshold grows each blob and
    # drags its intensity-weighted centroid off the ball. Measured on the US
    # Open clip, dropping this to 0.2 raised the tracked-frame count from
    # 386 to 400 but moved enough centroids to lose a hand-confirmed bounce
    # at 3.45s. Extra coverage is not worth degrading every position.
    _SCORE_THRESHOLD = 0.5
    _MIN_BLOB_AREA = 1
    _MAX_BLOB_AREA = 100
    # How many blobs to hand on per frame. The true ball is often not the
    # strongest peak when a player, a line or the net cord lights up too.
    _MAX_CANDIDATES = 5

    def __init__(
        self,
        weights_path: str | Path,
        device: Optional[str] = None,
        score_threshold: Optional[float] = None,
        max_candidates: Optional[int] = None,
    ):
        self.device = resolve_device(device)
        self.score_threshold = self._SCORE_THRESHOLD if score_threshold is None else score_threshold
        self.max_candidates = self._MAX_CANDIDATES if max_candidates is None else max_candidates

        self.model = HRNet(_HRNET_CONFIG)
        # weights_only=True breaks loading this checkpoint's legacy (.pth.tar,
        # pre-zipfile) format - a PyTorch bug where the restricted unpickler's
        # location-tag lookup comes back empty regardless of map_location,
        # raising "don't know how to restore data location ... (tagged with
        # )". False is safe here: this is a local file you supplied yourself,
        # not an untrusted download fetched at request time.
        checkpoint = torch.load(str(weights_path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self._frame_buffer: deque[np.ndarray] = deque(maxlen=FRAMES_IN)

    def detect(self, frame: np.ndarray) -> Optional[Detection]:
        """The single best candidate, for callers that want one point per
        frame. `detect_candidates` is the richer interface."""
        candidates = self.detect_candidates(frame)
        return candidates[0] if candidates else None

    def detect_candidates(self, frame: np.ndarray) -> list[Detection]:
        """Every plausible ball position in this frame, strongest first.

        Feed frames of a video in order; returns an empty list until the
        3-frame buffer fills."""
        height, width = frame.shape[:2]
        center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        scale = float(max(height, width))
        trans_in = _get_affine_transform(center, scale, (MODEL_WIDTH, MODEL_HEIGHT))
        warped = cv2.warpAffine(frame, trans_in, (MODEL_WIDTH, MODEL_HEIGHT), flags=cv2.INTER_LINEAR)

        normalized = (cv2.cvtColor(warped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
        self._frame_buffer.append(np.rollaxis(normalized, 2, 0))  # HWC -> CHW
        if len(self._frame_buffer) < FRAMES_IN:
            return []

        stacked = np.concatenate(list(self._frame_buffer), axis=0)  # oldest..newest, matches training order
        inp = torch.from_numpy(stacked).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            out = self.model(inp)[0]  # scale 0 (the only one WASB tennis uses)
        heatmap = torch.sigmoid(out[0, FRAMES_IN - 1]).cpu().numpy()  # newest frame's channel

        trans_inv = _get_affine_transform(center, scale, (MODEL_WIDTH, MODEL_HEIGHT), inv=True)
        return [
            Detection(x=x, y=y, confidence=confidence)
            for x, y, confidence in self._postprocess_candidates(
                heatmap, trans_inv, self.score_threshold, self.max_candidates
            )
        ]

    @classmethod
    def _postprocess(
        cls, heatmap: np.ndarray, trans_inv: np.ndarray
    ) -> Optional[tuple[float, float, float]]:
        """Threshold + connected-components + intensity-weighted centroid -
        WASB-SBDT's own default postprocessing (`blob_det_method: concomp`,
        `use_hm_weight: True` in `configs/detector/tracknetv2.yaml`), same
        approach `TrackNetBallDetector._postprocess` now uses for the same
        reason (a shape-fitting method like Hough circles is unstable on a
        coarse heatmap; a weighted centroid isn't)."""
        found = cls._postprocess_candidates(heatmap, trans_inv, cls._SCORE_THRESHOLD, 1)
        return found[0] if found else None

    @classmethod
    def _postprocess_candidates(
        cls,
        heatmap: np.ndarray,
        trans_inv: np.ndarray,
        score_threshold: float,
        max_candidates: int,
    ) -> list[tuple[float, float, float]]:
        """Every blob in the heatmap that could be the ball, strongest
        first, each as an intensity-weighted centroid.

        WASB-SBDT's own postprocessing (`blob_det_method: concomp`,
        `use_hm_weight: True`) but returning all the blobs rather than only
        the strongest - a weighted centroid over connected components is
        stable on a coarse heatmap where a shape-fitting method like Hough
        circles is not, and which blob is the ball is a question the
        trajectory can answer far better than a single frame can.
        """
        binary = (heatmap > score_threshold).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        if num_labels <= 1:
            return []

        rows, cols = np.indices(heatmap.shape)
        found: list[tuple[float, float, float, float]] = []  # score, x, y, peak
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if not (cls._MIN_BLOB_AREA <= area <= cls._MAX_BLOB_AREA):
                continue
            mask = labels == label
            weights = heatmap * mask
            total_weight = float(weights.sum())
            if total_weight <= 0:
                continue
            cx = float((cols * weights).sum() / total_weight)
            cy = float((rows * weights).sum() / total_weight)
            ox, oy = _affine_transform_point((cx, cy), trans_inv)
            found.append((total_weight, ox, oy, float(heatmap[mask].max())))

        found.sort(key=lambda item: item[0], reverse=True)
        return [(x, y, peak) for _, x, y, peak in found[:max_candidates]]
