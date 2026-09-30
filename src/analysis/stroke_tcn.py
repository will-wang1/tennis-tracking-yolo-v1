"""A stroke classifier trained on pose sequences: forehand, backhand, both
slices, both volleys, serve, overhead - and "not a swing".

WHY A NEW MODEL. BST (src/analysis/bst_strokes.py) knows only TenniSet's
near/far x left/right hit and serve - no slice, no volley - and got 80% of
the hand-labelled Dingles swings right. E2E-Spot's tennis labels
(data/external/e2e_spot_tennis) give ~9,800 broadcast strokes WITH slice and
volley, so a model can be pretrained on those and fine-tuned on labels from
our own camera. The recipe is CourtCheck's (a small temporal convolution
network over the hitter's keypoints; github.com/AggieSportsAnalytics/CourtCheck),
with two changes for our setting:

1. FAR PLAYERS ARE MIRRORED left-right, so every player is seen "from
   behind" like a near-end one - a far player's right hand is on the
   picture's left. Only coordinates are mirrored; keypoint names stay, as
   they name the person's own joints.
2. "NOT A SWING" IS A CLASS. A quarter of our detected swings were not
   swings (hand labels); the model learns to reject them, from our labels
   and from broadcast windows where the player is not hitting.

INPUT: a window of SEQ_LEN frames at 25fps around the swing - BEFORE_S
before to AFTER_S after - of one player's keypoints (17 COCO, pixel x/y and
confidence) and person box. Features per frame: keypoints centred on the
hips and divided by the player's height (so near and far, big and small
players compare), their frame-to-frame velocities, and the confidences.
"""

from typing import Optional

import numpy as np

CLASSES = ("forehand", "backhand", "forehand_slice", "backhand_slice",
           "forehand_volley", "backhand_volley", "serve", "overhead", "not_a_swing")
FPS = 25.0
BEFORE_S, AFTER_S = 1.2, 0.8
SEQ_LEN = int(round((BEFORE_S + AFTER_S) * FPS)) + 1  # 51
N_FEATURES = 17 * 2 + 17 * 2 + 17
MIN_CONF = 0.3
LEFT_RIGHT = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))

# E2E-Spot's comment -> our class
E2E_TO_CLASS = {
    "forehand_topspin": "forehand", "backhand_topspin": "backhand",
    "forehand_slice": "forehand_slice", "backhand_slice": "backhand_slice",
    "forehand_volley": "forehand_volley", "backhand_volley": "backhand_volley",
    "serve": "serve", "overhead": "overhead",
}


def class_of_label(label: dict) -> Optional[str]:
    """Our labelling page's label (stroke + volley/slice toggles) -> a class.
    A volley that is also a slice counts as a volley."""
    stroke = label.get("stroke")
    if stroke in ("not a swing",):
        return "not_a_swing"
    if stroke in ("serve", "overhead"):
        return stroke
    if stroke in ("forehand", "backhand"):
        if label.get("volley"):
            return f"{stroke}_volley"
        if label.get("slice"):
            return f"{stroke}_slice"
        return stroke
    return None  # skipped


def window_times(center_s: float) -> np.ndarray:
    """The SEQ_LEN sample times, in seconds, of a window around `center_s`."""
    return center_s + np.linspace(-BEFORE_S, AFTER_S, SEQ_LEN)


def features(keypoints: np.ndarray, conf: np.ndarray, boxes: np.ndarray, far: bool) -> np.ndarray:
    """(T,17,2) pixels, (T,17) confidences, (T,4) boxes (NaN rows where the
    player was not seen) -> (T, N_FEATURES)."""
    kp = keypoints.astype(np.float64).copy()
    c = np.nan_to_num(conf.astype(np.float64), nan=0.0)
    c[~np.isfinite(kp).all(axis=2)] = 0.0
    heights = boxes[:, 3] - boxes[:, 1]
    scale = float(np.nanmedian(heights)) if np.isfinite(heights).any() else 1.0
    scale = scale if scale > 1 else 1.0
    hips = kp[:, [11, 12]]
    hip_ok = (c[:, [11, 12]] >= MIN_CONF).all(axis=1)
    centre = np.where(hip_ok[:, None], hips.mean(axis=1),
                      np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1))
    rel = (kp - centre[:, None, :]) / scale
    if far:
        rel[:, :, 0] = -rel[:, :, 0]  # seen from behind, like a near-end player
    seen = c >= MIN_CONF
    rel[~seen] = 0.0
    rel = np.nan_to_num(rel)
    vel = np.zeros_like(rel)
    both = seen[1:] & seen[:-1]
    vel[1:] = np.where(both[:, :, None], rel[1:] - rel[:-1], 0.0)
    out = np.concatenate([rel.reshape(len(rel), -1), vel.reshape(len(vel), -1), np.where(seen, c, 0.0)], axis=1)
    return out.astype(np.float32)


def mirror_hand(feats: np.ndarray) -> np.ndarray:
    """The same stroke by an opposite-handed player: mirror left-right AND
    swap left/right joints. A forehand stays a forehand (training
    augmentation; Nadal's forehands are left-handed)."""
    f = feats.copy()
    xy = f[:, :34].reshape(-1, 17, 2)
    v = f[:, 34:68].reshape(-1, 17, 2)
    cf = f[:, 68:]
    for arr in (xy, v):
        arr[:, :, 0] *= -1
        for a, b in LEFT_RIGHT:
            arr[:, [a, b]] = arr[:, [b, a]]
    for a, b in LEFT_RIGHT:
        cf[:, [a, b]] = cf[:, [b, a]]
    return np.concatenate([xy.reshape(len(f), -1), v.reshape(len(f), -1), cf], axis=1)


def build_model(n_classes: int = len(CLASSES), width: int = 128, levels: int = 4, dropout: float = 0.3):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation), nn.BatchNorm1d(width), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation), nn.BatchNorm1d(width), nn.ReLU(),
            )

        def forward(self, x):
            return torch.relu(x + self.net(x))

    class StrokeTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Conv1d(N_FEATURES + 1, width, 1), nn.ReLU())  # +1: which end
            self.blocks = nn.Sequential(*[Block(2 ** i) for i in range(levels)])
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * width, n_classes))

        def forward(self, x, far):
            # x: (B, T, F); far: (B,)
            end = far.float()[:, None, None].expand(-1, x.shape[1], 1)
            h = self.blocks(self.inp(torch.cat([x, end], dim=2).transpose(1, 2)))
            return self.head(torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1))

    return StrokeTCN()
