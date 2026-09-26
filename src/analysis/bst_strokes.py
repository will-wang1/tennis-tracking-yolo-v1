"""Stroke type from BST (src/third_party/bst), trained on TenniSet.

WHAT THE MODEL KNOWS. TenniSet's six classes, from broadcast Olympic
tennis: a hit or a serve, by the near or the far player, and for hits the
side - left or right. Nothing about volleys, slice or smashes.

Left/right is the side AS THE CAMERA SEES IT: checked by eye on
dingles_serve_volley, every clear forehand by a right-hander came out
near-right (seen from behind) or far-left (seen from the front) - 9 of 9.
So forehand/backhand needs the player's handedness; `stroke_name` assumes
right-handed until a player is marked otherwise.

WHAT IT NEEDS, per frame of a short clip around the stroke, for EXACTLY two
people - the far one first, then the near one:
  - 17 COCO keypoints, relative to that person's box, centred on it and
    divided by its diagonal (BST's normalize_joints, center_align=True),
    plus the 19 COCO bones (joint differences) - "JnB_bone"
  - their position on court: feet through the homography, divided by the
    doubles court's size, so the far baseline is y=0 and the near one y=1
  - the ball's pixel position divided by the frame size
Missing anything in a frame is zeros, as in BST's own preprocessing.
Clips are padded to SEQ_LEN frames; the real length goes in separately.

Squads have more than two people on court. The hitter is one of the two;
the other is a person in the opposite half (the one the ball goes to), so
the input looks like the singles broadcast the model was trained on.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

CLASSES = ("HFL", "HFR", "HNL", "HNR", "SF", "SN")  # BST's TenniSet class ids 0..5
SEQ_LEN = 100
# TenniSet's labelled hits last a median 27-30 frames and serves 61-62. A
# clip for a hit is centred a little before contact, where the swing is.
HIT_BEFORE, HIT_AFTER = 18, 10
SERVE_BEFORE, SERVE_AFTER = 50, 10
BONE_PAIRS = (
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    (3, 5), (4, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
# Final-label gates, from the same by-eye check (42 candidate hits, one
# clip - provisional): a serve-length clip scoring serve at 0.8+ was a real
# serve 5/5 times; a hit clip whose near/far disagrees with the tracking
# was not a real stroke 13 of 16 times (pre-serve ball bounces, the coach
# holding balls, the ball cart); and a real hit's top class almost always
# scored 0.55+ where the non-strokes that slipped through scored ~0.5.
SERVE_MIN = 0.8
HIT_MIN = 0.55
FOREHAND_RIGHT_HANDED = {"HNR", "HFL"}
DOUBLES_WIDTH_M = 10.97
COURT_LENGTH_M = 23.77


@dataclass
class PersonFrame:
    keypoints: Optional[np.ndarray]  # (17, 2) frame pixels, or None
    box: Optional[tuple[float, float, float, float]]  # x1, y1, x2, y2
    court_m: Optional[tuple[float, float]]  # feet on the court, metres


def normalize_joints(keypoints: np.ndarray, box) -> np.ndarray:
    """BST's normalize_joints for one person, center_align=True."""
    x1, y1, x2, y2 = box
    dist = float(np.hypot(x2 - x1, y2 - y1)) or 1.0
    kp = np.asarray(keypoints, dtype=np.float64)
    out = np.where(kp != 0.0, (kp - np.array([x1, y1])) / dist, 0.0)
    centre = np.array([(x2 - x1) / 2.0, (y2 - y1) / 2.0]) / dist
    return out - centre


def bones(joints: np.ndarray) -> np.ndarray:
    """(..., 17, 2) -> (..., 19, 2), zero where either end is missing."""
    out = []
    for a, b in BONE_PAIRS:
        ja, jb = joints[..., a, :], joints[..., b, :]
        out.append(np.where((ja != 0.0) & (jb != 0.0), jb - ja, 0.0))
    return np.stack(out, axis=-2)


def build_input(
    far: Sequence[PersonFrame],
    near: Sequence[PersonFrame],
    ball: Sequence[Optional[tuple[float, float]]],
    frame_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """One clip as BST's (JnB (t, 2, 72), pos (t, 2, 2), shuttle (t, 2),
    video_len), padded to SEQ_LEN."""
    width, height = frame_size
    t = len(ball)
    joints = np.zeros((t, 2, 17, 2))
    pos = np.zeros((t, 2, 2))
    shuttle = np.zeros((t, 2))
    for i in range(t):
        for p, person in enumerate((far[i], near[i])):
            if person.keypoints is not None and person.box is not None:
                joints[i, p] = normalize_joints(person.keypoints, person.box)
            if person.court_m is not None:
                pos[i, p] = (person.court_m[0] / DOUBLES_WIDTH_M, person.court_m[1] / COURT_LENGTH_M)
        if ball[i] is not None:
            shuttle[i] = (ball[i][0] / width, ball[i][1] / height)
    jnb = np.concatenate((joints, bones(joints)), axis=-2).reshape(t, 2, -1)
    video_len = min(t, SEQ_LEN)
    if t > SEQ_LEN:  # BST keeps the END of an over-long clip
        jnb, pos, shuttle = jnb[-SEQ_LEN:], pos[-SEQ_LEN:], shuttle[-SEQ_LEN:]
    else:
        pad = SEQ_LEN - t
        jnb = np.pad(jnb, ((0, pad), (0, 0), (0, 0)))
        pos = np.pad(pos, ((0, pad), (0, 0), (0, 0)))
        shuttle = np.pad(shuttle, ((0, pad), (0, 0)))
    return jnb.astype(np.float32), pos.astype(np.float32), shuttle.astype(np.float32), video_len


def stroke_name(label: str, probability: float, serve_probability: float, half_agrees: bool,
                left_handed: bool = False) -> str:
    """forehand | backhand | serve | unsure, from one hit's two clips."""
    if serve_probability >= SERVE_MIN or (label.startswith("S") and probability >= SERVE_MIN):
        return "serve"
    if not half_agrees or probability < HIT_MIN or label.startswith("S"):
        return "unsure"
    forehand = (label in FOREHAND_RIGHT_HANDED) != left_handed
    return "forehand" if forehand else "backhand"


def load_model(weights: str, device: str = "cpu"):
    import torch

    from src.third_party.bst.bst import BST_AP

    model = BST_AP(in_dim=(17 + len(BONE_PAIRS)) * 2, n_class=len(CLASSES), seq_len=SEQ_LEN, depth_tem=2, depth_inter=1)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()


def classify(model, clips: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]], device: str = "cpu") -> np.ndarray:
    """Class probabilities, (n, 6), in CLASSES order."""
    import torch

    if not clips:
        return np.zeros((0, len(CLASSES)))
    jnb, pos, shuttle, lens = (np.stack(x) for x in zip(*clips))
    with torch.no_grad():
        logits = model(
            torch.from_numpy(jnb).to(device),
            torch.from_numpy(shuttle).to(device),
            torch.from_numpy(pos).to(device),
            torch.from_numpy(lens.astype(np.int64)).to(device),
        )
    return torch.softmax(logits, dim=1).cpu().numpy()
