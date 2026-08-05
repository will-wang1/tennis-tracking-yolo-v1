"""Turn a short window of pose keypoints into a fixed-length feature vector
for stroke classification.

Feature values are angles (already scale-invariant) or pixel quantities
normalized by shoulder width, so the same stroke performed closer to or
farther from the camera produces the same feature vector - the single most
important property for the classifier to generalize across camera
distances/zoom levels, since the two example clips in this repo already
frame players at very different scales.

This module is imported by BOTH scripts/train_stroke_classifier.py and
src/analysis/stroke_classifier.py - training and inference must always
compute features the exact same way, or the model silently degrades.
"""

from typing import Sequence

import numpy as np

from src.detection.pose_detector import PersonPose

LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12

FEATURE_NAMES = [
    "elbow_angle",
    "shoulder_angle",
    "wrist_height_rel_shoulder",
    "wrist_dx_per_frame",
    "wrist_dy_per_frame",
    "torso_lean",
]


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex `b`, between rays b->a and b->c, in radians."""
    v1 = a - b
    v2 = c - b
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _active_side(pose_window: Sequence[PersonPose]) -> str:
    """Whichever wrist moved more over the window is doing the swinging."""
    left = np.array([p.keypoints[LEFT_WRIST] for p in pose_window])
    right = np.array([p.keypoints[RIGHT_WRIST] for p in pose_window])
    left_travel = np.sum(np.linalg.norm(np.diff(left, axis=0), axis=1))
    right_travel = np.sum(np.linalg.norm(np.diff(right, axis=0), axis=1))
    return "left" if left_travel >= right_travel else "right"


def extract_features(pose_window: Sequence[PersonPose]) -> np.ndarray:
    """`pose_window`: consecutive striker PersonPose samples (e.g. 7 frames
    centered on the stroke). Returns a fixed-length vector, len(FEATURE_NAMES)."""
    if len(pose_window) < 2:
        raise ValueError("extract_features needs at least 2 frames to compute wrist velocity")

    side = _active_side(pose_window)
    shoulder_idx = LEFT_SHOULDER if side == "left" else RIGHT_SHOULDER
    elbow_idx = LEFT_ELBOW if side == "left" else RIGHT_ELBOW
    wrist_idx = LEFT_WRIST if side == "left" else RIGHT_WRIST

    center = pose_window[len(pose_window) // 2]
    kp = center.keypoints

    shoulder_width = float(np.linalg.norm(kp[LEFT_SHOULDER] - kp[RIGHT_SHOULDER])) + 1e-6

    elbow_angle = _angle_between(kp[shoulder_idx], kp[elbow_idx], kp[wrist_idx])
    shoulder_angle = float(
        np.arctan2(
            kp[RIGHT_SHOULDER][1] - kp[LEFT_SHOULDER][1],
            kp[RIGHT_SHOULDER][0] - kp[LEFT_SHOULDER][0],
        )
    )
    wrist_height_rel_shoulder = float(kp[shoulder_idx][1] - kp[wrist_idx][1]) / shoulder_width

    wrists = np.array([p.keypoints[wrist_idx] for p in pose_window])
    velocities = np.diff(wrists, axis=0) / shoulder_width
    mean_velocity = velocities.mean(axis=0)
    wrist_dx_per_frame, wrist_dy_per_frame = float(mean_velocity[0]), float(mean_velocity[1])

    shoulder_mid = (kp[LEFT_SHOULDER] + kp[RIGHT_SHOULDER]) / 2
    hip_mid = (kp[LEFT_HIP] + kp[RIGHT_HIP]) / 2
    torso_vector = shoulder_mid - hip_mid
    torso_lean = float(np.arctan2(torso_vector[0], -torso_vector[1]))  # 0 = perfectly upright

    return np.array(
        [
            elbow_angle,
            shoulder_angle,
            wrist_height_rel_shoulder,
            wrist_dx_per_frame,
            wrist_dy_per_frame,
            torso_lean,
        ]
    )
