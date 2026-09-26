"""Stable player identity: join a clip's tracklets into players.

The person tracker (scripts/track_people.py) gives one id per continuous
stretch of tracking - a tracklet - and starts a new one whenever someone
is hidden too long, leaves frame, or the camera cuts. On
dingles_serve_volley that was 66 ids (37 lasting a second or more) for 7
people. Every "per player" number needs those joined back up.

HOW, in three parts:

1. Who is a person at all. The yellow ball cart was detected as a person
   in six tracklets. It never has a human skeleton (pose on 0-3% of its
   frames, against 78-99% for every real person), so a tracklet posed on
   under MIN_POSED_SHARE of its frames is dropped.

2. An appearance fingerprint per tracklet: HSV colour histograms of the
   torso and the legs, separately, averaged over sampled frames. Chosen by
   measurement against hand-labelled identities on the Dingles clip -
   nearest-neighbour tracklet is the same person for 90% of tracklets
   (28/31), against 74% for ImageNet ResNet-50 features and 55% for
   DINOv2-small, and it separated the two players in the same teal top
   perfectly (0/10 wrong). General-purpose image models encode pose and
   background as much as the person; clothing colour, split into top and
   bottom, is what tells these players apart. The user has confirmed
   their squads wear different clothing, which is the condition this
   relies on; a squad in fully identical kit - same top AND bottoms -
   would defeat it, and a person-re-identification model would be the
   upgrade then.

3. Clustering with one hard rule: two tracklets that are on screen at the
   same time, in DIFFERENT places, cannot be the same person. Two
   exceptions come from the data: a tracker handing one person over from
   an old id to a new one overlaps them by a few frames (<=9 on Dingles),
   so up to HANDOVER_FRAMES of co-occurrence is allowed; and a person can
   be boxed twice at once (the coach carried ids 54 AND 115 for 2.8s),
   so co-occurring boxes that overlap heavily (IoU >= DUPLICATE_IOU) are
   one person seen twice, not a conflict. Within that rule, the most
   similar pair of clusters is merged until no compatible pair is closer
   than the distance threshold.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

MIN_POSED_SHARE = 0.3
HANDOVER_FRAMES = 12
DUPLICATE_IOU = 0.5
# 1 - cosine similarity. Measured on dingles_serve_volley against
# hand-labelled identities (scripts/assign_players.py --truth): 100%
# pairwise precision at every setting tried, recall 100% from 0.35 to 0.6
# (7 players, matching the labels exactly), below 0.35 people start
# splitting. 0.45 is the middle of that plateau rather than its edge.
DEFAULT_MERGE_DISTANCE = 0.45


def colour_fingerprint(crop_bgr: np.ndarray) -> np.ndarray:
    """Torso and leg HSV histograms of one person crop, concatenated.
    The crop is resized to 128x256 first so regions are comparable; the
    torso band is rows 40-130, legs 130-220, central half of the width to
    keep background out."""
    import cv2

    crop = cv2.resize(np.ascontiguousarray(crop_bgr), (128, 256))
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    parts = []
    for y0, y1 in ((40, 130), (130, 220)):
        h = cv2.calcHist([hsv[y0:y1, 32:96]], [0, 1], None, [12, 6], [0, 180, 0, 256]).flatten()
        parts.append(h / (h.sum() + 1e-9))
    return np.concatenate(parts)


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def conflicts(boxes_by_track: dict[int, dict[int, tuple]]) -> set[frozenset]:
    """Pairs of tracklets that cannot be one person: on screen together,
    in different places, for longer than a handover."""
    ids = sorted(boxes_by_track)
    out = set()
    for i, a in enumerate(ids):
        fa = boxes_by_track[a]
        for b in ids[i + 1:]:
            fb = boxes_by_track[b]
            shared = fa.keys() & fb.keys()
            if len(shared) <= HANDOVER_FRAMES:
                continue
            apart = sum(1 for f in shared if box_iou(fa[f], fb[f]) < DUPLICATE_IOU)
            if apart > HANDOVER_FRAMES:
                out.add(frozenset((a, b)))
    return out


@dataclass
class Players:
    player_of: dict[int, int]  # tracklet id -> player number (1..n), people only
    not_people: list[int]  # tracklets dropped as not a person (no skeleton)
    merge_distance: float

    def members(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for t, p in self.player_of.items():
            out.setdefault(p, []).append(t)
        return {p: sorted(ts) for p, ts in sorted(out.items())}


def cluster_tracklets(
    fingerprints: dict[int, np.ndarray],
    boxes_by_track: dict[int, dict[int, tuple]],
    merge_distance: float = DEFAULT_MERGE_DISTANCE,
    cannot_link: Optional[set[frozenset]] = None,
) -> dict[int, int]:
    """Constrained average-linkage clustering. Returns tracklet -> cluster
    index (0-based, by size). Two clusters merge only if no pair across
    them conflicts, and only while their average distance is under
    `merge_distance`."""
    ids = sorted(fingerprints)
    if not ids:
        return {}
    if cannot_link is None:
        cannot_link = conflicts({t: boxes_by_track[t] for t in ids})
    unit = {t: fingerprints[t] / (np.linalg.norm(fingerprints[t]) + 1e-12) for t in ids}
    dist = {(a, b): 1.0 - float(unit[a] @ unit[b]) for a in ids for b in ids}
    clusters = [[t] for t in ids]

    def linkage(x, y):
        return float(np.mean([dist[(a, b)] for a in x for b in y]))

    def compatible(x, y):
        return not any(frozenset((a, b)) in cannot_link for a in x for b in y)

    while True:
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if not compatible(clusters[i], clusters[j]):
                    continue
                d = linkage(clusters[i], clusters[j])
                if d < merge_distance and (best is None or d < best[0]):
                    best = (d, i, j)
        if best is None:
            break
        _, i, j = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
    clusters.sort(key=lambda c: -sum(len(boxes_by_track[t]) for t in c))
    return {t: k for k, c in enumerate(clusters) for t in c}


def pairwise_scores(assignment: dict[int, int], truth: dict[int, str]) -> dict:
    """Pairwise precision/recall of 'same person' decisions against labels,
    over tracklets present in both."""
    ids = sorted(set(assignment) & set(truth))
    tp = fp = fn = 0
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            same_pred = assignment[a] == assignment[b]
            same_true = truth[a] == truth[b]
            tp += same_pred and same_true
            fp += same_pred and not same_true
            fn += same_true and not same_pred
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return {"precision": precision, "recall": recall, "clusters": len(set(assignment[t] for t in ids))}
