"""Stroke type for every hit in a session, with BST trained on TenniSet
(src/analysis/bst_strokes.py says what the model can and cannot tell).

EXPERIMENTAL: the model has never seen a fence camera, a squad, or amateur
players. Every hit gets two independent sanity checks to judge it by:
  - near/far: the model's class says which end hit; the tracking already
    knows. Disagreement means the model is not reading the clip.
  - serve: a serve clip is scored separately with a serve-length window.

Hits are the report's impacts that are a confirmed racket contact, or
unclassified with a person within reach (the same reach test
attribute_contacts uses); bounces are skipped.

    python scripts/classify_strokes_bst.py \\
        --video data/videos/dingles_serve_volley.mp4 \\
        --ball-cache outputs/dingles_serve_volley/replay_cache.pkl \\
        --people outputs/dingles_serve_volley/person_tracks.pkl \\
        --report outputs/dingles_serve_volley/session_report.json \\
        --players outputs/dingles_serve_volley/players.json \\
        --out outputs/dingles_serve_volley/strokes_bst.json
"""

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.bst_strokes import (  # noqa: E402
    BASELINE_ZONE_M, CLASSES, STROKES, HIT_AFTER, HIT_BEFORE, SERVE_AFTER, SERVE_BEFORE, PersonFrame, build_input, classify,
    fold_serve_tosses, load_model, stroke_name,
)
from src.analysis.court_calibration import CourtCalibration  # noqa: E402
from src.analysis.impact_pipeline import analyze_impacts  # noqa: E402
from src.analysis.person_tracks import boxes_by_frame  # noqa: E402
from src.detection._tracknet_arch import resolve_device  # noqa: E402
from src.tracking.candidate_tracker import track_ball_paths, track_candidates  # noqa: E402

REACH = 0.6  # ball-to-box distance in box heights, as attribute_contacts
NET_Y_M = 23.77 / 2
NET_ZONE_M = 6.4  # inside the service line: a hit here is likely a volley
POSE_CROP_HEIGHT = 384
POSE_PAD = 0.25


def nearest_in_reach(row, x, y):
    best = None
    for track_id, x1, y1, x2, y2, _c in row:
        if track_id is None:
            continue
        dx, dy = max(x1 - x, 0.0, x - x2), max(y1 - y, 0.0, y - y2)
        ratio = float(np.hypot(dx, dy)) / max(y2 - y1, 1.0)
        if best is None or ratio < best[0]:
            best = (ratio, int(track_id))
    return best[1] if best and best[0] <= REACH else None


def ball_path(tracks, frame, x, y, frames):
    """The ball through `frames`: from the impact point, step outwards
    frame by frame taking the candidate nearest the last position."""
    by_frame: dict[int, list] = {}
    for t in tracks:
        for f, d in t.detections.items():
            by_frame.setdefault(f, []).append((d.x, d.y))
    out = {}
    for direction in (range(frame, frames[-1] + 1), range(frame - 1, frames[0] - 1, -1)):
        last = (x, y)
        for f in direction:
            cands = by_frame.get(f, [])
            if cands:
                c = min(cands, key=lambda p: np.hypot(p[0] - last[0], p[1] - last[1]))
                if np.hypot(c[0] - last[0], c[1] - last[1]) < 250:
                    out[f] = c
                    last = c
    return [out.get(f) for f in frames]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--ball-cache", required=True)
    parser.add_argument("--people", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--players")
    parser.add_argument("--swings", help="scripts/detect_swings.py output: classify these swings instead of the ball-based hits")
    parser.add_argument("--weights", default=str(REPO_ROOT / "weights" / "bst" / "bst_AP_JnB_bone.pt"))
    parser.add_argument("--pose-model", default=str(REPO_ROOT / "weights" / "yolo" / "yolov8s-pose.pt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    cache = pickle.load(open(args.ball_cache, "rb"))
    people = pickle.load(open(args.people, "rb"))
    report = json.loads(Path(args.report).read_text())
    fps, n = cache["fps"], cache["num_frames"]
    court = CourtCalibration(homography=np.array(report["court"]["homography"]))
    skip = {f for a, b in report["clip"]["cutaways_s"] for f in range(int(round(a * fps)), int(round(b * fps)))}
    player_of = {}
    if args.players:
        for t, p in json.loads(Path(args.players).read_text())["player_of_tracklet"].items():
            player_of[int(t)] = p

    rows = people["people"]
    impacts = analyze_impacts(
        track_candidates(cache["candidates"]), fps,
        calibrations_by_frame={f: court for f in range(n)}, player_boxes_by_frame=boxes_by_frame(rows),
    )
    tracks = track_ball_paths(cache["candidates"])

    def feet(box):
        x1, y1, x2, y2 = box
        return court.pixel_to_world((x1 + x2) / 2.0, y2)

    def boxes_at(f):
        return {int(t): (x1, y1, x2, y2) for t, x1, y1, x2, y2, _c in (rows[f] if 0 <= f < len(rows) else []) if t is not None}

    # What gets classified: swings (scripts/detect_swings.py) when given,
    # else the ball-based impacts with a person in reach.
    events = []  # (frame, hitter tracklet, ball x, ball y, kind)
    if args.swings:
        by_frame: dict[int, list] = {}
        for t in tracks:
            for f, d in t.detections.items():
                by_frame.setdefault(f, []).append((d.x, d.y))
        for sw in json.loads(Path(args.swings).read_text())["swings"]:
            f, t = sw["frame"], sw["tracklet"]
            box = boxes_at(f).get(t)
            if box is None:
                continue
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            # The ball nearest the player around the swing anchors its path.
            near = [p for g in range(f - 8, f + 9) for p in by_frame.get(g, [])]
            x, y = min(near, key=lambda p: np.hypot(p[0] - cx, p[1] - cy)) if near else (cx, cy)
            events.append((f, t, x, y, "swing"))
    else:
        for imp in impacts.impacts:
            f = imp.frame_idx
            if f in skip or imp.kind == "bounce":
                continue
            hitter = nearest_in_reach(rows[f], imp.x, imp.y)
            if hitter is not None:
                events.append((f, hitter, imp.x, imp.y, imp.kind))

    hits = []
    for f, hitter, x0, y0, kind in events:
        hx, hy = feet(boxes_at(f)[hitter])
        hitter_far = hy < NET_Y_M
        # The other player: someone in the opposite half, on or near this
        # court, nearest where the ball is HIT_AFTER frames later (else
        # nearest the middle of that half).
        later = ball_path(tracks, f, x0, y0, list(range(f, min(n, f + HIT_AFTER + 1))))[-1]
        candidates = []
        for t, box in boxes_at(f).items():
            if t == hitter:
                continue
            x, y = feet(box)
            if (y < NET_Y_M) != hitter_far and -2 <= x <= 12.97 and -6 <= y <= 29.77:
                if later is not None:
                    key = np.hypot((box[0] + box[2]) / 2 - later[0], (box[1] + box[3]) / 2 - later[1])
                else:
                    key = abs(x - 5.485)
                candidates.append((key, t))
        other = min(candidates)[1] if candidates else None
        hits.append({"frame": f, "x": x0, "y": y0, "kind": kind, "hitter": hitter, "other": other,
                     "hitter_half": "far" if hitter_far else "near", "at_net": abs(hy - NET_Y_M) <= NET_ZONE_M,
                     "at_baseline": abs(min(hy, 2 * NET_Y_M - hy)) <= BASELINE_ZONE_M})
    print(f"{len(hits)} to classify ({', '.join(f'{k}: {v}' for k, v in sorted(Counter(h['kind'] for h in hits).items()))})")

    # Poses for the two people over each clip's frames (serve window is the widest).
    need: dict[int, set[int]] = {}
    for h in hits:
        for f in range(max(0, h["frame"] - SERVE_BEFORE), min(n, h["frame"] + SERVE_AFTER + 1)):
            need.setdefault(f, set()).update(t for t in (h["hitter"], h["other"]) if t is not None)
    device = resolve_device(args.device)
    pose_model = YOLO(args.pose_model)
    capture = cv2.VideoCapture(args.video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    poses: dict[tuple[int, int], np.ndarray] = {}
    for f in range(n):
        ok, frame = capture.read()
        if not ok:
            break
        if f not in need:
            continue
        here = boxes_at(f)
        crops, meta = [], []
        for t in need[f]:
            if t not in here:
                continue
            x1, y1, x2, y2 = here[t]
            w, hh = x2 - x1, y2 - y1
            cx1, cy1 = int(max(0, x1 - POSE_PAD * w)), int(max(0, y1 - POSE_PAD * hh))
            cx2, cy2 = int(min(width, x2 + POSE_PAD * w)), int(min(height, y2 + POSE_PAD * hh))
            if cx2 - cx1 < 16 or cy2 - cy1 < 16:
                continue
            crop = np.ascontiguousarray(frame[cy1:cy2, cx1:cx2])
            scale = POSE_CROP_HEIGHT / crop.shape[0]
            crops.append(cv2.resize(crop, (max(32, int(crop.shape[1] * scale)), POSE_CROP_HEIGHT)))
            meta.append((t, cx1, cy1, scale))
        if not crops:
            continue
        for (t, cx1, cy1, scale), r in zip(meta, pose_model.predict(crops, imgsz=POSE_CROP_HEIGHT, device=device, verbose=False)):
            if r.keypoints is None or len(r.boxes) == 0:
                continue
            centres = r.boxes.xywh[:, :2].cpu().numpy()
            best = int(np.argmin(np.linalg.norm(centres - np.array([r.orig_shape[1] / 2, r.orig_shape[0] / 2]), axis=1)))
            poses[(f, t)] = r.keypoints.xy[best].cpu().numpy() / scale + np.array([cx1, cy1])
    capture.release()

    def person_frames(t, frames):
        out = []
        for f in frames:
            box = boxes_at(f).get(t) if t is not None else None
            out.append(PersonFrame(
                keypoints=poses.get((f, t)) if box is not None else None,
                box=box,
                court_m=feet(box) if box is not None else None,
            ))
        return out

    clips = {"hit": [], "serve": []}
    for h in hits:
        for kind, before, after in (("hit", HIT_BEFORE, HIT_AFTER), ("serve", SERVE_BEFORE, SERVE_AFTER)):
            frames = list(range(max(0, h["frame"] - before), min(n, h["frame"] + after + 1)))
            hitter, other = person_frames(h["hitter"], frames), person_frames(h["other"], frames)
            far, near = (hitter, other) if h["hitter_half"] == "far" else (other, hitter)
            ball = ball_path(tracks, h["frame"], h["x"], h["y"], frames)
            clips[kind].append(build_input(far, near, ball, (width, height)))
    model = load_model(args.weights, "cpu")
    probs = {k: classify(model, v) for k, v in clips.items()}

    out_hits = []
    for i, h in enumerate(hits):
        p_hit, p_serve = probs["hit"][i], probs["serve"][i]
        label = CLASSES[int(np.argmax(p_hit))]
        model_half = "far" if label[1] == "F" else "near"
        serve_prob = float(p_serve[4] + p_serve[5])
        # No player (not a person - the ball cart) is never a stroke.
        stroke = stroke_name(label, float(p_hit.max()), serve_prob, model_half == h["hitter_half"],
                             at_baseline=h["at_baseline"], at_net=h["at_net"]) \
            if player_of.get(h["hitter"]) is not None or not player_of else "unsure"
        out_hits.append({
            "t_s": round(h["frame"] / fps, 2),
            "frame": h["frame"],
            "impact_kind": h["kind"],
            "tracklet": h["hitter"],
            "player": player_of.get(h["hitter"]),
            "hitter_half": h["hitter_half"],
            "other_tracklet": h["other"],
            "stroke": stroke,
            "at_net": bool(h["at_net"]),
            "at_baseline": bool(h["at_baseline"]),
            "label": label,
            "probabilities": {c: round(float(v), 3) for c, v in zip(CLASSES, p_hit)},
            "serve_window_label": CLASSES[int(np.argmax(p_serve))],
            "serve_window_serve_prob": round(serve_prob, 3),
            "half_agrees": model_half == h["hitter_half"],
        })
    before = len(out_hits)
    out_hits = fold_serve_tosses(out_hits, fps)
    if before != len(out_hits):
        print(f"{before - len(out_hits)} swings folded into the serve they were the ball toss of")
    agree = sum(h["half_agrees"] for h in out_hits)
    print(f"model's near/far agrees with tracking on {agree}/{len(out_hits)} hits")
    for h in out_hits:
        print(f"  {h['t_s']:6.2f}s {h['impact_kind']:8s} player {h['player']!s:4s} {h['hitter_half']:4s} "
              f"{h['stroke']:8s} <- {h['label']} "
              f"{max(h['probabilities'].values()):.2f}  serve-window {h['serve_window_label']} ({h['serve_window_serve_prob']:.2f})"
              f"{'' if h['half_agrees'] else '  HALF MISMATCH'}")
    Path(args.out).write_text(json.dumps({
        "model": "BST-AP trained on TenniSet (github.com/Va6lue/BST-Badminton-Stroke-type-Transformer)",
        "classes": {"HFL": "hit, far player, left", "HFR": "hit, far player, right", "HNL": "hit, near player, left",
                    "HNR": "hit, near player, right", "SF": "serve, far player", "SN": "serve, near player"},
        "half_agreement": [agree, len(out_hits)],
        "assumes": "right-handed players (forehand = near-right or far-left as the camera sees it)",
        "strokes": {k: sum(h["stroke"] == k for h in out_hits) for k in STROKES},
        "hits": out_hits,
    }, indent=1))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
