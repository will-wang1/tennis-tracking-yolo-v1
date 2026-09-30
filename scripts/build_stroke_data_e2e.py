"""Stroke training data from E2E-Spot's labelled broadcast points.

For every stroke E2E-Spot labelled (data/external/e2e_spot_tennis), find
the hitter - the near or far player, as labelled - and save their pose over
the window around the stroke (src/analysis/stroke_tcn.py: 1.2s before to
0.8s after, sampled at 25fps whatever the video's rate). One "not a swing"
window per player per point is saved too, from a moment with none of their
events nearby.

Which person is the hitter: people are detected on every frame of the
point, and our court detector maps their feet onto the court; the near
player is the person on the near half, inside the court area (ball kids and
the umpire stand outside it), largest in the picture, followed frame to
frame by nearest position. Without a court fit the point is skipped.

Runs on the GPU laptop, one output file per match, so it can be stopped and
restarted (a finished match is skipped):

    python scripts/build_stroke_data_e2e.py --videos <folder of the 9 .mp4s> --out data/strokes/e2e
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.court_calibration import CourtCalibration  # noqa: E402
from src.analysis.stroke_tcn import CLASSES, E2E_TO_CLASS, FPS, SEQ_LEN, window_times  # noqa: E402
from src.detection._tracknet_arch import resolve_device  # noqa: E402

LABELS = REPO_ROOT / "data" / "external" / "e2e_spot_tennis"
NET_Y, LENGTH, WIDTH = 23.77 / 2, 23.77, 10.97
PAD_S = 1.5  # read this much either side of the point, for windows at its edges
CROP_HEIGHT, PAD = 384, 0.25
BATCH = 16


def pick_player(people_px, feet_m, far: bool, last=None):
    """Index of the player among this frame's people, or None."""
    best = None
    for i, ((x1, y1, x2, y2), (fx, fy)) in enumerate(zip(people_px, feet_m)):
        if not -3.0 <= fx <= WIDTH + 3.0:
            continue
        if far and not (-6.0 <= fy <= NET_Y - 0.5):
            continue
        if not far and not (NET_Y + 0.5 <= fy <= LENGTH + 6.0):
            continue
        h = y2 - y1
        key = h if last is None else -np.hypot(fx - last[0], fy - last[1]) * 10 + h * 0.01
        if best is None or key > best[0]:
            best = (key, i)
    return None if best is None else best[1]


def pose_of(pose_model, frame, box, device):
    import cv2

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1, cy1 = int(max(0, x1 - PAD * w)), int(max(0, y1 - PAD * h))
    cx2, cy2 = int(min(width, x2 + PAD * w)), int(min(height, y2 + PAD * h))
    if cx2 - cx1 < 16 or cy2 - cy1 < 16:
        return None
    crop = np.ascontiguousarray(frame[cy1:cy2, cx1:cx2])
    scale = CROP_HEIGHT / crop.shape[0]
    crop = cv2.resize(crop, (max(32, int(crop.shape[1] * scale)), CROP_HEIGHT))
    r = pose_model.predict(crop, imgsz=CROP_HEIGHT, device=device, verbose=False)[0]
    if r.keypoints is None or len(r.boxes) == 0:
        return None
    centres = r.boxes.xywh[:, :2].cpu().numpy()
    k = int(np.argmin(np.linalg.norm(centres - np.array([crop.shape[1] / 2, crop.shape[0] / 2]), axis=1)))
    return r.keypoints.xy[k].cpu().numpy() / scale + np.array([cx1, cy1]), r.keypoints.conf[k].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", required=True, help="Folder with <match name>.mp4 for the matches in videos.csv")
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "strokes" / "e2e"))
    parser.add_argument("--matches", nargs="*", help="Only these matches (default: all 9)")
    parser.add_argument("--max-points", type=int, default=None, help="Per match, for a quick test")
    parser.add_argument("--device", default=None)
    parser.add_argument("--preview", type=int, default=12,
                        help="Save a strip of this many labelled strokes per match (the hitter at the stroke, with "
                             "its label) to check the video lines up with the labels; 0 for none")
    parser.add_argument("--labels", default=str(LABELS), help="Folder with stroke_clips.json and videos.csv")
    parser.add_argument("--person-model", default=str(REPO_ROOT / "weights" / "yolo" / "yolov8s.pt"))
    parser.add_argument("--pose-model", default=str(REPO_ROOT / "weights" / "yolo" / "yolov8s-pose.pt"))
    parser.add_argument("--court-weights", default=str(REPO_ROOT / "weights" / "court_net_pretrained.pt"))
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    from src.detection.tennis_court_net import TennisCourtNetDetector

    device = resolve_device(args.device)
    print(f"device: {device}")
    person_model, pose_model = YOLO(args.person_model), YOLO(args.pose_model)
    court_detector = TennisCourtNetDetector(args.court_weights, device=device)
    labels = Path(args.labels)
    clips = json.loads((labels / "stroke_clips.json").read_text())
    by_match = defaultdict(list)
    for c in clips:
        by_match[c["video"].rsplit("_", 2)[0]].append(c)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    matches = [r["name"] for r in csv.DictReader(open(labels / "videos.csv"))]
    if args.matches:
        matches = [m for m in matches if m in args.matches]

    for match in matches:
        out_path = out_dir / f"{match}.npz"
        if out_path.exists():
            print(f"[skip] {match}: {out_path} exists")
            continue
        video = next((p for p in Path(args.videos).glob(f"{match}.*")), None)
        if video is None:
            print(f"[missing] {match}: no video in {args.videos}")
            continue
        capture = cv2.VideoCapture(str(video))
        vfps = capture.get(cv2.CAP_PROP_FPS)
        n_video = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        samples = {"kp": [], "conf": [], "box": [], "far": [], "label": [], "meta": []}
        previews = []
        started, skipped = time.time(), defaultdict(int)
        points = by_match[match][: args.max_points]
        for p_i, clip in enumerate(points):
            _, start, end = clip["video"].rsplit("_", 2)
            lfps = clip["fps"]
            t0, t1 = int(start) / lfps - PAD_S, int(end) / lfps + PAD_S
            f0, f1 = max(0, int(t0 * vfps)), min(n_video, int(t1 * vfps) + 1)
            capture.set(cv2.CAP_PROP_POS_FRAMES, f0)
            frames = {}
            for f in range(f0, f1):
                ok, img = capture.read()
                if not ok:
                    break
                frames[f] = img
            if not frames:
                skipped["unreadable"] += 1
                continue
            # The court, from a few frames of the point (the camera holds still during a point).
            court = None
            for f in list(frames)[len(frames) // 4:: max(1, len(frames) // 4)][:3]:
                pts = court_detector.detect(frames[f]) or {}
                if len(pts) >= 4:
                    try:
                        court = CourtCalibration.from_keypoints(pts)
                        break
                    except ValueError:
                        pass
            if court is None:
                skipped["no court"] += 1
                continue
            # People on every frame of the point, and each end's player followed through it.
            order = sorted(frames)
            player_box = {False: {}, True: {}}
            def detections():
                for i in range(0, len(order), BATCH):  # in chunks: a whole point at once needs >10GB
                    yield from person_model.predict([frames[f] for f in order[i:i + BATCH]], imgsz=1280,
                                                    classes=[0], conf=0.2, device=device, verbose=False)
            last = {False: None, True: None}
            for f, r in zip(order, detections()):
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
                feet = [court.pixel_to_world((b[0] + b[2]) / 2, b[3]) for b in boxes]
                for far in (False, True):
                    i = pick_player(boxes, feet, far, last[far])
                    if i is not None:
                        player_box[far][f] = boxes[i]
                        last[far] = feet[i]
            # Windows: every labelled stroke, and one quiet moment per player.
            wanted = []
            events = [(e["frame"], e["label"].startswith("far"), e.get("comment")) for e in clip["events"]]
            for fr, far, comment in events:
                if comment in E2E_TO_CLASS:
                    wanted.append(((int(start) + fr) / lfps, far, E2E_TO_CLASS[comment]))
            for far in (False, True):
                busy = [(int(start) + fr) / lfps for fr, f_, c in events if f_ == far]
                for t in np.linspace(int(start) / lfps + 1.0, int(end) / lfps - 1.0, 7):
                    if all(abs(t - b) > 1.0 for b in busy):
                        wanted.append((float(t), far, "not_a_swing"))
                        break
            for t, far, cls in wanted:
                kp = np.full((SEQ_LEN, 17, 2), np.nan, np.float32)
                conf = np.zeros((SEQ_LEN, 17), np.float32)
                box = np.full((SEQ_LEN, 4), np.nan, np.float32)
                for k, ts in enumerate(window_times(t)):
                    f = int(round(ts * vfps))
                    b = player_box[far].get(f)
                    if b is None or f not in frames:
                        continue
                    box[k] = b
                    pose = pose_of(pose_model, frames[f], b, device)
                    if pose is not None:
                        kp[k], conf[k] = pose
                if np.isfinite(box[:, 0]).mean() < 0.6:
                    skipped["player not found"] += 1
                    continue
                if cls != "not_a_swing" and len(previews) < args.preview:
                    mid = int(round(t * vfps))
                    b = player_box[far].get(mid)
                    if b is not None and mid in frames:
                        x1, y1, x2, y2 = (int(v) for v in b)
                        h, w = y2 - y1, x2 - x1
                        tile = frames[mid][max(0, y1 - h // 2):y2 + h // 5, max(0, x1 - w):x2 + w]
                        if tile.size:
                            tile = cv2.resize(tile, (200, 260))
                            cv2.rectangle(tile, (0, 0), (200, 24), (0, 0, 0), -1)
                            cv2.putText(tile, f"{cls} {'far' if far else 'near'}", (4, 17),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            previews.append(tile)
                samples["kp"].append(kp); samples["conf"].append(conf); samples["box"].append(box)
                samples["far"].append(far); samples["label"].append(CLASSES.index(cls))
                samples["meta"].append(f"{clip['video']}@{t:.2f}")
            if p_i % 20 == 0:
                print(f"  {match}: point {p_i + 1}/{len(points)}, {len(samples['label'])} windows, "
                      f"{time.time() - started:.0f}s, skipped {dict(skipped)}", flush=True)
        capture.release()
        if previews:
            while len(previews) % 6:
                previews.append(np.zeros_like(previews[0]))
            grid = np.vstack([np.hstack(previews[i:i + 6]) for i in range(0, len(previews), 6)])
            cv2.imwrite(str(out_dir / f"{match}_preview.jpg"), grid)
            print(f"  check {out_dir / (match + '_preview.jpg')}: each picture should show that stroke")
        if not samples["label"]:
            print(f"[empty] {match}")
            continue
        np.savez_compressed(out_path, kp=np.stack(samples["kp"]), conf=np.stack(samples["conf"]),
                            box=np.stack(samples["box"]), far=np.array(samples["far"]),
                            label=np.array(samples["label"]), meta=np.array(samples["meta"]),
                            source=np.array("e2e"), fps=np.array(FPS))
        counts = np.bincount(samples["label"], minlength=len(CLASSES))
        print(f"[done] {match}: {len(samples['label'])} windows -> {out_path}  "
              + ", ".join(f"{c} {n}" for c, n in zip(CLASSES, counts) if n) + f"  skipped {dict(skipped)}")


if __name__ == "__main__":
    main()
