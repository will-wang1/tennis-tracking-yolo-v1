"""Pose (17 COCO keypoints) for EVERY tracked person in a clip, cached once.

Built for stroke classification, which needs the arms of far-court players
as much as near ones. Running a pose model on the whole frame does not
give that: measured on dingles_serve_volley, YOLOv8s-pose over the full
1080p frame finds only the 2 near players (boxes ~280px tall) and misses
all 5 at the far end (~76-111px tall). Cropping each tracked person's box
with 25% padding and enlarging it to CROP_HEIGHT before running the same
model found a pose for all 21 people in 3 test frames, with 65-100% of
keypoints above 0.5 confidence, and the skeletons land on the body on
visual inspection.

Input is a scripts/track_people.py cache, so poses inherit its person ids;
output is keyed the same way:

    {"fps", "num_frames", "model",
     "poses": [ {track_id: (keypoints_xy[17,2] in FRAME pixels, confidence[17])}, ...per frame ]}

    python scripts/track_poses.py \\
        --input data/videos/dingles_serve_volley.mp4 \\
        --people outputs/dingles_serve_volley/person_tracks.pkl \\
        --output outputs/dingles_serve_volley/poses.pkl
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.detection._tracknet_arch import resolve_device  # noqa: E402

CROP_HEIGHT = 384
PAD = 0.25


def crop_box(frame_shape, x1, y1, x2, y2, pad=PAD):
    height, width = frame_shape[:2]
    w, h = x2 - x1, y2 - y1
    return (
        int(max(0, x1 - pad * w)),
        int(max(0, y1 - pad * h)),
        int(min(width, x2 + pad * w)),
        int(min(height, y2 + pad * h)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--people", required=True, help="scripts/track_people.py cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(REPO_ROOT / "weights" / "yolo" / "yolov8s-pose.pt"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    with open(args.people, "rb") as handle:
        people = pickle.load(handle)
    model = YOLO(args.model)
    device = resolve_device(args.device)

    capture = cv2.VideoCapture(args.input)
    poses: list[dict] = []
    started = time.time()
    for index, row in enumerate(people["people"]):
        ok, frame = capture.read()
        if not ok:
            break
        crops, meta = [], []
        for track_id, x1, y1, x2, y2, _conf in row:
            if track_id is None:
                continue
            cx1, cy1, cx2, cy2 = crop_box(frame.shape, x1, y1, x2, y2)
            if cx2 - cx1 < 8 or cy2 - cy1 < 8:
                continue
            crop = frame[cy1:cy2, cx1:cx2]
            scale = CROP_HEIGHT / crop.shape[0]
            crops.append(cv2.resize(crop, (max(32, int(crop.shape[1] * scale)), CROP_HEIGHT), interpolation=cv2.INTER_CUBIC))
            meta.append((int(track_id), cx1, cy1, scale))
        frame_poses = {}
        if crops:
            for (track_id, cx1, cy1, scale), result in zip(
                meta, model.predict(crops, imgsz=CROP_HEIGHT, device=device, verbose=False)
            ):
                if result.keypoints is None or len(result.boxes) == 0:
                    continue
                # Several people can share a padded crop; keep the detection
                # nearest the crop centre, which is the one it was cut for.
                centres = result.boxes.xywh[:, :2].cpu().numpy()
                target = np.array([result.orig_shape[1] / 2.0, result.orig_shape[0] / 2.0])
                best = int(np.argmin(np.linalg.norm(centres - target, axis=1)))
                xy = result.keypoints.xy[best].cpu().numpy() / scale + np.array([cx1, cy1])
                conf = result.keypoints.conf[best].cpu().numpy()
                frame_poses[track_id] = (xy.astype(np.float32), conf.astype(np.float32))
        poses.append(frame_poses)
        if index % 250 == 0:
            print(f"  frame {index}/{people['num_frames']}  ({time.time() - started:.0f}s, {len(frame_poses)} poses)", flush=True)
    capture.release()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as handle:
        pickle.dump({"fps": people["fps"], "num_frames": len(poses), "model": args.model, "poses": poses}, handle)
    print(f"Wrote {out}: {len(poses)} frames")


if __name__ == "__main__":
    main()
