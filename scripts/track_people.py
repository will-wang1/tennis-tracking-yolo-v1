"""Detect and track every person in a clip once, and cache the result.

Same shape of job as `replay_impacts.py --build` for the ball: the only
expensive part is the neural network, so run it once, keep what it said,
and do all the analysis - distance, coverage, who hit what - against the
cache instead of the video.

MODEL: YOLOv8s person detection with ultralytics' built-in ByteTrack.
Chosen by measurement on dingles_serve_volley, three sample frames:

    yolov8s        7 people/frame   62 ms/frame   smallest box ~76px tall
    yolov8m        7 people/frame  118 ms/frame   (same people, half the speed)
    yolov8s-pose   2 people/frame   65 ms/frame   near players only

The pose model finds only the two near-court players and misses everyone
at the far end - worth knowing well beyond this script, because it means
pose-based stroke classification will not see the far side of the court
on a fence-height camera without a better pose model or a crop-and-zoom
pass.

TRACKER: ByteTrack, with `track_buffer` raised from ultralytics' default
of 30 frames to 75 (3s at 25fps) - see configs/bytetrack_coaching.yaml.
In a coaching session players walk behind each other and behind the
coach constantly; at the default a person hidden for longer than 1.2s
comes back as a NEW id, and every id switch splits one player's distance
and touches across two identities.

HONEST LIMIT: ByteTrack matches on box position and motion only. Two
players in identical kit who cross paths can swap ids, and nothing here
can detect that it happened. An appearance model (re-identification,
e.g. BoT-SORT with ReID) is the upgrade if the id-stability numbers
`summarise_person_tracks` prints come back bad.

    python scripts/track_people.py \\
        --input data/videos/dingles_serve_volley.mp4 \\
        --output outputs/dingles_serve_volley/person_tracks.pkl
"""

import argparse
import pickle
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.detection._tracknet_arch import resolve_device  # noqa: E402

_COCO_PERSON = 0


def track(args) -> dict:
    # Imported here so --help works without ultralytics installed, matching
    # how the rest of the repo keeps detector imports out of module scope.
    import cv2
    from ultralytics import YOLO

    capture = cv2.VideoCapture(args.input)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    device = resolve_device(args.device)
    model = YOLO(args.model)
    people: list[list[tuple]] = []
    started = time.time()
    # stream=True yields one result per frame without holding the video in
    # memory - the whole-video-in-RAM pattern is what killed jobs silently
    # on the old laptop.
    for i, result in enumerate(
        model.track(
            source=args.input,
            stream=True,
            persist=True,
            classes=[_COCO_PERSON],
            imgsz=args.imgsz,
            conf=args.conf,
            tracker=args.tracker,
            device=device,
            verbose=False,
        )
    ):
        boxes = result.boxes
        ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
        frame_people = []
        for track_id, (x1, y1, x2, y2), confidence in zip(
            ids, boxes.xyxy.tolist(), boxes.conf.tolist()
        ):
            frame_people.append((track_id, float(x1), float(y1), float(x2), float(y2), float(confidence)))
        people.append(frame_people)
        if i % 250 == 0:
            elapsed = time.time() - started
            print(f"  frame {i}/{frame_count}  ({elapsed:.0f}s, {len(frame_people)} people)", flush=True)

    return {
        "video": str(Path(args.input).resolve()),
        "fps": fps,
        "num_frames": len(people),
        "model": args.model,
        "tracker": args.tracker,
        "imgsz": args.imgsz,
        "conf": args.conf,
        # per frame: [(track_id | None, x1, y1, x2, y2, confidence), ...]
        "people": people,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=str(REPO_ROOT / "weights" / "yolo" / "yolov8s.pt"))
    parser.add_argument("--tracker", default=str(REPO_ROOT / "configs" / "bytetrack_coaching.yaml"))
    parser.add_argument("--imgsz", type=int, default=1280, help="Far-end people are ~80px tall; 640 loses them")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cache = track(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as handle:
        pickle.dump(cache, handle)
    print(f"Wrote {out}: {cache['num_frames']} frames")


if __name__ == "__main__":
    main()
