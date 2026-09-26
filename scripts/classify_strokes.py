"""Find and name every person's strokes in a clip (src/analysis/strokes.py).

    python scripts/classify_strokes.py \\
        --video data/videos/dingles_serve_volley.mp4 \\
        --poses outputs/dingles_serve_volley/poses.pkl \\
        --people outputs/dingles_serve_volley/person_tracks.pkl \\
        --ball-cache outputs/dingles_serve_volley/replay_cache.pkl \\
        --report outputs/dingles_serve_volley/session_report.json \\
        --out outputs/dingles_serve_volley/strokes.json \\
        --sheet outputs/dingles_serve_volley/stroke_sheet

The session report supplies the court (fitted from court-view shots only)
and the cutaways, where no stroke is looked for. --sheet writes one image
strip per stroke - the person, cropped, from backswing to follow-through,
with the call written on it - for checking calls by eye or labelling.

Racket hand is decided by detecting the racket ("tennis racket", COCO
class 38) in each person's crop at their swing peaks and taking the wrist
nearest it, by majority over that person's swings.
"""

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.analysis.court_calibration import CourtCalibration, static_calibration_from_frames  # noqa: E402
from src.analysis.strokes import (  # noqa: E402
    L_WRIST,
    R_WRIST,
    Stroke,
    ball_within_reach,
    body_frames,
    classify_swing,
    resting_lateral,
    swing_peaks,
)
from src.detection._tracknet_arch import resolve_device  # noqa: E402
from src.tracking.candidate_tracker import track_candidates  # noqa: E402

COCO_RACKET = 38
CROP_PAD = 0.35
# Racket-hand voting. The first version voted only at swing peaks and came
# out ~50/50 left/right on both halves of the court (16 of 33 people "left
# handed", against ~10% in the population): at a swing's peak the racket is
# flung away from both hands, two-handed shots put both wrists on it, and
# pose left/right labels flip most during fast motion. So votes are taken
# across a person's whole time on camera, a vote only counts when the
# racket sits close to a wrist, and LEFT is called only on strong evidence -
# right-handedness is the ~90% prior.
RACKET_SAMPLE_EVERY = 5
RACKET_NEAR_WRIST_TORSO = 1.0
LEFT_MIN_VOTES = 5
LEFT_MIN_SHARE = 0.7


def _crop(frame, box, pad=CROP_PAD):
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1, cy1 = int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h))
    cx2, cy2 = int(min(w_img, x2 + pad * w)), int(min(h_img, y2 + pad * h))
    return frame[cy1:cy2, cx1:cx2], (cx1, cy1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--poses", required=True)
    parser.add_argument("--people", required=True)
    parser.add_argument("--ball-cache", required=True)
    parser.add_argument("--report", help="session_report.json - court and cutaways")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sheet", help="Directory for one review image strip per stroke")
    parser.add_argument("--detector", default=str(REPO_ROOT / "weights" / "yolo" / "yolov8s.pt"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    poses = pickle.load(open(args.poses, "rb"))
    people = pickle.load(open(args.people, "rb"))
    ball_cache = pickle.load(open(args.ball_cache, "rb"))
    fps = poses["fps"]
    num_frames = poses["num_frames"]

    cutaway = set()
    if args.report:
        report = json.loads(Path(args.report).read_text())
        court = CourtCalibration(homography=np.array(report["court"]["homography"], dtype=float))
        for a, b in report["clip"]["cutaways_s"]:
            cutaway.update(range(int(round(a * fps)), int(round(b * fps))))
    else:
        court = static_calibration_from_frames({
            f: (v if isinstance(v, CourtCalibration) else CourtCalibration(homography=v))
            for f, v in ball_cache["calibrations"].items()
        })
    ball = track_candidates(ball_cache["candidates"])

    boxes: dict[int, dict[int, tuple]] = defaultdict(dict)
    for f, row in enumerate(people["people"]):
        for track_id, x1, y1, x2, y2, _c in row:
            if track_id is not None:
                boxes[int(track_id)][f] = (x1, y1, x2, y2)

    # ---- swings per person
    track_ids = sorted({t for frame in poses["poses"] for t in frame})
    frames_by_track = {t: body_frames(poses["poses"], t) for t in track_ids}
    peaks_by_track = {}
    for t in track_ids:
        peaks = [(f, v) for f, v in swing_peaks(frames_by_track[t], fps) if f not in cutaway and f in boxes[t]]
        if peaks:
            peaks_by_track[t] = peaks
    total = sum(len(p) for p in peaks_by_track.values())
    print(f"{total} swings across {len(peaks_by_track)} people")

    # ---- racket detection at each swing peak, and strip frames for the sheet
    wanted: dict[int, list[tuple[int, int]]] = defaultdict(list)  # frame -> [(track, role)]
    strip_offsets = [int(round(o * fps)) for o in (-0.4, -0.24, -0.12, 0.0, 0.12, 0.24)]
    for t in peaks_by_track:
        # racket-hand votes across this person's whole time on camera
        for f in sorted(boxes[t])[::RACKET_SAMPLE_EVERY]:
            if f not in cutaway:
                wanted[f].append((t, -1))
    for t, peaks in peaks_by_track.items():
        for f, _ in peaks:
            if args.sheet:
                for o in strip_offsets:
                    if 0 <= f + o < num_frames:
                        wanted[f + o].append((t, f))
    detector = YOLO(args.detector)
    device = resolve_device(args.device)
    racket_votes: dict[int, Counter] = defaultdict(Counter)
    strip_crops: dict[tuple[int, int], dict[int, np.ndarray]] = defaultdict(dict)
    capture = cv2.VideoCapture(args.video)
    for f in range(num_frames):
        ok, frame = capture.read()
        if not ok:
            break
        if f not in wanted:
            continue
        for t, peak in wanted[f]:
            box = boxes[t].get(f)
            if box is None:
                continue
            crop, (ox, oy) = _crop(frame, box)
            # Skip slivers, and hand OpenCV a contiguous copy: resizing a
            # few-pixel-wide, non-contiguous slice of the frame (a person
            # at the frame edge) segfaulted inside OpenCV 5's ARM resize
            # path (kleidicv) once racket votes sampled whole tracks.
            if crop.shape[0] < 16 or crop.shape[1] < 16:
                continue
            crop = np.ascontiguousarray(crop)
            if args.sheet and peak >= 0:
                strip_crops[(t, peak)][f] = crop
            if peak != -1:
                continue
            scale = 480.0 / max(crop.shape[0], 1)
            big = cv2.resize(crop, (max(32, int(crop.shape[1] * scale)), 480))
            result = detector.predict(big, imgsz=480, classes=[COCO_RACKET], conf=0.2, device=device, verbose=False)[0]
            if len(result.boxes) == 0 or t not in poses["poses"][f]:
                continue
            rx, ry = (result.boxes.xywh[0, :2].cpu().numpy() / scale) + np.array([ox, oy])
            xy, conf = poses["poses"][f][t]
            torso = float(np.linalg.norm((xy[5] + xy[6]) / 2 - (xy[11] + xy[12]) / 2))
            dl = float(np.hypot(*(xy[L_WRIST] - (rx, ry)))) if conf[L_WRIST] > 0.5 else 9e9
            dr = float(np.hypot(*(xy[R_WRIST] - (rx, ry)))) if conf[R_WRIST] > 0.5 else 9e9
            # Count only a racket clearly in one hand: near a wrist, and
            # nearer that wrist than the other by a margin.
            near, far_ = min(dl, dr), max(dl, dr)
            if torso > 5 and near < RACKET_NEAR_WRIST_TORSO * torso and far_ > 1.5 * near:
                racket_votes[t]["left" if dl < dr else "right"] += 1
    capture.release()

    # ---- classify
    strokes: list[Stroke] = []
    hand_by_track = {}
    rest_by_track = {}
    for t, peaks in peaks_by_track.items():
        votes = racket_votes.get(t) or Counter()
        total_votes = sum(votes.values())
        if total_votes >= LEFT_MIN_VOTES and votes["left"] / total_votes >= LEFT_MIN_SHARE:
            hand, source = "left", "racket"
        elif total_votes >= LEFT_MIN_VOTES and votes["right"] / total_votes >= LEFT_MIN_SHARE:
            hand, source = "right", "racket"
        else:
            hand, source = "right", "assumed"  # too little evidence: the population prior
        hand_by_track[t] = (hand, source, dict(votes or {}))
        rest_by_track[t] = resting_lateral(frames_by_track[t], hand)
        for f, speed in peaks:
            x1, y1, x2, y2 = boxes[t][f]
            wx, wy = court.pixel_to_world((x1 + x2) / 2.0, y2)
            position = (float(wx), float(wy))
            stroke, side, evidence = classify_swing(
                frames_by_track[t], f, fps, hand, position, rest_lateral=rest_by_track[t]
            )
            strokes.append(Stroke(
                track_id=t, frame=f, t_s=f / fps, stroke=stroke, side=side,
                hit_confirmed=ball_within_reach(ball, boxes[t], f, fps),
                peak_speed=speed, court_position_m=position, racket_hand=hand,
                racket_hand_source=source, evidence=evidence,
            ))
    strokes.sort(key=lambda s: s.frame)

    confirmed = [s for s in strokes if s.hit_confirmed]
    print(f"\nall swings:        {dict(Counter(s.stroke for s in strokes))}")
    print(f"confirmed hits:    {dict(Counter(s.stroke for s in confirmed))}")
    hands = Counter(src for _h, src, _v in hand_by_track.values())
    print(f"racket hand from:  {dict(hands)}")

    per_person = defaultdict(Counter)
    for s in confirmed:
        per_person[s.track_id][s.stroke] += 1
    out = {
        "fps": fps,
        "method": "rules over pose, racket detection and court position - see src/analysis/strokes.py",
        "strokes": [s.to_dict() for s in strokes],
        "per_person_confirmed": {str(t): dict(c) for t, c in per_person.items()},
        "racket_hand": {str(t): {"hand": h, "source": src, "votes": v} for t, (h, src, v) in hand_by_track.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"Wrote {args.out}")

    if args.sheet:
        sheet = Path(args.sheet)
        sheet.mkdir(parents=True, exist_ok=True)
        for s in strokes:
            crops = strip_crops.get((s.track_id, s.frame), {})
            panels = []
            for o in strip_offsets:
                c = crops.get(s.frame + o)
                if c is None:
                    continue
                p = cv2.resize(c, (int(c.shape[1] * 220 / c.shape[0]), 220))
                if o == 0:
                    p = cv2.copyMakeBorder(p, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=(0, 255, 255))
                    p = cv2.resize(p, (p.shape[1], 220))
                panels.append(p)
            if not panels:
                continue
            strip = cv2.hconcat(panels)
            label = f"{s.t_s:6.2f}s  P{s.track_id}  {s.stroke.upper()}  {'HIT' if s.hit_confirmed else 'no ball'}  ({s.racket_hand}-handed)"
            bar = np.full((26, strip.shape[1], 3), 30, np.uint8)
            cv2.putText(bar, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imwrite(str(sheet / f"{s.frame:06d}_P{s.track_id}.jpg"), np.vstack([bar, strip]))
        print(f"Review strips -> {sheet}")


if __name__ == "__main__":
    main()
