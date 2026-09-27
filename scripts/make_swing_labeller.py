"""A local page for labelling every detected swing of a session.

Each swing becomes a short looping clip cropped around the player (a
sprite sheet of frames, so it plays in any browser from a plain file - no
video codec to depend on), with keys for the stroke:

    F forehand   B backhand   S serve   O overhead   N not a swing
    V volley (toggle)   L slice (toggle)

so a backhand slice is B+L and a forehand volley F+V. The model's guess
(scripts/classify_strokes_bst.py --swings) is filled in when given, so most
swings are one keypress to confirm. Labels are kept in the browser as you
go and exported with "Save labels" as JSON - the training data for a stroke
model fine-tuned on this camera.

The page and its clips stay on this machine: they show the people in the
session.

    python scripts/make_swing_labeller.py \\
        --video data/videos/dingles_serve_volley.mp4 \\
        --swings outputs/dingles_serve_volley/swings.json \\
        --people outputs/dingles_serve_volley/person_tracks.pkl \\
        --strokes outputs/dingles_serve_volley/strokes_swings.json \\
        --name dingles_serve_volley \\
        --out outputs/dingles_serve_volley/labeller
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BEFORE, AFTER, STEP = 30, 20, 2  # frames around the swing; every STEP-th frame
TILE_H = 240
TILE_W = 180
PAD_X, PAD_TOP, PAD_BOTTOM = 0.9, 0.5, 0.15  # of the player's box, around the union of their boxes
FOREHAND_LABELS = {"HNR", "HFL"}  # BST's camera-side classes that are a right-hander's forehand


def guess(hit: dict | None) -> dict:
    """The model's call as a pre-filled label: stroke plus volley."""
    if not hit:
        return {}
    if hit["stroke"] == "serve":
        return {"stroke": "serve"}
    if hit["stroke"] in ("forehand", "backhand"):
        return {"stroke": hit["stroke"]}
    if hit["stroke"] == "net":
        side = "forehand" if hit["label"] in FOREHAND_LABELS else "backhand"
        return {"stroke": side, "volley": True}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--swings", required=True)
    parser.add_argument("--people", required=True)
    parser.add_argument("--strokes", help="classify_strokes_bst.py --swings output, to pre-fill the model's guess")
    parser.add_argument("--name", required=True, help="Session name, stored with every label")
    parser.add_argument("--out", required=True, help="Folder for the page and its clips")
    args = parser.parse_args()

    import cv2

    swings = json.loads(Path(args.swings).read_text())["swings"]
    rows = pickle.load(open(args.people, "rb"))["people"]
    by_frame = {}
    if args.strokes:
        by_frame = {h["frame"]: h for h in json.loads(Path(args.strokes).read_text())["hits"]}
    out = Path(args.out)
    (out / "clips").mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    wanted = {}
    for i, sw in enumerate(swings):
        for f in range(sw["frame"] - BEFORE, sw["frame"] + AFTER + 1, STEP):
            wanted.setdefault(f, []).append(i)
    frames: dict[int, np.ndarray] = {}
    last = max(wanted) if wanted else -1
    f = 0
    while f <= last:
        ok, img = capture.read()
        if not ok:
            break
        if f in wanted:
            frames[f] = img
        f += 1
    capture.release()

    items = []
    for i, sw in enumerate(swings):
        t = sw["tracklet"]
        boxes = [b for g in range(sw["frame"] - BEFORE, sw["frame"] + AFTER + 1) if 0 <= g < len(rows)
                 for b in rows[g] if b[0] == t]
        if not boxes:
            continue
        x1 = min(b[1] for b in boxes); y1 = min(b[2] for b in boxes)
        x2 = max(b[3] for b in boxes); y2 = max(b[4] for b in boxes)
        h = float(np.median([b[4] - b[2] for b in boxes])); w = float(np.median([b[3] - b[1] for b in boxes]))
        cx1, cx2 = x1 - PAD_X * w, x2 + PAD_X * w
        cy1, cy2 = y1 - PAD_TOP * h, y2 + PAD_BOTTOM * h
        # Keep the tile's shape so nobody is stretched.
        need_w = (cy2 - cy1) * TILE_W / TILE_H
        if cx2 - cx1 < need_w:
            c = (cx1 + cx2) / 2; cx1, cx2 = c - need_w / 2, c + need_w / 2
        else:
            need_h = (cx2 - cx1) * TILE_H / TILE_W
            c = (cy1 + cy2) / 2; cy1, cy2 = c - need_h / 2, c + need_h / 2
        # Whole-pixel crop, fixed once, so every slice below has the same size.
        ix1, iy1 = int(round(cx1)), int(round(cy1))
        cw, ch = max(1, int(round(cx2 - cx1))), max(1, int(round(cy2 - cy1)))
        sx1, sy1 = max(0, ix1), max(0, iy1)
        sx2, sy2 = min(width, ix1 + cw), min(height, iy1 + ch)
        tiles = []
        for g in range(sw["frame"] - BEFORE, sw["frame"] + AFTER + 1, STEP):
            img = frames.get(g)
            if img is None:
                tiles.append(np.zeros((TILE_H, TILE_W, 3), np.uint8))
                continue
            # Crop with black padding where the box runs off the frame.
            canvas = np.zeros((ch, cw, 3), np.uint8)
            if sx2 > sx1 and sy2 > sy1:
                canvas[sy1 - iy1:sy2 - iy1, sx1 - ix1:sx2 - ix1] = img[sy1:sy2, sx1:sx2]
            tiles.append(cv2.resize(canvas, (TILE_W, TILE_H), interpolation=cv2.INTER_CUBIC))
        sprite = f"clips/{i:04d}.jpg"
        cv2.imwrite(str(out / sprite), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])
        items.append({
            "id": f"{args.name}:{sw['frame']}:{t}", "session": args.name, "frame": sw["frame"], "t_s": sw["t_s"],
            "tracklet": t, "player": sw.get("player"), "sprite": sprite, "n": len(tiles),
            "peak": BEFORE // STEP, "guess": guess(by_frame.get(sw["frame"])),
        })

    template = (REPO_ROOT / "src" / "labelling" / "swing_labeller.html").read_text()
    page = template.replace("__TITLE__", f"Label swings: {args.name}").replace(
        "__ITEMS__", json.dumps(items)).replace("__SESSION__", json.dumps(args.name)).replace(
        "__FPS__", str(round(fps / STEP, 2)))
    (out / "index.html").write_text(page)
    print(f"{len(items)} swings -> {out / 'index.html'}")


if __name__ == "__main__":
    main()
