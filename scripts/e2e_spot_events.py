"""Tennis events from E2E-Spot (Hong et al., ECCV 2022) - EXPERIMENTAL.

E2E-Spot looks at the whole video frame and marks, to the frame, six kinds
of event: a bounce, a swing and a serve, each at the near or the far end.
It needs no ball tracking, which is why it is worth trying where our own
bounce and contact detection is weakest. It was trained on broadcast
matches (Wimbledon at 25fps, US Open at 30fps), so how it transfers to a
fence camera is the question this script exists to answer.

The model is vendored in src/third_party/e2e_spot (BSD-3) with its tennis
weights in weights/e2e_spot:

    python scripts/e2e_spot_events.py --video data/videos/dingles_serve_volley.mp4 \\
        --out outputs/dingles_serve_volley/e2e_spot.json [--full-width]

MEASURED on dingles_serve_volley against the hand-labelled swings: near
end 13 of 21 swings found and 13 of 14 calls real; far end 2 of 34 -
nearly blind there, and full-width or larger frames did not help. Its
bounces looked better than ours on video.

Frames are shrunk to 224px high, as the model was trained. By default each
is centre-cropped to 224x224 - also as trained - which on a 16:9 frame
keeps only the middle 56% of the width; --full-width feeds the whole frame
instead (the network is convolutional, so it accepts it).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CLASSES = ["far_court_bounce", "far_court_swing", "far_court_serve",
           "near_court_bounce", "near_court_swing", "near_court_serve"]  # data/tennis/class.txt
CLIP_LEN = 100
HEIGHT = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
PEAK_WINDOW = 6  # an event is the highest score within this many frames either side
MIN_SCORE = 0.25


def peaks(scores: np.ndarray, min_score: float = MIN_SCORE, window: int = PEAK_WINDOW) -> list[int]:
    out = []
    for f in range(len(scores)):
        if scores[f] < min_score:
            continue
        lo, hi = max(0, f - window), min(len(scores), f + window + 1)
        if scores[f] == scores[lo:hi].max() and not any(abs(f - g) <= window for g in out):
            out.append(f)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", default=str(REPO_ROOT / "weights" / "e2e_spot" / "tennis_rny008gsm_gru_rgb.pt"))
    parser.add_argument("--full-width", action="store_true")
    parser.add_argument("--height", type=int, default=HEIGHT,
                        help="Frame height fed to the model (trained at 224; larger makes far players bigger)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import cv2
    import torch

    from src.detection._tracknet_arch import half_precision
    from src.third_party.e2e_spot.model import load

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = load(args.weights, device)

    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, img = capture.read()
        if not ok:
            break
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(round(w * args.height / h)), args.height), interpolation=cv2.INTER_AREA)
        if not args.full_width:
            x0 = (img.shape[1] - args.height) // 2
            img = img[:, x0:x0 + args.height]
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    capture.release()
    n = len(frames)
    print(f"{n} frames at {fps:g}fps, {frames[0].shape[1]}x{frames[0].shape[0]}")

    scores = np.zeros((n, len(CLASSES) + 1), np.float32)
    counts = np.zeros(n, np.float32)
    stride = CLIP_LEN // 2
    starts = list(range(0, max(1, n - CLIP_LEN + 1), stride))
    if starts[-1] + CLIP_LEN < n:
        starts.append(n - CLIP_LEN)
    for s in starts:
        clip = np.stack(frames[s:s + CLIP_LEN]).astype(np.float32) / 255.0
        clip = (clip - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(clip).permute(0, 3, 1, 2).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=half_precision(device)):
            pred = model(x)
            prob = torch.softmax(pred, dim=2)[0].float().cpu().numpy()
        scores[s:s + len(prob)] += prob
        counts[s:s + len(prob)] += 1
    scores /= np.maximum(counts, 1)[:, None]

    events = []
    for c, name in enumerate(CLASSES):
        for f in peaks(scores[:, c + 1]):
            events.append({"frame": f, "t_s": round(f / fps, 2), "label": name, "score": round(float(scores[f, c + 1]), 3)})
    events.sort(key=lambda e: e["frame"])
    Path(args.out).write_text(json.dumps({
        "model": "E2E-Spot tennis_rny008gsm_gru_rgb (checkpoint 41)", "full_width": args.full_width, "height": args.height,
        "fps": fps, "min_score": MIN_SCORE, "events": events,
        "scores": np.round(scores[:, 1:], 3).tolist(),
    }))
    from collections import Counter
    print(Counter(e["label"] for e in events))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
