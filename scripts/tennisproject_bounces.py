"""Detect bounces with yastrebksv/TennisProject's exact method.

Upstream (https://github.com/yastrebksv/TennisProject, main.py) does two
steps, and both are reproduced unchanged:

  1. Ball track: TrackNet on 640x360 resizes of 1280x720 frames, three
     frames stacked, HoughCircles on the thresholded heatmap, and a
     nearest-to-previous filter at 80px -
     `track_ball_tennisproject` in src/detection/tracknet_ball_detector.py.
  2. Bounces: a pretrained CatBoost regressor over lag/lead pixel
     differences of that track, threshold 0.45, with consecutive hits
     merged to their peak - `CatBoostBounceDetector.predict_frames`, which
     was already a line-for-line port and is used as is.

Step 1 is NOT this project's usual ball tracker on purpose. The CatBoost
model learned what a bounce looks like from tracks made by step 1, in
step 1's pixel units, so feeding it a different tracker's output would be
a different method that happens to share a model file.

INPUT MUST BE 1280x720 (upstream's README says so, and its x2 scaling
assumes it) and should be ~30fps, the frame rate of the TrackNet dataset
the model's frame-lag features were learned at. Make one with:

    python scripts/normalise_clip.py --input SOURCE --start 12 --end 120 \\
        --fps 29.97 --width 1280 --height 720 --output clip_720p30.mp4

    python scripts/tennisproject_bounces.py --input clip_720p30.mp4 \\
        --cache outputs/clip/tennisproject_ball_track.pkl

The ball track is cached, since it is the only slow step; a second run with
the same --cache skips straight to the bounce model.

ONE ASSUMPTION THAT COULD NOT BE CHECKED: that weights/tracknet_pretrained.pt
is the same checkpoint TennisProject's README links for ball detection.
It is yastrebksv's TrackNet release, by the same author, loaded into an
architecture checked layer-for-layer against the upstream model; the
Google Drive file itself could not be compared.
"""

import argparse
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def frames_of(path: str):
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise SystemExit(f"Could not open {path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="A 1280x720 clip")
    parser.add_argument("--cache", required=True, help="Where the ball track is (or will be) cached")
    parser.add_argument("--ball-weights", default=str(REPO_ROOT / "weights" / "tracknet_pretrained.pt"))
    parser.add_argument("--bounce-weights", default=str(REPO_ROOT / "weights" / "bounce_catboost_pretrained.cbm"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--rebuild", action="store_true", help="Re-run TrackNet even if the cache exists")
    args = parser.parse_args()

    import cv2

    capture = cv2.VideoCapture(args.input)
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()

    cache_path = Path(args.cache)
    if cache_path.exists() and not args.rebuild:
        with open(cache_path, "rb") as handle:
            cache = pickle.load(handle)
        print(f"Loaded ball track from {cache_path}")
    else:
        from src.detection.tracknet_ball_detector import track_ball_tennisproject

        print("Running TrackNet (TennisProject method)...", flush=True)
        ball_track = track_ball_tennisproject(frames_of(args.input), args.ball_weights, args.device)
        cache = {"video": str(Path(args.input).resolve()), "fps": fps, "ball_track": ball_track}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as handle:
            pickle.dump(cache, handle)
        print(f"Wrote {cache_path}")

    from src.analysis.catboost_bounce_detector import CatBoostBounceDetector

    ball_track = cache["ball_track"]
    fps = cache["fps"]
    detected = sum(1 for x, _ in ball_track if x is not None)
    print(f"ball found in {detected}/{len(ball_track)} frames ({100 * detected / len(ball_track):.1f}%)")

    # Exactly upstream main.py: split the track into x/y lists and predict.
    # Copies, because predict smooths the lists in place.
    x_ball = [point[0] for point in ball_track]
    y_ball = [point[1] for point in ball_track]
    bounces = sorted(CatBoostBounceDetector(args.bounce_weights).predict_frames(x_ball, y_ball))

    print(f"\n{len(bounces)} bounces:")
    for frame in bounces:
        x, y = x_ball[frame], y_ball[frame]
        where = f"({x:7.1f}, {y:6.1f}) px" if x is not None else "(no position)"
        print(f"  {frame / fps:7.2f}s  frame {frame:5d}  {where}")

    cache["bounces"] = bounces
    cache["smoothed_xy"] = list(zip(x_ball, y_ball))
    with open(cache_path, "wb") as handle:
        pickle.dump(cache, handle)


if __name__ == "__main__":
    main()
