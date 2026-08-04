# tennis-tracking-yolo-v1

Tennis ball tracking using a YOLO detector fine-tuned specifically for the
ball, with no other model in the loop. Most tennis-analysis projects (e.g.
those built around TrackNet) use a dedicated heatmap-regression CNN for the
ball because it's small, fast, and frequently occluded - this repo takes the
harder-but-simpler path of pushing a fine-tuned YOLO far enough on its own to
track it reliably, using tracking-side post-processing (outlier rejection +
gap interpolation) to smooth over the frames it misses.

This currently covers ball detection and tracking only. Player detection,
court-line detection, and bounce detection are the natural next pieces if
this gets extended into a full match-analysis pipeline, and the code is
structured (`src/detection/`, `src/tracking/`, `src/video/`,
`src/visualize/`) so they can be added as siblings to the ball modules
without restructuring what's here.

## How it works

1. **Detection** (`src/detection/ball_detector.py`) - a YOLO checkpoint
   fine-tuned on ball-only footage runs on every frame at a low confidence
   threshold. Missed frames are expected and handled downstream; a frame
   the detector never even flagged as a low-confidence candidate can't be
   recovered later, so recall is favored over precision here.
2. **Tracking** (`src/tracking/ball_tracker.py`) - raw per-frame detections
   are cleaned up in two passes: detections implying a physically
   implausible jump since the last accepted point are discarded as false
   positives, then short gaps (a handful of missed frames) are filled by
   linear interpolation between the surrounding accepted points. Gaps
   longer than `max_interpolation_gap` are left unfilled rather than
   guessed at.
3. **Visualization** (`src/visualize/draw.py`) - the tracked position and a
   fading trail are drawn back onto the video.

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Get a dataset

See `data/README.md`. Short version: either pull the public Roboflow
tennis-ball dataset with `scripts/download_dataset.py`, or extract frames
from your own footage with `scripts/extract_frames.py` and label them
yourself.

## 2. Fine-tune YOLO on the ball

```
python scripts/train.py --epochs 150 --imgsz 1280
```

Run from the repo root - `configs/ball_dataset.yaml` is resolved relative to
the working directory. Defaults are tuned for a small, fast-moving object
(high resolution, mosaic augmentation disabled - see the docstring in
`scripts/train.py` for why). Copy the resulting `runs/ball_detector/weights/best.pt`
to `weights/ball_detector.pt` (the script prints the exact command) once
you're happy with it, or point `main.py --weights` at it directly.

If accuracy plateaus, the usual levers are: more/more-varied training
footage (different courts, lighting, camera angles), more epochs, and a
larger base checkpoint (`yolov8s.pt` / `yolov8m.pt` instead of `yolov8n.pt`).

## 3. Track the ball in a video

```
python main.py --input path/to/match.mp4 --output outputs/tracked.mp4
```

Prints raw detection rate and how much of the trajectory got recovered by
interpolation, so you can see how well the fine-tuned model is doing before
even watching the output video.

## Tests

```
pytest tests/
```

Covers the tracker's interpolation and outlier-rejection logic directly
(no trained model needed to run these).
