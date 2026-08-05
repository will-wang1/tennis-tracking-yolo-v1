# tennis-tracking-yolo-v1

Tennis ball tracking using a YOLO detector fine-tuned specifically for the
ball, with no other model in the loop. Most tennis-analysis projects (e.g.
those built around TrackNet) use a dedicated heatmap-regression CNN for the
ball because it's small, fast, and frequently occluded - this repo takes the
harder-but-simpler path of pushing a fine-tuned YOLO far enough on its own to
track it reliably, using tracking-side post-processing (outlier rejection +
gap interpolation) to smooth over the frames it misses.

Beyond ball tracking, the pipeline optionally adds match analysis: player
pose + stroke classification, real-world shot speed, and bounce/landing
detection (`src/analysis/`, `src/detection/pose_detector.py`). Each is
opt-in via a `main.py` flag and off by default, so ball-only tracking is
unaffected. Court-line detection is still the one piece not automated -
shot speed in real-world units needs a one-time manual court calibration
(see "Shot speed and bounce detection" below).

## How it works

1. **Detection** (`src/detection/ball_detector.py`) - a YOLO checkpoint
   fine-tuned on ball-only footage runs on every frame at a low confidence
   threshold. Missed frames are expected and handled downstream; a frame
   the detector never even flagged as a low-confidence candidate can't be
   recovered later, so recall is favored over precision here.
2. **Tracking** (`src/tracking/ball_tracker.py`) - raw per-frame detections
   are cleaned up in three passes: detections implying a physically
   implausible jump since the last accepted point are discarded as false
   positives; a run of detections confined to a small area for many
   consecutive frames is discarded as a lock-on onto something fixed on
   screen (a line, the net cord, a broadcast graphic) rather than trusted as
   a stationary ball; then short gaps (a handful of missed frames) are
   filled by linear interpolation between the surrounding accepted points.
   Gaps longer than `max_interpolation_gap` are left unfilled rather than
   guessed at. All four thresholds are exposed as `main.py` flags
   (`--max-jump`, `--interp-gap`, `--lockon-frames`, `--lockon-radius`) so
   they can be tuned to your footage without touching code.
3. **Visualization** (`src/visualize/draw.py`) - the tracked position and a
   fading trail are drawn back onto the video.
4. **Match analysis** (`src/analysis/`, `src/detection/pose_detector.py`,
   opt-in) - `--bounce` finds landing spots directly from the tracked
   trajectory (a local maximum in screen-y with the ball's motion flipping
   from falling to rising, cross-referenced against the nearest player's
   ankle height when `--pose` is also given, to tell a real landing apart
   from a racket contact); `--pose` runs YOLOv8-pose and picks whichever
   player is nearest the ball each frame as the striker; `--stroke-classifier`
   classifies their stroke from a short window of pose keypoints; `--speed`
   reports peak ball speed per shot (segmented at any trajectory
   direction-change, bounce or contact), in real-world km/h if
   `--calibration` is given, otherwise px/s; `--sidebar` composites a panel
   showing the current stroke and a stable per-shot speed reading. See
   "Match analysis" below for the full setup.

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

**If you're seeing false-positive dots that cluster in the same screen
region across multiple runs** (e.g. always near the net), that's not random
noise - it means something fixed in that spot (net cord/tape, a sponsor
logo, court hardware) consistently looks like the ball to the model. Two
levers, cheapest first:

- Tune the tracker without retraining: raise `--confidence` (fewer weak
  detections to begin with), or tighten `--lockon-frames`/`--lockon-radius`
  so a shorter/tighter false lock-on gets discarded sooner.
- Fix it at the source: pull frames from right where the false positives
  keep appearing (`scripts/extract_frames.py` on that clip), label them
  (bounding box on the real ball if present, otherwise no label - a true
  negative example), add them to `data/raw`, and retrain. This teaches the
  model what that specific distractor looks like, which tracker tuning
  alone can't fully substitute for.

## Match analysis: pose, stroke, speed, bounce

All of this is opt-in - omit these flags and `main.py` behaves exactly as
before (ball tracking only).

**Bounce detection** needs nothing extra on its own - it's post-processing
on the already-tracked trajectory (a local maximum in screen-y):

```
python main.py --input match.mp4 --output outputs/tracked.mp4 --bounce
```

A local max in y is necessary but not sufficient: for a broadcast camera
looking down the court from behind a baseline, the ball's height off the
ground and its depth (near/far court position) are both encoded in the same
2D y-pixel, so a racket **contact** that redirects the ball back across the
court reverses y almost the same way a real **bounce** does - they can look
identical in trajectory alone. Two filters help distinguish them:
horizontal direction (`--bounce-min-x-reversal`) is a weak signal for this
camera angle specifically, since lateral motion isn't where the primary
court-crossing action is. **Add `--pose` alongside `--bounce`** for the
strong signal instead: it cross-references each candidate against the
nearest player's ankle height that frame (a standing player's feet are at
court level), and rejects anything more than `--bounce-max-height` (default
160px) above it as too high to be a real landing - almost certainly a
contact. That threshold is deliberately generous: telling a low bounce from
a contact from pixels alone is inherently approximate (monocular depth
ambiguity, not a bug to fully solve here), and a stricter threshold in
testing rejected every candidate in some clips outright. Without `--pose`,
bounce detection still runs, but only has the weaker trajectory-only
signals to go on.

`--bounce-min-prominence` (default 15px) is sensitive to camera framing -
how many pixels the ball's y-position needs to swing to count as a real
bounce depends on how zoomed in/far the broadcast camera is. If you're
seeing zero bounces on a clip, try lowering it; if you're seeing bounces
that are obviously just noise, raise it. There's no single default that
fits every camera angle.

**Pose + stroke skeleton overlay** uses YOLOv8-pose (`yolov8n-pose.pt`,
auto-downloaded by `ultralytics` on first use - unlike `weights/ball_detector.pt`
this isn't a fine-tuned repo artifact) to draw the striker's skeleton:

```
python main.py --input match.mp4 --output outputs/tracked.mp4 --pose
```

**Stroke classification** needs a trained classifier, which needs labeled
data first - there's no pre-existing labeled stroke dataset, so this is a
three-step workflow:

1. Find candidate stroke-contact frames (crude, high-recall - a human
   filters next):
   ```
   python scripts/extract_stroke_candidates.py --input match.mp4 \
       --out outputs/stroke_candidates.csv --thumbnails-out outputs/stroke_candidates/
   ```
2. Open the thumbnails, fill in the `label` column of the CSV for every
   candidate you can confidently identify
   (`forehand`/`backhand`/`serve`/`volley`/`other`; see
   `src/analysis/stroke_classifier.py:STROKE_LABELS`), leave ambiguous rows
   blank, save as `outputs/stroke_candidates_labeled.csv`.
3. Train and use it:
   ```
   python scripts/train_stroke_classifier.py --labels outputs/stroke_candidates_labeled.csv \
       --out weights/stroke_classifier.pkl
   python main.py --input match.mp4 --output outputs/tracked.mp4 \
       --pose --stroke-classifier weights/stroke_classifier.pkl
   ```

With only a couple of short clips to label, expect a few dozen examples at
most - `train_stroke_classifier.py` prints cross-validation scores so you
can see how limited that makes it, rather than asserting a confident
accuracy. More labeled clips directly improve it, the same way more training
footage improves the ball detector itself.

**Real-world shot speed** needs a one-time court calibration per camera
angle, since there's no automatic court-line detection built into the main
pipeline (though see "Training an automatic court-keypoint detector" below
for a trainable alternative to reading pixels by hand). Two ways to build
one:

Manual - read four corners off a still frame by eye:
```
python scripts/extract_calibration_frame.py --input match.mp4 --out outputs/calibration_frame.jpg
# open outputs/calibration_frame.jpg in an image viewer, read off the pixel
# (x, y) of the baseline-left/right and service-line-left/right corners
# nearest the camera, then:
python scripts/calibrate_court.py --frame outputs/calibration_frame.jpg \
    --baseline-left 340,980 --baseline-right 1580,980 \
    --service-left 520,650 --service-right 1400,650 \
    --out configs/court_calibration.json
```

Automatic - once you've trained a court-keypoint detector (see below):
```
python scripts/calibrate_court_auto.py --input match.mp4 \
    --keypoint-weights runs/court_keypoint_detector/weights/best.pt \
    --out configs/court_calibration.json
```

Then, either way:
```
python main.py --input match.mp4 --output outputs/tracked.mp4 \
    --bounce --speed --sidebar --calibration configs/court_calibration.json \
    --speed-window 5
```

Without `--calibration`, `--speed` still works but reports px/s instead of
km/h. Speed is computed by mapping the ball's pixel position through a
**ground-plane** homography - exact at a bounce, increasingly approximate
the higher the ball is above the court (e.g. a serve toss or smash), since
there's no calibrated depth/height model. That's an accepted limitation, not
a bug. Readings above 300 km/h (faster than any tennis shot ever recorded)
are excluded from peak-speed selection outright - almost always a single
noisy pixel jump, not a real shot.

`--speed-window` (default 1, i.e. raw frame-to-frame displacement) trades
responsiveness for robustness to detector jitter: a coarser/more zoomed-out
camera calibration maps the same few pixels of jitter to more real-world
meters, which can otherwise swing the reported speed by 100+ km/h frame to
frame. If a clip's speed readings look noisy, raise this - there's no
single value that's right for every camera's zoom level, same as
`--bounce-min-prominence`.

`--speed` prints each shot's peak speed to the console once processing
finishes. `--sidebar` shows a **stable per-shot** speed reading (not a raw
per-frame instant) - held constant for each shot's whole duration, only
updating when the trajectory changes direction. Shots are segmented at
*any* trajectory direction-change (bounce or contact - see
`find_trajectory_breakpoints`), not just confirmed bounces, since those are
often sparse; the sidebar works even with `--bounce` omitted.

### Training an automatic court-keypoint detector

`scripts/calibrate_court.py`'s manual pixel-reading works but is tedious
and only as precise as the human doing it. `src/detection/court_keypoint_detector.py`
detects the 14 standard tennis court keypoints (both baselines, both
singles and doubles sidelines, both service lines, and the two center
marks - see `src.analysis.court_calibration.FULL_COURT_REFERENCE_POINTS`
for their exact real-world layout) directly, the same way `pose_detector.py`
detects player joints.

No pretrained checkpoint ships with this repo - train one:

```
python scripts/download_court_dataset.py
python scripts/prepare_court_keypoint_dataset.py
python scripts/train_court_keypoints.py --epochs 60
```

The dataset is [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)
(free Google Drive download, no API key) rather than one of the smaller
Roboflow tennis-court-keypoint datasets (500-2.5k images, usually a single
court surface) - it has 8,841 images spanning hard, clay, *and* grass
courts, which matters here since this repo's own test clips are hard court
and grass. `scripts/prepare_court_keypoint_dataset.py` converts its raw
JSON keypoint format into YOLO-pose format.

This needs a real GPU - the ball/pose detectors in this repo are small
enough to fine-tune on CPU in a pinch, but 8.8k images at `imgsz=1280` is
not.  Once trained, `scripts/calibrate_court_auto.py` replaces the manual
calibration workflow above.

## Diagnosing false positives

If you're seeing dots that don't belong, don't guess from screenshots -
run the detector's raw output through `scripts/analyze_detections.py`
(bypasses the tracker's filtering entirely) to get a CSV of every detection
plus a heatmap image of where they cluster on screen:

```
python scripts/analyze_detections.py --input path/to/match.mp4
```

A diffuse spread following the court is real ball detections. A bright,
tight hotspot sitting in one fixed screen location means something specific
there (net cord/tape, a sponsor logo, court hardware) is consistently
fooling the model.

Once you've confirmed a hotspot is a real false positive (spot-check a few
of its frames - if the actual ball isn't there, it's confirmed), stage those
exact frames as training negatives in one step:

```
python scripts/add_hard_negatives.py --input match.mp4 --csv outputs/detections.csv \
    --x-min 870 --x-max 900 --y-min 210 --y-max 240 --out data/raw/train --prefix net_marker
```

This writes the flagged frames plus an empty label file each - YOLO's
convention for "no ball in this image" - straight into your training split.
Retrain afterward and re-run `analyze_detections.py` on the same clip; the
hotspot should shrink or disappear. That fixes it at the source, which no
amount of tracker tuning fully replaces.

## Tests

```
pytest tests/
```

Covers the tracker's interpolation/outlier-rejection logic, bounce
detection, court calibration geometry, speed estimation, striker selection,
and stroke feature engineering directly with hand-built data - no trained
model or video file needed to run any of these.
