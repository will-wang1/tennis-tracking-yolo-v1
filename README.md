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

**Bounce detection** is post-processing on the already-tracked trajectory,
and it works best with the court:

```
python main.py --input match.mp4 --output outputs/tracked.mp4 \
    --bounce --contacts --show-court --minimap
```

The hard part is not finding where the ball was struck - it is saying WHAT
struck it. For a camera looking down the court, the ball's height off the
ground and its depth down the court are both encoded in the same y-pixel,
so a racket **contact** that sends the ball back reverses y almost exactly
as a **bounce** does. Every method that judges a single frame's shape
confuses the two, which is what the earlier ones here did.

The default (`--bounce-method parabolic`) instead fits a whole free-flight
arc either side of each candidate and keeps only transitions a court
surface could physically have produced - see
`src/analysis/parabolic_bounce_detector.py`. Fitting arcs rather than
reading one frame is what separates a real bounce from both detector noise
and a racket contact.

`--show-court` then upgrades the verdict from "something hit the ball" to
"the court hit the ball", by measuring the ball's PROJECTED court position
rather than its screen position: the homography maps the ground plane, so
it is exact for a ball on the court and wrong for one above it, and that
error is itself the height signal (`src/analysis/touchdown_detector.py`).
`--minimap` additionally collects player boxes, which are used in one
direction only - to withhold the "contact" verdict from an impact nobody
could have reached.

Verdicts are three-valued. "Bounce" and "contact" are positive claims
backed by a measurement; an impact that can be found but not attributed is
reported as neither and draws no marker, because a marker is a claim.
`--contacts` prints every impact with the reason behind its verdict, which
is the quickest way to see where a run is failing: a rally alternates
bounce and contact, so a contact with no bounce before the next one is
either a volley or a bounce that was missed.


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

With `--show-court`, the headline reading for a shot is taken where its
fitted flight crosses the NET (`estimate_flight_net_speeds`). That is the
best moment available in a monocular view: the calibration is a
ground-plane homography, so its error grows with the ball's height, and a
rally ball is lowest there between the two strikes. Solving the crossing on
the fitted curve rather than looking for detections near the net means it
does not matter whether the ball was actually seen at that instant - it
usually is not, since the ball is small, fast and often occluded exactly
there.

A flight that never crosses the net - bounce to racket, about half of
them - still gets the coarser whole-flight reading, so every shot has a
number. `--speed-window` (default 1, i.e. raw frame-to-frame displacement)
controls that fallback, and trades responsiveness for robustness to
detector jitter: a more zoomed-out calibration maps the same few pixels of
jitter to more real-world metres, which can otherwise swing the reading by
100+ km/h frame to frame. If a clip's speeds look noisy, raise it.

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

## Checking bounce/contact accuracy without a GPU

Every impact verdict downstream of the detectors is arithmetic over a few
hundred tracked points; only the detectors themselves are expensive. So
`scripts/replay_impacts.py` runs them once, caches what they said, and
replays the rest as often as you like - a change to a threshold is measured
in about a second instead of a render:

```
# once (the only step that wants a GPU)
python scripts/replay_impacts.py --build --input match.mp4     --cache outputs/match/replay_cache.pkl

# then, on any laptop
python scripts/replay_impacts.py --cache outputs/match/replay_cache.pkl     --labels data/labels/video_input2_impacts.csv
python scripts/replay_impacts.py --cache ... --labels ... --max-reach-ratio 0.4
```

It prints every impact with the measurements the verdict came from - the
projected approach rate either side, and how close the nearest player
was - and, given `--labels`, scores those verdicts against hand-labelled
events. `--add-player-boxes` tops the cache up with the person detector on
just the frames around this run's impacts, which is the only place player
boxes are read.

The label files live in `data/labels/` - see that directory's README for
what they are, and for the more important question of what a good score on
them does and does not prove.

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
detection, bounce-vs-contact attribution, label scoring, court calibration
geometry, speed estimation, striker selection, and stroke feature
engineering directly with hand-built data - no trained model or video file
needed to run any of these.

## Web UI

A browser-based upload-and-track UI lives alongside the CLI: `backend/`
(FastAPI + Celery/Redis + Postgres + S3-compatible storage) and `frontend/`
(React + Vite). It wraps the same pipeline the CLI uses -
`src/pipeline.py::run_pipeline`, extracted from `main.py` so both can call
it - so nothing about the detection/tracking/analysis logic is duplicated.

A user registers, uploads a video, picks bounce/speed/sidebar/minimap
toggles, optionally clicks 4 court corners in the browser to calibrate real
km/h speeds and a bounce landing heatmap (`CourtCalibration.from_points`,
the same math `scripts/calibrate_court.py` uses), then watches a Celery
worker's progress and gets back the annotated video plus a stats table,
speed chart, and heatmap.

**This needs real model weights to do anything** - none ship in this repo
(see "Setup" above). Point `BALL_WEIGHTS_PATH`/`WASB_WEIGHTS_PATH`/etc at
your checkpoints via `.env` (`cp .env.example .env`); `COURT_WEIGHTS_PATH`
is optional - leave it unset and the minimap/court-overlay toggle is simply
hidden as unavailable rather than failing jobs.

Run the whole stack:

```
cp .env.example .env   # fill in JWT_SECRET and the weight paths
docker compose up --build
```

Frontend at `http://localhost:5173`, API at `http://localhost:8000/docs`.
The worker mounts `./weights` into the container - drop your checkpoints
there. Uncomment the GPU block in `docker-compose.yml`'s `worker` service if
running on an NVIDIA GPU host.

For local dev without Docker: `cd backend && cp ../.env.example .env`, run
Postgres/Redis/MinIO yourself (or point `DATABASE_URL`/`REDIS_URL`/
`S3_ENDPOINT_URL` at existing ones), then
`PYTHONPATH=.. python -m uvicorn app.main:app --reload` and, in another
shell, `PYTHONPATH=.. python -m celery -A app.celery_app worker --loglevel=info`.
`cd frontend && npm install && npm run dev` for the UI.

### Public access over Tailscale

Before sharing this beyond your own machine: set `INVITE_CODE` and rotate
`JWT_SECRET`/`POSTGRES_PASSWORD`/`S3_SECRET_KEY` in `.env` away from their
placeholder values (see the comments in `.env.example`) - the app logs a
warning at startup if you forget.

To actually put it on the internet without touching your router, buying a
domain, or installing anything on the host (no `.pkg`, no `sudo`): the
`tailscale` service in `docker-compose.yml` runs Tailscale Funnel inside a
container, reaching the `frontend` container over the compose network.

1. Create a free account at [tailscale.com](https://tailscale.com) and
   generate an auth key at
   [login.tailscale.com/admin/settings/keys](https://login.tailscale.com/admin/settings/keys)
   (a reusable, non-ephemeral key is easiest if you'll restart the
   container). Set `TS_AUTHKEY` in `.env` to it.
2. In the Tailscale admin console, confirm **HTTPS Certificates** and
   **Funnel** are enabled for your tailnet (on by default for most personal
   accounts).
3. `docker compose --profile tunnel up -d` (the `tunnel` profile keeps this
   service out of a normal `docker compose up` for anyone not using it).
4. `docker compose exec tailscale tailscale funnel --bg --https=443 http://frontend:80`
   (flags vary by Tailscale version - run
   `docker compose exec tailscale tailscale funnel --help` if this errors).

That gives you a stable `https://<name>.<your-tailnet>.ts.net` URL. No
other config changes are needed: the frontend already proxies `/api/` to
the backend over a relative same-origin path, and nginx's
`client_max_body_size 2G` (in `frontend/nginx.conf`) already covers large
video uploads through the tunnel.
