# Coaching session analysis - backend

Evaluates **how a tennis coaching session was run**, from video of a fixed
camera on the back fence behind the baseline. Video only, no audio. Works for
squad sessions (usually two players a side) and 1-on-1 lessons.

It measures **activity** - ball in play, who hit how many balls and from
where, where shots landed, errors, movement, waiting, feeding vs live play -
and a local language model (Qwen3 8B) writes feedback from those numbers.
It cannot judge whether a drill was a good choice or whether anyone improved,
and nothing it writes claims to.

## Run a session

```bash
python scripts/run_session.py --input session.mp4 --name tuesday_squad --theme netplay,consistency
open outputs/tuesday_squad/coach_report.html
```

Steps whose output exists are skipped, so an interrupted run resumes
(`--force` redoes everything). About 8 minutes of processing per minute of
video on an M2 MacBook.

| Step | Script | Writes (`outputs/<name>/`) |
|---|---|---|
| 1. normalise to 1080p/25fps | `normalise_clip.py` | `data/videos/<name>.mp4` |
| 2. ball (WASB) + court (TennisCourtNet), every frame / once a second | `replay_impacts.py --build` | `replay_cache.pkl` |
| 3. people (YOLOv8s + ByteTrack) | `track_people.py` | `person_tracks.pkl` |
| 4. pose for every person, every frame (YOLOv8s-pose) | `track_poses.py` | `poses.pkl` |
| 5. bounces and swings (E2E-Spot, a video model) | `e2e_spot_events.py` | `e2e_spot.json` |
| 6. court fit, camera cutaways, first report | `session_report.py` | `session_report.json` |
| 7. stable players (clothing colour) | `assign_players.py` | `players.json` |
| 8. swings and contacts (wrist movement + ball) | `detect_swings.py` | `swings.json` |
| 9. stroke type, once a trained model exists | `classify_strokes_tcn.py` | `strokes_tcn.json` |
| 10. full report | `session_report.py` | `session_report.json` |
| 11. AI feedback (Qwen3 8B through MLX, on this machine) | `session_feedback.py` | `session_feedback.json` |
| 12. coach page | `render_coach_report.py` | `coach_report.html` |
| 13. annotated video (H.264) | `render_from_cache.py` | `annotated.mp4` |

## Web app

`web/` is the app: the head coach compares coaches against a benchmark
learned from exemplary sessions, coaches see their own sessions. Session
videos are uploaded in the app and analysed by `scripts/worker.py` (which
runs `run_session.py` on any computer that can reach the app); sessions
analysed by hand go in with `scripts/publish_session.py`. See `web/README.md`.

## What the report contains

- **Verdicts first**: a headline, what went well, what to look at - each judged
  against a target (`--targets` to change them) or against the group.
- **Lesson themes** (`--theme`): net play and consistency, start vs end of the
  session. Adjustment (slice) and around-the-backhand wait for the stroke model.
- **Per player**: movement, time moving, waiting off court, time at the net,
  balls hit, where they hit from, side of the body (near end), shots deep/out,
  errors (out / into the net, on the 1st-2nd ball), shots not returned.
- **Session**: ball in play, drills and breaks, feeding vs live play, rallies
  and how they ended, bounce landing map with depth and in/out.
- Every number carries its **basis** (measured / estimated / lower bound),
  method and caveats; the page and the AI feedback keep to them.

## Code

| Where | What |
|---|---|
| `src/detection/` | ball (WASB), court keypoints (TennisCourtNet), people, pose |
| `src/analysis/` | court calibration and refinement, swings and contacts, E2E-Spot bounces, player identity, person tracks, stroke models |
| `src/session/` | report, insights, structure, feeding, contacts, outcomes, themes, feedback, coach page template |
| `src/third_party/` | vendored E2E-Spot and BST (licences inside) |
| `scripts/` | the pipeline steps above, plus stroke-training and labelling tools |
| `data/labels/` | hand labels (swings, player identities) |
| `data/external/e2e_spot_tennis/` | E2E-Spot's stroke-labelled broadcast points, for training |

## Stroke classification (in progress)

- `scripts/make_swing_labeller.py` - a local page to label every detected
  swing (forehand, backhand, serve, overhead, volley, slice, not a swing).
- `scripts/build_stroke_data_e2e.py` / `build_stroke_data_ours.py` and
  `train_stroke_tcn.py` - pretrain on E2E-Spot's ~9,800 labelled broadcast
  strokes, fine-tune on our labels. Run on a GPU machine:
  `docs/gpu_stroke_training.html`.
- `classify_strokes_bst.py` - the current baseline (BST, TenniSet weights).

## Accuracy, as measured on the one session so far (Dingles, 108s, hand labels)

- Swings: 55 of 73 detections real; about 1 in 6 real swings missed.
- Stroke type (BST baseline): about 80% on real swings.
- Contact side (left/right): reliable at the near end only.
- Bounce positions: court fit refined onto the painted lines; bounces placed
  between frames. Not yet measured against ground truth.
- Contact height is not reported: from one fence camera it cannot be measured.

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Weights go in `weights/` (not in git): `wasb_tennis_pretrained.pth.tar`,
`court_net_pretrained.pt`, `yolo/yolov8s.pt`, `yolo/yolov8s-pose.pt`,
`e2e_spot/tennis_rny008gsm_gru_rgb.pt`, `bst/bst_AP_JnB_bone.pt`.
Qwen3 8B downloads from Hugging Face on first use (~4.6GB).
