# Training the stroke-type model on the Acer

The model learns forehand / backhand / serve / etc. from players' body poses.
It trains on ~9,800 labelled strokes from 9 televised matches (the E2E-Spot
labels are in `data/external/e2e_spot_tennis`; the videos come from YouTube),
then is scored on the 73 labelled Dingles swings, which it never trains on.
That score is the real test, since TV cameras film from a different angle
than our fence camera.

Analysis is paused while this runs: both need the GPU.

## 1. Update, then pause the worker

Uploads wait in the queue while it's paused.

```
cd ~/coaching
bash deploy/update.sh
sudo systemctl stop coaching-worker
df -h ~
```

You need about **40 GB free** (the 9 full matches are a few GB each).

## 2. Install the downloader

```
~/.local/bin/uv pip install --python .venv/bin/python yt-dlp
```

## 3. Download the 9 matches

```
mkdir -p ~/tennis_videos
tail -n +2 data/external/e2e_spot_tennis/videos.csv | while IFS=, read -r name res fps id; do
  .venv/bin/yt-dlp -f "bv*[height=1080][fps<=30][vcodec^=avc1]/bv*[height=1080][fps<=30]" \
    -o "$HOME/tennis_videos/$name.%(ext)s" "https://www.youtube.com/watch?v=$id"
done
ls -lh ~/tennis_videos
```

If one match is no longer on YouTube, carry on without it. If yt-dlp says
"Sign in to confirm you're not a bot" or that the format is not available,
send Claude the message.

## 4. Copy the Dingles test file from the Mac

It is not in git. Copy it the same way as the weights folder (USB stick):

- from the Mac: `tennis-tracking-yolo-v1/data/strokes/ours/dingles_serve_volley.npz` (640 KB)
- to the Acer: `~/coaching/data/strokes/ours/dingles_serve_volley.npz`

```
mkdir -p ~/coaching/data/strokes/ours
```

## 5. Quick check on one match (~5 min)

```
.venv/bin/python scripts/build_stroke_data_e2e.py --videos ~/tennis_videos --out data/strokes/e2e_check \
  --matches wimbledon_2019_womens_final_halep_williams --max-points 20 --batch 4
xdg-open data/strokes/e2e_check/wimbledon_2019_womens_final_halep_williams_preview.jpg
```

- **Good:** each picture shows a player at the moment of the stroke named above it.
- **Bad:** crowd shots, replays, players standing around. The video and the
  labels don't line up: stop and tell Claude.

## 6. Build the training data (a few hours)

Ctrl+C stops it; run the same command again to carry on (finished matches
are skipped).

```
.venv/bin/python scripts/build_stroke_data_e2e.py --videos ~/tennis_videos --out data/strokes/e2e --batch 4
```

If it fails with "CUDA out of memory", run it again with `--batch 2`.

## 7. Train (10–30 min)

```
.venv/bin/python scripts/train_stroke_tcn.py --train "data/strokes/e2e/*.npz" \
  --val-match wimbledon_2019_womens_final_halep_williams \
  --test data/strokes/ours/dingles_serve_volley.npz --out weights/stroke_tcn.pt
```

Each round prints an accuracy on the held-out match; it should climb over
the first 10–20 rounds. At the end it prints a score per stroke type on
Dingles and saves `weights/stroke_tcn.pt` and `weights/stroke_tcn.json`.

## 8. Turn analysis back on

```
sudo systemctl start coaching-worker
rm -rf ~/tennis_videos      # optional: frees the space
```

## Afterwards

Send Claude the Dingles scores from the end of step 7. Every upload now also
writes the model's stroke types to `strokes_tcn.json`, kept beside the report
and not shown in it yet. If the model beats the current stroke classifier on
those 73 swings, the report switches to it.
