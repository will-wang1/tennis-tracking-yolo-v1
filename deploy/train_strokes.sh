#!/usr/bin/env bash
# Steps 5-7 of deploy/train_strokes.md, so nothing long has to be typed:
#
#   bash deploy/train_strokes.sh check    # 5. quick check on one match (~5 min), opens the picture
#   bash deploy/train_strokes.sh build    # 6. training data from all matches (a few hours; run again to carry on)
#   bash deploy/train_strokes.sh train    # 7. train, and score on the Dingles labels (10-30 min)
#   bash deploy/train_strokes.sh score    # score the trained model on the Dingles labels again (seconds)
#
# Add a batch size to check/build if the GPU runs out of memory:
#   bash deploy/train_strokes.sh build 2
#
# Everything printed also goes to outputs/train_strokes_<step>.log.

set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/.venv/bin/python"
VIDEOS="$HOME/tennis_videos"
CHECK_MATCH=wimbledon_2019_womens_final_halep_williams
DINGLES="$REPO/data/strokes/ours/dingles_serve_volley.npz"
cd "$REPO"
mkdir -p outputs

step="${1:-}"
batch="${2:-4}"
log="outputs/train_strokes_${step:-none}.log"

if [ "$step" != score ] && systemctl is-active -q coaching-worker 2>/dev/null; then
  echo "The worker is running and would share the GPU. Stop it first: sudo systemctl stop coaching-worker"
  exit 1
fi

videos_there() {
  if ! ls "$VIDEOS"/* >/dev/null 2>&1; then
    echo "No videos in $VIDEOS - run: bash deploy/download_matches.sh"
    exit 1
  fi
  echo "Videos in $VIDEOS:"
  ls -lh "$VIDEOS" | tail -n +2 | awk '{print "  " $5 "  " $9}'
  echo
}

case "$step" in
  check)
    videos_there
    match=$CHECK_MATCH
    if ! ls "$VIDEOS/$match".* >/dev/null 2>&1; then
      # Not downloaded: check whichever match is there instead.
      match=$(ls "$VIDEOS" | grep -v '\.part$' | head -1 | sed 's/\.[^.]*$//')
      echo "$CHECK_MATCH is not downloaded - checking $match instead"
    fi
    rm -rf data/strokes/e2e_check  # a check is always made afresh
    "$PY" -u scripts/build_stroke_data_e2e.py --videos "$VIDEOS" --out data/strokes/e2e_check \
      --matches "$match" --max-points 20 --batch "$batch" 2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    preview="data/strokes/e2e_check/${match}_preview.jpg"
    if [ "$status" = 0 ] && [ -f "$preview" ]; then
      echo
      echo "Opening $preview - each picture should show a player hitting the stroke named above it."
      xdg-open "$preview" >/dev/null 2>&1 &
    else
      echo
      echo "It did not work. Show the end of it with: tail -30 $REPO/$log"
    fi
    ;;
  build)
    videos_there
    "$PY" -u scripts/build_stroke_data_e2e.py --videos "$VIDEOS" --out data/strokes/e2e --batch "$batch" 2>&1 | tee "$log"
    [ "${PIPESTATUS[0]}" = 0 ] && echo "Done. Next: bash deploy/train_strokes.sh train" \
      || echo "Stopped. Run the same command to carry on; if it keeps failing, show the end with: tail -30 $REPO/$log"
    ;;
  train)
    if ! ls data/strokes/e2e/*.npz >/dev/null 2>&1; then
      echo "No training data yet - run: bash deploy/train_strokes.sh build"
      exit 1
    fi
    test=()
    if [ -f "$DINGLES" ]; then
      test=(--test "$DINGLES")
    else
      echo "No Dingles test file at $DINGLES (step 4) - training without the Dingles score."
    fi
    "$PY" -u scripts/train_stroke_tcn.py --train "data/strokes/e2e/*.npz" --val-match "$CHECK_MATCH" \
      "${test[@]}" --out weights/stroke_tcn.pt 2>&1 | tee "$log"
    [ "${PIPESTATUS[0]}" = 0 ] && echo "Done. Send Claude the scores above, then: sudo systemctl start coaching-worker" \
      || echo "It did not work. Show the end of it with: tail -30 $REPO/$log"
    [ -f "$DINGLES" ] || echo "No Dingles score: copy the file (step 4), then: bash deploy/train_strokes.sh score"
    ;;
  score)
    if [ ! -f weights/stroke_tcn.pt ]; then
      echo "No trained model yet - run: bash deploy/train_strokes.sh train"
      exit 1
    fi
    if [ ! -f "$DINGLES" ]; then
      echo "No Dingles test file at $DINGLES - copy it from the Mac by USB (step 4 of deploy/train_strokes.md)."
      exit 1
    fi
    "$PY" -u scripts/train_stroke_tcn.py --init weights/stroke_tcn.pt --epochs 0 --test "$DINGLES" 2>&1 | tee "$log"
    [ "${PIPESTATUS[0]}" = 0 ] && echo "Send Claude the scores above." \
      || echo "It did not work. Show the end of it with: tail -30 $REPO/$log"
    ;;
  *)
    sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
