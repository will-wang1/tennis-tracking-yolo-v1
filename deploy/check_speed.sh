#!/usr/bin/env bash
# Runs one clip through the whole analysis twice - 32-bit models, then the
# 16-bit ones the GPU now uses by default - shows each step's time side by
# side, and checks the two runs found the same things (scripts/compare_runs.py).
# The worker must be stopped first: both need the GPU.
#
#   sudo systemctl stop coaching-worker
#   bash deploy/check_speed.sh ~/Downloads/dingles_serve_volley.mp4
#   sudo systemctl start coaching-worker
#
# Takes about twice as long as one upload. No AI feedback is written.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLIP="${1:?usage: bash deploy/check_speed.sh <clip.mp4>}"
PY="$REPO/.venv/bin/python"
cd "$REPO"

if systemctl is-active -q coaching-worker 2>/dev/null; then
  echo "The worker is running and would share the GPU: sudo systemctl stop coaching-worker, then run this again."
  exit 1
fi

run() {  # name, extra env...
  local name=$1; shift
  echo "== $name"
  local start=$SECONDS
  env "$@" "$PY" -u scripts/run_session.py --input "$CLIP" --name "$name" --no-feedback --force \
    | tee "outputs/$name.log" | grep --line-buffered '^\[done\]'
  echo "   total $((SECONDS - start))s"
}

mkdir -p outputs
run speed_fp32 TENNIS_FP32=1
run speed_fp16 TENNIS_FP32=0

echo
echo "Step times (seconds): 32-bit vs 16-bit"
paste <(grep '^\[done\]' outputs/speed_fp32.log) <(grep '^\[done\]' outputs/speed_fp16.log | sed 's/.* in //') \
  | sed -E 's/^\[done\] (.*) in ([0-9]+)s\t(.*)s$/  \1: \2 -> \3/'

echo
labels="$REPO/data/labels/swing_labels_dingles_serve_volley.json"
extra=()
[[ "$(basename "$CLIP")" == dingles_serve_volley.* && -f "$labels" ]] && extra=(--labels "$labels")
"$PY" scripts/compare_runs.py outputs/speed_fp32 outputs/speed_fp16 "${extra[@]}"
