#!/usr/bin/env bash
# Downloads the 9 broadcast matches the stroke model trains on
# (deploy/train_strokes.md, step 3) into ~/tennis_videos. Run it again to
# fetch any that failed; finished ones are skipped.
#
#   bash deploy/download_matches.sh

set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$HOME/tennis_videos"
mkdir -p "$OUT"

# videos.csv has Windows line endings; tr drops the \r that would end up in each id.
tail -n +2 "$REPO/data/external/e2e_spot_tennis/videos.csv" | tr -d '\r' | while IFS=, read -r name res fps id; do
  [ -n "$id" ] || continue
  echo "== $name"
  "$REPO/.venv/bin/yt-dlp" --no-overwrites \
    -f "bv*[height=1080][fps<=30][vcodec^=avc1]/bv*[height=1080][fps<=30]" \
    -o "$OUT/$name.%(ext)s" "https://www.youtube.com/watch?v=$id" \
    || echo "!! $name failed - carrying on"
done
ls -lh "$OUT"
