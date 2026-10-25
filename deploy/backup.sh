#!/usr/bin/env bash
# Nightly (coaching-backup.timer, 3am): back up the database, copy the videos
# to BACKUP_DIR, and delete what the retention settings say has expired
# (web/scripts/maintenance.mjs). Run by hand any time: bash deploy/backup.sh

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
set -a; . "$REPO/deploy/coaching.env"; set +a

node "$REPO/web/scripts/maintenance.mjs"

if [ -n "${BACKUP_DIR:-}" ]; then
  MEDIA="${APP_MEDIA:-$REPO/web/data/media}"
  mkdir -p "$BACKUP_DIR/media"
  # --delete: a video removed under the retention rule leaves the backup too.
  rsync -a --delete "$MEDIA/" "$BACKUP_DIR/media/"
  echo "videos copied to $BACKUP_DIR/media ($(du -sh "$BACKUP_DIR/media" | cut -f1))"
fi
