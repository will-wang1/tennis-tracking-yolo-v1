#!/usr/bin/env bash
# Bring the Acer up to date after changes were tested on the Mac and pushed:
#
#   bash deploy/update.sh
#
# Pulls the code, updates packages, rebuilds and restarts. A clip being
# analysed is cut off and starts again from its last finished step.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
cd "$REPO"

git pull --ff-only
# System packages added since setup (installing again is a no-op).
sudo apt-get install -y -q ffmpeg >/dev/null
uv pip install -q --python .venv/bin/python -r requirements.txt

set -a; . deploy/coaching.env; set +a
(cd web && npm ci --no-audit --no-fund && npm run build)

# Pick up changed service files too.
for unit in coaching-web.service coaching-worker.service coaching-backup.service coaching-backup.timer; do
  sed -e "s|@REPO@|$REPO|g" -e "s|@USER@|$USER|g" "deploy/systemd/$unit" | sudo tee "/etc/systemd/system/$unit" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl restart coaching-web.service coaching-worker.service
echo "Updated to $(git log -1 --format='%h %s')"
bash deploy/status.sh
