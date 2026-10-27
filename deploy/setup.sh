#!/usr/bin/env bash
# One-time setup of the Acer as the coaching server: Ubuntu 24.04, either
# installed on the laptop or inside WSL2 (deploy/README.md). Run it from the
# cloned repo as your normal user (it asks for sudo when it needs it):
#
#   bash deploy/setup.sh
#
# Safe to run again: each step skips what is already done.

set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="$REPO/deploy/coaching.env"
WSL=$(grep -qi microsoft /proc/version && echo 1 || echo 0)
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "System packages"
sudo apt-get update -q
sudo apt-get install -y -q git curl ca-certificates build-essential libgl1 libglib2.0-0 rsync openssl ffmpeg

step "GPU"
if ! nvidia-smi >/dev/null 2>&1; then
  if [ "$WSL" = 1 ]; then
    echo "No GPU inside WSL: install the NVIDIA driver in Windows (not in Ubuntu) and restart. deploy/README.md, step 2."
  else
    echo "No NVIDIA driver: run 'sudo ubuntu-drivers install', restart, then run this again."
  fi
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "$WSL" = 0 ]; then
  step "Never sleep, ignore the lid"
  sudo systemctl mask -q sleep.target suspend.target hibernate.target hybrid-sleep.target
  sudo mkdir -p /etc/systemd/logind.conf.d
  printf '[Login]\nHandleLidSwitch=ignore\nHandleLidSwitchExternalPower=ignore\nHandleLidSwitchDocked=ignore\n' \
    | sudo tee /etc/systemd/logind.conf.d/coaching.conf >/dev/null
fi

step "Settings file"
if [ ! -f "$ENV_FILE" ]; then
  sed -e "s/^IMPORT_TOKEN=$/IMPORT_TOKEN=$(openssl rand -hex 32)/" \
      -e "s/^SETUP_TOKEN=$/SETUP_TOKEN=$(openssl rand -hex 12)/" \
      "$REPO/deploy/coaching.env.example" > "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  echo "Wrote $ENV_FILE with new secrets."
else
  echo "Keeping the existing $ENV_FILE"
fi
set -a; . "$ENV_FILE"; set +a

step "Python 3.14 and the analysis packages (a few GB, the first time)"
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
[ -x "$REPO/.venv/bin/python" ] || uv venv --python 3.14 "$REPO/.venv"
uv pip install --python "$REPO/.venv/bin/python" -r "$REPO/requirements.txt"
"$REPO/.venv/bin/python" -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'; print('PyTorch on', torch.cuda.get_device_name(0))"

step "Model weights"
missing=0
for f in wasb_tennis_pretrained.pth.tar court_net_pretrained.pt yolo/yolov8s.pt yolo/yolov8s-pose.pt \
         e2e_spot/tennis_rny008gsm_gru_rgb.pt e2e_spot/tennis_rny008gsm_gru_rgb.json; do
  [ -f "$REPO/weights/$f" ] || { echo "missing weights/$f"; missing=1; }
done
if [ "$missing" = 1 ]; then
  echo "Copy the weights/ folder from the Mac first (deploy/README.md, step 4), then run this again."
  exit 1
fi
echo "all there"

step "Node 24"
if ! node --version 2>/dev/null | grep -q '^v24\.'; then
  curl -fsSL https://deb.nodesource.com/setup_24.x | sudo -E bash -
  sudo apt-get install -y -q nodejs
fi
node --version

step "Ollama and $FEEDBACK_MODEL"
command -v ollama >/dev/null || curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
for _ in $(seq 30); do curl -fs http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done
ollama pull "$FEEDBACK_MODEL"

step "Build the web app"
cd "$REPO/web"
npm ci --no-audit --no-fund
npm run build

step "Services (start at boot, restart if they stop)"
for unit in coaching-web.service coaching-worker.service coaching-backup.service coaching-backup.timer; do
  sed -e "s|@REPO@|$REPO|g" -e "s|@USER@|$USER|g" "$REPO/deploy/systemd/$unit" | sudo tee "/etc/systemd/system/$unit" >/dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable -q --now coaching-web.service coaching-worker.service coaching-backup.timer
sudo systemctl restart coaching-web.service coaching-worker.service

step "Tailscale (how coaches reach the server)"
command -v tailscale >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh
if ! tailscale status >/dev/null 2>&1; then
  echo "Sign in to Tailscale - open the link it prints:"
  sudo tailscale up
fi
sudo tailscale serve --bg "$PORT"
HOST=$(tailscale status --json | "$REPO/.venv/bin/python" -c "import json,sys; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))")
if [ "${APP_HOSTS:-}" != "$HOST" ]; then
  sed -i "s|^APP_HOSTS=.*|APP_HOSTS=$HOST|" "$ENV_FILE"
  echo "Set APP_HOSTS=$HOST - rebuilding so forms through it are accepted"
  set -a; . "$ENV_FILE"; set +a
  npm run build
  sudo systemctl restart coaching-web.service
fi

step "Done"
echo "Open https://$HOST and create the head coach account."
echo "The setup code it asks for: $(grep '^SETUP_TOKEN=' "$ENV_FILE" | cut -d= -f2)"
echo "Check on things with: bash deploy/status.sh"
