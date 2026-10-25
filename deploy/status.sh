#!/usr/bin/env bash
# Is everything running? bash deploy/status.sh

REPO="$(cd "$(dirname "$0")/.." && pwd)"
for unit in coaching-web coaching-worker ollama; do
  printf '%-16s %s\n' "$unit" "$(systemctl is-active "$unit")"
done
printf '%-16s %s\n' "next backup" "$(systemctl list-timers coaching-backup.timer --no-legend | awk '{print $1, $2, $3}')"
printf '%-16s %s\n' "web app" "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3000/login)"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader | sed 's/^/gpu              /'
df -h "$REPO/web/data" | awk 'NR==2 {print "disk             " $4 " free of " $2}'
echo
echo "Worker, last lines (all of it: journalctl -u coaching-worker -f):"
journalctl -u coaching-worker -n 8 --no-pager -o cat 2>/dev/null
