#!/usr/bin/env bash
# Start FRAM3 like an app: start the local server if it isn't running yet and
# open the editor in its own window (Chromium/Chrome/Brave app mode, else the
# default browser). Used by the FRAM3 entry in the app menu.
set -uo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${FRAM3_PORT:-8733}"
URL="http://127.0.0.1:$PORT"
LOG="${XDG_STATE_HOME:-$HOME/.local/state}/fram3/server.log"
mkdir -p "$(dirname "$LOG")"

up() { curl -fs -o /dev/null "$URL/api/status"; }

if ! up; then
  if [ ! -x "$DIR/.venv/bin/python" ]; then
    notify-send -i fram3 "FRAM3" "Eerste keer opstarten: Python-omgeving wordt gemaakt…" 2>/dev/null || true
    "$DIR/fram3.sh" --no-browser --port "$PORT" >>"$LOG" 2>&1 &
  else
    (cd "$DIR" && nohup .venv/bin/python -m fram3 --no-browser --port "$PORT" >>"$LOG" 2>&1 &)
  fi
  for _ in $(seq 1 120); do up && break; sleep 0.5; done
  if ! up; then
    notify-send -i fram3 "FRAM3" "De server start niet. Zie $LOG" 2>/dev/null || true
    exit 1
  fi
fi

# Own window without browser chrome, when a Chromium-based browser is available.
for b in chromium-browser chromium google-chrome google-chrome-stable brave-browser microsoft-edge; do
  if command -v "$b" >/dev/null; then
    exec "$b" --app="$URL" --class=FRAM3 --user-data-dir="$HOME/.local/share/fram3/browser" >/dev/null 2>&1
  fi
done
for f in org.chromium.Chromium com.google.Chrome com.brave.Browser; do
  if flatpak info "$f" >/dev/null 2>&1; then
    exec flatpak run "$f" --app="$URL" --class=FRAM3 >/dev/null 2>&1
  fi
done
exec xdg-open "$URL"
