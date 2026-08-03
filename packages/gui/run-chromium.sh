#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-5173}"
CHROMIUM="${CHROMIUM:-chromium}"
PAGE="${1:-/}"
PROFILE_DIR="${CHOMATO_CHROME_PROFILE:-/tmp/chomato-chromium-${PORT}}"

cleanup() {
  if [[ -n "${VITE_PID:-}" ]]; then
    kill "$VITE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

bunx vite --config "$HERE/vite.config.ts" --host 127.0.0.1 --port "$PORT" &
VITE_PID=$!

# Wait until Vite accepts connections. No GPU work happens here.
for _ in $(seq 1 100); do
  if curl -fsS "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
    break
  fi
  sleep 0.1
done

"$CHROMIUM" \
  --disable-gpu-sandbox \
  --user-data-dir="$PROFILE_DIR" \
  --no-first-run \
  --no-default-browser-check \
  "http://127.0.0.1:${PORT}${PAGE}"
