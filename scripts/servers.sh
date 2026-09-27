#!/usr/bin/env bash
# Start/stop the GPU decision-model servers used by benchmark arms.
# Models load on start and are unloaded on stop, so no RAM is held when idle.
#
#   scripts/servers.sh start   # kev.serve (:8009) + decider.serve (:8010), wait for health
#   scripts/servers.sh stop    # kill both + unload ollaya models
#   scripts/servers.sh status  # ports + loaded ollaya models
#
# Override companion-repo locations with KEV_SERVE_DIR / DECIDER_DIR if needed.
set -euo pipefail

KEV_DIR="${KEV_SERVE_DIR:-$HOME/Code/kev-server}"
DECIDER_DIR="${DECIDER_DIR:-$HOME/Code/decider-apple}"
KEV_PORT=8009
DECIDER_PORT=8010

wait_http() {  # url, seconds
  local url=$1 deadline=$((SECONDS + $2))
  until curl -sf -m 2 "$url" >/dev/null 2>&1; do
    (( SECONDS < deadline )) || { echo "timeout waiting for $url"; return 1; }
    sleep 2
  done
}

start() {
  if ! curl -sf -m 2 "http://localhost:$KEV_PORT/v1/models" >/dev/null 2>&1; then
    echo "starting kev.serve (MLX) ..."
    (cd "$KEV_DIR" && nohup uv run --extra serve python -m kev.serve \
        --run jaredpalmer/kev-4b --port "$KEV_PORT" > serve.log 2>&1 &)
    wait_http "http://localhost:$KEV_PORT/v1/models" 300
  fi
  echo "kev.serve ready on :$KEV_PORT"

  if ! curl -sf -m 2 "http://localhost:$DECIDER_PORT/health" >/dev/null 2>&1; then
    echo "starting decider.serve (MPS) ..."
    (cd "$DECIDER_DIR" && DECIDER_MODEL=Mapika/decider-2b nohup uv run uvicorn \
        decider.serve:app --host 127.0.0.1 --port "$DECIDER_PORT" > serve.log 2>&1 &)
    wait_http "http://localhost:$DECIDER_PORT/health" 300
  fi
  echo "decider.serve ready on :$DECIDER_PORT"
}

stop() {
  pkill -f "kev.serve --run jaredpalmer/kev-4b" 2>/dev/null && echo "kev.serve stopped" || echo "kev.serve not running"
  pkill -f "uvicorn decider.serve:app" 2>/dev/null && echo "decider.serve stopped" || echo "decider.serve not running"
  ollaya stop >/dev/null 2>&1 && echo "ollaya models unloaded" || true
}

status() {
  for port in "$KEV_PORT" "$DECIDER_PORT"; do
    if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then echo ":$port listening"; else echo ":$port free"; fi
  done
  ollaya ps 2>/dev/null || true
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|status}"; exit 1 ;;
esac
