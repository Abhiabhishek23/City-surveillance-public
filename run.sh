#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VENV=".venv"

setup() {
  if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
  fi
  source "$VENV/bin/activate"
  pip install --upgrade pip
  pip install -r requirements.txt || pip install -r ci-requirements.txt
  echo "Python deps installed."
}

start() {
  source "$VENV/bin/activate" || true
  (cd backend && UVICORN_LOG_LEVEL=info uvicorn main:app --host 127.0.0.1 --port 8000 & echo $! > ../.pids_backend)
  echo "Backend starting on http://127.0.0.1:8000"
}

open_ui() {
  python3 - "$@" <<'PY'
import webbrowser, time
urls = [
  'http://127.0.0.1:8000/welcome',
]
for u in urls:
  webbrowser.open(u)
  time.sleep(0.2)
PY
}

stop() {
  if [ -f .pids_backend ]; then
    kill $(cat .pids_backend) 2>/dev/null || true
    rm -f .pids_backend
    echo "Backend stopped"
  fi
}

logs() {
  lsof -i :8000 || true
}

case "${1:-}" in
  setup) setup ;;
  start) start ;;
  open) open_ui ;;
  stop) stop ;;
  logs) logs ;;
  *) echo "Usage: $0 {setup|start|open|stop|logs}" ; exit 1 ;;
esac
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
BACKEND_DIR="$SCRIPT_DIR/backend"
PID_FILE="$SCRIPT_DIR/.backend.pid"
HOST="127.0.0.1"
PORT="8000"

usage(){
  cat <<EOF
Usage: ./run.sh <command>

Commands:
  setup         Create venv and install requirements (root and backend)
  start         Start FastAPI backend (background) on $HOST:$PORT
  stop          Stop backend started by this script
  status        Print backend status and health
  open          Open Client and Admin UIs in your default browser
  logs          Tail recent backend logs when started via this script (best-effort)

Examples:
  ./run.sh setup
  ./run.sh start && ./run.sh open
  ./run.sh stop
EOF
}

ensure_venv(){
  if [[ ! -x "$PY" ]]; then
    echo "[setup] Creating virtualenv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
}

cmd_setup(){
  ensure_venv
  source "$VENV_DIR/bin/activate"
  echo "[setup] Upgrading pip"
  "$PIP" install --upgrade pip
  if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
    echo "[setup] Installing root requirements.txt"
    "$PIP" install -r "$SCRIPT_DIR/requirements.txt"
  fi
  if [[ -f "$BACKEND_DIR/requirements.txt" ]]; then
    echo "[setup] Installing backend/requirements.txt"
    "$PIP" install -r "$BACKEND_DIR/requirements.txt"
  fi
  if [[ ! -f "$BACKEND_DIR/.env" && -f "$BACKEND_DIR/.env.example" ]]; then
    echo "[setup] Creating backend/.env from .env.example"
    cp "$BACKEND_DIR/.env.example" "$BACKEND_DIR/.env"
  fi
  echo "[setup] Done"
}

is_running(){
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(cat "$PID_FILE" || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

cmd_start(){
  ensure_venv
  if is_running; then
    echo "[start] Backend already running (pid $(cat "$PID_FILE"))"
    exit 0
  fi
  source "$VENV_DIR/bin/activate"
  pushd "$BACKEND_DIR" >/dev/null
  echo "[start] Launching backend on http://$HOST:$PORT"
  UVICORN_LOG_LEVEL=${UVICORN_LOG_LEVEL:-info} nohup uvicorn main:app --host "$HOST" --port "$PORT" --reload > "$SCRIPT_DIR/.backend.out" 2>&1 &
  echo $! > "$PID_FILE"
  popd >/dev/null
  echo "[start] PID $(cat "$PID_FILE")"
  echo -n "[start] Waiting for health"; for i in {1..40}; do
    sleep 0.25
    if curl -fsS "http://$HOST:$PORT/" >/dev/null; then echo " OK"; break; else echo -n "."; fi
  done
  echo "[start] Client UI: http://$HOST:$PORT/client_ui"
  echo "[start] Admin UI:  http://$HOST:$PORT/admin"
}

cmd_stop(){
  if ! is_running; then
    echo "[stop] Backend not running"
    exit 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  echo "[stop] Stopping PID $pid"
  kill "$pid" 2>/dev/null || true
  sleep 0.5
  if kill -0 "$pid" 2>/dev/null; then
    echo "[stop] Force killing PID $pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "[stop] Done"
}

cmd_status(){
  if is_running; then
    echo "[status] Backend running (pid $(cat "$PID_FILE"))"
  else
    echo "[status] Backend not running"
  fi
  if curl -fsS "http://$HOST:$PORT/" >/dev/null; then
    echo "[status] Health: OK"
  else
    echo "[status] Health: unavailable"
  fi
}

cmd_open(){
  local url_client="http://$HOST:$PORT/client_ui"
  local url_admin="http://$HOST:$PORT/admin"
  if command -v open >/dev/null 2>&1; then
    open "$url_client"
    open "$url_admin"
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url_client"
    xdg-open "$url_admin"
  else
    echo "[open] Please open these URLs:"
    echo "  $url_client"
    echo "  $url_admin"
  fi
}

cmd_logs(){
  if [[ -f "$SCRIPT_DIR/.backend.out" ]]; then
    tail -n 100 -f "$SCRIPT_DIR/.backend.out"
  else
    echo "[logs] No log file yet. Start the server first with ./run.sh start"
  fi
}

case "${1:-}" in
  setup)  cmd_setup ;;
  start)  cmd_start ;;
  stop)   cmd_stop ;;
  status) cmd_status ;;
  open)   cmd_open ;;
  logs)   cmd_logs ;;
  *) usage; exit 1 ;;
esac
