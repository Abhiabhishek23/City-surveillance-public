#!/usr/bin/env zsh
set -euo pipefail
ROOT="${0:A:h}"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

# Start backend in background
cd backend
BACK_HOST=${BACK_HOST:-127.0.0.1}
BACK_PORT=${BACK_PORT:-8000}
echo "Starting backend at http://${BACK_HOST}:${BACK_PORT} ..."
UVICORN_LOG_LEVEL=info uvicorn main:app --host "$BACK_HOST" --port "$BACK_PORT" &
PID=$!
sleep 1

# Open UI
open "http://${BACK_HOST}:${BACK_PORT}/client_ui/"

echo "Backend PID: $PID"
wait $PID || true
