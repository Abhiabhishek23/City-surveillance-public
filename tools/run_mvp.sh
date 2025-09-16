#!/usr/bin/env bash
# tools/run_mvp.sh - One-click trigger to start Backend, Dashboard, and GUI client in parallel
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Prefer project venv Python if present
PY="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
  PY="python3"
fi

MODEL="${MODEL:-yolov12n.pt}"
SOURCE="${SOURCE:-Test samples/7.mp4}"
CAMERA="${CAMERA:-CAM_DEMO}"
THRESH="${THRESH:-5}"

exec "$PY" "$REPO_ROOT/urban_monitoring_deployment.py" \
  --project-root "$REPO_ROOT" \
  --model "$MODEL" \
  --source "$SOURCE" \
  --camera "$CAMERA" \
  --crowd-threshold "$THRESH"
