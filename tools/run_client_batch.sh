#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CLIENT_DIR="$ROOT/kaggle/working/urban_monitoring_deployment"
VENVD="$CLIENT_DIR/.venv_deploy"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

if [ ! -d "$VENVD" ]; then
  echo "Creating venv and installing requirements..."
  python3 -m venv "$VENVD"
  . "$VENVD/bin/activate"
  pip install -r "$CLIENT_DIR/requirements.txt"
else
  . "$VENVD/bin/activate"
fi

for f in "$ROOT/Test samples"/*.mp4; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .mp4)
  echo "Running client on $name"
  python "$CLIENT_DIR/urban_monitor_deploy.py" --model "$CLIENT_DIR/urban_monitor.onnx" --source "$f" --backend "http://127.0.0.1:8000/alerts" --camera "BATCH_${name}" > "$LOGDIR/urban_client_${name}.log" 2>&1
  echo "Finished $name"
done

echo "Batch complete. Logs in $LOGDIR"
