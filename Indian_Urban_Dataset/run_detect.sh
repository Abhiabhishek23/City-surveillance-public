#!/usr/bin/env bash
set -euo pipefail
# Usage:
# 1) Slugs mode:   bash run_detect.sh <extras-slug> <yolo-dataset-slug>
# 2) Path mode:    bash run_detect.sh </kaggle/input/.../Indian_Urban_Dataset> </kaggle/input/.../data.yaml>

EXTRAS_ARG=${1:-}
YOLO_ARG=${2:-}

if [ -z "${EXTRAS_ARG}" ] || [ -z "${YOLO_ARG}" ]; then
  echo "Usage: bash run_detect.sh <extras-slug|extras-path> <yolo-dataset-slug|data.yaml-path>" >&2
  exit 2
fi

# Resolve extras base folder (contains weights/ and this script)
if [[ "${EXTRAS_ARG}" == /kaggle/* ]]; then
  EXTRAS_BASE="${EXTRAS_ARG}"
else
  EXTRAS_BASE="/kaggle/input/${EXTRAS_ARG}/Indian_Urban_Dataset"
fi

# Resolve data.yaml path
if [[ "${YOLO_ARG}" == *.yaml ]] || [[ -f "${YOLO_ARG}" ]]; then
  DATA_YAML="${YOLO_ARG}"
else
  DATA_YAML="/kaggle/input/${YOLO_ARG}/data.yaml"
fi

if [ ! -f "${DATA_YAML}" ]; then
  echo "data.yaml not found at: ${DATA_YAML}" >&2
  exit 3
fi

# Prefer included yolov12n.pt, else yolov8n.pt
MODEL="${EXTRAS_BASE}/weights/yolov12n.pt"
if [ ! -f "$MODEL" ]; then
  MODEL="${EXTRAS_BASE}/weights/yolov8n.pt"
fi
if [ ! -f "$MODEL" ]; then
  echo "Model weights not found under ${EXTRAS_BASE}/weights (expected yolov12n.pt or yolov8n.pt)" >&2
  exit 4
fi

echo "Using MODEL: $MODEL"
echo "Using DATA:  $DATA_YAML"

yolo detect train \
  model="$MODEL" \
  data="$DATA_YAML" \
  epochs=100 imgsz=640 batch=16 device=0 workers=4 plots=False \
  project="runs/detect" name="indian-urban-y12"
