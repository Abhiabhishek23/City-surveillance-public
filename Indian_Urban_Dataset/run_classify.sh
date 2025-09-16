#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run_classify.sh <slug>
SLUG=${1:-<slug>}
DATA_DIR="/kaggle/input/${SLUG}/Indian_Urban_Dataset"

yolo classify train \
  model=yolov8n-cls.pt \
  data="$DATA_DIR" \
  epochs=50 batch=128 imgsz=224 device=0 workers=4 plots=False \
  project="runs/classify" name="indian-urban-cls"
