#!/usr/bin/env bash
set -euo pipefail
DATA_YAML="/kaggle/input/<your-kaggle-dataset-slug>/data.yaml"
echo "Training with $DATA_YAML"
yolo detect train     model=weights/yolov12n.pt     data="$DATA_YAML"     epochs=100 imgsz=640 batch=16 device=0 workers=4 plots=False     project="runs/detect" name="indian-urban-y12"
