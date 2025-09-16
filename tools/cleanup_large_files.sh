#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKUP=~/cs_backup
mkdir -p "$BACKUP"
echo "Moving large known duplicates to $BACKUP (non-destructive)"
# Examples (only move if present)
[ -f "$ROOT/Data/combined_dataset.zip" ] && mv -n "$ROOT/Data/combined_dataset.zip" "$BACKUP/" && echo moved Data/combined_dataset.zip
[ -d "$ROOT/kaggle/working/urban_monitoring_deployment/path" ] && mv -n "$ROOT/kaggle/working/urban_monitoring_deployment/path" "$BACKUP/" && echo moved embedded venv
[ -f "$ROOT/Edge/yolov8n.pt.bak" ] && mv -n "$ROOT/Edge/yolov8n.pt.bak" "$BACKUP/" && echo moved Edge/yolov8n.pt.bak
echo "Done. Review $BACKUP and delete if OK."
