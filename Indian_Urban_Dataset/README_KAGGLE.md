# Indian_Urban_Dataset — Kaggle Extras (images untouched)

This folder is your full dataset (train/val/test). We only add helper files to train on Kaggle:

- `weights/` — starting checkpoints here if present: `yolov12n.pt`, `yolov8n.pt`
- `run_classify.sh` — trains image classification (uses this folder directly)
- `run_detect.sh` — fine-tunes YOLO detector (requires a YOLO-format dataset with `data.yaml` attached separately)

## A) Classification training (uses this dataset directly)
Run in a Kaggle Notebook shell cell after attaching THIS dataset:

```bash
# Replace <slug> with your Kaggle dataset slug for this folder
DATA_DIR="/kaggle/input/<slug>/Indian_Urban_Dataset"

yolo classify train \
  model=yolov8n-cls.pt \
  data="$DATA_DIR" \
  epochs=50 batch=128 imgsz=224 device=0 workers=4 plots=False \
  project="runs/classify" name="indian-urban-cls"

# Optional
yolo classify val model="/kaggle/working/runs/classify/indian-urban-cls/weights/best.pt" data="$DATA_DIR" device=0
cp -f /kaggle/working/runs/classify/indian-urban-cls/weights/best.pt /kaggle/working/best_cls.pt || true
```

## B) Detection fine-tuning (YOLOv12n or YOLOv8n)
Attach BOTH datasets:
- This dataset (for `weights/`), and
- Your YOLO-format dataset with `data.yaml` (e.g., `/kaggle/input/<yolo-slug>/data.yaml`).

Then run:

```bash
DATA_YAML="/kaggle/input/<yolo-slug>/data.yaml"
EXTRAS_SLUG="<slug>"  # slug of this dataset

yolo detect train \
  model="/kaggle/input/${EXTRAS_SLUG}/Indian_Urban_Dataset/weights/yolov12n.pt" \
  data="$DATA_YAML" \
  epochs=100 imgsz=640 batch=16 device=0 workers=4 plots=False \
  project="runs/detect" name="indian-urban-y12"

# Validate and export
yolo detect val model="/kaggle/working/runs/detect/indian-urban-y12/weights/best.pt" data="$DATA_YAML" device=0 workers=4
cp -f /kaggle/working/runs/detect/indian-urban-y12/weights/best.pt /kaggle/working/best_det.pt || true
```

Tip: For faster epochs, set `plots=False`, consider a writable clean copy + `cache=True`, and tune `workers`/`batch`.

---
Only helper files are added; images remain untouched.
