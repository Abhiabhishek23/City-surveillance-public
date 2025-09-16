# Kaggle YOLO Detection Dataset: indian-urban-y12

This folder is packaged for Kaggle. It contains:
- `data.yaml` with relative paths
- `images/` and `labels/` in YOLO format
- `weights/` with optional starting checkpoints: yolov12n.pt, yolov8n.pt

Classes (nc=18): ['car', 'person', 'barricade', 'tent_pandal', 'garbage_pile', 'hawker', 'truck', 'bus', 'open_fire', 'bicycle', 'excavator', 'loudspeaker', 'motorbike', 'sand_pile', 'idol_statue', 'boat', 'banner_hoarding', 'tractor']

## How to train on Kaggle (single T4 GPU)

Run this in a Kaggle Notebook cell (shell):

```bash
yolo detect train model=weights/yolov12n.pt data=/kaggle/input/<your-kaggle-dataset-slug>/data.yaml epochs=100 imgsz=640 batch=16 device=0 workers=4 plots=False project="runs/detect" name="indian-urban-y12"
```

Validate at the end:

```bash
yolo detect val model=/kaggle/working/runs/detect/indian-urban-y12/weights/best.pt data=/kaggle/input/<your-kaggle-dataset-slug>/data.yaml device=0 workers=4
```

Export best weights:

```bash
cp -f /kaggle/working/runs/detect/indian-urban-y12/weights/best.pt /kaggle/working/best_det.pt || true
ls -lh /kaggle/working/*.pt
```

Notes:
- Replace `<your-kaggle-dataset-slug>` with your actual Kaggle Dataset slug after you upload this folder.
- You can switch to `model=weights/yolov8n.pt` if `yolov12n.pt` is not present, or use `yolo11n.pt`.
