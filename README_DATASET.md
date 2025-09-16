# Encroachment Dataset Scaffold

This repository now includes a scaffold for building a 5‑class encroachment dataset:

Classes (index order in `dataset/data.yaml`):
0. `Permanent_Legal`
1. `Permanent_Illegal`
2. `Temporary_Legal`
3. `Temporary_Illegal`
4. `Natural_Area`

## Folder Structure
```
dataset/
  data.yaml
  images/
    train/
    val/
  labels/
    train/
    val/
```
Each label file: YOLO format lines -> `<class_id> <cx> <cy> <w> <h>` (normalized 0–1).

## Subclasses and Ontology → 5 Classes
- Define rich attributes (permanence, zone, permit_status, structure_type, area_type) in `dataset/ontology.yaml`.
- Annotate using attributes in a JSONL file, then flatten to YOLO classes via the mapper:
  - Use `tools/ontology_mapper.py` to apply `mapping_rules` and emit class ids from `class_ids`.

JSONL example (one record per image):
```
{"image": "images/train/img_0001.jpg", "boxes": [
 {"bbox": [0.42, 0.55, 0.30, 0.28], "attributes": {"permanence":"temporary","zone":"road_footpath","permit_status":"unknown","area_type":"built","structure_type":"stall"}}
]}
```
Run the mapper:
```
python tools/ontology_mapper.py --jsonl dataset/samples/annotations_template.jsonl \
  --out dataset/labels/train --images-root dataset/images/train
```

## Adding Data
1. Place training images into `dataset/images/train/` and validation images into `dataset/images/val/`.
2. Create corresponding label `.txt` files under `dataset/labels/train/` or `dataset/labels/val/` using identical stem names.
3. Keep unannotated images out or label them with only relevant structures; avoid mixing background unless you intentionally include Natural_Area boxes.

## Natural_Area Strategy
Two options:
- Do NOT label large background: rely on detector learning background implicitly.
- OR add a few bounding boxes for major open zones to encourage discrimination (advanced; start simple first).

## Preparing a Kaggle Upload
Two reliable options:

1) Automated (recommended) — Scripted upload via Kaggle API
- Prereq: Create Kaggle API token from https://www.kaggle.com/settings/account → Create New API Token
- Place `kaggle.json` in `~/.kaggle/kaggle.json` and set permissions:
  ```bash
  mkdir -p ~/.kaggle
  mv ~/Downloads/kaggle.json ~/.kaggle/
  chmod 600 ~/.kaggle/kaggle.json
  ```
- Install deps in your venv:
  ```bash
  pip install -r requirements.txt
  ```
- Upload your dataset folder (e.g., `dataset/`) as a Kaggle dataset:
  ```bash
  python tools/kaggle_upload.py \
    --dataset-dir dataset \
    --owner <your_kaggle_username> \
    --slug urban-encroachment-yolo \
    --title "Urban Encroachment YOLO Dataset" \
    --private \
    --message "Initial upload"
  ```
  Notes:
  - Re-run with a new `--message` to version an existing dataset.
  - Use `--public` if you want it public.
  - By default it zips everything into `data.zip` and includes metadata + README for Kaggle.

2) Manual fallback — Zip locally, then upload via Kaggle UI
- Create a clean zip (avoid hidden files and macOS artifacts):
  ```bash
  cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
  zip -r kaggle_dataset.zip dataset \
    -x "*/__pycache__/*" "*.pyc" "*~" "*/.DS_Store" "*/.git/*" "*/__MACOSX/*"
  ```
- Go to https://www.kaggle.com/datasets → Create Dataset → Upload Files → select `kaggle_dataset.zip`.
- If uploading a whole folder via UI fails, uploading a single `.zip` file is the reliable path.

## Kaggle Training Notebook
Use `kaggle/notebooks/train_encroachment_yolo.ipynb`:
- Installs ultralytics, shapely, pyproj
- Points YOLO to `dataset/data.yaml`
- Trains and saves weights (`best.pt`) to Kaggle output

## Training (Example Ultralytics)
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # or yolov8s / yolov12 variant when available
model.train(data='dataset/data.yaml', epochs=80, imgsz=640, batch=32, lr0=0.002, patience=20)
model.val()
model.export(format='onnx')
```

### Using the dataset in a Kaggle notebook
After adding the dataset to your notebook (Add data → Your datasets), Kaggle mounts it under `/kaggle/input/<owner>-<slug>/`.

If you uploaded via the script, the data is inside `data.zip`:
```python
import os, zipfile
root = '/kaggle/input/<owner>-<slug>'
zip_path = os.path.join(root, 'data.zip')
with zipfile.ZipFile(zip_path, 'r') as zf:
  zf.extractall('/kaggle/working')

data_yaml = '/kaggle/working/dataset/data.yaml'
```
Pass `data_yaml` to your YOLO trainer.

## Active Learning Loop
- After first pass, collect low‑confidence or misclassified images.
- Add them back into train set, re‑train with lower learning rate.

## GIS / Legality Fusion (Later)
This dataset only captures *appearance*. Final legality = detector output + geofence overlays.

## Next Steps
- Populate a small pilot set (≈200 images) to smoke test pipeline.
- Validate label consistency (no class drift).
- Expand toward several thousand images per class before production.

---
Feel free to extend this README with annotation guidelines or class decision trees.
