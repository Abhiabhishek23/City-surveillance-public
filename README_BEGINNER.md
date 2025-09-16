# Quick Start (Beginner Friendly)

Follow these steps to build your first encroachment dataset and train a model.

## 1) Collect Images
- Put your images into `dataset/images/train/` (you can start with ~100–200 images).
- Keep a separate small set (20–40) for `dataset/images/val/`.

## 2) Choose Annotation Style
Option A (Simple): Label directly into 5 classes.
- For each image `dataset/images/*/*.jpg`, create a `.txt` file in `dataset/labels/*/` with the same name.
- Lines are: `class cx cy w h` (all normalized 0–1). See `dataset/data.yaml` for class order.

Option B (Recommended): Use attributes and auto-map to 5 classes.
- Edit `dataset/samples/annotations_template.jsonl` and add one JSON line per image:
  - `image`: path relative to repo root (e.g., `images/train/pic.jpg`)
  - `boxes`: list of objects with `bbox` (cx,cy,w,h) and `attributes` (permanence, zone, permit_status, area_type, structure_type).
- Run the mapper to generate YOLO labels:
```
python tools/ontology_mapper.py --jsonl dataset/samples/annotations_template.jsonl \
  --out dataset/labels/train --images-root dataset/images/train
```
- Validate:
```
python tools/validate_dataset.py --data dataset/data.yaml --labels dataset/labels/train
```

## 3) Optional: GIS Zones
- Update `dataset/zones.geojson` with your city polygons (EPSG:4326).
- Use the Kaggle notebook (Section 5) to classify points into zones, which you can then use as `zone` attribute values.

## 4) Package and Upload to Kaggle
- Zip the dataset:
```
python tools/prepare_kaggle_dataset.py --out kaggle_dataset.zip
```
- Create a Private Dataset on Kaggle and upload the zip.

## 5) Train on Kaggle
- Open `kaggle/notebooks/train_encroachment_yolo.ipynb` on Kaggle.
- In Section 3, set `DATASET_DIR = '/kaggle/input/<YOUR_DATASET_NAME>/dataset'`.
- Run Section 11 to train; it saves `/kaggle/working/weights/best.pt`.

## 6) Use Your Weights
- Download `best.pt` from Kaggle and point your client/orchestrator to it. If you want, I’ll wire the client to accept a `--model` path.

Tips:
- Keep class order stable (see `dataset/data.yaml`).
- Start small, iterate quickly, and validate labels with the validator script.
