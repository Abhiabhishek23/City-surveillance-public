# Indian Urban Dataset - Classification

This repository contains a cleaned, deduplicated classification dataset with `train/`, `val/`, and `test/` splits.

## Structure
- `train/`, `val/`, `test/`: class folders (flattened), images inside
- `annotations/`:
  - `classes.txt`, `class_to_index.json`
  - `train.csv`, `val.csv`, `test.csv`
  - `dataset.yaml` (names + paths)
  - `dedupe_report.json`
- `scripts/`: helper scripts
  - `prepare_dataset.py` — split, dedupe, flatten, annotate
  - `inspect_dataset.py` — quick stats
  - `export_for_ultralytics.py` — export/symlink dataset for Ultralytics training

## Rebuild annotations only
With raw removed, you can rebuild annotations from existing splits:

```bash
python3 scripts/prepare_dataset.py --root . --val-ratio 0.2 --test-ratio 0.1
```

## Export for Ultralytics (YOLOv12n classification)
Create an export folder with a standard dataset.yaml:

```bash
python3 scripts/export_for_ultralytics.py --root . --out ../UrbanDatasetExport --mode symlink
```

- Use `--mode copy` if symlinks are not desired.

### Train with Ultralytics
Install Ultralytics and train a small model (e.g., YOLOv12n) for classification:

```bash
python -m pip install -U ultralytics
ultralytics train cls --model yolov12n.pt --data ../UrbanDatasetExport/dataset.yaml --epochs 50 --imgsz 224 --batch 64
```

Common flags:
- `--lr0 0.001 --optimizer adamw`
- `--patience 10` early stopping
- `--device 0` for a single GPU

## Notes
- Dataset is deduplicated by content; `dedupe_report.json` logs any duplicates within or across splits.
- Splits are deterministic with `--seed` in the prepare script.
- Nested class folders (e.g., `People/Pedestrians/walking`) were flattened (e.g., `People_Pedestrians_walking`).
