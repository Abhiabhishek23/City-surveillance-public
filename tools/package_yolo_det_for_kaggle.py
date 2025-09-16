#!/usr/bin/env python3
"""
Package a YOLO detection dataset into a Kaggle-ready folder structure, include model weights,
normalize data.yaml to relative paths, and optionally create a ZIP for upload.

Usage (example):
  python tools/package_yolo_det_for_kaggle.py \
    --src "Indian_Urban_Dataset_yolo" \
    --out "dist/Indian_Urban_Dataset_yolo_kaggle" \
    --include-weights \
    --zip

This will produce:
  dist/Indian_Urban_Dataset_yolo_kaggle/
    data.yaml
    images/train|val|test/...
    labels/train|val|test/...
    weights/yolov12n.pt (if found)
    weights/yolov8n.pt  (if found)
    README.md (with Kaggle run commands)
    pack_info.json (summary)
  dist/Indian_Urban_Dataset_yolo_kaggle.zip (if --zip)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Dict, List, Tuple

try:
    import yaml  # PyYAML
except Exception as e:
    print("ERROR: PyYAML is required. Add PyYAML to requirements.txt and install it.")
    raise


ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_yaml(p: Path) -> Dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: Dict, p: Path) -> None:
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_split(src_root: Path, out_root: Path, split: str) -> Tuple[int, int, int, int]:
    """Copy images and labels for a split.
    Returns (images_copied, labels_copied, missing_label_count, empty_labels_created).
    If a label is missing for an image, create an empty label file to preserve dataset size.
    """
    src_imgs = src_root / "images" / split
    src_lbls = src_root / "labels" / split
    if not src_imgs.exists():
        return (0, 0, 0, 0)

    out_imgs = out_root / "images" / split
    out_lbls = out_root / "labels" / split
    ensure_dir(out_imgs)
    ensure_dir(out_lbls)

    images_copied = 0
    labels_copied = 0
    missing = 0
    empty_created = 0

    for img in sorted(src_imgs.iterdir()):
        if not img.is_file():
            continue
        if img.suffix.lower() not in ALLOWED_EXTS:
            continue
        stem = img.stem
        lbl = src_lbls / f"{stem}.txt"
        # Copy image
        shutil.copy2(img, out_imgs / img.name)
        images_copied += 1

        # Label: copy if exists else create empty
        out_lbl_path = out_lbls / f"{stem}.txt"
        if lbl.exists():
            shutil.copy2(lbl, out_lbl_path)
            labels_copied += 1
        else:
            missing += 1
            out_lbl_path.write_text("", encoding="utf-8")
            empty_created += 1
            labels_copied += 1

    return (images_copied, labels_copied, missing, empty_created)


def normalize_data_yaml(src_yaml: Path, out_dir: Path) -> Dict:
    """Read data.yaml and rewrite it to use relative paths inside the packaged folder."""
    data = load_yaml(src_yaml) or {}

    # Normalize to relative paths inside packaged folder
    data["path"] = "."
    data["train"] = "images/train"
    data["val"] = "images/val"
    if "test" in data and data.get("test"):
        data["test"] = "images/test"

    # Normalize names and nc
    names = data.get("names")
    nc = data.get("nc")

    # If names is a dict like {0: 'a', 1: 'b'} or {'0': 'a', '1': 'b'}, convert to list sorted by key
    if isinstance(names, dict):
        try:
            # sort keys numerically if possible
            def _key(k):
                try:
                    return int(k)
                except Exception:
                    return str(k)
            names_list = [names[k] for k in sorted(names.keys(), key=_key)]
        except Exception:
            # fallback: values as-is order
            names_list = list(names.values())
        names = [str(x) for x in names_list]
        data["names"] = names

    # If names is not a list but truthy (e.g., a string), wrap or reset
    if names is not None and not isinstance(names, list):
        names = [str(names)]
        data["names"] = names

    # Derive nc if missing or inconsistent
    if nc is None and names is not None:
        nc = len(names)
        data["nc"] = nc
    elif nc is not None:
        try:
            nc = int(nc)
        except Exception:
            # if nc is non-castable, try from names
            nc = len(names) if isinstance(names, list) else 0
        data["nc"] = nc

    # If names missing but nc present, synthesize placeholder names
    if (not names) and isinstance(nc, int) and nc > 0:
        data["names"] = [f"class_{i}" for i in range(nc)]

    dump_yaml(data, out_dir / "data.yaml")
    return data


def copy_weights(repo_root: Path, out_dir: Path, weights: List[str]) -> List[str]:
    found = []
    wdir = out_dir / "weights"
    ensure_dir(wdir)
    for w in weights:
        wpath = (repo_root / w).resolve()
        if wpath.exists():
            shutil.copy2(wpath, wdir / wpath.name)
            found.append(wpath.name)
    return found


def write_run_script(out_dir: Path, dataset_name_hint: str):
        """Create a run_detect.sh helper script for Kaggle notebook shells."""
        sh = f"""#!/usr/bin/env bash
set -euo pipefail
DATA_YAML="/kaggle/input/<your-kaggle-dataset-slug>/data.yaml"
echo "Training with $DATA_YAML"
yolo detect train \
    model=weights/yolov12n.pt \
    data="$DATA_YAML" \
    epochs=100 imgsz=640 batch=16 device=0 workers=4 plots=False \
    project="runs/detect" name="{dataset_name_hint}"
"""
        path = out_dir / "run_detect.sh"
        path.write_text(sh, encoding="utf-8")
        path.chmod(0o755)


def write_readme(out_dir: Path, dataset_name_hint: str, weights_found: List[str], nc: int, names: List[str]):
    slug_placeholder = "<your-kaggle-dataset-slug>"
    model_choice = "weights/yolov12n.pt" if "yolov12n.pt" in weights_found else (
        "weights/yolov8n.pt" if "yolov8n.pt" in weights_found else "yolo11n.pt")

    cmd_train = f"yolo detect train model={model_choice} data=/kaggle/input/{slug_placeholder}/data.yaml epochs=100 imgsz=640 batch=16 device=0 workers=4 plots=False project=\"runs/detect\" name=\"{dataset_name_hint}\""
    cmd_val = f"yolo detect val model=/kaggle/working/runs/detect/{dataset_name_hint}/weights/best.pt data=/kaggle/input/{slug_placeholder}/data.yaml device=0 workers=4"

    readme = f"""
# Kaggle YOLO Detection Dataset: {dataset_name_hint}

This folder is packaged for Kaggle. It contains:
- `data.yaml` with relative paths
- `images/` and `labels/` in YOLO format
- `weights/` with optional starting checkpoints: {', '.join(weights_found) if weights_found else 'none included'}

Classes (nc={nc}): {names}

## How to train on Kaggle (single T4 GPU)

Run this in a Kaggle Notebook cell (shell):

```bash
{cmd_train}
```

Validate at the end:

```bash
{cmd_val}
```

Export best weights:

```bash
cp -f /kaggle/working/runs/detect/{dataset_name_hint}/weights/best.pt /kaggle/working/best_det.pt || true
ls -lh /kaggle/working/*.pt
```

Notes:
- Replace `{slug_placeholder}` with your actual Kaggle Dataset slug after you upload this folder.
- You can switch to `model=weights/yolov8n.pt` if `yolov12n.pt` is not present, or use `yolo11n.pt`.
"""
    (out_dir / "README.md").write_text(readme.strip() + "\n", encoding="utf-8")


def make_zip(out_dir: Path) -> Path:
    zip_path = out_dir.with_suffix("")
    # Ensure .zip suffix
    if not str(zip_path).endswith(".zip"):
        zip_path = Path(str(out_dir) + ".zip")
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(base_name=str(out_dir), format="zip", root_dir=str(out_dir))
    return zip_path


def main():
    ap = argparse.ArgumentParser(description="Package YOLO detection dataset for Kaggle upload")
    ap.add_argument("--src", required=True, help="Source YOLO dataset folder (contains data.yaml, images/, labels/)")
    ap.add_argument("--out", required=True, help="Output folder to create")
    ap.add_argument("--include-weights", action="store_true", help="Include weights/yolov12n.pt and weights/yolov8n.pt if present at repo root")
    ap.add_argument("--no-zip", action="store_true", help="Do not create a zip archive")
    ap.add_argument("--name", default="indian-urban-y12", help="Name to use in README/run directory")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    src_root = Path(args.src).resolve()
    out_root = Path(args.out).resolve()

    if not (src_root / "data.yaml").exists():
        print(f"ERROR: {src_root}/data.yaml not found.")
        sys.exit(1)

    # Prepare output structure
    ensure_dir(out_root)

    # Normalize and write data.yaml
    data_info = normalize_data_yaml(src_root / "data.yaml", out_root)
    nc = int(data_info.get("nc", 0))
    names = data_info.get("names", [])

    # Copy splits
    totals = {}
    missing_total = 0
    empty_total = 0
    for split in ["train", "val", "test"]:
        ic, lc, miss, empty = copy_split(src_root, out_root, split)
        totals[split] = {"images": ic, "labels": lc, "missing_labels": miss, "empty_labels_created": empty}
        missing_total += miss
        empty_total += empty

    # Copy weights if requested
    weights_found: List[str] = []
    if args.include_weights:
        weights_found = copy_weights(repo_root, out_root, ["yolov12n.pt", "yolov8n.pt"])

    # Write pack_info.json
    pack_info = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "src": str(src_root),
        "out": str(out_root),
        "totals": totals,
        "missing_labels_total": missing_total,
        "empty_labels_created_total": empty_total,
        "weights_included": weights_found,
        "nc": nc,
        "names": names,
    }
    (out_root / "pack_info.json").write_text(json.dumps(pack_info, indent=2), encoding="utf-8")

    # README with usage
    write_readme(out_root, args.name, weights_found, nc, names)
    # Run script helper
    write_run_script(out_root, args.name)

    # Zip
    if not args.no_zip:
        zip_path = make_zip(out_root)
        print(f"Packaged and zipped at: {zip_path}")
    else:
        print(f"Packaged at: {out_root}")

    # Summary
    print("Summary:")
    for split, info in totals.items():
        print(f"  {split:5s} -> images: {info['images']}, labels: {info['labels']}, missing_labels: {info['missing_labels']}")
    if weights_found:
        print("  weights:", ", ".join(weights_found))
    if missing_total > 0:
        print(f"WARNING: Skipped {missing_total} images without labels.")


if __name__ == "__main__":
    main()
