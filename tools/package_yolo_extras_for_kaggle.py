#!/usr/bin/env python3
"""
Create a Kaggle-ready "extras" bundle that does NOT include the dataset.
It packages:
  - weights/ (yolov12n.pt, yolov8n.pt if present at repo root)
  - README.md with exact Kaggle commands referencing your dataset slug
  - run_detect.sh helper script (optional)

Usage:
  python tools/package_yolo_extras_for_kaggle.py \
    --out "dist/indian_urban_yolo_extras_kaggle" \
    --name "indian-urban-y12" \
    --dataset-slug "<your-dataset-slug>"

Result:
  dist/indian_urban_yolo_extras_kaggle/
    weights/yolov12n.pt (if found)
    weights/yolov8n.pt  (if found)
    README.md
    run_detect.sh
  dist/indian_urban_yolo_extras_kaggle.zip
"""

from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import textwrap


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def copy_weights(repo_root: Path, out_dir: Path) -> list[str]:
    found = []
    wdir = out_dir / "weights"
    ensure_dir(wdir)
    for w in ["yolov12n.pt", "yolov8n.pt"]:
        wp = (repo_root / w)
        if wp.exists():
            shutil.copy2(wp, wdir / wp.name)
            found.append(wp.name)
    return found


def write_readme(out_dir: Path, dataset_slug: str, run_name: str, weights_found: list[str]):
    model_choice = "weights/yolov12n.pt" if "yolov12n.pt" in weights_found else (
        "weights/yolov8n.pt" if "yolov8n.pt" in weights_found else "yolo11n.pt")

    content = f"""
# Kaggle Extras for YOLO Detection: {run_name}

This bundle intentionally does NOT contain the dataset. Attach your dataset in Kaggle as:
  /kaggle/input/{dataset_slug}

Included:
- weights: {', '.join(weights_found) if weights_found else 'none (you can use yolo11n.pt)'}
- run_detect.sh helper (optional)

Train (detection) on Kaggle (shell cell):
```bash
DATA_YAML="/kaggle/input/{dataset_slug}/data.yaml"
yolo detect train \
  model={model_choice} \
  data="$DATA_YAML" \
  epochs=100 imgsz=640 batch=16 device=0 workers=4 plots=False \
  project="runs/detect" name="{run_name}"
```

Resume later:
```bash
yolo detect train resume=True project="runs/detect" name="{run_name}" device=0 plots=False
```

Validate and export:
```bash
yolo detect val model="/kaggle/working/runs/detect/{run_name}/weights/best.pt" data="$DATA_YAML" device=0 workers=4
cp -f /kaggle/working/runs/detect/{run_name}/weights/best.pt /kaggle/working/best_det.pt || true
ls -lh /kaggle/working/*.pt
```

Notes:
- Ensure your dataset contains images/ and labels/ with matching pairs and a valid data.yaml.
- You can switch to model=weights/yolov8n.pt or yolo11n.pt if yolov12n.pt isn't present.
"""
    (out_dir / "README.md").write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")


def write_run_script(out_dir: Path, dataset_slug: str, run_name: str, weights_found: list[str]):
    model_choice = "weights/yolov12n.pt" if "yolov12n.pt" in weights_found else (
        "weights/yolov8n.pt" if "yolov8n.pt" in weights_found else "yolo11n.pt")
    sh = f"""#!/usr/bin/env bash
set -euo pipefail
DATA_YAML="/kaggle/input/{dataset_slug}/data.yaml"
echo "Training with $DATA_YAML"
yolo detect train \
  model={model_choice} \
  data="$DATA_YAML" \
  epochs=100 imgsz=640 batch=16 device=0 workers=4 plots=False \
  project="runs/detect" name="{run_name}"
"""
    runfile = out_dir / "run_detect.sh"
    runfile.write_text(sh, encoding="utf-8")
    runfile.chmod(0o755)


def make_zip(out_dir: Path) -> Path:
    # shutil.make_archive expects base name without suffix
    base = str(out_dir)
    zip_base = base
    shutil.make_archive(base_name=zip_base, format="zip", root_dir=str(out_dir))
    return Path(zip_base + ".zip")


def main():
    ap = argparse.ArgumentParser(description="Package YOLO detection EXTRAS for Kaggle (no dataset)")
    ap.add_argument("--out", required=True, help="Output folder to create")
    ap.add_argument("--name", default="indian-urban-y12", help="Run name used in README/commands")
    ap.add_argument("--dataset-slug", default="<your-dataset-slug>", help="Kaggle dataset slug to reference in README")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.out).resolve()
    ensure_dir(out_dir)

    weights_found = copy_weights(repo_root, out_dir)
    write_readme(out_dir, args.dataset_slug, args.name, weights_found)
    write_run_script(out_dir, args.dataset_slug, args.name, weights_found)

    z = make_zip(out_dir)
    print("Built extras zip:", z)
    print("Contains weights:", ", ".join(weights_found) if weights_found else "none")


if __name__ == "__main__":
    main()
