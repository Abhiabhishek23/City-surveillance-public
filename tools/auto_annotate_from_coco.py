#!/usr/bin/env python3
"""
Auto-annotate images in Indian_Urban_Dataset_yolo using a COCO-pretrained YOLO model.
- Maps only overlapping classes from COCO -> dataset: person, car, bicycle, bus, truck, motorbike, boat
- Writes YOLO txt labels into labels/{train,val} aligned with data.yaml class IDs

Usage:
  python tools/auto_annotate_from_coco.py \
    --dataset-root Indian_Urban_Dataset_yolo \
    --weights yolov8n.pt \
    --conf 0.35 --iou 0.5 --imgsz 640 --batch 16 --device cpu
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import yaml

try:
    from ultralytics import YOLO
except Exception as e:
    print("[ERROR] ultralytics not installed or failed to import: ", e, file=sys.stderr)
    print("Try: pip install ultralytics", file=sys.stderr)
    sys.exit(1)


COCO_TO_DATASET = {
    # COCO name -> dataset name
    "person": "person",
    "car": "car",
    "bicycle": "bicycle",
    "bus": "bus",
    "truck": "truck",
    # COCO uses "motorcycle"; dataset uses "motorbike"
    "motorcycle": "motorbike",
    "boat": "boat",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_dataset_names(data_yaml_path: Path) -> List[str]:
    with data_yaml_path.open("r") as f:
        data = yaml.safe_load(f)
    names = data.get("names")
    if isinstance(names, dict):
        # if mapping like {0: name0, 1: name1, ...}
        names = [names[k] for k in sorted(names.keys())]
    if not isinstance(names, list):
        raise ValueError(f"Invalid names in {data_yaml_path}")
    return names


def find_images(split_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in split_dir.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            files.append(p)
    files.sort()
    return files


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def xyxy_to_xywhn(xyxy, w: int, h: int):
    x1, y1, x2, y2 = map(float, xyxy)
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = x1 + bw / 2.0
    yc = y1 + bh / 2.0
    # normalize
    return (
        xc / max(1, w),
        yc / max(1, h),
        bw / max(1, w),
        bh / max(1, h),
    )


def annotate_split(model: YOLO, dataset_root: Path, names: List[str], split: str, conf: float, iou: float, imgsz: int, batch: int, device: str):
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    ensure_dir(labels_dir)

    name_to_id: Dict[str, int] = {n: i for i, n in enumerate(names)}

    images = find_images(images_dir)
    if not images:
        print(f"[WARN] No images found in {images_dir}")
        return

    print(f"[INFO] Split={split} images={len(images)} -> annotating...")

    # Batch inference
    for i in range(0, len(images), batch):
        batch_paths = images[i : i + batch]
        # Call model; pass list of str paths
        results = model(
            [str(p) for p in batch_paths],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        # results is a list aligned with batch_paths
        for img_path, res in zip(batch_paths, results):
            h, w = res.orig_shape  # (h, w)
            lines: List[str] = []
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls_idx = int(b.cls)
                    coco_name = model.names.get(cls_idx, str(cls_idx))
                    ds_name = COCO_TO_DATASET.get(coco_name)
                    if ds_name is None:
                        continue  # skip non-overlapping classes
                    if ds_name not in name_to_id:
                        continue  # dataset doesn't contain this mapped class
                    ds_id = name_to_id[ds_name]
                    xyxy = b.xyxy[0].tolist()  # [x1,y1,x2,y2] in pixels
                    x, y, bw, bh = xyxy_to_xywhn(xyxy, w, h)
                    # clip to [0,1]
                    x = min(max(x, 0.0), 1.0)
                    y = min(max(y, 0.0), 1.0)
                    bw = min(max(bw, 0.0), 1.0)
                    bh = min(max(bh, 0.0), 1.0)
                    # filter tiny boxes
                    if bw <= 0 or bh <= 0:
                        continue
                    lines.append(f"{ds_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

            # write label file (overwrite)
            rel = img_path.relative_to(images_dir)
            lbl_path = labels_dir / rel.with_suffix(".txt")
            ensure_dir(lbl_path.parent)
            lbl_path.write_text("\n".join(lines))  # empty file if no lines

    print(f"[DONE] Annotated split={split} -> labels in {labels_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("Indian_Urban_Dataset_yolo"))
    ap.add_argument("--data-yaml", type=Path, default=None, help="Optional explicit path to data.yaml")
    ap.add_argument("--weights", type=str, default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    dataset_root: Path = args.dataset_root
    data_yaml_path = args.data_yaml or (dataset_root / "data.yaml")
    if not data_yaml_path.exists():
        print(f"[ERROR] data.yaml not found at {data_yaml_path}", file=sys.stderr)
        sys.exit(2)

    names = load_dataset_names(data_yaml_path)
    print(f"[INFO] Dataset classes ({len(names)}): {names}")

    model = YOLO(args.weights)
    # print model class names
    try:
        print(f"[INFO] Model classes ({len(model.names)}): {model.names}")
    except Exception:
        pass

    for split in ("train", "val"):
        annotate_split(model, dataset_root, names, split, args.conf, args.iou, args.imgsz, args.batch, args.device)

    print("[OK] Auto-annotation complete.")


if __name__ == "__main__":
    main()
