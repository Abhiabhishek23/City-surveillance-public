#!/usr/bin/env python3
"""
Auto-annotate YOLO dataset (train/val/test) using a COCO-pretrained YOLO model,
mapping COCO classes into your dataset's `objects_data.yaml` classes.

- Reads images from <dataset_root>/images/{train,val,test}
- Writes YOLO .txt labels to <dataset_root>/labels/{train,val,test}
  (skips existing non-empty labels unless --overwrite)

Default class mapping (COCO -> objects_data.yaml):
  person      -> pedestrian_walking
  bicycle     -> vehicle_bike
  motorcycle  -> vehicle_bike
  car         -> vehicle_car
  bus         -> vehicle_truck_bus
  truck       -> vehicle_truck_bus
  boat        -> boat

Usage examples:
  python tools/auto_annotate_objects_from_coco.py \
    --dataset-root Indian_Urban_Dataset_yolo_prepared \
    --data dataset/objects_data.yaml --weights yolov8n.pt --device auto --batch 16 --conf 0.35

  # Overwrite labels and include test split
  python tools/auto_annotate_objects_from_coco.py --dataset-root dataset --data dataset/objects_data.yaml --overwrite

Notes:
 - Only overlapping classes are labeled; other domain-specific classes (e.g., vendor carts, pandals)
   will remain empty and should be manually annotated later.
 - Ensure `ultralytics` is installed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Iterable

import yaml


def detect_device_arg(want: str) -> str:
    """Resolve device='auto' into '0' (CUDA) or 'mps' (Apple Silicon) or 'cpu'."""
    if want and want != 'auto':
        return want
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return '0'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'


def load_dataset_names(data_yaml_path: Path) -> List[str]:
    with data_yaml_path.open('r') as f:
        data = yaml.safe_load(f)
    names = data.get('names')
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names.keys())]
    if not isinstance(names, list):
        raise ValueError(f"Invalid names in {data_yaml_path}")
    return names


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def find_images(split_dir: Path) -> List[Path]:
    files: List[Path] = []
    for p in split_dir.rglob('*'):
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
    return (
        xc / max(1, w),
        yc / max(1, h),
        bw / max(1, w),
        bh / max(1, h),
    )


DEFAULT_COCO_TO_DATASET = {
    'person': 'pedestrian_walking',
    'bicycle': 'vehicle_bike',
    'motorcycle': 'vehicle_bike',
    'car': 'vehicle_car',
    'bus': 'vehicle_truck_bus',
    'truck': 'vehicle_truck_bus',
    'boat': 'boat',
}


def determine_mapping(names: List[str]) -> Dict[str, str]:
    """Pick a COCO->dataset mapping based on available dataset class names.

    - If dataset contains standard names like 'person','car','bus','truck','bicycle','motorbike','boat',
      map directly to those.
    - Else fallback to object-first schema (vehicle_* and pedestrian_*).
    """
    std = {'person', 'car', 'bus', 'truck', 'bicycle', 'motorbike', 'boat'}
    if std.intersection(set(names)):
        return {
            'person': 'person',
            'bicycle': 'bicycle',
            'motorcycle': 'motorbike',
            'car': 'car',
            'bus': 'bus',
            'truck': 'truck',
            'boat': 'boat',
        }
    return DEFAULT_COCO_TO_DATASET


def annotate_split(model, dataset_root: Path, names: List[str], split: str, conf: float, iou: float, imgsz: int, batch: int, device: str, overwrite: bool, limit: int, mapping: Dict[str, str]):
    images_dir = dataset_root / 'images' / split
    labels_dir = dataset_root / 'labels' / split
    if not images_dir.exists():
        print(f"[INFO] Split '{split}' not found at {images_dir}; skipping.")
        return
    ensure_dir(labels_dir)

    name_to_id: Dict[str, int] = {n: i for i, n in enumerate(names)}

    images = find_images(images_dir)
    if limit > 0:
        images = images[:limit]
    if not images:
        print(f"[WARN] No images found in {images_dir}")
        return

    print(f"[INFO] Split={split} images={len(images)} -> annotating (device={device})...")

    for i in range(0, len(images), batch):
        batch_paths = images[i : i + batch]
        results = model(
            [str(p) for p in batch_paths],
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )
        for img_path, res in zip(batch_paths, results):
            h, w = res.orig_shape  # (h, w)
            # Determine label path aligned to images/<split>/... structure
            rel = img_path.relative_to(images_dir)
            lbl_path = labels_dir / rel.with_suffix('.txt')
            ensure_dir(lbl_path.parent)

            if lbl_path.exists() and lbl_path.stat().st_size > 0 and not overwrite:
                continue

            lines: List[str] = []
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls_idx = int(b.cls)
                    # model.names: id->name
                    coco_name = model.names.get(cls_idx, str(cls_idx))
                    ds_name = mapping.get(coco_name)
                    if ds_name is None:
                        continue
                    if ds_name not in name_to_id:
                        continue
                    ds_id = name_to_id[ds_name]
                    xyxy = b.xyxy[0].tolist()
                    x, y, bw, bh = xyxy_to_xywhn(xyxy, w, h)
                    # clip to [0,1] and filter tiny boxes
                    x = min(max(x, 0.0), 1.0)
                    y = min(max(y, 0.0), 1.0)
                    bw = min(max(bw, 0.0), 1.0)
                    bh = min(max(bh, 0.0), 1.0)
                    if bw <= 0 or bh <= 0:
                        continue
                    lines.append(f"{ds_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")

            lbl_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description='Auto-annotate YOLO dataset with COCO-pretrained model (objects mapping)')
    ap.add_argument('--dataset-root', type=Path, required=True, help='YOLO dataset root containing images/ and labels/')
    ap.add_argument('--data', type=Path, default=Path('dataset/objects_data.yaml'), help='Path to data.yaml with names list')
    ap.add_argument('--weights', type=str, default='yolov8n.pt', help='Pretrained YOLO weights (COCO)')
    ap.add_argument('--splits', type=str, default='train,val,test', help='Comma-separated splits to process if present')
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', type=str, default='auto', help="'auto'|'cpu'|'mps'|'0'|...) ")
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing non-empty labels')
    ap.add_argument('--limit', type=int, default=0, help='Limit number of images per split (0=all)')
    args = ap.parse_args()

    # Lazy import ultralytics to speed help
    try:
        from ultralytics import YOLO
    except Exception as e:
        print('[ERROR] ultralytics not installed or failed to import: ', e, file=sys.stderr)
        print('Try: pip install ultralytics', file=sys.stderr)
        sys.exit(1)

    data_yaml_path = args.data
    if not data_yaml_path.exists():
        print(f"[ERROR] data.yaml not found at {data_yaml_path}", file=sys.stderr)
        sys.exit(2)

    names = load_dataset_names(data_yaml_path)
    print(f"[INFO] Dataset classes ({len(names)}): {names}")

    device = detect_device_arg(args.device)
    print(f"[INFO] Using device: {device}")

    model = YOLO(args.weights)
    try:
        print(f"[INFO] Model classes ({len(model.names)}): {model.names}")
    except Exception:
        pass

    mapping = determine_mapping(names)
    dataset_root: Path = args.dataset_root
    for split in [s.strip() for s in args.splits.split(',') if s.strip()]:
        annotate_split(
            model,
            dataset_root,
            names,
            split,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            overwrite=args.overwrite,
            limit=args.limit,
            mapping=mapping,
        )

    print('[OK] Auto-annotation complete.')


if __name__ == '__main__':
    main()
