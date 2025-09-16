#!/usr/bin/env python3
"""
Auto-label images by running a YOLO model and writing LabelMe JSON files.
Supports mapping COCO class names → Mahakumbh taxonomy subset for bootstrapping.

Usage:
  python tools/auto_label_labelme.py \
    --weights yolov8n.pt \
    --images dataset/images/train \
    --out-json dataset/labelme_json/train \
    --conf 0.3 --coco-to-mahakumbh
"""
import argparse
import os
from pathlib import Path
from typing import Dict, List
import base64

from PIL import Image


COCO_TO_MAHA: Dict[str, str] = {
    "person": "pedestrian_walking",
    "car": "vehicle_car",
    "motorcycle": "vehicle_bike",
    "bicycle": "vehicle_bike",
    "bus": "vehicle_truck_bus",
    "truck": "vehicle_truck_bus",
    "boat": "boat",
}


def img_to_base64(p: Path) -> str:
    with open(p, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def write_labelme_json(img_path: Path, shapes: List[Dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(img_path) as im:
        W, H = im.size
    data = {
        "version": "5.3.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageData": None,  # keep files small; LabelMe loads image from disk
        "imageHeight": H,
        "imageWidth": W,
    }
    (out_dir / f"{img_path.stem}.json").write_text(__import__('json').dumps(data))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='yolov8n.pt')
    ap.add_argument('--images', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--mirror-dirs', action='store_true', help='Mirror input subfolders under out-json')
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--coco-to-mahakumbh', action='store_true')
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except Exception:
        raise SystemExit('Please install ultralytics: pip install ultralytics')

    model = YOLO(args.weights)
    # Gather image files
    img_paths: List[Path] = []
    for root, _, files in os.walk(args.images):
        for n in files:
            if n.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_paths.append(Path(root) / n)
    img_paths.sort()

    results = model.predict(source=[str(p) for p in img_paths], conf=args.conf)
    out_root = Path(args.out_json)

    for r in results:
        ip = Path(getattr(r, 'path', ''))
        names = r.names
        shapes: List[Dict] = []
        for b in getattr(r, 'boxes', []):
            cls_id = int(b.cls[0])
            cname = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            if args.coco_to_mahakumbh:
                cname = COCO_TO_MAHA.get(cname)
                if not cname:
                    continue
            # rectangle shape for LabelMe
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            shape = {
                "label": cname,
                "points": [[x1, y1], [x2, y2]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            shapes.append(shape)
        if shapes:
            if args.mirror_dirs:
                try:
                    rel = ip.parent.relative_to(Path(args.images))
                except ValueError:
                    rel = Path()
                out_dir = out_root / rel
            else:
                out_dir = out_root
            write_labelme_json(ip, shapes, out_dir)
            print(f"Wrote auto labels: {(out_dir / (ip.stem + '.json')).as_posix()}")


if __name__ == '__main__':
    main()
