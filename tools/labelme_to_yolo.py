#!/usr/bin/env python3
"""
Convert LabelMe JSON annotations to YOLO txt labels, for 30-class Mahakumbh taxonomy.

Assumptions:
- LabelMe generates a JSON per image with shapes (polygons/rectangles).
- We handle rectangles as YOLO bboxes; for polygons, we compute bounding box.
- Requires a classes file with one class name per line in order.

Usage:
  python tools/labelme_to_yolo.py \
    --json-dir dataset/labelme_json/train \
    --images-root dataset/images/train \
    --labels-out dataset/labels/train \
    --classes dataset/classes_mahakumbh.txt
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from PIL import Image


def load_classes(path: str) -> Dict[str, int]:
    names = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    return {name: i for i, name in enumerate(names)}


def bbox_from_shape(shape: Dict[str, Any]) -> Tuple[float, float, float, float]:
    if shape.get('shape_type') == 'rectangle':
        (x1, y1), (x2, y2) = shape['points']
        return float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))
    # polygon fallback
    xs = [p[0] for p in shape['points']]
    ys = [p[1] for p in shape['points']]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def norm_bbox(x1, y1, x2, y2, W, H) -> Tuple[float, float, float, float]:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx / W, cy / H, w / W, h / H


def convert(json_dir: str, images_root: str, labels_out: str, classes_file: str, mirror_dirs: bool = True):
    name_to_id = load_classes(classes_file)
    os.makedirs(labels_out, exist_ok=True)

    for jf in Path(json_dir).glob('*.json'):
        data = json.loads(Path(jf).read_text())
        img_file = data.get('imagePath') or (jf.stem + '.jpg')
        img_path = Path(images_root) / img_file
        if not img_path.exists():
            # Try common alternatives
            for ext in ['.jpg', '.jpeg', '.png']:
                alt = Path(images_root) / (jf.stem + ext)
                if alt.exists():
                    img_path = alt
                    break
        if not img_path.exists():
            print(f"Image not found for {jf}, skipping")
            continue

        with Image.open(img_path) as im:
            W, H = im.size

        label_lines: List[str] = []
        for shp in data.get('shapes', []):
            label = shp.get('label')
            if label not in name_to_id:
                # Unknown class; optionally warn and skip
                # print(f"Unknown class {label} in {jf}")
                continue
            x1, y1, x2, y2 = bbox_from_shape(shp)
            cx, cy, nw, nh = norm_bbox(x1, y1, x2, y2, W, H)
            cls_id = name_to_id[label]
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if mirror_dirs:
            try:
                rel = img_path.parent.relative_to(Path(images_root))
            except ValueError:
                rel = Path()
            out_dir = Path(labels_out) / rel
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = Path(labels_out)
            out_dir.mkdir(parents=True, exist_ok=True)
        out_txt = out_dir / (img_path.stem + '.txt')
        Path(out_txt).write_text('\n'.join(label_lines))
        print(f"Wrote {out_txt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-dir', required=True)
    ap.add_argument('--images-root', required=True)
    ap.add_argument('--labels-out', required=True)
    ap.add_argument('--classes', default='dataset/classes_mahakumbh.txt')
    ap.add_argument('--no-mirror-dirs', action='store_true')
    args = ap.parse_args()

    convert(args.json_dir, args.images_root, args.labels_out, args.classes, mirror_dirs=not args.no_mirror_dirs)


if __name__ == '__main__':
    main()
