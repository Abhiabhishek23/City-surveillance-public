#!/usr/bin/env python3
"""
Convert JSONL annotations (from annotation_ui) to YOLO labels for object classes.

Each box must have attributes.top_class equal to one of the class names in objects_data.yaml.

Usage:
  python tools/jsonl_to_yolo_objects.py --jsonl dataset/annotations_ui.jsonl \
    --out dataset/labels/train --data dataset/objects_data.yaml
"""
import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Any

import yaml


def load_class_map(data_yaml: str) -> Dict[str, int]:
    with open(data_yaml, 'r') as f:
        y = yaml.safe_load(f)
    names = y.get('names')
    if isinstance(names, dict):
        id_to_name = {int(k): v for k, v in names.items()}
    else:
        id_to_name = {i: n for i, n in enumerate(names)}
    name_to_id = {v: k for k, v in id_to_name.items()}
    return name_to_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default='dataset/objects_data.yaml')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    name_to_id = load_class_map(args.data)

    # Aggregate boxes per image
    per_image = defaultdict(list)
    with open(args.jsonl, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            img = rec['image']
            for b in rec.get('boxes', []):
                attrs = b.get('attributes', {})
                cname = attrs.get('top_class') or attrs.get('object_class')
                if cname not in name_to_id:
                    # Skip unknown classes
                    continue
                cid = name_to_id[cname]
                cx, cy, w, h = b['bbox']
                per_image[img].append((cid, cx, cy, w, h))

    # Write label files
    cnt = 0
    for img, boxes in per_image.items():
        stem = os.path.splitext(os.path.basename(img))[0]
        outp = os.path.join(args.out, f'{stem}.txt')
        with open(outp, 'w') as f:
            for cid, cx, cy, w, h in boxes:
                f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        cnt += 1
    print(f"Wrote {cnt} label files to {args.out}")


if __name__ == '__main__':
    main()
