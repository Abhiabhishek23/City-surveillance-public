#!/usr/bin/env python3
"""
Compute class distribution recursively from YOLO label files using names from a data.yaml.

Usage:
  python tools/compute_class_distribution.py \
    --labels-root Indian_Urban_Dataset_yolo/annotations_hier/labels \
    --data-yaml Indian_Urban_Dataset_yolo/annotations_hier/data.yaml \
    --out Indian_Urban_Dataset_yolo/annotations_hier/class_distribution_recursive.json
"""
from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
import yaml

def load_names(data_yaml_path: str):
    with open(data_yaml_path, 'r') as f:
        y = yaml.safe_load(f)
    names = y.get('names')
    if isinstance(names, dict):
        # keys like 0..N-1, order by int
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    elif isinstance(names, list):
        return names
    else:
        raise ValueError('Unsupported names format in data.yaml')

def compute_counts(labels_root: str, names: list[str]):
    counts = {n: 0 for n in names}
    for root, _dirs, files in os.walk(labels_root):
        for f in files:
            if not f.endswith('.txt'):
                continue
            p = os.path.join(root, f)
            try:
                with open(p, 'r') as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            cls_id = int(parts[0])
                        except Exception:
                            continue
                        if 0 <= cls_id < len(names):
                            counts[names[cls_id]] += 1
            except Exception:
                pass
    return counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels-root', required=True)
    ap.add_argument('--data-yaml', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    names = load_names(args.data_yaml)
    counts = compute_counts(args.labels_root, names)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(counts, f, indent=2)
    print(f"[OK] Wrote {args.out}")

if __name__ == '__main__':
    main()
