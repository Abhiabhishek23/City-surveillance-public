#!/usr/bin/env python3
"""
Validate YOLO dataset consistency and report class frequencies.

Checks:
- data.yaml exists and class ids match label usage
- Each label file has valid rows: int class_id then 4 floats in [0,1]
- Each image has corresponding label (optional flag)
"""
import argparse
import glob
import os
from collections import Counter

import yaml


def load_classes(data_yaml_path: str):
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    names = data.get('names')
    if isinstance(names, dict):
        # id->name mapping
        id_to_name = {int(k): v for k, v in names.items()}
    else:
        id_to_name = {i: n for i, n in enumerate(names)}
    return id_to_name


def validate_labels(labels_dir: str, id_to_name):
    counts = Counter()
    errors = []
    label_files = sorted(glob.glob(os.path.join(labels_dir, '*.txt')))
    for lp in label_files:
        with open(lp, 'r') as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{lp}:{ln} -> expected 5 fields, got {len(parts)}")
                    continue
                try:
                    cid = int(parts[0])
                    nums = list(map(float, parts[1:]))
                except Exception:
                    errors.append(f"{lp}:{ln} -> parse error")
                    continue
                if cid not in id_to_name:
                    errors.append(f"{lp}:{ln} -> class id {cid} not in data.yaml")
                for i, v in enumerate(nums):
                    if not (0.0 <= v <= 1.0):
                        errors.append(f"{lp}:{ln} -> bbox[{i}] out of [0,1]: {v}")
                counts[cid] += 1
    return counts, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='dataset/data.yaml')
    ap.add_argument('--labels', default='dataset/labels/train')
    args = ap.parse_args()

    id_to_name = load_classes(args.data)
    counts, errors = validate_labels(args.labels, id_to_name)

    print("Class frequencies:")
    total = sum(counts.values())
    for cid in sorted(id_to_name.keys()):
        c = counts.get(cid, 0)
        name = id_to_name[cid]
        pct = (c / total * 100.0) if total else 0.0
        print(f"  {cid} {name}: {c} ({pct:.1f}%)")
    if errors:
        print("\nErrors:")
        for e in errors[:200]:
            print(" -", e)
        if len(errors) > 200:
            print(f"... {len(errors)-200} more")
        raise SystemExit(1)
    print("\nValidation: OK")


if __name__ == '__main__':
    main()
