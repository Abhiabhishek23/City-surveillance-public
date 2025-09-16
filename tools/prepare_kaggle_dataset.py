"""prepare_kaggle_dataset.py
Utility to bundle the local dataset folder (dataset/) into a single zip suitable for Kaggle upload.

Usage:
  python tools/prepare_kaggle_dataset.py --out kaggle_dataset.zip

It will:
- Validate directory structure
- Count images/labels
- Optionally generate a class frequency report (if labels exist)
- Zip into the provided output path

Add later: checksum manifest, train/val split automation.
"""
from __future__ import annotations
import argparse
import os
import sys
import zipfile
from pathlib import Path
from collections import Counter

DATASET_ROOT = Path('dataset')

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
LABEL_EXT = '.txt'


def collect_files():
    images = list((DATASET_ROOT / 'images' / 'train').rglob('*')) + list((DATASET_ROOT / 'images' / 'val').rglob('*'))
    labels = list((DATASET_ROOT / 'labels' / 'train').rglob('*')) + list((DATASET_ROOT / 'labels' / 'val').rglob('*'))
    images = [p for p in images if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    labels = [p for p in labels if p.is_file() and p.suffix.lower() == LABEL_EXT]
    return images, labels


def class_frequency(labels):
    freq = Counter()
    for lf in labels:
        try:
            for line in lf.read_text().strip().splitlines():
                if not line.strip():
                    continue
                cls_id = line.split()[0]
                freq[cls_id] += 1
        except Exception:
            pass
    return freq


def make_zip(out_path: Path):
    with zipfile.ZipFile(out_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(DATASET_ROOT):
            for f in files:
                fp = Path(root) / f
                arc = fp.relative_to(DATASET_ROOT.parent)
                zf.write(fp, arcname=str(arc))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='kaggle_dataset.zip', help='Output zip file')
    args = ap.parse_args()

    if not DATASET_ROOT.exists():
        print('Dataset root not found: dataset/')
        sys.exit(1)

    images, labels = collect_files()
    print(f'Images: {len(images)}  Labels: {len(labels)}')
    if labels:
        freq = class_frequency(labels)
        if freq:
            print('Class frequency (raw YOLO IDs):')
            for k, v in sorted(freq.items(), key=lambda x: int(x[0])):
                print(f'  {k}: {v}')

    out_path = Path(args.out)
    make_zip(out_path)
    print(f'Wrote zip: {out_path.resolve()}')

if __name__ == '__main__':
    main()
