#!/usr/bin/env python3
"""
Prepare LabelMe project structure and class list for the Mahakumbh dataset.
- Creates dataset/images/{train,val,test}
- Writes classes file for LabelMe labeling tool

Usage:
  python tools/labelme_prep.py --root dataset --classes dataset/classes_mahakumbh.txt
"""
import argparse
import os
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='dataset')
    ap.add_argument('--classes', default='dataset/classes_mahakumbh.txt')
    args = ap.parse_args()

    root = Path(args.root)
    for split in ['train','val','test']:
        for sub in ['images','labels']:
            (root / sub / split).mkdir(parents=True, exist_ok=True)

    # Ensure classes file exists
    cf = Path(args.classes)
    if not cf.exists():
        raise SystemExit(f"Classes file not found: {cf}")

    # LabelMe uses per-json labels; we just provide classes as reference
    print(f"Project ready under {root}. Use LabelMe on dataset/images/<split>.")

if __name__ == '__main__':
    main()
