#!/usr/bin/env python3
"""
Export dataset for Ultralytics YOLO classification training.
This script ensures structure and writes a dataset.yaml compatible with Ultralytics.
It can either copy or symlink the existing train/val/test folders into a target export directory.

Usage:
  python scripts/export_for_ultralytics.py --root . --out ../UrbanDatasetExport --mode symlink

"""
import argparse
import os
import shutil
from pathlib import Path
import yaml


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str):
    ensure_dir(dst.parent)
    if dst.exists():
        if dst.is_symlink() or dst.is_dir():
            return
        dst.unlink()
    if mode == 'symlink':
        if dst.exists():
            return
        os.symlink(src, dst)
    elif mode == 'copy':
        if dst.exists():
            return
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    else:
        raise ValueError('mode must be symlink or copy')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='.')
    ap.add_argument('--out', type=str, required=True, help='Target export directory')
    ap.add_argument('--mode', type=str, default='symlink', choices=['symlink', 'copy'], help='How to populate export dir')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()

    train = root / 'train'
    val = root / 'val'
    test = root / 'test'
    ann = root / 'annotations'

    assert train.exists() and val.exists(), 'train/ and val/ must exist'

    ensure_dir(out)
    # Create structure
    out_train = out / 'train'
    out_val = out / 'val'
    out_test = out / 'test'

    link_or_copy(train, out_train, args.mode)
    link_or_copy(val, out_val, args.mode)
    if test.exists():
        link_or_copy(test, out_test, args.mode)

    # Write dataset.yaml in export dir
    classes_txt = ann / 'classes.txt'
    with classes_txt.open('r', encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]

    ds_yaml = {
        'path': str(out),
        'train': 'train',
        'val': 'val',
        'test': 'test' if (out / 'test').exists() else None,
        'names': classes,
    }
    ds_yaml = {k: v for k, v in ds_yaml.items() if v is not None}
    with (out / 'dataset.yaml').open('w', encoding='utf-8') as f:
        yaml.safe_dump(ds_yaml, f, sort_keys=False, allow_unicode=True)

    print(f"Exported for Ultralytics at: {out}")


if __name__ == '__main__':
    main()
