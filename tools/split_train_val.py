#!/usr/bin/env python3
"""
Split images into train/val folders with a given ratio.

Usage:
  python tools/split_train_val.py --images dataset/images --val-ratio 0.2 --ext .jpg .png

It moves files into images/train and images/val.
"""
import argparse
import os
import random
import shutil
from glob import glob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', default='dataset/images')
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--ext', nargs='+', default=['.jpg', '.jpeg', '.png'])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    img_dir = args.images
    all_files = []
    for e in args.ext:
        all_files.extend(glob(os.path.join(img_dir, f'*{e}')))
    all_files = sorted(all_files)

    random.shuffle(all_files)
    n_val = int(len(all_files) * args.val_ratio)
    val_files = set(all_files[:n_val])

    train_out = os.path.join(img_dir, 'train')
    val_out = os.path.join(img_dir, 'val')
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out, exist_ok=True)

    moved_t = moved_v = 0
    for f in all_files:
        dest = val_out if f in val_files else train_out
        shutil.move(f, os.path.join(dest, os.path.basename(f)))
        moved_t += 1 if dest == train_out else 0
        moved_v += 1 if dest == val_out else 0

    print(f"Moved {moved_t} to train, {moved_v} to val")


if __name__ == '__main__':
    main()
