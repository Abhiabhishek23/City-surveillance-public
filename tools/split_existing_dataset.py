#!/usr/bin/env python3
import os
import shutil
import random
import argparse
from pathlib import Path


def collect_images(images_dir):
    items = []
    for root, _dirs, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                abs_img = os.path.join(root, f)
                rel_path = os.path.relpath(abs_img, images_dir)
                items.append((abs_img, rel_path))
    return items


def copy_pairs(pairs, labels_root, out_img_root, out_lbl_root):
    for abs_img, rel_path in pairs:
        rel_dir = os.path.dirname(rel_path)
        os.makedirs(os.path.join(out_img_root, rel_dir), exist_ok=True)
        os.makedirs(os.path.join(out_lbl_root, rel_dir), exist_ok=True)
        shutil.copy(abs_img, os.path.join(out_img_root, rel_dir, os.path.basename(rel_path)))
        base = os.path.splitext(os.path.basename(rel_path))[0]
        lbl_src = os.path.join(labels_root, rel_dir, base + '.txt')
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, os.path.join(out_lbl_root, rel_dir, base + '.txt'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='root images folder to split (recursive)')
    ap.add_argument('--labels', required=True, help='labels root produced by annotator (recursive)')
    ap.add_argument('--out', required=True, help='output dataset root (will create train/ and val/)')
    ap.add_argument('--val_ratio', type=float, default=0.2)
    args = ap.parse_args()

    items = collect_images(args.images)
    random.shuffle(items)
    split_idx = int(len(items) * (1 - args.val_ratio))
    train_items = items[:split_idx]
    val_items = items[split_idx:]

    train_img_dir = os.path.join(args.out, 'train/images')
    train_lbl_dir = os.path.join(args.out, 'train/labels')
    val_img_dir = os.path.join(args.out, 'val/images')
    val_lbl_dir = os.path.join(args.out, 'val/labels')

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    copy_pairs(train_items, args.labels, train_img_dir, train_lbl_dir)
    copy_pairs(val_items, args.labels, val_img_dir, val_lbl_dir)

    print(f"[OK] Split created → train: {len(train_items)} images, val: {len(val_items)} images")
    print('Train images:', train_img_dir)
    print('Val images  :', val_img_dir)


if __name__ == '__main__':
    main()
