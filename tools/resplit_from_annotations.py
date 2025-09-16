#!/usr/bin/env python3
"""
Resplit images + labels into proper YOLO train/val folders from existing annotations.

Use when labels were generated under annotations_hier/labels/{train,val,...} and you need
flat splits at:
  out/train/images, out/train/labels, out/val/images, out/val/labels

If the source images folder already has top-level train/val/test subfolders, we mirror them
and strip the top folder name from destination paths (no nested train/train).
Otherwise we perform a random split using --val-ratio.

Also writes a data.yaml and recomputes class_distribution_recursive.json using names from
the annotator's taxonomy file (data.yaml) if provided, else infers names from folder order.

Usage:
  python tools/resplit_from_annotations.py \
    --images Indian_Urban_Dataset_yolo/images \
    --labels-root Indian_Urban_Dataset_yolo/annotations_hier/labels \
    --data-yaml Indian_Urban_Dataset_yolo/annotations_hier/data.yaml \
    --out Indian_Urban_Dataset_yolo/annotations_hier \
    --val-ratio 0.2
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path
import json
import yaml


def has_top_level_splits(images_dir: str) -> bool:
    top = set(os.listdir(images_dir)) if os.path.isdir(images_dir) else set()
    return any(d in top for d in ("train", "val", "test"))


def load_names_from_yaml(data_yaml_path: str):
    with open(data_yaml_path, 'r') as f:
        y = yaml.safe_load(f)
    names = y.get('names')
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    elif isinstance(names, list):
        return names
    else:
        raise ValueError('Unsupported names format in data.yaml')


def copy_item(img_abs: str, rel_parts: list[str], dest_img_root: str, dest_lbl_root: str, labels_root: str):
    rel_dir = os.path.join(*rel_parts[:-1]) if len(rel_parts) > 1 else ""
    img_name = rel_parts[-1]
    base = os.path.splitext(img_name)[0]
    # ensure dirs
    os.makedirs(os.path.join(dest_img_root, rel_dir), exist_ok=True)
    os.makedirs(os.path.join(dest_lbl_root, rel_dir), exist_ok=True)
    # copy image
    shutil.copy(img_abs, os.path.join(dest_img_root, rel_dir, img_name))
    # label: look in labels_root with same rel_dir
    lbl_src = os.path.join(labels_root, *rel_parts[:-1], base + '.txt')
    if os.path.exists(lbl_src):
        shutil.copy(lbl_src, os.path.join(dest_lbl_root, rel_dir, base + '.txt'))


def mirror_splits(images_dir: str, labels_root: str, out_dir: str):
    train_img_out = os.path.join(out_dir, 'train/images')
    train_lbl_out = os.path.join(out_dir, 'train/labels')
    val_img_out = os.path.join(out_dir, 'val/images')
    val_lbl_out = os.path.join(out_dir, 'val/labels')
    for d in (train_img_out, train_lbl_out, val_img_out, val_lbl_out):
        os.makedirs(d, exist_ok=True)

    for split in ('train', 'val'):
        split_dir = os.path.join(images_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for root, _dirs, files in os.walk(split_dir):
            for f in files:
                if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                img_abs = os.path.join(root, f)
                rel_after_split = os.path.relpath(img_abs, split_dir)
                rel_parts = rel_after_split.split(os.sep)
                if split == 'train':
                    copy_item(img_abs, rel_parts, train_img_out, train_lbl_out, os.path.join(labels_root, 'train'))
                else:
                    copy_item(img_abs, rel_parts, val_img_out, val_lbl_out, os.path.join(labels_root, 'val'))
    return train_img_out, val_img_out


def random_split(images_dir: str, labels_root: str, out_dir: str, val_ratio: float):
    train_img_out = os.path.join(out_dir, 'train/images')
    train_lbl_out = os.path.join(out_dir, 'train/labels')
    val_img_out = os.path.join(out_dir, 'val/images')
    val_lbl_out = os.path.join(out_dir, 'val/labels')
    for d in (train_img_out, train_lbl_out, val_img_out, val_lbl_out):
        os.makedirs(d, exist_ok=True)

    items = []
    for root, _dirs, files in os.walk(images_dir):
        for f in files:
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_abs = os.path.join(root, f)
            rel_path = os.path.relpath(img_abs, images_dir)
            rel_parts = rel_path.split(os.sep)
            items.append((img_abs, rel_parts))

    random.shuffle(items)
    split_idx = int(len(items) * (1 - val_ratio))
    train_items, val_items = items[:split_idx], items[split_idx:]
    for img_abs, rel_parts in train_items:
        copy_item(img_abs, rel_parts, train_img_out, train_lbl_out, labels_root)
    for img_abs, rel_parts in val_items:
        copy_item(img_abs, rel_parts, val_img_out, val_lbl_out, labels_root)
    return train_img_out, val_img_out


def compute_distribution(labels_root: str, names: list[str]) -> dict:
    counts = {n: 0 for n in names}
    for root, _dirs, files in os.walk(labels_root):
        for f in files:
            if not f.endswith('.txt'):
                continue
            with open(os.path.join(root, f)) as fh:
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
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True)
    ap.add_argument('--labels-root', required=True)
    ap.add_argument('--data-yaml', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--val-ratio', type=float, default=0.2)
    args = ap.parse_args()

    names = load_names_from_yaml(args.data_yaml)

    if has_top_level_splits(args.images):
        train_path, val_path = mirror_splits(args.images, args.labels_root, args.out)
    else:
        train_path, val_path = random_split(args.images, args.labels_root, args.out, args.val_ratio)

    # Write data.yaml
    data_yaml_text = """train: {}
val: {}
nc: {}
names:
""".format(Path(train_path).resolve(), Path(val_path).resolve(), len(names))
    for i, n in enumerate(names):
        data_yaml_text += f"  {i}: {n}\n"
    with open(os.path.join(args.out, 'data.yaml'), 'w') as f:
        f.write(data_yaml_text)

    # Compute distribution recursively
    dist = compute_distribution(os.path.join(args.out, 'train/labels'), names)
    dist_val = compute_distribution(os.path.join(args.out, 'val/labels'), names)
    summary = {
        'train_counts': dist,
        'val_counts': dist_val,
    }
    with open(os.path.join(args.out, 'class_distribution_recursive.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('[OK] Wrote data.yaml and class_distribution_recursive.json')


if __name__ == '__main__':
    main()
