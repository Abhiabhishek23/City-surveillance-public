#!/usr/bin/env python3
"""
Prepare Indian_Urban_Dataset for YOLO detection training.

Assumptions:
- Dataset is classed by folders under Indian_Urban_Dataset/{train,val}/... and temp_download/ has extra classes without split.
- We perform a stratified split of temp_download/ into train/val (80/20), and flatten everything into:
    Indian_Urban_Dataset_yolo/
      images/{train,val}/*.jpg
      labels/{train,val}/*.txt  # YOLO bboxes (we create empty labels initially—classification-style, to be annotated later)

Notes:
- If you already have bounding boxes in a tool like CVAT/LabelStudio in YOLO format, drop them alongside images and this script will keep them.
- Otherwise, it will create empty label files so you can run detection training with weak supervision (not recommended) or use as a staging area for annotation.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict
import random
import argparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'Indian_Urban_Dataset'
OUT = ROOT / 'Indian_Urban_Dataset_yolo'

# Map complex folder taxonomy to YOLO class names (adjust as needed)
FOLDER_TO_CLASS = {
    # People
    'People': 'person',
    'Pedestrians': 'person',
    'walking': 'person',
    # Vehicles
    'bus': 'bus', 'buses': 'bus', 'Bus': 'bus',
    'truck': 'truck', 'Trucks': 'truck',
    'car': 'car', 'cars': 'car', 'Car': 'car',
    'motorbike': 'motorbike', 'bike': 'motorbike', 'two_wheeler': 'motorbike',
    'bicycle': 'bicycle',
    'boat': 'boat', 'boats': 'boat',
    'tractor': 'tractor', 'excavator': 'excavator',
    # Crowd infra / structures
    'Barricades': 'barricade', 'barricades': 'barricade', 'Structures_Barricades_bamboo': 'barricade',
    'traffic_cone': 'traffic_cone', 'cones': 'traffic_cone',
    'hawker': 'hawker', 'stall': 'hawker', 'stall_cart': 'hawker',
    'tent': 'tent_pandal', 'pandal': 'tent_pandal', 'Tent_Pandal': 'tent_pandal',
    'idol': 'idol_statue', 'idol_statue': 'idol_statue',
    'banner': 'banner_hoarding', 'hoarding': 'banner_hoarding',
    'loudspeaker': 'loudspeaker',
    'gas_cylinder': 'gas_cylinder',
    'garbage': 'garbage_pile', 'garbage_pile': 'garbage_pile',
    'open_fire': 'open_fire', 'smoke': 'smoke',
    'sand_pile': 'sand_pile', 'sand': 'sand_pile',
    'drone': 'drone',
    'rebar_pillar': 'rebar_pillar', 'brick_stack': 'brick_stack', 'concrete_mixer': 'concrete_mixer', 'cement_bag': 'cement_bag',
}

CLASS_INDEX = {}

# Additional prefix rules to match the updated taxonomy folders like
# "People_Pedestrians_walking", "Structures_Barricades_bamboo", etc.
PREFIX_TO_CLASS = [
    ("people", "person"),
    ("people_pedestrians", "person"),
    ("structures_barricades", "barricade"),
    ("vehicles_cars", "car"),
    ("vehicles_two_wheelers", "motorbike"),
    ("vehicles_buses", "bus"),
    ("vehicles_trucks", "truck"),
    ("vehicles_bicycles", "bicycle"),
    ("vehicles_tractor", "tractor"),
    ("vehicles_special_jcb_crane", "excavator"),
    ("religious_idols", "idol_statue"),
    ("religious_loudspeakers", "loudspeaker"),
    ("structures_pandals", "tent_pandal"),
    ("religious_fireworks", "open_fire"),
    ("waste_garbage", "garbage_pile"),
    ("environment_land_burning", "open_fire"),
    ("environment_sand_mining", "sand_pile"),
]

def infer_class_from_path(p: Path) -> str | None:
    parts = [x.lower() for x in p.parts]
    # 1) Direct exact folder name match
    for k, v in FOLDER_TO_CLASS.items():
        lk = k.lower()
        if lk in parts:
            return v
    # 2) Prefix-based match on composed taxonomy segments
    for part in parts:
        for prefix, v in PREFIX_TO_CLASS:
            if part.startswith(prefix):
                return v
    # 3) Token match: split parts by underscores and check equality to keys
    for part in parts:
        tokens = part.replace('-', '_').split('_')
        for k, v in FOLDER_TO_CLASS.items():
            lk = k.lower()
            if lk in tokens:
                return v
    # 4) Substring fallback (last resort)
    for part in parts:
        for k, v in FOLDER_TO_CLASS.items():
            if k.lower() in part:
                return v
    return None

def collect_images(base: Path):
    exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
    return [p for p in base.rglob('*') if p.suffix.lower() in exts]

def ensure_dirs():
    for split in ('train','val'):
        (OUT/'images'/split).mkdir(parents=True, exist_ok=True)
        (OUT/'labels'/split).mkdir(parents=True, exist_ok=True)

def write_empty_label(label_path: Path):
    if not label_path.exists():
        label_path.write_text('')

def stratified_split(paths):
    # paths: list of (image_path, class_name)
    by_cls = defaultdict(list)
    for p,c in paths:
        by_cls[c].append(p)
    train, val = [], []
    for c, arr in by_cls.items():
        random.shuffle(arr)
        n = len(arr)
        k = max(1, int(0.2 * n))
        val.extend((p,c) for p in arr[:k])
        train.extend((p,c) for p in arr[k:])
    return train, val

def copy_and_label(items, split):
    for img_path, cls in items:
        cls_idx = CLASS_INDEX.setdefault(cls, len(CLASS_INDEX))
        out_img = OUT/'images'/split/f"{img_path.stem}{img_path.suffix.lower()}"
        out_lbl = OUT/'labels'/split/f"{img_path.stem}.txt"
        # Copy image
        if not out_img.exists():
            shutil.copy2(img_path, out_img)
        # If existing label alongside the source image in YOLO format, prefer it
        src_lbl = img_path.with_suffix('.txt')
        if src_lbl.exists():
            shutil.copy2(src_lbl, out_lbl)
        else:
            write_empty_label(out_lbl)

def write_yaml():
    names = [None]*len(CLASS_INDEX)
    for name, idx in CLASS_INDEX.items():
        names[idx] = name
    yaml_path = OUT/'data.yaml'
    yaml_path.write_text(
        'path: '+str(OUT)+"\n"+
        'train: images/train\n' +
        'val: images/val\n' +
        'names:\n' + ''.join([f"  {i}: {n}\n" for i,n in enumerate(names)])
    )
    print('[OK] Wrote', yaml_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)

    ensure_dirs()

    # 1) Use existing split where present (Indian_Urban_Dataset/train|val)
    existing = []
    for split in ('train','val'):
        base = SRC/split
        if base.exists():
            for img in collect_images(base):
                cls = infer_class_from_path(img)
                if cls:
                    existing.append((img, cls, split))

    # 2) Extra images in temp_download without split → stratify
    temp = SRC/'temp_download'
    temp_items = []
    if temp.exists():
        for img in collect_images(temp):
            cls = infer_class_from_path(img)
            if cls:
                temp_items.append((img, cls))

    temp_train, temp_val = stratified_split(temp_items) if temp_items else ([], [])

    # Copy existing split
    for img, cls, split in existing:
        copy_and_label([(img, cls)], split)

    # Copy stratified
    copy_and_label(temp_train, 'train')
    copy_and_label(temp_val, 'val')

    write_yaml()
    print('[DONE] Prepared YOLO dataset at', OUT)

if __name__ == '__main__':
    main()
