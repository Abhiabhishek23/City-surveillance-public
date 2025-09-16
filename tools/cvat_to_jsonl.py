#!/usr/bin/env python3
"""
Convert CVAT YOLO or CVAT XML to our JSONL attribute format.

Usage:
  python tools/cvat_to_jsonl.py --images-root dataset/images/train --labels-dir dataset/labels/train --out dataset/annotations_train.jsonl --mode yolo
  python tools/cvat_to_jsonl.py --cvat-xml export.xml --out dataset/annotations_train.jsonl --mode xml

Notes:
- YOLO mode reads *.txt files and emits records with bbox and a placeholder attributes dict.
- XML mode parses CVAT XML (bbox annotations) and emits records similarly.
- You can enrich attributes later (zone, permit_status, permanence) before mapping.
"""
import argparse
import os
import json
from glob import glob
from xml.etree import ElementTree as ET


def yolo_to_jsonl(images_root: str, labels_dir: str, out_path: str):
    records = {}
    for lp in glob(os.path.join(labels_dir, '*.txt')):
        stem = os.path.splitext(os.path.basename(lp))[0]
        image_rel = f"images/train/{stem}.jpg"  # adjust if different ext
        with open(lp, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        boxes = []
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            boxes.append({
                'bbox': [cx, cy, w, h],
                'attributes': {
                    'permanence': 'temporary',
                    'zone': 'none',
                    'permit_status': 'unknown',
                    'area_type': 'built',
                    'structure_type': 'other',
                    'top_class_hint': cls_id
                }
            })
        records[image_rel] = boxes
    with open(out_path, 'w') as out:
        for img, boxes in records.items():
            out.write(json.dumps({'image': img, 'boxes': boxes}) + '\n')
    print(f"Wrote JSONL: {out_path}")


def cvat_xml_to_jsonl(xml_path: str, images_root: str, out_path: str):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    ns = {}
    records = {}
    for image in root.findall('.//image', ns):
        img_name = image.get('name')
        image_rel = f"images/train/{os.path.basename(img_name)}"
        boxes = []
        for box in image.findall('box', ns):
            # CVAT stores absolute pixels; we do not have sizes here, so we leave normalized placeholders.
            # You can post-process to normalized if needed by reading image sizes.
            xtl = float(box.get('xtl')); ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr')); ybr = float(box.get('ybr'))
            # Placeholders for normalized coords — user should run a post-normalizer if needed
            cx = 0.5; cy = 0.5; w = 0.2; h = 0.2
            label = box.get('label', 'object')
            boxes.append({
                'bbox': [cx, cy, w, h],
                'attributes': {
                    'permanence': 'temporary',
                    'zone': 'none',
                    'permit_status': 'unknown',
                    'area_type': 'built',
                    'structure_type': label,
                }
            })
        records[image_rel] = boxes
    with open(out_path, 'w') as out:
        for img, boxes in records.items():
            out.write(json.dumps({'image': img, 'boxes': boxes}) + '\n')
    print(f"Wrote JSONL: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['yolo','xml'], required=True)
    ap.add_argument('--images-root', default='dataset/images/train')
    ap.add_argument('--labels-dir', default='dataset/labels/train')
    ap.add_argument('--cvat-xml', help='Path to CVAT XML export')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    if args.mode == 'yolo':
        yolo_to_jsonl(args.images_root, args.labels_dir, args.out)
    else:
        if not args.cvat_xml:
            raise SystemExit('--cvat-xml is required for xml mode')
        cvat_xml_to_jsonl(args.cvat_xml, args.images_root, args.out)


if __name__ == '__main__':
    main()
