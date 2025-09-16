#!/usr/bin/env python3
"""
Flattens hierarchical encroachment attributes into 5 YOLO classes using dataset/ontology.yaml.

Usage:
  python tools/ontology_mapper.py --jsonl annotations.jsonl --out dataset/labels/train --images-root dataset/images/train

- Input JSONL format (one object per line):
{
  "image": "relative/path.jpg",
  "boxes": [
     {"bbox": [cx, cy, w, h], "attributes": {"permanence":"temporary", "zone":"road_footpath", "permit_status":"unknown", "area_type":"built", "structure_type":"stall"}},
     ...
  ]
}

Outputs YOLO .txt files alongside images under labels/ with mapped class ids.
"""
import argparse
import json
import os
from typing import Dict, Any, List

import yaml


def load_ontology(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def matches(rule_when: Dict[str, Any], attrs: Dict[str, Any]) -> bool:
    for k, v in rule_when.items():
        if isinstance(v, list):
            if attrs.get(k) not in v:
                return False
        else:
            if attrs.get(k) != v:
                return False
    return True


def map_attributes_to_class(attrs: Dict[str, Any], ontology: Dict[str, Any]) -> str:
    # If converter already provided a direct top_class name or id, prefer that
    if 'top_class' in attrs:
        return attrs['top_class']
    if 'top_class_hint' in attrs:
        try:
            # hint can be class id
            hint = attrs['top_class_hint']
            if isinstance(hint, int):
                inv = {v: k for k, v in ontology.get('class_ids', {}).items()}
                if hint in inv:
                    return inv[hint]
        except Exception:
            pass
    for rule in ontology.get('mapping_rules', []):
        when = rule.get('when', {})
        if matches(when, attrs):
            return rule['target']
    # No rule matched
    return 'Temporary_Illegal'


def class_name_to_id(name: str, ontology: Dict[str, Any]) -> int:
    try:
        return int(ontology['class_ids'][name])
    except Exception as e:
        raise KeyError(f"Class name {name} not found in ontology class_ids") from e


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def write_label_file(image_rel: str, boxes: List[Dict[str, Any]], out_dir: str, images_root: str, ontology: Dict[str, Any]):
    stem = os.path.splitext(os.path.basename(image_rel))[0]
    out_path = os.path.join(out_dir, f"{stem}.txt")
    lines = []
    for b in boxes:
        cls_name = map_attributes_to_class(b.get('attributes', {}), ontology)
        cls_id = class_name_to_id(cls_name, ontology)
        cx, cy, w, h = b['bbox']
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True, help='Path to JSONL annotations with attributes')
    ap.add_argument('--out', required=True, help='Output labels directory (e.g., dataset/labels/train)')
    ap.add_argument('--images-root', required=True, help='Images root directory for this split')
    ap.add_argument('--ontology', default='dataset/ontology.yaml', help='Path to ontology.yaml')
    args = ap.parse_args()

    ontology = load_ontology(args.ontology)
    ensure_dir(args.out)

    written = 0
    with open(args.jsonl, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            image_rel = rec['image']
            boxes = rec.get('boxes', [])
            write_label_file(image_rel, boxes, args.out, args.images_root, ontology)
            written += 1

    print(f"Wrote {written} label files to {args.out}")


if __name__ == '__main__':
    main()
