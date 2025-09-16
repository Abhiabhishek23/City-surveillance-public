#!/usr/bin/env python3
"""
Convert Label Studio JSON export to our JSONL attribute format.

Usage:
  python tools/label_studio_to_jsonl.py --ls-json export.json --out dataset/annotations_train.jsonl

Assumes rectangle labels with normalized coordinates {x,y,width,height} in percents (0-100) and a label/choices field.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ls-json', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    with open(args.ls_json, 'r') as f:
        data = json.load(f)

    with open(args.out, 'w') as out:
        for item in data:
            try:
                img_rel = item['data']['image']
                # Convert absolute URL or path into relative repo path if needed
                if '/dataset/images/' in img_rel:
                    # already relative
                    pass
                # parse annotations
                boxes = []
                for ann in item.get('annotations', []):
                    for r in ann.get('result', []):
                        if r.get('type') != 'rectanglelabels':
                            continue
                        val = r['value']
                        # Label Studio rectangle is in percentages
                        cx = (val['x'] + val['width']/2) / 100.0
                        cy = (val['y'] + val['height']/2) / 100.0
                        w = val['width'] / 100.0
                        h = val['height'] / 100.0
                        label = (val.get('rectanglelabels') or val.get('labels') or ['object'])[0]
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
                out.write(json.dumps({'image': img_rel, 'boxes': boxes}) + '\n')
            except Exception as e:
                # best-effort conversion; skip malformed entries
                continue

    print(f"Wrote JSONL: {args.out}")


if __name__ == '__main__':
    main()
