#!/usr/bin/env python3
"""
Run YOLO (Ultralytics) inference over images/video and export detections to CSV
with columns: image, class_name, conf, x1, y1, x2, y2, lon, lat

Usage:
  python tools/yolo_infer_to_csv.py --weights runs/detect/train/weights/best.pt \
    --source dataset/images/val --out detections.csv --gps-json camera_meta.json

gps-json (optional): JSON mapping image filename to {lat, lon, alt, yaw, pitch, h_fov, v_fov}
If not provided, lon/lat will be empty.
"""
import os
import json
import argparse
from typing import Dict, Any

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--source', required=True, help='Image folder or video file')
    ap.add_argument('--out', required=True)
    ap.add_argument('--gps-json', help='Optional JSON file with per-image GPS camera meta')
    ap.add_argument('--conf', type=float, default=0.25)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise SystemExit('Please install ultralytics: pip install ultralytics')

    model = YOLO(args.weights)
    results = model.predict(source=args.source, conf=args.conf)

    gps_map: Dict[str, Dict[str, Any]] = {}
    if args.gps_json and os.path.exists(args.gps_json):
        with open(args.gps_json, 'r') as f:
            gps_map = json.load(f)

    rows = []
    for r in results:
        img_path = r.path if hasattr(r, 'path') else ''
        names = r.names
        for b in getattr(r, 'boxes', []):
            try:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cname = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                meta = gps_map.get(os.path.basename(img_path), {})
                lat = meta.get('lat')
                lon = meta.get('lon')
                rows.append({
                    'image': img_path,
                    'class_name': cname,
                    'conf': conf,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'lon': lon, 'lat': lat,
                })
            except Exception:
                continue

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f'Wrote {len(rows)} detections to {args.out}')


if __name__ == '__main__':
    main()
