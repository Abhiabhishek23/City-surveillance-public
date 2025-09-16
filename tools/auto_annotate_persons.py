#!/usr/bin/env python3
"""
Auto-annotate 'person' bounding boxes for images in Indian_Urban_Dataset_yolo/images/{train,val}.
Writes YOLO txt labels with class 0 for person where labels are empty or --overwrite is set.
"""
from ultralytics import YOLO
from pathlib import Path
import argparse
import cv2
import os

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / 'Indian_Urban_Dataset_yolo'

def to_yolo(xyxy, w, h):
    x1,y1,x2,y2 = xyxy
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw/2
    cy = y1 + bh/2
    return cx/w, cy/h, bw/w, bh/h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='yolov8n.pt', help='pretrained model for person detection')
    ap.add_argument('--conf', type=float, default=0.35)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    model = YOLO(args.weights)
    for split in ('train','val'):
        imgdir = DS/'images'/split
        lbldir = DS/'labels'/split
        if not imgdir.exists():
            continue
        for imgp in imgdir.glob('*'):
            if imgp.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp','.webp'}:
                continue
            lblp = lbldir/f"{imgp.stem}.txt"
            if lblp.exists() and not args.overwrite and lblp.stat().st_size>0:
                continue
            im = cv2.imread(str(imgp))
            if im is None:
                continue
            h, w = im.shape[:2]
            res = model.predict(source=im, imgsz=640, conf=args.conf, verbose=False)[0]
            lines = []
            names = res.names
            for b in res.boxes:
                cls = int(b.cls[0].item())
                if names[cls].lower() != 'person':
                    continue
                xyxy = b.xyxy[0].tolist()
                cx, cy, bw, bh = to_yolo(xyxy, w, h)
                # class 0 reserved as 'person' in our yaml ordering
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            lblp.write_text("\n".join(lines))
            print('wrote', lblp, 'num=', len(lines))

if __name__ == '__main__':
    main()
