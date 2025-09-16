#!/usr/bin/env python3
"""Base Layer inference: run YOLO on images or a video and dump detections as JSON.
Usage:
  python inference/run_inference.py --source path/to/images_or_video --weights models/yolov12n_indian_urban/weights/best.pt
"""
import argparse, json, time
from pathlib import Path
from ultralytics import YOLO
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'base_layer'
OUT_DIR.mkdir(exist_ok=True)

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2))

def run_images(model, src: Path):
    exts = {'.jpg','.jpeg','.png','.bmp','.webp'}
    imgs = [p for p in src.rglob('*') if p.suffix.lower() in exts]
    for p in imgs:
        ts = time.time()
        res = model.predict(source=str(p), imgsz=640, conf=0.25, verbose=False)[0]
        dets = []
        names = res.names
        for b in res.boxes:
            xyxy = b.xyxy[0].tolist()
            cls = int(b.cls[0].item()); conf = float(b.conf[0].item())
            dets.append({
                'object': names[cls],
                'class': names[cls],
                'bbox_xyxy': xyxy,
                'confidence': conf,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(ts))
            })
        save_json(OUT_DIR / f"{p.stem}.json", {'image': str(p), 'detections': dets})

def run_video(model, src: Path):
    cap = cv2.VideoCapture(str(src))
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        ts = time.time()
        res = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)[0]
        dets = []
        names = res.names
        for b in res.boxes:
            xyxy = b.xyxy[0].tolist()
            cls = int(b.cls[0].item()); conf = float(b.conf[0].item())
            dets.append({
                'object': names[cls],
                'class': names[cls],
                'bbox_xyxy': xyxy,
                'confidence': conf,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(ts))
            })
        save_json(OUT_DIR / f"{src.stem}_{idx:06d}.json", {'video': str(src), 'frame': idx, 'detections': dets})
        idx += 1
    cap.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--weights', default=str(ROOT/'models/yolov12n_indian_urban/weights/best.pt'))
    args = ap.parse_args()
    model = YOLO(args.weights)
    src = Path(args.source)
    if src.is_dir(): run_images(model, src)
    else: run_video(model, src)

if __name__ == '__main__':
    main()
