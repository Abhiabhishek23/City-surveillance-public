#!/usr/bin/env python3
"""
Auto-annotate 'person' bounding boxes for an existing YOLO dataset split using a pretrained model.

- Reads images from Indian_Urban_Dataset_yolo/images/{train,val}
- Writes YOLO txt labels to Indian_Urban_Dataset_yolo/labels/{train,val}
  with class id 0 for 'person' only.
 - Skips files that already have a non-empty label unless --overwrite is provided.

Usage:
  python tools/auto_annotate_persons_yolo.py --weights yolov8n.pt --split train --conf 0.25
  python tools/auto_annotate_persons_yolo.py --split val

Requirements:
  pip install ultralytics pillow
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "Indian_Urban_Dataset_yolo"


def yolo_box_from_xyxy(x1, y1, x2, y2, w, h):
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def iter_images(folder: Path) -> Iterable[Path]:
    for p in folder.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolov8n.pt", help="Detector weights for auto-annotation")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing non-empty labels")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    args = ap.parse_args()

    images_dir = DATASET / "images" / args.split
    labels_dir = DATASET / "labels" / args.split
    labels_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except Exception:
        raise SystemExit("Please install ultralytics: pip install ultralytics")

    model = YOLO(args.weights)

    imgs = list(iter_images(images_dir))
    if args.limit:
        imgs = imgs[: args.limit]

    total, wrote, skipped = 0, 0, 0
    for img_path in imgs:
        total += 1
        lbl_path = labels_dir / (img_path.stem + ".txt")

        # Skip if label exists and has content
        if lbl_path.exists() and lbl_path.stat().st_size > 0 and not args.overwrite:
            skipped += 1
            continue

        # Validate image can be opened
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            print(f"[WARN] Skipping unreadable image {img_path.name}: {e}")
            skipped += 1
            continue

        # Run detection
        results = model.predict(source=str(img_path), conf=args.conf, verbose=False)
        if not results:
            lbl_path.write_text("")
            continue

        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or len(boxes) == 0:
            lbl_path.write_text("")
            continue

        # 'person' in COCO is class 0 for COCO-trained models
        lines = []
        try:
            for b in boxes:
                cls_id = int(b.cls[0])
                if cls_id != 0:  # keep only person
                    continue
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
                cx, cy, bw, bh = yolo_box_from_xyxy(x1, y1, x2, y2, w, h)
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        except Exception:
            pass

        lbl_path.write_text("\n".join(lines))
        wrote += 1
        if wrote % 25 == 0:
            print(f"[INFO] Wrote {wrote} labels (skipped {skipped}/{total})")

    print(f"[DONE] Split={args.split} total={total} wrote={wrote} skipped={skipped}")


if __name__ == "__main__":
    main()
