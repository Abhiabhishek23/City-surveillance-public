#!/usr/bin/env python3
"""
Local YOLO Fine-tuning Runner (Ultralytics)
------------------------------------------
Trains YOLO on your local dataset using Ultralytics APIs with sane defaults.
Detects device (CUDA/MPS/CPU), supports yolov12n.pt if available, falls back to yolov8n.pt.

Example:
  python tools/train_yolov12n_local.py \
    --data "Data/crowd/data.yaml" \
    --weights yolov12n.pt \
    --epochs 50 --imgsz 640 --batch 8 --name crowd_yolov12n

Smoke test:
  python tools/train_yolov12n_local.py --data "Data/crowd/data.yaml" --epochs 1 --batch 2 --imgsz 320
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def detect_device() -> str:
    try:
        import torch
    except Exception:
        return 'cpu'

    if torch.cuda.is_available():
        return '0'  # first CUDA GPU
    # macOS Metal Performance Shaders (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def resolve_weights(candidate: str) -> str:
    # If provided file exists, use it. Else try common fallbacks.
    if candidate and Path(candidate).exists():
        return candidate
    # Fall back order: yolov12n.pt at repo root, models/, then yolov8n.pt
    root = Path.cwd()
    candidates = [
        root / 'yolov12n.pt',
        root / 'models' / 'yolov12n.pt',
        root / 'yolov8n.pt',
        root / 'models' / 'yolov8n.pt',
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Nothing local; rely on Ultralytics to fetch by alias if possible
    return candidate or 'yolov8n.pt'


def main():
    parser = argparse.ArgumentParser(description='Local YOLO fine-tuning (Ultralytics)')
    parser.add_argument('--data', required=True, help='Path to data.yaml')
    parser.add_argument('--weights', default='yolov12n.pt', help='Initial weights (file or alias)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--device', default=None, help='Override device: 0|cpu|mps')
    parser.add_argument('--name', default='yolov12n_local', help='Run name')
    parser.add_argument('--project', default='models', help='Output dir base')
    parser.add_argument('--lr0', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--workers', type=int, default=2, help='Dataloader workers')
    parser.add_argument('--amp', action='store_true', help='Enable AMP (mixed precision)')
    parser.add_argument('--mosaic', type=float, default=0.0, help='Mosaic augmentation ratio (0 disables)')
    parser.add_argument('--val', type=lambda x: str(x).lower() not in ['0', 'false', 'no'], default=True,
                        help='Run validation after each epoch (true/false)')
    args = parser.parse_args()

    # Lazy import after parsing to speed help
    from ultralytics import YOLO

    device = args.device or detect_device()
    weights = resolve_weights(args.weights)

    print(f"Using device: {device}")
    print(f"Weights: {weights}")
    print(f"Data: {args.data}")

    model = YOLO(weights)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        project=args.project,
        lr0=args.lr0,
        patience=args.patience,
        verbose=args.verbose,
        workers=args.workers,
        amp=args.amp,
        mosaic=args.mosaic,
        val=args.val,
        plots=False,
    )

    # Print artifact location
    try:
        save_dir = Path(str(results.save_dir))  # Ultralytics returns hub/box with .save_dir
    except Exception:
        save_dir = Path(args.project) / args.name
    print(f"Training complete. Artifacts: {save_dir}")
    best = save_dir / 'weights' / 'best.pt'
    if best.exists():
        print(f"Best weights: {best}")
    else:
        print("Best weights file not found yet; check run logs.")


if __name__ == '__main__':
    main()
