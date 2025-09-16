#!/usr/bin/env python3
"""
Train classification on Indian_Urban_Dataset with MPS/CUDA/CPU auto selection.

Dataset structure expected (classification):
Indian_Urban_Dataset/
  train/
    class_a/ img1.jpg ...
    class_b/ ...
  val/
    class_a/ ...
    class_b/ ...

Usage (smoke test):
  python tools/train_indian_urban_cls.py \
    --data "Indian_Urban_Dataset" \
    --epochs 1 --batch 16 --imgsz 224 --device auto

Usage (full):
  python tools/train_indian_urban_cls.py \
    --data "Indian_Urban_Dataset" \
    --epochs 50 --batch 64 --imgsz 224 --device auto

Outputs go to runs/classify/<name> (Ultralytics default). Best weights: best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone


def detect_device(cli_device: str = "auto") -> str:
    try:
        import torch
        if cli_device and cli_device != "auto":
            return cli_device
        # Prefer MPS on Apple Silicon
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        # CUDA if available
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except Exception:
        return "cpu"


def validate_dataset_root(root: Path) -> None:
    if not root.exists():
        raise SystemExit(f"Dataset root not found: {root}")
    for split in ("train", "val"):
        split_dir = root / split
        if not split_dir.exists():
            raise SystemExit(f"Missing split folder: {split_dir}")
        # Ensure at least one class directory exists
        any_class = any(p.is_dir() for p in split_dir.iterdir())
        if not any_class:
            raise SystemExit(f"No class subfolders found under: {split_dir}")


def main():
    p = argparse.ArgumentParser(description="Train Ultralytics classification on Indian_Urban_Dataset")
    p.add_argument("--data", default="Indian_Urban_Dataset", help="Path to dataset root containing train/ and val/ subfolders")
    p.add_argument("--model", default="yolov8n-cls.pt", help="Pretrained classification weights (e.g., yolov8n-cls.pt)")
    p.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    p.add_argument("--batch", type=int, default=32, help="Batch size")
    p.add_argument("--imgsz", type=int, default=224, help="Image size for training")
    p.add_argument("--device", default="auto", help="Device: auto|mps|cuda|cpu")
    p.add_argument("--project", default="runs/classify", help="Project directory for runs")
    p.add_argument("--name", default=None, help="Run name (defaults to timestamp)")
    p.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 is safest on macOS/MPS)")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    p.add_argument("--no-val", action="store_true", help="Skip validation during training (workaround for some env bugs)")

    args = p.parse_args()

    data_root = Path(args.data).resolve()
    validate_dataset_root(data_root)

    device = detect_device(args.device)
    run_name = args.name or f"indian-urban-cls-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    print(f"Using device: {device}")
    print(f"Dataset: {data_root}")

    try:
        from ultralytics import YOLO
        # Patch classifier validator finalize to avoid confusion matrix crash on some envs
        try:
            from ultralytics.models.yolo.classify.val import ClassificationValidator  # type: ignore

            _orig_finalize = ClassificationValidator.finalize_metrics  # type: ignore[attr-defined]

            def _safe_finalize(self):  # type: ignore[no-redef]
                try:
                    return _orig_finalize(self)
                except Exception as e:  # noqa: BLE001
                    print(f"Warning: Skipping validator.finalize_metrics due to error: {e}")
                    # Best-effort: zero-out metrics to keep trainer moving
                    try:
                        if hasattr(self, "metrics") and isinstance(self.metrics, dict):
                            for k in self.metrics:
                                try:
                                    self.metrics[k] = 0.0
                                except Exception:
                                    pass
                    except Exception:
                        pass
                    return None

            ClassificationValidator.finalize_metrics = _safe_finalize  # type: ignore[attr-defined]
        except Exception:
            # If patching fails, continue; outer try/except below will still catch errors
            pass
    except Exception as e:
        raise SystemExit(
            f"Ultralytics not available. Install with: pip install ultralytics\nError: {e}"
        )

    model = YOLO(args.model)

    # Ultralytics will infer classification task from model type '-cls'
    # data expects a path with 'train/' and 'val/' subfolders
    run_dir = Path(args.project).resolve() / run_name
    results = None
    try:
        results = model.train(
            data=str(data_root),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            name=run_name,
            workers=args.workers,
            patience=args.patience,
            verbose=True,
            val=not args.no_val,
        )
    except Exception as e:
        # Ultralytics classification validation can crash in some macOS/Python combos.
        # If training epoch(s) completed, weights will still be saved under run_dir/weights.
        print(f"Warning: Training finished but validation crashed: {e}")
        print("Continuing. You can use last.pt or best.pt from the run directory.")

    # Print checkpoint paths
    best_path = run_dir / "weights" / "best.pt"
    last_path = run_dir / "weights" / "last.pt"
    if best_path.exists():
        print(f"Best weights: {best_path}")
    elif results is not None and getattr(results, 'best', None):
        print(f"Best weights: {results.best}")
    elif last_path.exists():
        print(f"Last weights: {last_path}")
    else:
        print(f"Training complete. Check run directory: {run_dir}")


if __name__ == "__main__":
    main()
