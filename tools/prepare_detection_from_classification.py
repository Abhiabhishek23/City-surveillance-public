#!/usr/bin/env python3
"""
Prepare a YOLO detection dataset skeleton from a classification-style tree.

Input (classification):
  <src>/train/<class>/*.jpg|png|...
  <src>/val/<class>/*.jpg|png|...
  <src>/test/<class>/*.jpg|png|...  (optional)
  Optionally: <src>/annotations/classes.txt listing class names (not required)

Output (detection skeleton):
  <dst>/images/train/<class>/*.jpg|png|...
  <dst>/labels/train/<class>/*.txt           (empty label files to be annotated later)
  <dst>/images/val/<class>/*.jpg|png|...
  <dst>/labels/val/<class>/*.txt
  <dst>/images/test/... and <dst>/labels/test/... (optional)
  <dst>/data.yaml

Notes:
 - This does NOT auto-generate bounding boxes. It only creates empty label files. You must annotate in a tool that writes YOLO .txt labels.
 - Unsupported formats like GIF/AVIF are skipped (to avoid PIL/OpenCV issues).
 - You can run this locally or adapt the same logic in a Kaggle cell to create a skeleton under /kaggle/working for download and annotation.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple
import yaml

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".dng", ".mpo"}


def scan_classes(src_split: Path) -> List[str]:
    """Collect class folder names from a split (train or val)."""
    if not src_split.exists():
        return []
    classes = [p.name for p in src_split.iterdir() if p.is_dir()]
    classes.sort()
    return classes


def read_classes_txt(src: Path) -> List[str]:
    cand = src / "annotations" / "classes.txt"
    if cand.exists():
        names = [line.strip() for line in cand.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names
    return []


def merge_classes(*lists: List[str]) -> List[str]:
    s = []
    seen = set()
    for lst in lists:
        for name in lst:
            if name not in seen:
                seen.add(name)
                s.append(name)
    return s


def copy_split(src: Path, dst: Path, split: str) -> Tuple[int, int]:
    src_split = src / split
    if not src_split.exists():
        return 0, 0
    img_base = dst / "images" / split
    lab_base = dst / "labels" / split
    copied = 0
    skipped = 0
    for cls_dir in sorted([p for p in src_split.iterdir() if p.is_dir()]):
        for img in cls_dir.rglob("*"):
            if not img.is_file():
                continue
            ext = img.suffix.lower()
            if ext not in SUPPORTED:
                skipped += 1
                continue
            rel = img.relative_to(src_split)
            img_out = img_base / rel
            lab_out = (lab_base / rel).with_suffix(".txt")
            img_out.parent.mkdir(parents=True, exist_ok=True)
            lab_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, img_out)
            # Create empty label file to be filled by annotators later
            if not lab_out.exists():
                lab_out.write_text("", encoding="utf-8")
            copied += 1
    return copied, skipped


def write_data_yaml(dst: Path, names: List[str], base_path: Path):
    data = {
        "path": str(base_path),
        "train": str(dst / "images" / "train"),
        "val": str(dst / "images" / "val"),
    }
    if (dst / "images" / "test").exists():
        data["test"] = str(dst / "images" / "test")
    if names:
        data["names"] = names
    with open(dst / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main():
    ap = argparse.ArgumentParser(description="Prepare YOLO detection skeleton from classification dataset")
    ap.add_argument("--src", required=True, help="Source classification dataset root (contains train/val[/test])")
    ap.add_argument("--dst", required=True, help="Destination folder for YOLO dataset skeleton")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    if not (src / "train").exists() or not (src / "val").exists():
        raise SystemExit(f"Expected classification splits at {src}/train and {src}/val")

    # Gather class list
    names_from_txt = read_classes_txt(src)
    names_from_train = scan_classes(src / "train")
    names_from_val = scan_classes(src / "val")
    names = merge_classes(names_from_txt, names_from_train, names_from_val)

    # Copy splits and create empty label files
    dst.mkdir(parents=True, exist_ok=True)
    tr_copied, tr_skipped = copy_split(src, dst, "train")
    vl_copied, vl_skipped = copy_split(src, dst, "val")
    ts_copied, ts_skipped = copy_split(src, dst, "test")

    # Write data.yaml
    write_data_yaml(dst, names, dst)

    print("Prepared detection skeleton at:", dst)
    print("Classes (names):", names)
    print(f"Copied train: {tr_copied} images (skipped {tr_skipped} unsupported)")
    print(f"Copied val:   {vl_copied} images (skipped {vl_skipped} unsupported)")
    print(f"Copied test:  {ts_copied} images (skipped {ts_skipped} unsupported)")
    print("Next steps:")
    print("  - Annotate bounding boxes in labels/*.txt using a YOLO-compatible tool (Label Studio, CVAT, or your annotation_ui)")
    print("  - Verify each image has a corresponding .txt file with box lines, or empty if no objects apply")
    print("  - Train using ultralytics with data.yaml under this folder")


if __name__ == "__main__":
    main()
