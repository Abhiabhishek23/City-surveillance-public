#!/usr/bin/env python3
import os
from pathlib import Path
import hashlib

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

root = Path('.').resolve()
raw_root = root / 'temp_download'
train_root = root / 'train'
val_root = root / 'val'
test_root = root / 'test'


def list_images(dir_path: Path):
    for r, _, files in os.walk(dir_path):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
                yield Path(r) / fn

def file_hash(p: Path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with p.open('rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

# Build hash set of images present in splits
present = {}
for split_root in [train_root, val_root, test_root]:
    if not split_root.exists():
        continue
    for img in list_images(split_root):
        try:
            h = file_hash(img)
        except Exception:
            continue
        present.setdefault(h, []).append(str(img))

missing_from_splits = []
raw_count = 0
for img in list_images(raw_root):
    raw_count += 1
    try:
        h = file_hash(img)
    except Exception:
        continue
    if h not in present:
        missing_from_splits.append(str(img))

print(f"Raw images total: {raw_count}")
print(f"Raw images NOT present in train/val/test by content: {len(missing_from_splits)}")
if missing_from_splits:
    print("Example missing (up to 20):")
    for p in missing_from_splits[:20]:
        print(p)
