#!/usr/bin/env python3
"""
Prepare dataset: split temp_download/* class folders into train/ and val/ in an ideal structure.
Features:
- Discovers classes from temp_download's immediate subfolders.
- Computes content hashes to deduplicate images within and across classes/splits.
- Moves files (default) from temp_download to train/ and val/ to avoid redundancy; can copy with --copy.
- Deterministic shuffling with --seed.
- Generates annotation helpers:
  * classes.txt (one class per line)
  * class_to_index.json
        * train.csv, val.csv, test.csv (image_path, class_index, class_name) by scanning existing splits (not just moved files)
  * dataset.yaml (YOLO-style with paths and names)
    * dedupe_report.json (any duplicate files by content hash across train/val)
- Idempotent: safe to re-run; won't duplicate; will skip files that already exist with same hash.

Usage:
  python scripts/prepare_dataset.py --root . --val-ratio 0.2 --move

"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Set, Iterable

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}


def file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def list_images(dir_path: Path) -> List[Path]:
    files = []
    for root, _, fnames in os.walk(dir_path):
        for fn in fnames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMAGE_EXTS:
                files.append(Path(root) / fn)
    return files


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def write_lines(path: Path, lines: List[str]):
    with path.open('w', encoding='utf-8') as f:
        for line in lines:
            f.write(f"{line}\n")


def flatten_nested_classes(split_root: Path, known_prefixes: Iterable[str], global_hashes: Dict[str, Path], copy: bool = False) -> Dict[str, str]:
    """
    Normalize nested directories like People/Pedestrians/walking into People_Pedestrians_walking.
    Returns a mapping of src_dir -> new_class_name.
    """
    mapping: Dict[str, str] = {}
    prefixes = set(known_prefixes)

    def do_move(src: Path, dst: Path):
        ensure_dir(dst.parent)
        if copy:
            shutil.copy2(src, dst)
        else:
            if dst.exists():
                try:
                    if file_hash(dst) == file_hash(src):
                        return
                except Exception:
                    pass
            shutil.move(str(src), str(dst))

    # walk split_root for nested dirs one level below known prefixes
    for pref in prefixes:
        base = split_root / pref
        if not base.exists() or not base.is_dir():
            continue
        # List nested paths containing images
        for root, dirs, files in os.walk(base):
            root_path = Path(root)
            rel_parts = root_path.relative_to(split_root).parts
            # Only consider deeper than 1 level (e.g., People/<...>)
            if len(rel_parts) <= 1:
                continue
            # If this dir has image files, plan to move them
            imgs = [Path(root) / f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
            if not imgs:
                continue
            # Build flattened class name from all parts
            flat_name = "_".join(rel_parts)
            mapping[str(root_path)] = flat_name
            flat_dir = split_root / flat_name
            ensure_dir(flat_dir)
            for img in imgs:
                try:
                    h = file_hash(img)
                except Exception:
                    continue
                if h in global_hashes:
                    # Skip duplicate content
                    continue
                global_hashes[h] = img
                dst = flat_dir / img.name
                do_move(img, dst)
    # Cleanup: remove empty directories under known prefixes
    for pref in prefixes:
        base = split_root / pref
        if not base.exists():
            continue
        # Remove empty dirs bottom-up
        for root, dirs, files in os.walk(base, topdown=False):
            # ignore macOS metadata files for emptiness check
            files = [f for f in files if f != '.DS_Store']
            if not dirs and not files:
                try:
                    Path(root).rmdir()
                except OSError:
                    pass
        # If prefix dir becomes empty, remove it too
        try:
            # consider empty if contains nothing or only .DS_Store
            contents = [p for p in base.iterdir()]
            non_meta = [p for p in contents if p.name != '.DS_Store']
            if base.exists() and not non_meta:
                base.rmdir()
        except OSError:
            pass
    return mapping


def save_csv(path: Path, rows: List[Tuple[str, int, str]]):
    ensure_dir(path.parent)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'class_index', 'class_name'])
        writer.writerows(rows)


def discover_classes(*roots: Path) -> List[str]:
    names: Set[str] = set()
    for root in roots:
        if root.exists():
            for d in root.iterdir():
                if d.is_dir() and not d.name.startswith('.'):
                    names.add(d.name)
    classes = sorted(names)
    return classes


def plan_moves_for_class(class_dir: Path, dst_train: Path, dst_val: Path, class_index: int, 
                         global_hashes: Dict[str, Path], val_ratio: float, seed: int, copy: bool) -> Tuple[List[Tuple[Path, Path, str]], List[Tuple[Path, Path, str]], List[Tuple[str, int, str]]]:
    images = list_images(class_dir)
    if not images:
        return [], [], []

    # Deduplicate within the class by content
    unique_images: List[Path] = []
    seen_hashes: Set[str] = set()
    local_map: Dict[str, Path] = {}
    for img in images:
        try:
            h = file_hash(img)
        except Exception:
            continue
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        local_map[h] = img
        unique_images.append(img)

    # Deterministic shuffle
    rnd = random.Random(seed)
    rnd.shuffle(unique_images)

    n = len(unique_images)
    n_val = max(1 if n > 1 else 0, int(round(n * val_ratio)))
    val_imgs = set(unique_images[:n_val])

    moves_train: List[Tuple[Path, Path, str]] = []
    moves_val: List[Tuple[Path, Path, str]] = []
    annotations: List[Tuple[str, int, str]] = []

    for img in unique_images:
        try:
            h = file_hash(img)
        except Exception:
            continue
        # Skip if globally seen (avoid duplicates across classes/splits)
        if h in global_hashes:
            # Already placed somewhere; skip duplicate.
            continue
        global_hashes[h] = img

        subdir = dst_val if img in val_imgs else dst_train
        ensure_dir(subdir)
        dst_path = subdir / class_dir.name / img.name
        ensure_dir(dst_path.parent)
        moves = moves_val if img in val_imgs else moves_train
        moves.append((img, dst_path, h))
        annotations.append((str(dst_path), class_index, class_dir.name))

    # For idempotency, filter out moves whose destination already exists with same content hash
    def needs_move(src: Path, dst: Path, h: str) -> bool:
        if dst.exists():
            try:
                if file_hash(dst) == h:
                    return False
            except Exception:
                pass
        return True

    moves_train = [m for m in moves_train if needs_move(*m)]
    moves_val = [m for m in moves_val if needs_move(*m)]

    return moves_train, moves_val, annotations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=str, default='.', help='Dataset root containing temp_download/train/val')
    ap.add_argument('--raw', type=str, default='temp_download', help='Raw classes root folder')
    ap.add_argument('--train', type=str, default='train', help='Train folder name or path')
    ap.add_argument('--val', type=str, default='val', help='Val folder name or path')
    ap.add_argument('--test', type=str, default='test', help='Test folder name or path')
    ap.add_argument('--val-ratio', type=float, default=0.2, help='Validation split ratio (0-1)')
    ap.add_argument('--test-ratio', type=float, default=0.0, help='Test split ratio (0-1); train share = 1 - val - test')
    ap.add_argument('--seed', type=int, default=42, help='Random seed for deterministic split')
    ap.add_argument('--copy', action='store_true', help='Copy instead of move (default is move)')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    raw_root = (root / args.raw).resolve()
    train_root = (root / args.train).resolve()
    val_root = (root / args.val).resolve()
    test_root = (root / args.test).resolve()

    # If raw root is missing, fall back to annotation-only mode: scan existing splits and write annotations.
    annotation_only = not raw_root.exists()

    ensure_dir(train_root)
    ensure_dir(val_root)
    ensure_dir(test_root)

    classes = discover_classes(raw_root, train_root, val_root, test_root)
    if not classes:
        if annotation_only:
            raise SystemExit("No class folders found under train/val/test for annotation.")
        else:
            raise SystemExit(f"No class folders found under {raw_root}")

    class_to_index = {c: i for i, c in enumerate(classes)}

    global_hashes: Dict[str, Path] = {}
    all_rows_train: List[Tuple[str, int, str]] = []
    all_rows_val: List[Tuple[str, int, str]] = []
    all_rows_test: List[Tuple[str, int, str]] = []

    total_moves_train: List[Tuple[Path, Path, str]] = []
    total_moves_val: List[Tuple[Path, Path, str]] = []

    if not annotation_only:
        for cls in classes:
            class_dir = raw_root / cls
            idx = class_to_index[cls]
            # We will reuse val_ratio for validation size; test will be carved out from remaining after val.
            mt, mv, ann = plan_moves_for_class(
                class_dir, train_root, val_root, idx, global_hashes,
                val_ratio=args.val_ratio, seed=args.seed, copy=args.copy
            )
            total_moves_train.extend(mt)
            total_moves_val.extend(mv)
            # We will rebuild annotations by scanning after moves; ann retained for logging only.

    # Execute moves/copies
    def do_move(src: Path, dst: Path):
        ensure_dir(dst.parent)
        if args.copy:
            shutil.copy2(src, dst)
        else:
            # Use move; if same file already at dst with same content, skip
            if dst.exists():
                try:
                    if file_hash(dst) == file_hash(src):
                        return
                except Exception:
                    pass
            shutil.move(str(src), str(dst))

    if not annotation_only:
        for src, dst, _ in total_moves_train:
            do_move(src, dst)
        for src, dst, _ in total_moves_val:
            do_move(src, dst)

    # If test-ratio > 0, move a portion from train into test per class (post-move, deterministic)
    if (args.test_ratio and args.test_ratio > 0) and not annotation_only:
        rnd = random.Random(args.seed)
        for cn in classes:
            cdir_train = train_root / cn
            if not cdir_train.exists():
                continue
            imgs = sorted(list_images(cdir_train))
            if not imgs:
                continue
            k = int(round(len(imgs) * args.test_ratio))
            if k <= 0:
                continue
            rnd.shuffle(imgs)
            take = imgs[:k]
            cdir_test = test_root / cn
            ensure_dir(cdir_test)
            for img in take:
                dst = cdir_test / img.name
                do_move(img, dst)

    # Normalize any nested class folders into flattened class names (e.g., People/Pedestrians/walking -> People_Pedestrians_walking)
    known_prefixes = {"People", "Structures", "Vehicles", "Urban", "Environment", "Law", "Religious", "Waste", "Commerce"}
    flatten_nested_classes(train_root, known_prefixes, global_hashes, copy=args.copy)
    flatten_nested_classes(val_root, known_prefixes, global_hashes, copy=args.copy)
    flatten_nested_classes(test_root, known_prefixes, global_hashes, copy=args.copy)

    # Build annotations by scanning train/val to include pre-existing files
    def rows_from_split(split_root: Path, classes: Iterable[str]) -> List[Tuple[str, int, str]]:
        rows: List[Tuple[str, int, str]] = []
        for cn in classes:
            cdir = split_root / cn
            if not cdir.exists():
                continue
            idx = class_to_index[cn]
            for img in list_images(cdir):
                rows.append((str(img), idx, cn))
        return rows

    # Rediscover classes in case flattening created new class dirs and removed generic ones
    classes = discover_classes(train_root, val_root, test_root)
    class_to_index = {c: i for i, c in enumerate(classes)}
    all_rows_train = rows_from_split(train_root, classes)
    all_rows_val = rows_from_split(val_root, classes)
    all_rows_test = rows_from_split(test_root, classes)

    # Write annotations and helpers
    meta_dir = root / 'annotations'
    ensure_dir(meta_dir)

    write_lines(meta_dir / 'classes.txt', classes)
    with (meta_dir / 'class_to_index.json').open('w', encoding='utf-8') as f:
        json.dump(class_to_index, f, indent=2, ensure_ascii=False)

    save_csv(meta_dir / 'train.csv', all_rows_train)
    save_csv(meta_dir / 'val.csv', all_rows_val)
    save_csv(meta_dir / 'test.csv', all_rows_test)

    # Filter classes to only those present in annotations
    used_classes: List[str] = sorted({cn for _, _, cn in all_rows_train + all_rows_val})
    classes = used_classes
    class_to_index = {c: i for i, c in enumerate(classes)}

    # Re-save class mapping after filtering
    write_lines(meta_dir / 'classes.txt', classes)
    with (meta_dir / 'class_to_index.json').open('w', encoding='utf-8') as f:
        json.dump(class_to_index, f, indent=2, ensure_ascii=False)

    # YOLO dataset.yaml style
    dataset_yaml = {
        'path': str(root),
        'train': str(train_root.relative_to(root)),
        'val': str(val_root.relative_to(root)),
        'test': str(test_root.relative_to(root)) if any((test_root / c).exists() for c in classes) else None,
        'names': classes,
    }
    dataset_yaml_path = meta_dir / 'dataset.yaml'
    # Remove None fields (e.g., test when not present)
    dataset_yaml = {k: v for k, v in dataset_yaml.items() if v is not None}
    try:
        import yaml  # Optional
        with dataset_yaml_path.open('w', encoding='utf-8') as f:
            yaml.safe_dump(dataset_yaml, f, sort_keys=False, allow_unicode=True)
    except Exception:
        # Minimal manual YAML writer
        def manual_yaml(d: Dict) -> str:
            lines = [f"path: {d['path']}", f"train: {d['train']}", f"val: {d['val']}", "names:"]
            for i, name in enumerate(d['names']):
                lines.append(f"  - {name}")
            return "\n".join(lines) + "\n"
        dataset_yaml_path.write_text(manual_yaml(dataset_yaml), encoding='utf-8')

    # Dedupe report across train/val by content hash
    def gather_hashes(split_root: Path) -> Dict[str, List[str]]:
        mapping: Dict[str, List[str]] = {}
        for cn in classes:
            cdir = split_root / cn
            if not cdir.exists():
                continue
            for img in list_images(cdir):
                try:
                    h = file_hash(img)
                except Exception:
                    continue
                mapping.setdefault(h, []).append(str(img))
        return mapping

    h_train = gather_hashes(train_root)
    h_val = gather_hashes(val_root)
    dupes: Dict[str, List[str]] = {}
    for h, paths in {**h_train}.items():
        if len(paths) > 1:
            dupes[h] = paths
    for h, paths in h_val.items():
        if h in dupes:
            dupes[h].extend(paths)
        elif len(paths) > 1 or h in h_train:
            dupes[h] = h_train.get(h, []) + paths
    # include test duplicates
    if test_root.exists():
        h_test = gather_hashes(test_root)
        for h, paths in h_test.items():
            if h in dupes:
                dupes[h].extend(paths)
            elif len(paths) > 1 or h in h_train or h in h_val:
                dupes[h] = h_train.get(h, []) + h_val.get(h, []) + paths
    with (meta_dir / 'dedupe_report.json').open('w', encoding='utf-8') as f:
        json.dump(dupes, f, indent=2)

    print(f"Classes: {len(classes)} | Train moves: {len(total_moves_train)} | Val moves: {len(total_moves_val)}")
    print(f"CSV written to {meta_dir}/train.csv and {meta_dir}/val.csv")
    print(f"Class mapping at {meta_dir}/classes.txt and class_to_index.json")
    print(f"YOLO dataset.yaml at {dataset_yaml_path}")
    print(f"Dedupe report at {meta_dir}/dedupe_report.json")


if __name__ == '__main__':
    main()
