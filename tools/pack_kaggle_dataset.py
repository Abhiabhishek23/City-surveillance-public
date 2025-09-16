#!/usr/bin/env python3
"""
Pack a YOLO dataset folder into a Kaggle Dataset-ready structure and optionally emit dataset-metadata.json.

Usage:
  python tools/pack_kaggle_dataset.py \
    --dataset-dir Data/combined_dataset \
    --output-dir dist/kaggle_dataset \
    --owner <your_kaggle_username> \
    --title "Indian Urban Dataset YOLO" \
    --slug indian-urban-dataset-yolo

This will:
- Copy (or zip) the input folder
- Create a zip archive for upload
- Create a dataset-metadata.json suitable for Kaggle CLI

Then upload using Kaggle CLI:
  kaggle datasets create -p dist/kaggle_dataset
Or, for updates:
  kaggle datasets version -p dist/kaggle_dataset -m "Update"
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-dir', required=True, help='Path to your YOLO dataset root (contains data.yaml, images/, labels/)')
    p.add_argument('--output-dir', default='dist/kaggle_dataset', help='Where to place the packaged dataset folder')
    p.add_argument('--owner', required=True, help='Kaggle username (owner)')
    p.add_argument('--title', required=True, help='Kaggle dataset title')
    p.add_argument('--slug', required=True, help='Kaggle dataset slug (lowercase, hyphens)')
    p.add_argument('--zip-name', default=None, help='Optional custom zip name (default: <slug>.zip)')
    p.add_argument('--copy', action='store_true', help='Copy dataset tree into output folder (default just zips from source)')
    args = p.parse_args()

    ds_root = Path(args.dataset_dir).resolve()
    if not ds_root.exists():
        print(f"Dataset dir not found: {ds_root}")
        sys.exit(1)

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Create dataset-metadata.json
    meta = {
        "title": args.title,
        "id": f"{args.owner}/{args.slug}",
        "licenses": [{"name": "CC0-1.0"}],
        "subtitle": "YOLO formatted dataset",
        "description": f"Packaged on {datetime.utcnow().isoformat()}Z",
    }
    meta_path = out_root / 'dataset-metadata.json'
    meta_path.write_text(json.dumps(meta, indent=2))
    print('Wrote', meta_path)

    # Optionally copy the tree into output folder for clarity
    staging_dir = out_root / 'content'
    if args.copy:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        shutil.copytree(ds_root, staging_dir)
        zip_src = staging_dir
    else:
        zip_src = ds_root

    zip_name = args.zip_name or args.slug
    zip_base = out_root / zip_name
    created = shutil.make_archive(str(zip_base), 'zip', root_dir=zip_src)
    print('Created zip:', created)

    print('\nNext steps:')
    print(f"  1) Ensure you are logged in: kaggle config view (and kaggle.json is configured)")
    print(f"  2) Create dataset: kaggle datasets create -p {out_root}")
    print(f"     or version existing: kaggle datasets version -p {out_root} -m 'Update'")
    print(f"  3) In your Kaggle Notebook, attach dataset '{args.owner}/{args.slug}'.")


if __name__ == '__main__':
    main()
