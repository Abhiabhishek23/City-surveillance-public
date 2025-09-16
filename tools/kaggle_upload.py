#!/usr/bin/env python3
"""
Kaggle Dataset Uploader
-----------------------

Purpose:
- Package a dataset folder into a single zip file and publish it as a Kaggle dataset.
- Works around web UI folder upload issues by using the Kaggle CLI/API.

Features:
- Create or update an existing dataset (automatic fallback when dataset exists).
- Private or public visibility.
- Custom README and license.
- Excludes hidden/system files like .DS_Store and __MACOSX.

Usage examples:
  python tools/kaggle_upload.py \
    --dataset-dir dataset \
    --owner your_kaggle_username \
    --slug urban-encroachment-yolo \
    --title "Urban Encroachment YOLO Dataset" \
    --private \
    --message "Initial upload (train/val split, YOLO labels)"

  # Update existing dataset with a new version
  python tools/kaggle_upload.py \
    --dataset-dir dataset \
    --owner your_kaggle_username \
    --slug urban-encroachment-yolo \
    --private \
    --message "Added 350 new images and labels"

Before running:
- Ensure Kaggle API is configured: place kaggle.json in ~/.kaggle with chmod 600.
  See: https://www.kaggle.com/settings/account (Create New API Token)

"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9\-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "dataset"


def ensure_kaggle_credentials() -> None:
    # Respect KAGGLE_CONFIG_DIR if set, else default to ~/.kaggle
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    config_file = config_dir / "kaggle.json"
    if not config_file.exists():
        raise SystemExit(
            f"Kaggle API token not found. Expected at: {config_file}\n"
            "Create a token from Kaggle: https://www.kaggle.com/settings/account -> Create New API Token\n"
            f"Then place it at {config_file} and set permissions: chmod 600 {config_file}"
        )


def zip_dataset(src_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    base_dir = src_dir

    def _is_skippable(p: Path) -> bool:
        name = p.name
        if name.startswith("."):
            return True
        if name in {"__pycache__", "__MACOSX"}:
            return True
        if name.endswith(".pyc"):
            return True
        return False

    import zipfile

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(base_dir):
            # filter dirs in-place to skip hidden/system dirs
            dirs[:] = [d for d in dirs if not _is_skippable(Path(d))]
            for fn in files:
                fp = Path(root) / fn
                if _is_skippable(fp):
                    continue
                arcname = fp.relative_to(base_dir)
                zf.write(fp, arcname)


def write_metadata(build_dir: Path, owner: str, slug: str, title: str, license_name: str) -> Path:
    # Kaggle requires dataset-metadata.json with minimum fields: title, id, licenses
    meta = {
        "title": title,
        "id": f"{owner}/{slug}",
        "licenses": [
            {"name": license_name}
        ],
    }
    meta_path = build_dir / "dataset-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def write_readme(build_dir: Path, readme_text: Optional[str], dataset_subdir: str = "dataset/") -> Optional[Path]:
    if not readme_text:
        # Provide a helpful default README with unzip instructions
        readme_text = f"""
# Dataset Contents

This dataset contains a zipped folder `data.zip` with the data under `{dataset_subdir}` (images and labels in YOLO format).

## Usage (Kaggle Notebook)
```python
import zipfile, os
zip_path = '/kaggle/input/'  # auto-mounted datasets live here
# Replace <owner>-<slug> with your dataset slug used on Kaggle\n
ds_root = '/kaggle/input/<owner>-<slug>'
zip_file = os.path.join(ds_root, 'data.zip')
extract_to = '/kaggle/working'
with zipfile.ZipFile(zip_file, 'r') as zf:
    zf.extractall(extract_to)

!ls -R /kaggle/working/{dataset_subdir}
```

Then point your training config (e.g., Ultralytics YOLO) to the extracted `{dataset_subdir}` path.
""".strip()

    readme_path = build_dir / "README.md"
    readme_path.write_text(readme_text, encoding="utf-8")
    return readme_path


def run_kaggle_cli(args: list[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        # Bubble up with combined output for easier debugging
        print(e.stdout or "", file=sys.stderr)
        raise


def create_or_update_dataset(build_dir: Path, visibility: str, message: str) -> str:
    # Try creating first; if 409 conflict, version it instead
    # Kaggle CLI: create uses -u/--public to make public; default is private. No --private flag exists.
    try:
        print("Creating new Kaggle dataset...")
        args = ["kaggle", "datasets", "create", "-p", str(build_dir)]
        if visibility == "public":
            args.append("--public")
        res = run_kaggle_cli(args)
        print(res.stdout)
        # Extract URL if present
        for line in res.stdout.splitlines():
            if line.strip().startswith("https://www.kaggle.com/datasets/"):
                return line.strip()
        return "Created (URL not detected). Check your Kaggle Datasets page."
    except subprocess.CalledProcessError as e:
        out = e.stdout or ""
        if "409" in out or "Conflict" in out or "already exists" in out:
            print("Dataset exists. Creating a new version...")
            # `version` does not accept visibility flags; it inherits existing visibility
            res = run_kaggle_cli([
                "kaggle", "datasets", "version", "-p", str(build_dir), "-m", message
            ])
            print(res.stdout)
            for line in res.stdout.splitlines():
                if line.strip().startswith("https://www.kaggle.com/datasets/"):
                    return line.strip()
            return "Versioned (URL not detected). Check your Kaggle Datasets page."
        raise


def main():
    parser = argparse.ArgumentParser(description="Zip a dataset folder and publish to Kaggle.")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset folder to zip and upload.")
    parser.add_argument("--owner", required=True, help="Kaggle username (dataset owner).")
    parser.add_argument("--slug", required=True, help="Dataset slug (lowercase, dash-separated).")
    parser.add_argument("--title", default=None, help="Dataset title (defaults to slug pretty-cased).")
    parser.add_argument("--license", dest="license_name", default="CC-BY-4.0", help="Kaggle license name (e.g., CC0-1.0, CC-BY-4.0).")
    vis = parser.add_mutually_exclusive_group()
    vis.add_argument("--public", action="store_true", help="Make dataset public.")
    vis.add_argument("--private", action="store_true", help="Make dataset private (default).")
    parser.add_argument("--message", default=None, help="Version message for updates (optional). If omitted, a timestamped message is used.")
    parser.add_argument("--readme", default=None, help="Path to a README.md to include; if omitted, a default README is generated.")
    parser.add_argument("--build-dir", default=None, help="Directory to stage the upload (defaults to ./kaggle_build/<slug>-YYYYmmddHHMMSS).")
    parser.add_argument("--zip-name", default="data.zip", help="Name of the zip file inside the dataset (default: data.zip).")
    parser.add_argument("--dataset-subdir", default="dataset/", help="A hint for README about where data lives inside the zip.")

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise SystemExit(f"Dataset dir not found or not a directory: {dataset_dir}")

    owner = args.owner.strip()
    slug = _slugify(args.slug)
    title = args.title or slug.replace("-", " ").title()
    license_name = args.license_name
    visibility = "public" if args.public else "private"
    message = args.message or f"Update {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}"

    ensure_kaggle_credentials()

    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    default_build_root = Path.cwd() / "kaggle_build"
    build_dir = Path(args.build_dir).resolve() if args.build_dir else (default_build_root / f"{slug}-{timestamp}")
    build_dir.mkdir(parents=True, exist_ok=True)

    # Create zip
    zip_path = build_dir / args.zip_name
    print(f"Zipping dataset from {dataset_dir} -> {zip_path} ...")
    zip_dataset(dataset_dir, zip_path)

    # Write metadata & README
    write_metadata(build_dir, owner=owner, slug=slug, title=title, license_name=license_name)
    if args.readme:
        readme_src = Path(args.readme).resolve()
        if not readme_src.exists():
            raise SystemExit(f"README file not found: {readme_src}")
        shutil.copy2(readme_src, build_dir / "README.md")
    else:
        write_readme(build_dir, None, dataset_subdir=args.dataset_subdir)

    # Create or update the Kaggle dataset
    url = create_or_update_dataset(build_dir, visibility=visibility, message=message)
    print("Done.")
    print(url)


if __name__ == "__main__":
    main()
