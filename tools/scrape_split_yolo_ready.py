#!/usr/bin/env python3
import os
import shutil
import random
import argparse
import uuid
import time
import random
from pathlib import Path
from typing import Dict, List, Optional

import requests

# Prefer Bing downloader if importable; else fall back to DuckDuckGo search
try:
    from bing_image_downloader import downloader as bing_downloader
except Exception:
    bing_downloader = None

DDGS = None
DDGS_ALT = None
try:
    # Preferred new package name
    from ddgs import DDGS as _DDGS
    DDGS = _DDGS
except Exception:
    DDGS = None
try:
    # Legacy/alternative package that often yields results where ddgs fails
    from duckduckgo_search import DDGS as _DDGS_ALT
    DDGS_ALT = _DDGS_ALT
except Exception:
    DDGS_ALT = None

# ---------------------------
# Class-to-search-query mapping (~30 classes)
# ---------------------------
SEARCH_QUERIES: Dict[str, List[str]] = {
    "vehicle_car": ["car on Indian road", "illegal car parking India", "parked car ghat Ujjain"],
    "vehicle_bike": ["motorbike India traffic", "bike parked footpath", "crowd motorcycles India"],
    "vehicle_auto": ["auto rickshaw India", "auto parked roadside", "autorickshaw Ujjain fair"],
    "vehicle_truck_bus": ["truck parking India", "bus parked ghat India", "festival bus traffic India"],
    "vehicle_tractor": ["tractor road India", "tractor festival Indore", "tractor parking ghat"],
    "emergency_vehicle": ["police jeep India", "ambulance festival India", "fire brigade truck India"],
    "pedestrian_walking": ["people walking Indore street", "pilgrims walking Ujjain Kumbh", "pedestrians crossing India"],
    "pedestrian_queue": ["pilgrims queue ghat India", "people standing temple line", "queue Kumbh Mela India"],
    "beggar_squatter": ["beggar roadside India", "squatter Indian street", "homeless sleeping footpath India"],
    "pandal_tent": ["pandal tent festival India", "temporary tent Kumbh Mela", "festival shamiyana India"],
    "stage_platform": ["festival stage platform India", "temporary stage Kumbh Mela"],
    "idol_statue": ["idol installation ghat India", "religious statue roadside India", "Ganesh idol immersion ghat"],
    "flag_banner": ["festival flag India street", "religious banner India", "Kumbh Mela flags"],
    "religious_marker": ["roadside shrine India", "small temple public land", "unauthorized religious structure India"],
    "vendor_cart_thela": ["street vendor cart India", "thela Indore", "hawker pushcart India"],
    "food_stall": ["food stall festival India", "poha jalebi stall Indore", "chaat stall Ujjain"],
    "kiosk_cabin": ["wooden kiosk shop India", "temporary cabin stall India"],
    "shop_house": ["shop Indore street", "house near ghat India", "roadside shop India"],
    "barricade_fence": ["police barricade India", "festival crowd barricade", "temporary fence India"],
    "portable_toilet": ["portable toilet cabin India", "temporary toilet block festival"],
    "dustbin_dump": ["dustbin roadside India", "municipal garbage bin India"],
    "garbage_heap": ["garbage heap roadside India", "waste dump ghat India", "open garbage pile festival India"],
    "water_tank_tap": ["drinking water tank India", "public water tap Indore", "festival water supply India"],
    "bus_shelter_signage": ["bus stop India", "public signage Indore", "bus shelter India"],
    "cctv_tower_drone_station": ["cctv tower India", "festival surveillance tower", "drone station Kumbh Mela"],
    "open_fire_stove": ["open cooking stove India", "festival cooking fire India", "roadside chulha India"],
    "sand_heap": ["sand mining riverbed India", "sand heap construction India", "illegal sand pile riverbank"],
    "sewage_pipe_drain": ["sewage pipe river India", "open drain India", "wastewater pipe ghat India"],
    "boat": ["boat Ujjain ghat", "overloaded boat India", "fishing boat river India"],
    "hoarding_poster_banner": ["festival hoarding India", "political poster wall India", "banner street India"],
}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
RAW_DIR = os.path.join(BASE_DIR, "raw")  # raw downloads
SPLITS = ["train", "val", "test"]


def _download_ddg(query: str, save_dir: Path, limit: int):
    """Try DuckDuckGo image search using both ddgs and duckduckgo_search backends.

    We attempt the 'ddgs' package first when available, then fall back to
    'duckduckgo_search' which often returns results even when 'ddgs' does not.
    """
    if DDGS is None and DDGS_ALT is None:
        raise SystemExit("duckduckgo-search not installed. pip install ddgs or duckduckgo-search")

    save_dir.mkdir(parents=True, exist_ok=True)
    ua = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15'
    }

    backends = []
    if DDGS is not None:
        backends.append(("ddgs", DDGS))
    if DDGS_ALT is not None:
        backends.append(("duckduckgo_search", DDGS_ALT))

    got_total = 0
    for name, klass in backends:
        retries = 3
        for attempt in range(retries):
            try:
                with klass() as dd:
                    got = 0
                    try:
                        iterator = dd.images(query=query, max_results=limit)
                    except TypeError:
                        iterator = dd.images(keywords=query, max_results=limit)
                    for r in iterator:
                        url = r.get("image") or r.get("thumbnail") or r.get("url")
                        if not url:
                            continue
                        try:
                            resp = requests.get(url, timeout=20, headers=ua)
                            if resp.status_code != 200 or not resp.content:
                                continue
                            ext = ".jpg"
                            fn = f"{uuid.uuid4().hex}{ext}"
                            (save_dir / fn).write_bytes(resp.content)
                            got += 1
                            got_total += 1
                            if got_total >= limit:
                                return
                        except Exception:
                            continue
                    # If this backend yielded some but not enough, continue to next attempt/backend
                    if got == 0:
                        raise RuntimeError("No results found.")
                    else:
                        break
            except Exception as e:
                sleep_s = 5 * (attempt + 1) + random.uniform(0, 3)
                print(f"DDG[{name}] error: {e}. Retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
        # next backend after retries
    if got_total == 0:
        print("DDG: exhausted retries across backends; continuing.")


def _download_bing(query: str, out_root: Path, limit: int):
    # bing_image_downloader expects an output_dir and creates subfolder by query
    # We'll call it per-label; we'll later gather under RAW_DIR/label
    assert bing_downloader is not None
    bing_downloader.download(
        query=query,
        limit=int(limit),
        output_dir=str(out_root),
        adult_filter_off=True,
        force_replace=False,
        timeout=60,
        verbose=True,
    )


def download_all(limit_per_query: int, subset: Optional[List[str]] = None, sleep_between: float = 0.0):
    os.makedirs(RAW_DIR, exist_ok=True)
    items = SEARCH_QUERIES.items()
    if subset:
        items = [(k, SEARCH_QUERIES[k]) for k in subset if k in SEARCH_QUERIES]
    for label, queries in items:
        label_dir = Path(RAW_DIR) / label
        label_dir.mkdir(parents=True, exist_ok=True)
        for q in queries:
            print(f"Downloading for class: {label}, query: {q}")
            if bing_downloader is not None:
                # Use a temp folder created by bing package under RAW_DIR; then move files
                prev = set([p.name for p in Path(RAW_DIR).glob('**/*') if p.is_file()])
                try:
                    _download_bing(q, Path(RAW_DIR), limit_per_query)
                except Exception as e:
                    print(f"Bing downloader failed, falling back to DuckDuckGo. Error: {e}")
                    _download_ddg(q, label_dir, limit_per_query)
                    continue
                # Move any newly created files under a subdir to our label_dir
                new_files = [p for p in Path(RAW_DIR).glob('**/*') if p.is_file() and p.name not in prev]
                moved = 0
                for p in new_files:
                    try:
                        # Skip files already under our label_dir
                        if label_dir in p.parents:
                            continue
                        # Only move image-like files
                        if p.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                            continue
                        dest = label_dir / f"{uuid.uuid4().hex}{p.suffix.lower()}"
                        p.replace(dest)
                        moved += 1
                    except Exception:
                        continue
                if moved == 0:
                    # Nothing detected; try fallback
                    _download_ddg(q, label_dir, limit_per_query)
            else:
                _download_ddg(q, label_dir, limit_per_query)
            if sleep_between > 0:
                try:
                    time.sleep(sleep_between)
                except Exception:
                    pass


def split_and_prepare():
    split_ratios = {"train": 0.7, "val": 0.2, "test": 0.1}

    # Create split dirs (images + labels)
    for split in SPLITS:
        for label in SEARCH_QUERIES.keys():
            os.makedirs(os.path.join(BASE_DIR, "images", split, label), exist_ok=True)
            os.makedirs(os.path.join(BASE_DIR, "labels", split, label), exist_ok=True)

    # Move and split images
    for label in SEARCH_QUERIES.keys():
        class_dir = os.path.join(RAW_DIR, label)
        if not os.path.exists(class_dir):
            continue

        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * split_ratios["train"])
        n_val = int(n * split_ratios["val"])

        for i, img in enumerate(images):
            split = "test"
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"

            src = os.path.join(class_dir, img)
            dst_img = os.path.join(BASE_DIR, "images", split, label, img)
            dst_lbl = os.path.join(BASE_DIR, "labels", split, label, os.path.splitext(img)[0] + ".txt")

            try:
                shutil.move(src, dst_img)
            except Exception:
                # handle dupes or move errors by skipping
                continue

            # Create empty YOLO label file
            with open(dst_lbl, "w") as f:
                pass

    print("YOLOv12 dataset ready with images + empty label files!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit-per-query', type=int, default=20)
    parser.add_argument('--subset', type=str, help='Comma-separated subset of classes to download')
    parser.add_argument('--sleep-between', type=float, default=0.0, help='Seconds to sleep between queries to avoid rate limits')
    args = parser.parse_args()

    subset_list = [s.strip() for s in args.subset.split(',')] if args.subset else None
    download_all(args.limit_per_query, subset_list, args.sleep_between)
    split_and_prepare()
