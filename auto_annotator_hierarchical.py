#!/usr/bin/env python3
"""
Auto-annotator with scene-level hierarchy (CLIP) + YOLO objects.

Saves per image:
 - YOLO .txt labels (object boxes in your taxonomy)
 - scene JSON (scene tags + detection metadata)

Also saves:
 - class_distribution.json (counts per taxonomy class)
 - data.yaml (ready for YOLO training)
 - train/ and val/ split folders under --out

Usage example:
  python auto_annotator_hierarchical.py \
    --images Indian_Urban_Dataset_yolo/images/train \
    --out annotations_hier \
    --yolo yolov12n.pt \
    --conf 0.25 \
    --person_limit 8 \
    --suppress_people_in_priority

Requirements:
  pip install ultralytics==8.* Pillow ftfy regex tqdm numpy PyYAML
  pip install git+https://github.com/openai/CLIP.git  # or: pip install openai-clip
  (CLIP and YOLO run faster with GPU; CPU works but slower.)
"""

from __future__ import annotations

import os
import json
import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from PIL import Image

# YOLO
from ultralytics import YOLO

# CLIP
import torch
import clip  # from openai/CLIP or openai-clip


# --------------------------
# Configuration / Taxonomy
# --------------------------
# Expandable taxonomy with synonyms used for CLIP prompt matching
CLASS_MAP: Dict[str, List[str]] = {
    # --- PEOPLE ---
    "People/Pedestrians/walking": ["person walking", "people walking", "pedestrian"],
    "People/Pedestrians/standing": ["person standing", "people waiting", "people standing"],
    "People/Pedestrians/sitting": ["person sitting", "sitting people"],
    "People/Crowds/procession": ["procession", "religious procession", "march", "yatra"],
    "People/Crowds/pilgrim": ["pilgrim", "pilgrimage crowd"],
    "People/Special_actions/arti": ["arti", "aarti", "worship ceremony", "prayer"],
    "People/Special_actions/immersion": ["idol immersion", "visarjan", "river immersion"],
    "People/Informal_presence/hawkers": ["hawker", "street vendor", "fruit seller", "shopkeeper", "stall"],
    "People/Informal_presence/beggars": ["beggar", "homeless person"],
    "People/Informal_presence/street_children": ["street child", "child labor", "orphan child"],

    # --- WASTE ---
    "Waste/Garbage_bins/overflowing": ["overflowing dustbin", "garbage pile"],
    "Waste/Garbage_bins/normal": ["dustbin", "garbage bin"],
    "Waste/Open_dumping": ["open garbage dump", "landfill"],
    "Waste/Litter/roadside": ["roadside litter", "plastic waste"],
    "Waste/Sewage/open_drain": ["open drain", "dirty drain", "sewage"],

    # --- TRANSPORT ---
    "Transport/Vehicles/e-rickshaw": ["e-rickshaw", "electric rickshaw"],
    "Transport/Vehicles/auto": ["auto rickshaw", "autorickshaw"],
    "Transport/Vehicles/bus": ["bus", "public bus"],
    "Transport/Vehicles/truck": ["truck", "lorry"],
    "Transport/Vehicles/car": ["car", "taxi"],
    "Transport/Vehicles/two_wheeler": ["bike", "scooter", "motorcycle", "motorbike"],
    "Transport/Vehicles/cycle": ["bicycle", "cycle"],
    "Transport/Vehicles/cart": ["handcart", "pushcart", "thela"],

    # --- FESTIVALS ---
    "Urban/Festivals/Holi": ["Holi", "color festival", "throwing colors", "holi festival"],
    "Urban/Festivals/Diwali": ["Diwali", "festival of lights", "diyas", "diwali festival"],
    "Urban/Festivals/Ganesh_Chaturthi": ["Ganesh idol", "Ganesh festival"],
    "Urban/Festivals/Durga_Puja": ["Durga idol", "Durga pandal"],
    "Urban/Festivals/Eid": ["Eid prayer", "namaz"],
    "Urban/Festivals/Christmas": ["Christmas", "church celebration"],

    # --- STRUCTURES ---
    "Structures/Pandals/religious": ["pandal", "festival pandal", "decorated pandal"],
    "Structures/Temples": ["temple", "mandir"],
    "Structures/Mosques": ["mosque", "masjid"],
    "Structures/Churches": ["church"],
    "Structures/Public_stage": ["public stage", "concert stage"],

    # --- RIVER/ENVIRONMENT ---
    "River/Activities/bathing": ["river bathing", "holy dip", "snan"],
    "River/Activities/washing_clothes": ["washing clothes in river"],
    "River/Activities/boating": ["boat", "rowing"],
    "River/Conditions/polluted": ["polluted river", "dirty river"],
    "River/Conditions/clean": ["clean river", "fresh water"],

    # --- URBAN/OTHER ---
    "Urban/Markets": ["market", "bazaar", "mandi", "night bazaar", "crowded shops"],
    "Urban/Roads/encroachment": ["roadside encroachment", "illegal stalls", "footpath encroachment"],
    "Urban/Events/protest": ["protest", "rally", "dharna"],
    "Urban/Events/wedding": ["wedding procession", "baraat"],
    "Urban/Events/funeral": ["funeral procession", "shav yatra"],
}

# Scene-level prompts (image-level tags)
SCENE_PROMPTS: List[str] = [
    "arti", "aarti", "worship ceremony", "religious procession",
    "idol immersion", "visarjan", "Holi festival", "Diwali festival",
    "protest", "election rally", "wedding procession", "funeral procession",
    "market", "night bazaar", "sand mining site", "garbage dump"
]

# Priority scenes influence suppression rules (avoid generic 'person' dominance)
PRIORITY_SCENES = [
    "arti", "aarti", "idol immersion", "visarjan", "religious procession",
    "Holi festival", "Diwali festival", "protest", "wedding procession"
]


# --------------------------
# Helpers
# --------------------------

def preferred_device(force_cpu: bool = False) -> str:
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(yolo_path: str, device: str):
    yolo = YOLO(yolo_path)
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    return yolo, clip_model, clip_preprocess


def run_yolo_inference(yolo: YOLO, image_path: str, conf: float = 0.25, iou: float = 0.45):
    results = yolo(image_path, conf=conf, iou=iou)
    res = results[0]
    boxes = []
    if hasattr(res, "boxes") and res.boxes is not None:
        xyxy = res.boxes.xyxy.cpu().numpy()
        cls_idx = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()
        names = res.names
        for (x1, y1, x2, y2), cidx, confv in zip(xyxy, cls_idx, confs):
            label = names.get(int(cidx), str(cidx))
            boxes.append({
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "label": label,
                "conf": float(confv),
            })
    return boxes


@torch.no_grad()
def run_clip_image_scene(clip_model, preprocess, device: str, image: Image.Image, prompts: List[str]):
    img = preprocess(image).unsqueeze(0).to(device)
    text = clip.tokenize(prompts).to(device)
    image_features = clip_model.encode_image(img)
    text_features = clip_model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    sims_np = sims.cpu().numpy().flatten()
    ranked = sorted(zip(prompts, sims_np), key=lambda x: x[1], reverse=True)
    return ranked


@torch.no_grad()
def run_clip_crop_classify(clip_model, preprocess, device: str, crop_img: Image.Image, prompts: List[str]) -> Tuple[int, float]:
    img = preprocess(crop_img).unsqueeze(0).to(device)
    text = clip.tokenize(prompts).to(device)
    image_features = clip_model.encode_image(img)
    text_features = clip_model.encode_text(text)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    sims_np = sims.cpu().numpy().flatten()
    best_idx = int(np.argmax(sims_np))
    return best_idx, float(sims_np[best_idx])


def map_detection_to_custom(box_label: str, crop_img: Image.Image, clip_model, preprocess, device: str):
    """Map YOLO label to taxonomy via synonyms; fallback to CLIP crop classification."""
    low = box_label.lower()
    for tax, synonyms in CLASS_MAP.items():
        for s in synonyms:
            if s.lower() in low or low in s.lower():
                return tax, 0.99

    # Fallback: CLIP-best among all synonyms
    flat_prompts: List[str] = []
    tax_by_prompt: List[str] = []
    for tax, syns in CLASS_MAP.items():
        for s in syns:
            flat_prompts.append(s)
            tax_by_prompt.append(tax)
    if not flat_prompts:
        return None, 0.0
    best_idx, score = run_clip_crop_classify(clip_model, preprocess, device, crop_img, flat_prompts)
    mapped_tax = tax_by_prompt[best_idx]
    return mapped_tax, float(score)


def save_yolo_txt_for_image(img_path: str, detections: List[Dict], class_list: List[str], out_labels_dir: str, rel_dir: str = ""):
    im = Image.open(img_path)
    W, H = im.size
    basename = os.path.splitext(os.path.basename(img_path))[0]
    target_dir = os.path.join(out_labels_dir, rel_dir) if rel_dir else out_labels_dir
    os.makedirs(target_dir, exist_ok=True)
    txt_path = os.path.join(target_dir, f"{basename}.txt")
    lines: List[str] = []
    for d in detections:
        tax = d.get("mapped_tax")
        if not tax:
            continue
        try:
            cls_id = class_list.index(tax)
        except ValueError:
            continue
        x1, y1, x2, y2 = d["xyxy"]
        xc = ((x1 + x2) / 2.0) / W
        yc = ((y1 + y2) / 2.0) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        # clip to [0,1]
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        bw = min(max(bw, 0.0), 1.0)
        bh = min(max(bh, 0.0), 1.0)
        if bw <= 0 or bh <= 0:
            continue
        lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))


def save_scene_json(img_path: str, scene_tags: List[Dict], detections: List[Dict], out_scene_dir: str, rel_dir: str = ""):
    basename = os.path.splitext(os.path.basename(img_path))[0]
    target_dir = os.path.join(out_scene_dir, rel_dir) if rel_dir else out_scene_dir
    os.makedirs(target_dir, exist_ok=True)
    json_path = os.path.join(target_dir, f"{basename}.json")
    payload = {
        "image": os.path.basename(img_path),
        "scenes": scene_tags,
        "detections": detections,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)


def split_dataset(images_dir: str, labels_dir: str, out_dir: str, val_ratio: float = 0.2) -> Tuple[str, str]:
    train_img_dir = os.path.join(out_dir, "train/images")
    train_lbl_dir = os.path.join(out_dir, "train/labels")
    val_img_dir = os.path.join(out_dir, "val/images")
    val_lbl_dir = os.path.join(out_dir, "val/labels")

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # Collect images recursively with relative paths
    items: List[Tuple[str, str]] = []  # (abs_img, rel_path)
    for root, _dirs, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                abs_img = os.path.join(root, f)
                rel_path = os.path.relpath(abs_img, images_dir)
                items.append((abs_img, rel_path))
    random.shuffle(items)
    split_idx = int(len(items) * (1 - val_ratio))
    train_items = items[:split_idx]
    val_items = items[split_idx:]

    def copy_pairs(pairs: List[Tuple[str, str]], img_dest_root: str, lbl_dest_root: str):
        for abs_img, rel_path in pairs:
            rel_dir = os.path.dirname(rel_path)
            os.makedirs(os.path.join(img_dest_root, rel_dir), exist_ok=True)
            os.makedirs(os.path.join(lbl_dest_root, rel_dir), exist_ok=True)
            shutil.copy(abs_img, os.path.join(img_dest_root, rel_dir, os.path.basename(rel_path)))
            base = os.path.splitext(os.path.basename(rel_path))[0]
            lbl_src = os.path.join(labels_dir, rel_dir, base + ".txt")
            if os.path.exists(lbl_src):
                shutil.copy(lbl_src, os.path.join(lbl_dest_root, rel_dir, base + ".txt"))

    copy_pairs(train_items, train_img_dir, train_lbl_dir)
    copy_pairs(val_items, val_img_dir, val_lbl_dir)

    print(f"[INFO] Split done → {len(train_items)} train, {len(val_items)} val")
    return train_img_dir, val_img_dir


# --------------------------
# Main pipeline
# --------------------------

def process_images(args):
    device = preferred_device(args.cpu)
    yolo, clip_model, clip_preprocess = load_models(args.yolo, device)

    # Prepare output dirs
    labels_dir = os.path.join(args.out, "labels")
    scenes_dir = os.path.join(args.out, "scenes")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(scenes_dir, exist_ok=True)

    class_list = list(CLASS_MAP.keys())  # class id = index in this list
    scene_prompts = SCENE_PROMPTS

    # Images to process (recursive)
    image_files = []
    for root, _dirs, files in os.walk(args.images):
        for p in files:
            if p.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(os.path.join(root, p))
    image_files.sort()
    print(f"[INFO] Found {len(image_files)} images, device={device}")

    for img_path in tqdm(image_files):
        rel_dir = os.path.dirname(os.path.relpath(img_path, args.images))
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("[WARN] Failed to open", img_path, e)
            continue

        # 1) Scene-level classification
        ranked_scenes = run_clip_image_scene(clip_model, clip_preprocess, device, pil_img, scene_prompts)
        scene_tags = [{"scene": s, "score": float(sc)} for s, sc in ranked_scenes if sc >= args.scene_thresh][: args.topk_scenes]
        scene_names = [st["scene"] for st in scene_tags]

        # 2) YOLO object detection
        yolo_boxes = run_yolo_inference(yolo, img_path, conf=args.conf, iou=args.iou)

        # 3) Map detections to taxonomy
        final_detections = []
        for box in yolo_boxes:
            x1, y1, x2, y2 = [int(max(0, round(v))) for v in box["xyxy"]]
            crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
            mapped_tax, map_score = map_detection_to_custom(box["label"], crop, clip_model, clip_preprocess, device)
            det = {
                "xyxy": [x1, y1, x2, y2],
                "yolo_label": box["label"],
                "yolo_conf": box["conf"],
                "mapped_tax": mapped_tax,
                "mapping_score": map_score,
            }
            final_detections.append(det)

        # 4) Scene-priority suppression of generic people
        present_priority = any(s.lower() in (p.lower() for p in PRIORITY_SCENES) for s in scene_names)
        if present_priority and args.suppress_people_in_priority:
            filtered = []
            for d in final_detections:
                if d["mapped_tax"] and not d["mapped_tax"].startswith("People/Pedestrians"):
                    filtered.append(d)
                    continue
                if d["yolo_label"].lower() != "person":
                    filtered.append(d)
                    continue
                # else drop person-only
            final_detections = filtered

        # 5) Cap person boxes per image
        person_boxes = [d for d in final_detections if d.get("yolo_label", "").lower() == "person" or (d.get("mapped_tax") or "").startswith("People/Pedestrians")]
        other_boxes = [d for d in final_detections if d not in person_boxes]
        if len(person_boxes) > args.person_limit:
            person_keep = random.sample(person_boxes, args.person_limit)
            final_detections = other_boxes + person_keep
        else:
            final_detections = other_boxes + person_boxes

        # 6) Try mapping again for unmapped detections with lower threshold
        for d in final_detections:
            if d.get("mapped_tax") is None:
                x1, y1, x2, y2 = d["xyxy"]
                crop = pil_img.crop((x1, y1, x2, y2)).convert("RGB")
                mapped_tax, score = map_detection_to_custom(d["yolo_label"], crop, clip_model, clip_preprocess, device)
                if mapped_tax and score >= args.clip_map_thresh:
                    d["mapped_tax"] = mapped_tax
                    d["mapping_score"] = score

        # 7) Save outputs for this image
        save_yolo_txt_for_image(img_path, final_detections, class_list, labels_dir, rel_dir=rel_dir)
        save_scene_json(img_path, scene_tags, final_detections, scenes_dir, rel_dir=rel_dir)

    # --- Class distribution report ---
    freq = {c: 0 for c in class_list}
    # Walk recursively to count all labels under labels_dir
    for root, _dirs, files in os.walk(labels_dir):
        for lbl_file in files:
            if not lbl_file.endswith(".txt"):
                continue
            with open(os.path.join(root, lbl_file)) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(parts[0])
                    except Exception:
                        continue
                    if 0 <= cls_id < len(class_list):
                        cls_name = class_list[cls_id]
                        freq[cls_name] += 1
    with open(os.path.join(args.out, "class_distribution.json"), "w") as f:
        json.dump(freq, f, indent=2)
    print("[INFO] Class distribution saved to class_distribution.json")

    # --- Auto train/val split and data.yaml ---
    if 0.0 < args.val_ratio < 1.0:
        train_path, val_path = split_dataset(args.images, labels_dir, args.out, val_ratio=args.val_ratio)
    else:
        # No split requested: point both train and val to the source images folder
        train_path = args.images
        val_path = args.images
    data_yaml_text = """train: {}
val: {}
nc: {}
names:
""".format(Path(train_path).resolve(), Path(val_path).resolve(), len(class_list))
    for i, n in enumerate(class_list):
        data_yaml_text += f"  {i}: {n}\n"
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(data_yaml_text)
    print("[INFO] data.yaml generated for YOLO training.")
    print("[DONE] Auto-annotation finished.")
    print("[INFO] Labels dir:", labels_dir)
    print("[INFO] Scenes dir:", scenes_dir)
    print("[INFO] Split under:", args.out)


# --------------------------
# CLI
# --------------------------

def main():
    p = argparse.ArgumentParser(description="Hierarchical auto-annotator (CLIP scenes + YOLO objects)")
    p.add_argument("--images", type=str, required=True, help="path to images folder")
    p.add_argument("--out", type=str, default="annotations", help="output folder for labels/scenes/splits")
    p.add_argument("--yolo", type=str, required=True, help="path to YOLO model (e.g., yolov12n.pt or yolov8n.pt)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--scene_thresh", type=float, default=0.05, help="min softmax score to include scene tag")
    p.add_argument("--topk_scenes", type=int, default=3, help="max number of scene tags per image")
    p.add_argument("--person_limit", type=int, default=10, help="max person boxes to keep per image")
    p.add_argument("--clip_map_thresh", type=float, default=0.2, help="min CLIP score to accept mapping fallback")
    p.add_argument("--suppress_people_in_priority", action="store_true", help="suppress generic person boxes if priority scene detected")
    p.add_argument("--cpu", action="store_true", help="force CPU usage")
    p.add_argument("--val_ratio", type=float, default=0.2, help="validation split ratio (0-1)")
    args = p.parse_args()
    process_images(args)


if __name__ == "__main__":
    main()
