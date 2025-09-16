import os
import random
import shutil
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# === CONFIG ===
PROJECT_DIR = Path("/Users/abhishekx/Desktop/City-surveillance/Data/crowd")
RAW_IMAGES = PROJECT_DIR / "images_raw"
DATASET_DIR = PROJECT_DIR / "dataset"
TRAIN_SPLIT = 0.8  # 80% train, 20% val

# YOLO class index for "person" in COCO
PERSON_CLASS_ID = 0

# === PREPARE PATHS ===
for folder in ["images/train", "images/val", "labels/train", "labels/val"]:
    (DATASET_DIR / folder).mkdir(parents=True, exist_ok=True)

# === LOAD YOLO MODEL (Pretrained COCO) ===
model = YOLO("yolov8n.pt")  # Lightest model for annotation

# === STEP 1: Annotate Images ===
print("[INFO] Annotating images (person-only)...")
for img_path in RAW_IMAGES.glob("*.*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    try:
        # Validate image
        Image.open(img_path).verify()

        # Run YOLO inference
        results = model(img_path)
        detections = results[0].boxes.data.cpu().numpy()

        # Prepare label file
        label_path = img_path.with_suffix(".txt")
        with open(label_path, "w") as f:
            for det in detections:
                cls, x1, y1, x2, y2 = int(det[5]), det[0], det[1], det[2], det[3]
                if cls != PERSON_CLASS_ID:  # Skip non-person
                    continue
                # Normalize bbox
                cx = ((x1 + x2) / 2) / results[0].orig_shape[1]
                cy = ((y1 + y2) / 2) / results[0].orig_shape[0]
                bw = (x2 - x1) / results[0].orig_shape[1]
                bh = (y2 - y1) / results[0].orig_shape[0]
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

        print(f"[OK] Annotated {img_path.name}")
    except Exception as e:
        print(f"[ERROR] Skipping {img_path.name}: {e}")

# === STEP 2: Clean Images Without Labels ===
print("[INFO] Cleaning unlabeled images...")
for img_path in RAW_IMAGES.glob("*.*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue
    label_path = img_path.with_suffix(".txt")
    if not label_path.exists() or os.path.getsize(label_path) == 0:
        print(f"[CLEAN] Removing {img_path.name} (no persons detected)")
        img_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)

# === STEP 3: Split Train/Val ===
print("[INFO] Splitting dataset...")
all_images = list(RAW_IMAGES.glob("*.jpg")) + list(RAW_IMAGES.glob("*.png"))
random.shuffle(all_images)

split_idx = int(len(all_images) * TRAIN_SPLIT)
train_imgs, val_imgs = all_images[:split_idx], all_images[split_idx:]

for img_list, split in [(train_imgs, "train"), (val_imgs, "val")]:
    for img_path in img_list:
        label_path = img_path.with_suffix(".txt")
        shutil.copy(img_path, DATASET_DIR / f"images/{split}/{img_path.name}")
        shutil.copy(label_path, DATASET_DIR / f"labels/{split}/{label_path.name}")

# === STEP 4: Write crowd.yaml ===
yaml_content = f"""
path: {DATASET_DIR}
train: images/train
val: images/val
names:
  0: person
background: true
"""
with open(PROJECT_DIR / "crowd.yaml", "w") as f:
    f.write(yaml_content)

print(f"[DONE] Dataset ready at {DATASET_DIR}")
