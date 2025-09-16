import os
import random
import shutil
from ultralytics import YOLO

# --------------------------
# CONFIG
# --------------------------
BASE_DIR = "/Users/abhishekx/Desktop/City-surveillance/Data/crowd"
IMAGES_DIR = os.path.join(BASE_DIR, "images_raw")  # Keep raw images here
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_SPLIT = 0.8
MODEL = YOLO("yolov8n.pt")  # Pretrained YOLO model for auto-annotation

# --------------------------
# STEP 1: Setup folders
# --------------------------
for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(DATASET_DIR, sub), exist_ok=True)

# --------------------------
# STEP 2: Annotate images
# --------------------------
print("[INFO] Annotating images...")
temp_labels_dir = os.path.join(BASE_DIR, "temp_labels")
os.makedirs(temp_labels_dir, exist_ok=True)

for img_file in os.listdir(IMAGES_DIR):
    if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(IMAGES_DIR, img_file)
    results = MODEL(img_path)

    label_file = os.path.join(temp_labels_dir, img_file.rsplit('.', 1)[0] + ".txt")
    with open(label_file, "w") as f:
        for r in results:
            for box in r.boxes:
                cls = int(box.cls.item())
                if cls == 0:  # class 0 = person
                    x_center, y_center, w, h = box.xywhn[0].tolist()
                    f.write(f"{cls} {x_center} {y_center} {w} {h}\n")

    print(f"[OK] {img_file} annotated.")

# --------------------------
# STEP 3: Train/Val Split
# --------------------------
print("[INFO] Splitting dataset into train/val...")

images = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(images)
split_idx = int(TRAIN_SPLIT * len(images))

for idx, img_file in enumerate(images):
    subset = "train" if idx < split_idx else "val"

    img_src = os.path.join(IMAGES_DIR, img_file)
    lbl_src = os.path.join(temp_labels_dir, img_file.rsplit('.', 1)[0] + ".txt")

    img_dst = os.path.join(DATASET_DIR, "images", subset, img_file)
    lbl_dst = os.path.join(DATASET_DIR, "labels", subset, img_file.rsplit('.', 1)[0] + ".txt")

    shutil.copy(img_src, img_dst)
    shutil.copy(lbl_src, lbl_dst)

# --------------------------
# STEP 4: Create YAML file
# --------------------------
yaml_content = f"""
path: {DATASET_DIR}
train: images/train
val: images/val

nc: 1
names: ["person"]
"""

yaml_path = os.path.join(BASE_DIR, "crowd.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"[DONE] Dataset ready at {DATASET_DIR}")
print(f"[NEXT] Train your model using:\n"
      f"yolo detect train data={yaml_path} model=yolov8n.pt epochs=50 imgsz=640")
