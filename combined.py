import os
import shutil
import random
from pathlib import Path

# ===============================
# CONFIGURATION
# ===============================
# Source folders
datasets = {
    "crowd": "/Users/abhishekx/Desktop/City-surveillance/Data/crowd/images_raw",
    "construction_legal": "/Users/abhishekx/Desktop/City-surveillance/Data/Construction/ images_raw/legal",
    "construction_illegal": "/Users/abhishekx/Desktop/City-surveillance/Data/Construction/ images_raw/illegal",
    "river": "/Users/abhishekx/Desktop/City-surveillance/Data/Aerial_Landscapes/River"
}

# Class IDs for YOLO
class_ids = {
    "crowd": 0,
    "construction_legal": 1,
    "construction_illegal": 2,
    "river": 3
}

# Combined dataset path
combined_path = Path("/Users/abhishekx/Desktop/City-surveillance/Data/combined_dataset")
train_path = combined_path / "train"
val_path = combined_path / "val"

# Train/Val split ratio
train_ratio = 0.8

# Random seed for reproducibility
random.seed(42)

# ===============================
# CREATE FOLDERS
# ===============================
for folder in [train_path / "images", train_path / "labels", val_path / "images", val_path / "labels"]:
    folder.mkdir(parents=True, exist_ok=True)

# ===============================
# FUNCTION TO COPY & RENAME
# ===============================
def process_dataset(name, src_folder):
    src_folder = Path(src_folder)
    images = list(src_folder.glob("*.[jJ][pP][gG]")) + \
             list(src_folder.glob("*.[jJ][pP][eE][gG]")) + \
             list(src_folder.glob("*.[pP][nN][gG]"))

    print(f"Processing {name}: {len(images)} images")
    
    for i, img_path in enumerate(images, 1):
        prefix = f"{name}_"
        new_name = f"{prefix}{i:04d}.jpg"
        
        # Determine if train or val
        if random.random() < train_ratio:
            dst_img = train_path / "images" / new_name
            dst_label = train_path / "labels" / new_name.replace(".jpg", ".txt")
        else:
            dst_img = val_path / "images" / new_name
            dst_label = val_path / "labels" / new_name.replace(".jpg", ".txt")
        
        # Ensure destination folders exist
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy image
        shutil.copy2(img_path, dst_img)
        print(f"Copied {img_path.name} -> {dst_img}")
        
        # Handle label
        original_label = img_path.with_suffix(".txt")
        if original_label.exists():
            with open(original_label, "r") as f:
                lines = f.readlines()
            if name != "crowd":
                # Update class ID
                lines = [f"{class_ids[name]} " + " ".join(line.split()[1:]) + "\n" for line in lines]
            with open(dst_label, "w") as f:
                f.writelines(lines)
        else:
            # Create empty label if none exists
            with open(dst_label, "w") as f:
                f.write("")

# ===============================
# PROCESS ALL DATASETS
# ===============================
for name, folder in datasets.items():
    process_dataset(name, folder)

# ===============================
# CREATE YOLOv8 YAML
# ===============================
yaml_file = combined_path / "combined.yaml"
yaml_content = f"""
path: {combined_path}
train: train
val: val
nc: 4
names: ["crowd", "construction_legal", "construction_illegal", "river"]
"""

with open(yaml_file, "w") as f:
    f.write(yaml_content.strip())

print("✅ Combined dataset ready!")
print(f"YOLOv8 YAML: {yaml_file}")
