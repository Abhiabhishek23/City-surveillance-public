import os
from ultralytics import YOLO
import yaml
import shutil

# =========================
# CONFIGURE YOUR DATASETS
# =========================
DATASETS = {
    "crowd": {
        "images_path": "/Users/abhishekx/Desktop/City-surveillance/Data/crowd/dataset/images",
        "labels_path": "/Users/abhishekx/Desktop/City-surveillance/Data/crowd/dataset/labels",
        "classes": ["crowd"],
        "train_val_split": 0.8
    },
    "construction": {
        "images_path": "/Users/abhishekx/Desktop/City-surveillance/Data/Construction/images_raw",
        "labels_path": "/Users/abhishekx/Desktop/City-surveillance/Data/Construction/labels",
        "classes": ["legal", "illegal"],
        "train_val_split": 0.8
    },
    "river": {
        "images_path": "/Users/abhishekx/Desktop/City-surveillance/Data/river/images_raw",
        "labels_path": "/Users/abhishekx/Desktop/City-surveillance/Data/river/labels",
        "classes": ["river"],  # non-encroachment reference
        "train_val_split": 0.8
    }
}

BASE_MODEL = "yolov8n.pt"  # pretrained YOLOv8
DEVICE = "mps"  # Apple M1
IMG_SIZE = 320
BATCH = 4
EPOCHS = 10

# =========================
# FUNCTIONS
# =========================
def clean_labels(labels_path, valid_classes):
    """Removes labels with invalid class ids and corresponding images"""
    os.makedirs(labels_path, exist_ok=True)
    for root, _, files in os.walk(labels_path):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                lines = []
                with open(path, "r") as f:
                    for line in f.readlines():
                        cls = int(line.split()[0])
                        if cls in valid_classes:
                            lines.append(line)
                if lines:
                    with open(path, "w") as f:
                        f.writelines(lines)
                else:
                    os.remove(path)
                    # remove corresponding image
                    img_file = os.path.join(root.replace("labels", "images"), file.replace(".txt", ".jpg"))
                    if os.path.exists(img_file):
                        os.remove(img_file)


def auto_annotate(dataset_name, images_path, labels_path, class_map):
    """Auto-annotates images using YOLO pretrained model and assigns correct classes"""
    model = YOLO(BASE_MODEL)
    os.makedirs(labels_path, exist_ok=True)
    
    for class_name, class_id in class_map.items():
        class_folder = os.path.join(images_path, class_name)
        for file in os.listdir(class_folder):
            if file.lower().endswith((".jpg", ".png")):
                img_path = os.path.join(class_folder, file)
                results = model.predict(img_path, save_txt=True, conf=0.25)
                
                # Move txt to labels folder and assign correct class
                txt_file = img_path.replace("images_raw", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
                os.makedirs(os.path.dirname(txt_file), exist_ok=True)
                
                if os.path.exists(txt_file):
                    with open(txt_file, "r") as f:
                        lines = f.readlines()
                    with open(txt_file, "w") as f:
                        for line in lines:
                            parts = line.strip().split()
                            parts[0] = str(class_id)
                            f.write(" ".join(parts) + "\n")
    
    # Clean after annotation
    clean_labels(labels_path, list(class_map.values()))
    print(f"[INFO] {dataset_name} annotation and cleaning done.")


def create_yaml(dataset_name, images_path, labels_path, classes):
    """Creates YOLO dataset YAML file"""
    # Split train/val
    images = []
    for root, _, files in os.walk(images_path):
        for file in files:
            if file.lower().endswith((".jpg", ".png")):
                images.append(os.path.join(root, file))
    
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Move to train/val folders
    train_folder = os.path.join(os.path.dirname(images_path), "train")
    val_folder = os.path.join(os.path.dirname(images_path), "val")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    
    for idx, img in enumerate(train_images):
        shutil.move(img, os.path.join(train_folder, os.path.basename(img)))
        label_file = img.replace("images_raw", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
        if os.path.exists(label_file):
            shutil.move(label_file, os.path.join(labels_path.replace("images_raw", "labels"), os.path.basename(label_file)))
    
    for idx, img in enumerate(val_images):
        shutil.move(img, os.path.join(val_folder, os.path.basename(img)))
        label_file = img.replace("images_raw", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
        if os.path.exists(label_file):
            shutil.move(label_file, os.path.join(labels_path.replace("images_raw", "labels"), os.path.basename(label_file)))
    
    yaml_path = os.path.join(os.path.dirname(images_path), f"{dataset_name}.yaml")
    data_dict = {
        "path": os.path.dirname(images_path),
        "train": "train",
        "val": "val",
        "nc": len(classes),
        "names": classes
    }
    with open(yaml_path, "w") as f:
        yaml.dump(data_dict, f)
    print(f"[INFO] YAML created: {yaml_path}")
    return yaml_path


def train_model(dataset_name, yaml_path):
    """Trains YOLO model on dataset"""
    print(f"[INFO] Starting training for {dataset_name}...")
    model = YOLO(BASE_MODEL)
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        name=f"{dataset_name}_model"
    )
    print(f"[INFO] Training completed for {dataset_name}")


# =========================
# RUN AUTOMATION
# =========================
for dataset_name, config in DATASETS.items():
    print(f"[INFO] Processing dataset: {dataset_name}")
    classes = config["classes"]
    class_map = {cls_name: idx for idx, cls_name in enumerate(classes)}
    
    # 1. Auto-annotate
    auto_annotate(dataset_name, config["images_path"], config["labels_path"], class_map)
    
    # 2. Create YAML
    yaml_path = create_yaml(dataset_name, config["images_path"], config["labels_path"], classes)
    
    # 3. Train YOLO
    train_model(dataset_name, yaml_path)

print("[INFO] All datasets processed and models trained successfully.")
