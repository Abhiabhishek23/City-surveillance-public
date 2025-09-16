from ultralytics import YOLO
import os

# Configurable model path - set YOLO_MODEL env var to switch models (e.g. yolov12n.pt)
DEFAULT_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
# Load pre-trained model
model = YOLO(DEFAULT_MODEL)

# Paths
legal_folder = "/Users/abhishekx/Desktop/City-surveillance/Data/Construction/images_raw/legal"
illegal_folder = "/Users/abhishekx/Desktop/City-surveillance/Data/Construction/images_raw/illegal"

# Function to auto-annotate
def auto_annotate(folder_path, class_id):
    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(folder_path, file)
            results = model.predict(img_path, save_txt=True, conf=0.25)  # Predict & save YOLO txt
            # Update txt to correct class
            txt_file = os.path.join(folder_path.replace("images_raw", "labels"), file.replace(".jpg", ".txt"))
            if os.path.exists(txt_file):
                with open(txt_file, "r") as f:
                    lines = f.readlines()
                with open(txt_file, "w") as f:
                    for line in lines:
                        parts = line.strip().split()
                        parts[0] = str(class_id)  # Replace class
                        f.write(" ".join(parts) + "\n")

# Create labels folder
os.makedirs(legal_folder.replace("images_raw", "labels"), exist_ok=True)
os.makedirs(illegal_folder.replace("images_raw", "labels"), exist_ok=True)

# Auto-annotate
auto_annotate(legal_folder, class_id=0)
auto_annotate(illegal_folder, class_id=1)
