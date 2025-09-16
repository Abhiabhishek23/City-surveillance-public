from ultralytics import YOLO
import os

# Paths
img_folder = "/Users/abhishekx/Desktop/City-surveillance/Data/crowd/dataset/images"
label_folder = "/Users/abhishekx/Desktop/City-surveillance/Data/crowd/dataset/labels"
os.makedirs(label_folder, exist_ok=True)

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

for img_file in os.listdir(img_folder):
    if not img_file.lower().endswith(('.jpg','.png')):
        continue
    img_path = os.path.join(img_folder, img_file)
    results = model.predict(img_path, classes=[0], save=False)  # class 0 = person
    boxes = results[0].boxes.xyxy.cpu().numpy()
    h, w = results[0].orig_shape
    label_path = os.path.join(label_folder, img_file.replace(".jpg",".txt"))
    with open(label_path, "w") as f:
        for box in boxes:
            x1, y1, x2, y2 = box
            # YOLO format: class x_center y_center width height (normalized)
            x_center = ((x1+x2)/2)/w
            y_center = ((y1+y2)/2)/h
            bw = (x2-x1)/w
            bh = (y2-y1)/h
            f.write(f"0 {x_center} {y_center} {bw} {bh}\n")
print("Auto-annotation complete")
