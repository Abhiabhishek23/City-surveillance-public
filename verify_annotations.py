# verify_annotations.py
import os
import cv2
import json

# Path to your crowd dataset
DATA_PATH = "/Users/abhishekx/Desktop/City-surveillance/Data/crowd"

# List all images
images = [f for f in os.listdir(DATA_PATH) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for img_file in images:
    img_path = os.path.join(DATA_PATH, img_file)
    annotation_path = os.path.join(DATA_PATH, os.path.splitext(img_file)[0] + ".json")
    
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read image: {img_file}")
        continue

    # Draw bounding boxes if annotation exists
    if os.path.exists(annotation_path):
        with open(annotation_path, "r") as f:
            data = json.load(f)
            for obj in data.get("objects", []):
                bbox = obj.get("bbox", [])
                if len(bbox) == 4:
                    x, y, w, h = map(int, bbox)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, obj.get("label", "unknown"), (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Show image
    cv2.imshow("Crowd Annotation Check", img)
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
