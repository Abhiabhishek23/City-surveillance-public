import cv2
import json
import os

DATA_DIR = "predictions/"
LABELS_FILE = "labels.json"

labels = {}
for file in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, file)
    img = cv2.imread(path)
    cv2.imshow("Image", img)
    key = cv2.waitKey(0)
    if key == ord('c'):
        labels[file] = "crowd"
    elif key == ord('i'):
        labels[file] = "illegal"
    else:
        labels[file] = "ignore"
    cv2.destroyAllWindows()

with open(LABELS_FILE, "w") as f:
    json.dump(labels, f)
