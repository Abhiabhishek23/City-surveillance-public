import os
import glob
from ultralytics import YOLO
import cv2

# Runs inference on all .mp4 files in 'Test samples' and writes results to runs/test_inference
TEST_DIR = os.path.join(os.path.dirname(__file__), 'Test samples')
MODEL_PATH = os.getenv('YOLO_MODEL', 'yolov8n.pt')

os.makedirs('runs/test_inference', exist_ok=True)

print(f"Using model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

mp4s = glob.glob(os.path.join(TEST_DIR, '*.mp4'))
if not mp4s:
    print('No mp4 files found in Test samples. Place files in the folder and rerun.')

for mp4 in mp4s:
    print(f"Processing: {mp4}")
    try:
        # run inference and save annotated video output
        results = model.predict(source=mp4, save=True, conf=0.25, imgsz=640)
        print(f"Saved results for {mp4}")
    except Exception as e:
        print(f"Failed for {mp4}: {e}")

print('Done')
