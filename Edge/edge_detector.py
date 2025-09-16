# edge/enhanced_detector.py
import cv2
import time
import os
import json
import requests
import uuid
import numpy as np
from ultralytics import YOLO
from PIL import Image
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Redis import kept for reference but usage commented out
try:
    import redis
    redis_available = True
except ImportError:
    redis_available = False


class DummyRedis:
    def exists(self, key):
        return False

    def setex(self, key, time, value):
        return True


class EnhancedUrbanMonitor:
    def __init__(self):
        # Configuration
        self.CAMERA_ID = os.getenv("CAMERA_ID", "CAM_01")
        self.VIDEO_SRC = int(os.getenv("VIDEO_SRC", "0"))
        self.BACKEND_ALERT_URL = os.getenv("BACKEND_ALERT_URL", "http://localhost:8000/alerts")
        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.PERSON_LIMIT = int(os.getenv("PERSON_LIMIT", "5"))
        self.PERSON_COOLDOWN = int(os.getenv("PERSON_COOLDOWN", "60"))
        self.CONSTRUCTION_COOLDOWN = int(os.getenv("CONSTRUCTION_COOLDOWN", "86400"))
        self.CONSTRUCTION_CONFIDENCE = float(os.getenv("CONSTRUCTION_CONFIDENCE", "0.1"))
        self.UPLOAD_DIR = os.getenv("EDGE_UPLOAD_DIR", "./edge_uploads")

        # Create directories
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)

        # Redis setup (commented out for testing)
        # if redis_available:
        #     self.redis = redis.Redis(host=self.REDIS_HOST, port=6379, decode_responses=True)
        # else:
        #     self.redis = DummyRedis()
        print("Redis is disabled for testing. Alerts cooldowns will not be enforced.")

        # Load models (configurable via YOLO_MODEL env var)
        yolo_model_path = os.getenv("YOLO_MODEL", "yolov8n.pt")
        try:
            self.yolo_model = YOLO(yolo_model_path)
        except Exception as e:
            print(f"Failed to load YOLO model {yolo_model_path}: {e}")
            self.yolo_model = None

        # Load Mask R-CNN (optional)
        self.mask_rcnn_model = self._load_mask_rcnn()

        # Background model
        self.background = None
        self.background_stable_frames = 0
        self.background_ready = False

        print("Enhanced Urban Monitor initialized")

    # --- Methods below remain the same ---
    def _load_mask_rcnn(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_paths = [
            "./models/construction_detection_model.pth",
            "./models/maskrcnn_pretrained.pth"
        ]

        model = None
        loaded_model_path = None

        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    model = maskrcnn_resnet50_fpn(pretrained=False)
                    state_dict = torch.load(model_path, map_location=device)
                    if 'model' in state_dict:
                        model.load_state_dict(state_dict['model'])
                    else:
                        model.load_state_dict(state_dict)
                    model.to(device).eval()
                    loaded_model_path = model_path
                    print(f"Loaded Mask R-CNN from: {model_path}")
                    break
                except Exception as e:
                    print(f"Error loading {model_path}: {e}")
                    continue

        if model is None:
            try:
                model = maskrcnn_resnet50_fpn(pretrained=True)
                model.to(device).eval()
                print("Loaded pretrained COCO Mask R-CNN")
            except Exception as e:
                print(f"Failed to load any Mask R-CNN model: {e}")
                model = None

        return model

    def detect_construction_mask_rcnn(self, frame):
        if self.mask_rcnn_model is None:
            return False, [], [], 0.0

        try:
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = F.to_tensor(image_pil).to(next(self.mask_rcnn_model.parameters()).device)

            with torch.no_grad():
                predictions = self.mask_rcnn_model([image_tensor])

            construction_classes = [11, 12, 24, 25]
            construction_detected = False
            construction_masks = []
            construction_scores = []
            construction_boxes = []

            for i, score in enumerate(predictions[0]['scores']):
                if score > self.CONSTRUCTION_CONFIDENCE:
                    label = predictions[0]['labels'][i].item()
                    if label in construction_classes:
                        construction_detected = True
                        mask = predictions[0]['masks'][i, 0].mul(255).byte().cpu().numpy()
                        construction_masks.append(mask)
                        construction_scores.append(score.item())
                        construction_boxes.append(predictions[0]['boxes'][i].cpu().numpy())

            avg_confidence = np.mean(construction_scores) if construction_scores else 0.0

            return construction_detected, construction_masks, construction_boxes, avg_confidence

        except Exception as e:
            print(f"Mask R-CNN inference error: {e}")
            return False, [], [], 0.0

    def detect_construction_background_diff(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.background is None:
            self.background = gray.copy().astype(np.float32)
            return False, []

        cv2.accumulateWeighted(gray, self.background, 0.05)
        background_uint8 = cv2.convertScaleAbs(self.background)
        diff = cv2.absdiff(background_uint8, gray)

        thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 5.0:
                    significant_contours.append(cnt)

        return len(significant_contours) > 0, significant_contours

    def save_snapshot(self, frame, event_type):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{self.CAMERA_ID}_{event_type}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
        fpath = os.path.join(self.UPLOAD_DIR, fname)
        cv2.imwrite(fpath, frame)
        return fpath

    def send_alert(self, event_type, frame, details=None):
        """Send alert to backend - Redis cooldown disabled for testing"""
        snapshot_path = self.save_snapshot(frame, event_type)
        alert_data = {
            "camera_id": self.CAMERA_ID,
            "event": event_type,
            "count": details.get("count") if details else None,
            "confidence": details.get("confidence") if details else None,
            "details": json.dumps(details) if details else "{}"
        }

        files = {}
        if snapshot_path and os.path.exists(snapshot_path):
            files['snapshot'] = open(snapshot_path, 'rb')

        try:
            response = requests.post(self.BACKEND_ALERT_URL, data=alert_data, files=files, timeout=10)
            if response.status_code == 200:
                print(f"Alert sent: {event_type} (Redis disabled for testing)")
                return True
        except Exception as e:
            print(f"Failed to send alert: {e}")
        finally:
            for f in files.values():
                try:
                    f.close()
                except:
                    pass
        return False

    def visualize_frame(self, frame, people_count, construction_detected, construction_masks=None, contours=None):
        cv2.putText(frame, f"People: {people_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if construction_detected:
            if construction_masks:
                for mask in construction_masks:
                    color_mask = np.zeros_like(frame)
                    color_mask[mask > 0] = [0, 0, 255]
                    frame = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
            elif contours:
                for cnt in contours:
                    cv2.drawContours(frame, [cnt], -1, (0, 0, 255), 2)
            cv2.putText(frame, "CONSTRUCTION DETECTED", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"{self.CAMERA_ID} | {timestamp}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def process_frame(self, frame):
        results = self.yolo_model(frame, classes=[0])
        people_boxes = results[0].boxes if results[0].boxes is not None else None
        people_count = len(people_boxes) if people_boxes is not None else 0

        if people_count >= self.PERSON_LIMIT:
            self.send_alert("overcrowding", frame, {"count": people_count, "threshold": self.PERSON_LIMIT})

        construction_detected = False
        construction_masks = []
        construction_contours = []
        construction_confidence = 0.0

        construction_detected, construction_masks, construction_boxes, construction_confidence = self.detect_construction_mask_rcnn(frame)
        if not construction_detected:
            construction_detected, construction_contours = self.detect_construction_background_diff(frame)

        if construction_detected:
            alert_details = {
                "method": "mask_rcnn" if construction_masks else "background_diff",
                "confidence": construction_confidence,
                "mask_count": len(construction_masks),
                "contour_count": len(construction_contours)
            }
            self.send_alert("illegal_construction", frame, alert_details)

        visualized_frame = self.visualize_frame(frame, people_count, construction_detected,
                                                construction_masks, construction_contours)

        return visualized_frame, people_count, construction_detected

    def run(self):
        cap = cv2.VideoCapture(self.VIDEO_SRC)
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.VIDEO_SRC}")
            return

        print("Starting enhanced urban monitoring. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            processed_frame, people_count, construction_detected = self.process_frame(frame)
            cv2.imshow('Enhanced Urban Monitor', processed_frame)
            status = f"People: {people_count}, Construction: {'Yes' if construction_detected else 'No'}"
            print(status, end='\r')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    monitor = EnhancedUrbanMonitor()
    monitor.run()
