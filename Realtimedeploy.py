# ===== REAL-TIME URBAN MONITORING SYSTEM =====
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time

class UrbanMonitoringSystem:
    def __init__(self, model_path=None, config_path=None):
        # allow env override or explicit path
        model_path = model_path or os.getenv("YOLO_MODEL", "yolov8n.pt")
        self.model = YOLO(model_path)
        self.class_names = {0: 'person', 1: 'permitted_construction', 2: 'illegal_construction', 3: 'river'}
        self.alerts = {
            'illegal_construction': "🚨 ILLEGAL CONSTRUCTION DETECTED!",
            'crowd': "⚠️  LARGE CROWD DETECTED!",
            'river_encroachment': "🌊 RIVER ENCROACHMENT DETECTED!"
        }
    
    def analyze_frame(self, frame, confidence_threshold=0.3):
        results = self.model(frame, conf=confidence_threshold, imgsz=640, verbose=False)
        return results[0] if results else None
    
    def generate_alerts(self, detections):
        alerts = []
        counts = {'person': 0, 'illegal_construction': 0, 'permitted_construction': 0, 'river': 0}
        
        if detections and detections.boxes is not None:
            for box in detections.boxes:
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, 'unknown')
                counts[class_name] += 1
        
        # Alert logic
        if counts['illegal_construction'] > 0:
            alerts.append(self.alerts['illegal_construction'])
        if counts['person'] > 15:  # Crowd threshold
            alerts.append(self.alerts['crowd'])
        if counts['illegal_construction'] > 0 and counts['river'] > 0:
            alerts.append(self.alerts['river_encroachment'])
            
        return alerts, counts
    
    def draw_detections(self, frame, detections, alerts, counts):
        # Draw bounding boxes
        if detections and detections.boxes is not None:
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, 'unknown')
                
                # Color coding
                colors = {
                    'person': (0, 255, 0),        # Green
                    'illegal_construction': (0, 0, 255),  # Red
                    'permitted_construction': (255, 255, 0), # Yellow
                    'river': (0, 255, 255)        # Cyan
                }
                
                color = colors.get(class_name, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw alerts and counters
        y_offset = 30
        for alert in alerts:
            cv2.putText(frame, alert, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
        
        # Draw counters
        counter_text = f"People: {counts['person']} | Illegal: {counts['illegal_construction']} | Permitted: {counts['permitted_construction']} | River: {counts['river']}"
        cv2.putText(frame, counter_text, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

# Usage example for webcam/drone:
# monitor = UrbanMonitoringSystem('urban_monitor.engine', 'dataset.yaml')
# cap = cv2.VideoCapture(0)  # Webcam, or use RTSP URL for drone
# while True:
#     ret, frame = cap.read()
#     if not ret: break
#     
#     detections = monitor.analyze_frame(frame)
#     alerts, counts = monitor.generate_alerts(detections)
#     frame = monitor.draw_detections(frame, detections, alerts, counts)
#     
#     cv2.imshow('Urban Monitoring', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'): break
# 
# cap.release()
# cv2.destroyAllWindows()