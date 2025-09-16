# edge/combined_detector.py
import cv2
import time
import os
import json
import requests
import uuid
import numpy as np
from ultralytics import YOLO
from geopy.geocoders import Nominatim
import threading
import queue

class CombinedUrbanMonitor:
    def __init__(self):
        # Configuration
        self.CAMERA_ID = os.getenv("CAMERA_ID", "CAM_01")
        self.VIDEO_SRC = int(os.getenv("VIDEO_SRC", "0"))
        self.BACKEND_ALERT_URL = os.getenv("BACKEND_ALERT_URL", "http://localhost:8000/alerts")
        self.UPLOAD_DIR = os.getenv("EDGE_UPLOAD_DIR", "./edge_uploads")
        
        # Create directories
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        
        # Load combined model
        self.model = self._load_model()
        
        # Camera locations with GPS coordinates
        self.camera_locations = self.load_camera_locations()
        self.geolocator = Nominatim(user_agent="city_surveillance")
        
        # Frame processing settings
        self.frame_skip = 2  # Process every 2nd frame for better performance
        self.frame_count = 0
        
        # Multi-camera support
        self.camera_sources = {
            "CAM_01": 0,  # Webcam
            # Add more cameras: "CAM_02": "rtsp://camera2_url", etc.
        }
        self.frame_queues = {cam_id: queue.Queue(maxsize=1) for cam_id in self.camera_sources}
        
        print("Combined Urban Monitor initialized")
    
    def _load_model(self):
        """Load the trained combined model, auto-detecting best.pt from common paths.
        Priority order:
        1) YOLO_MODEL env var if exists
        2) KAGGLE_BEST_PT env var
        3) ./weights/best.pt, ./Edge/weights/best.pt, ./models/best_combined.pt
        4) latest runs/detect/*/weights/best.pt
        5) fallback yolov8n.pt
        """
        def try_load(path):
            if not path:
                return None
            if os.path.exists(path):
                try:
                    print(f"Trying model: {path}")
                    return YOLO(path)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            return None

        # 1) Explicit env var
        env_model = os.getenv("YOLO_MODEL")
        m = try_load(env_model)
        if m: return m

        # 2) Kaggle-style path via env
        kaggle_best = os.getenv("KAGGLE_BEST_PT")
        m = try_load(kaggle_best)
        if m: return m

        # 3) Common local paths
        for p in [
            "./weights/best.pt",
            "./Edge/weights/best.pt",
            "./models/best_combined.pt",
            "./models/yolov8n.pt",
        ]:
            m = try_load(p)
            if m: return m

        # 4) Latest from runs
        try:
            import glob
            runs = sorted(glob.glob('runs/detect/*/weights/best.pt'))
            if runs:
                m = try_load(runs[-1])
                if m: return m
        except Exception as e:
            print(f"Error searching runs: {e}")

        # 5) Final fallback
        fallback = "yolov8n.pt"
        print(f"No suitable model found, using fallback model: {fallback}")
        return YOLO(fallback)
    
    def load_camera_locations(self):
        """Load camera GPS coordinates and location references"""
        return {
            "CAM_01": {
                "coordinates": (23.1765, 75.7885),
                "location_ref": "Ram Ghat Area"
            },
            "CAM_02": {
                "coordinates": (23.1780, 75.7900), 
                "location_ref": "Mangalnath Ghat"
            },
            "CAM_03": {
                "coordinates": (23.1740, 75.7860),
                "location_ref": "Triveni Ghat"
            }
        }
    
    def get_nearby_landmarks(self, coordinates):
        """Get nearby landmarks using reverse geocoding"""
        try:
            location = self.geolocator.reverse(f"{coordinates[0]}, {coordinates[1]}")
            if location and location.raw.get('address'):
                address = location.raw['address']
                return {
                    "road": address.get('road', ''),
                    "suburb": address.get('suburb', ''),
                    "city": address.get('city', 'Ujjain'),
                    "state": address.get('state', 'MP')
                }
        except Exception as e:
            print(f"Geocoding error: {e}")
        
        return {"city": "Ujjain", "state": "MP"}
    
    def save_snapshot(self, frame, event_type, camera_id):
        """Save snapshot with timestamp and event type"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{camera_id}_{event_type}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
        fpath = os.path.join(self.UPLOAD_DIR, fname)
        cv2.imwrite(fpath, frame)
        return fpath
    
    def send_alert(self, camera_id, event_type, frame, details=None):
        """Send alert to backend with location information"""
        # Get location data
        cam_data = self.camera_locations.get(camera_id, {})
        coordinates = cam_data.get("coordinates", (0, 0))
        nearby_landmarks = self.get_nearby_landmarks(coordinates)
        
        # Enhance details with location
        enhanced_details = details or {}
        enhanced_details.update({
            "gps_coordinates": {
                "latitude": coordinates[0],
                "longitude": coordinates[1]
            },
            "location_reference": cam_data.get("location_ref", "Unknown"),
            "nearby_landmarks": nearby_landmarks,
            "camera_id": camera_id
        })
        
        # Save snapshot
        snapshot_path = self.save_snapshot(frame, event_type, camera_id)
        
        # Prepare alert data
        alert_data = {
            "camera_id": camera_id,
            "event": event_type,
            "count": enhanced_details.get("count"),
            "confidence": enhanced_details.get("confidence"),
            "details": json.dumps(enhanced_details)
        }
        
        # Send to backend
        files = {}
        if snapshot_path and os.path.exists(snapshot_path):
            files['snapshot'] = open(snapshot_path, 'rb')
        
        try:
            response = requests.post(self.BACKEND_ALERT_URL, data=alert_data, files=files, timeout=10)
            if response.status_code == 200:
                print(f"Alert sent: {event_type} from {camera_id}")
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
    
    def process_detections(self, results, camera_id, frame):
        """Process detection results and send alerts"""
        # Class mappings (adjust based on your combined.yaml)
        class_names = {
            0: "person",
            1: "permitted_construction", 
            2: "illegal_construction"
        }
        
        # Count objects by class
        counts = {class_id: 0 for class_id in class_names}
        boxes_by_class = {class_id: [] for class_id in class_names}
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id in class_names and confidence > 0.5:
                    counts[class_id] += 1
                    boxes_by_class[class_id].append(box.xyxy[0].cpu().numpy())
        
        # Check for overcrowding (people count)
        if counts[0] >= 5:  # Adjust threshold as needed
            self.send_alert(camera_id, "overcrowding", frame, {
                "count": counts[0],
                "threshold": 5,
                "bounding_boxes": boxes_by_class[0],
                "confidence": np.mean([box.conf[0] for box in results[0].boxes if int(box.cls[0]) == 0]) if counts[0] > 0 else 0
            })
        
        # Check for illegal construction
        if counts[2] > 0:
            self.send_alert(camera_id, "illegal_construction", frame, {
                "count": counts[2],
                "bounding_boxes": boxes_by_class[2],
                "confidence": np.mean([box.conf[0] for box in results[0].boxes if int(box.cls[0]) == 2]) if counts[2] > 0 else 0
            })
        
        # Check for permitted construction (optional monitoring)
        if counts[1] > 0:
            # Just log, don't alert for permitted construction
            print(f"Permitted construction detected: {counts[1]} instances")
        
        return counts
    
    def visualize_frame(self, frame, counts, camera_id):
        """Visualize detections on frame"""
        # Add camera ID and timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"{camera_id} | {timestamp}", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add counts overlay
        y_offset = 30
        for class_id, count in counts.items():
            if count > 0:
                class_name = ["People", "Permitted", "Illegal"][class_id]
                cv2.putText(frame, f"{class_name}: {count}", (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
        
        return frame
    
    def camera_thread(self, camera_id, source):
        """Thread for capturing frames from a camera"""
        cap = cv2.VideoCapture(source)
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Keep only the latest frame in the queue
            if not self.frame_queues[camera_id].empty():
                try:
                    self.frame_queues[camera_id].get_nowait()
                except:
                    pass
            
            self.frame_queues[camera_id].put(frame)
        
        cap.release()
    
    def run(self):
        """Main monitoring loop"""
        # Start camera threads
        for cam_id, src in self.camera_sources.items():
            t = threading.Thread(target=self.camera_thread, args=(cam_id, src), daemon=True)
            t.start()
            print(f"Started camera thread for {cam_id}")
        
        print("Starting combined urban monitoring. Press 'q' to quit.")
        
        while True:
            for cam_id in self.camera_sources:
                if not self.frame_queues[cam_id].empty():
                    frame = self.frame_queues[cam_id].get()
                    
                    # Skip frames for better performance
                    self.frame_count += 1
                    if self.frame_count % self.frame_skip != 0:
                        continue
                    
                    # Run inference
                    results = self.model(frame, verbose=False)
                    
                    # Process detections and send alerts
                    counts = self.process_detections(results, cam_id, frame)
                    
                    # Visualize results
                    vis_frame = self.visualize_frame(frame.copy(), counts, cam_id)
                    
                    # Display
                    cv2.imshow(f"{cam_id} Feed", vis_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = CombinedUrbanMonitor()
    monitor.run()