"""urban_monitor_deploy.py
Lightweight client for real-time urban monitoring with improved error handling.
"""

import argparse
import json
import os
import time
import threading
from collections import deque
import cv2
import numpy as np
import requests

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️ ultralytics not available. Please install: pip install ultralytics")

class UrbanMonitoringClient:
    def __init__(self, model_path, backend_url, camera_id, enable_verifier=False, crowd_threshold: int = 5):
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics package required but not available")

        # Model + metadata
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.backend_url = backend_url
        self.camera_id = camera_id
        self.enable_verifier = enable_verifier
        self._crowd_threshold = int(crowd_threshold)

        # Threading / buffering
        self.frame_buffer = deque(maxlen=4)
        self.buffer_lock = threading.Lock()
        self.stop_worker = False

        # State trackers
        self.last_sent = {}
        self.cooldowns = {'illegal_construction': 60, 'overcrowding': 30}
        self.frame_history = deque(maxlen=16)
        self.bg_estimate = None
        self.bg_count = 0

        # Heuristic illegal construction evidence accumulator
        self.illegal_score = 0.0
        self.last_illegal_flag = False

        print(f"✅ Model loaded: {model_path}")
        print(f"✅ Backend: {backend_url}")
        print(f"✅ Camera: {camera_id}")

    def inference_worker(self):
        """Worker thread for async inference processing"""
        while not self.stop_worker:
            frame = None
            with self.buffer_lock:
                if self.frame_buffer:
                    frame = self.frame_buffer.popleft()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            try:
                # Run inference (lower confidence for MVP sensitivity)
                detections = self.model(frame, conf=0.25, imgsz=640, verbose=False)[0]
                self.process_detections(frame, detections)
            except Exception as e:
                print(f"Inference error: {e}")

    def process_detections(self, frame, detections):
        """Process detection results and trigger alerts"""
        try:
            counts = {}
            if detections is not None and getattr(detections, 'boxes', None) is not None:
                cls = getattr(detections.boxes, 'cls', None)
                if cls is not None:
                    try:
                        cls_ids = [int(c) for c in cls.cpu().numpy().tolist()]
                    except Exception:
                        try:
                            cls_ids = [int(c) for c in cls.tolist()]
                        except Exception:
                            cls_ids = []
                    def id_to_name(cid: int) -> str:
                        names = self.class_names
                        if isinstance(names, dict):
                            return str(names.get(cid, 'unknown')).lower()
                        if isinstance(names, (list, tuple)) and 0 <= cid < len(names):
                            return str(names[cid]).lower()
                        return 'unknown'
                    for cid in cls_ids:
                        cname = id_to_name(cid)
                        counts[cname] = counts.get(cname, 0) + 1
            
            # Update background estimation
            self.update_background_estimate(frame)
            
            # Check for illegal construction
            construction_detected, confidence = self.detect_illegal_construction(frame, detections)
            if construction_detected and confidence >= 0.5:
                equipment_count = sum(1 for name in counts if any(kw in name for kw in 
                    ['crane', 'truck', 'excavator', 'bulldozer', 'backhoe', 'loader', 'concrete', 'mixer']))
                self.maybe_send_alert('illegal_construction', max(1, equipment_count), frame)
            
            # Check for overcrowding (uses configurable threshold)
            # More robust people counting across label variants
            people_aliases = ('person', 'people', 'pedestrian', 'crowd')
            people_count = sum(counts.get(k, 0) for k in people_aliases)
            # lightweight log
            print(f"[AI] cam={self.camera_id} people={people_count} thr={self._crowd_threshold} counts={counts}")
            if people_count >= self._crowd_threshold:
                self.maybe_send_alert('overcrowding', people_count, frame)
                
        except Exception as e:
            print(f"Error processing detections: {e}")

    def update_background_estimate(self, frame):
        """Update running background estimate"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame_history.append(gray)
            
            if self.bg_estimate is None:
                self.bg_estimate = gray.astype('float32')
                self.bg_count = 1
            elif self.bg_count < 120:
                cv2.accumulateWeighted(gray.astype('float32'), self.bg_estimate, 0.05)
                self.bg_count += 1
                
        except Exception as e:
            print(f"Background update error: {e}")

    def detect_illegal_construction(self, frame, detections=None):
        """Heuristic multi-cue illegal construction detection.

        This is a lightweight interim approach until a dedicated model with
        classes (Permanent_Legal, Permanent_Illegal, etc.) is trained.

        Heuristics (adds evidence):
          1. Presence of construction equipment keywords in YOLO class names.
          2. Low global motion (static scene) while equipment persists (suggests structure work vs transient vehicle).
          3. New large static foreground blob appearing (primitive background model diff).

        Returns:
            (flag: bool, confidence: float)
        """
        try:
            if detections is None or getattr(detections, 'boxes', None) is None:
                return False, 0.0

            equipment_keywords = [
                'crane','truck','excavator','bulldozer','backhoe','loader',
                'cement','concrete','mixer','forklift','dump','grader','jcb','roller','paver'
            ]
            names = self.class_names
            boxes = getattr(detections, 'boxes', None)
            cls = getattr(boxes, 'cls', None)
            equip_hits = 0
            if cls is not None:
                try:
                    cls_ids = [int(c) for c in cls.cpu().numpy().tolist()]
                except Exception:
                    try:
                        cls_ids = [int(c) for c in cls.tolist()]
                    except Exception:
                        cls_ids = []
                for cid in cls_ids:
                    cname = None
                    if isinstance(names, dict):
                        cname = str(names.get(cid, '')).lower()
                    elif isinstance(names, (list, tuple)) and 0 <= cid < len(names):
                        cname = str(names[cid]).lower()
                    if cname and any(k in cname for k in equipment_keywords):
                        equip_hits += 1

            # Background motion metric (simple): mean absolute diff to background if established
            motion_level = 0.0
            static_bonus = 0.0
            if self.bg_estimate is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(gray.astype('float32'), self.bg_estimate)
                    motion_level = float(diff.mean())  # 0..255
                    if motion_level < 6.0:  # very low motion
                        static_bonus = 0.5
                except Exception:
                    pass

            # Foreground emergence: crude threshold of diff > 25
            emergence_bonus = 0.0
            if self.bg_estimate is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(gray.astype('float32'), self.bg_estimate)
                    fg_mask = (diff > 25).astype('uint8')
                    fg_ratio = fg_mask.mean()  # 0..1
                    if 0.02 < fg_ratio < 0.25:  # some new structure occupying 2%-25% of frame
                        emergence_bonus = 0.4
                except Exception:
                    pass

            # Accumulate score with decay, but require equipment evidence to consider
            decay = 0.90
            self.illegal_score *= decay
            if equip_hits > 0:
                # Only accumulate if equipment is present; include context bonuses
                self.illegal_score += 0.6 + 0.2 * min(equip_hits, 3)
                self.illegal_score += static_bonus + emergence_bonus
            else:
                # Without equipment, clamp to zero to avoid false positives from background only
                self.illegal_score = 0.0

            # Normalize rough confidence (cap)
            confidence = min(1.0, self.illegal_score / 3.0)
            flag = (equip_hits > 0) and (confidence >= 0.75)
            self.last_illegal_flag = flag
            return flag, confidence
        except Exception:
            return False, 0.0

    def maybe_send_alert(self, event_type, count, frame):
        """Send alert with cooldown management"""
        current_time = time.time()
        last_sent = self.last_sent.get(event_type, 0)
        cooldown = self.cooldowns.get(event_type, 30)
        
        if current_time - last_sent >= cooldown:
            self.send_alert_to_backend(event_type, count, frame)
            self.last_sent[event_type] = current_time

    def send_alert_to_backend(self, event_type, count, frame):
        """Send alert to backend server"""
        try:
            # Encode frame as JPEG
            success, encoded_image = cv2.imencode('.jpg', frame)
            if not success:
                print("❌ Failed to encode frame")
                return
            
            # Prepare request
            files = {'snapshot': ('alert.jpg', encoded_image.tobytes(), 'image/jpeg')}
            data = {
                'camera_id': self.camera_id,
                'event': event_type,
                'count': count
            }
            
            # Send request
            response = requests.post(self.backend_url, data=data, files=files, timeout=10)
            if response.status_code == 200:
                print(f"✅ Alert sent: {event_type} (count: {count})")
            else:
                print(f"⚠️ Backend error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")

def resolve_model(model_arg):
    """Resolve model path from alias or search common locations"""
    # [Keep your improved implementation here]
    return model_arg  # Placeholder

def main():
    parser = argparse.ArgumentParser(description='Urban Monitoring AI Client')
    parser.add_argument('--model', default='yolov8s.pt', help='Model file or alias')
    parser.add_argument('--source', default='0', help='Video source')
    parser.add_argument('--backend', default='http://127.0.0.1:8000/alerts', help='Backend URL')
    parser.add_argument('--camera', default='CAM01', help='Camera ID')
    parser.add_argument('--enable-verifier', action='store_true', help='Enable verification')
    parser.add_argument('--no-gui', action='store_true', help='Run without OpenCV windows')
    parser.add_argument('--crowd-threshold', type=int, default=5, help='Min people count to trigger overcrowding alert')
    
    args = parser.parse_args()
    
    try:
        model_path = resolve_model(args.model)
        monitor = UrbanMonitoringClient(model_path, args.backend, args.camera, args.enable_verifier, crowd_threshold=args.crowd_threshold)
        
        # Start worker thread
        worker_thread = threading.Thread(target=monitor.inference_worker, daemon=True)
        worker_thread.start()
        
        # Main capture loop
        cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {args.source}")
        
        print("🚀 Starting monitoring. Press 'q' to quit.")
        
        # Prepare a visible GUI window
        win_title = f"Urban Monitoring - {args.camera}"
        if not args.no_gui:
            try:
                cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_title, 960, 540)
                cv2.moveWindow(win_title, 100, 80)
            except Exception as _e:
                print(f"GUI setup warning: {_e}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add to buffer
            with monitor.buffer_lock:
                if len(monitor.frame_buffer) < monitor.frame_buffer.maxlen:
                    monitor.frame_buffer.append(frame.copy())
            
            # Display (optional)
            if not args.no_gui:
                cv2.imshow(win_title, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    except Exception as e:
        print(f"❌ Fatal error: {e}")
    finally:
        if 'monitor' in locals():
            monitor.stop_worker = True
        if 'worker_thread' in locals():
            worker_thread.join(timeout=1.0)
        if 'cap' in locals():
            cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == '__main__':
    main()