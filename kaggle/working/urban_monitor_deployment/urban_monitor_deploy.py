"""urban_monitor_deploy.py

Lightweight client for real-time monitoring.
Supports selecting YOLOv12-n or YOLOv12-s (aliases), with an optional
Mask R-CNN verifier used only for flagged frames.

Usage example:
  python urban_monitor_deploy.py --model yolov12-n --source "Test samples/1.mp4" --backend http://127.0.0.1:8000/alerts --camera CAM01 --enable-verifier
"""

import argparse
import json
import os
import time
from collections import deque
from threading import Thread

import cv2
import numpy as np
import requests

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class UrbanMonitoringClient:
    def __init__(self, model_path, backend_url, camera_id, enable_verifier=False):
        if YOLO is None:
            raise RuntimeError('ultralytics package not available; install it or adapt code')

        self.model = YOLO(model_path)
        self.class_names = getattr(self.model, 'names', {})
        self.backend_url = backend_url
        self.camera_id = camera_id
        self.enable_verifier = enable_verifier

        print(f"\u2705 Model '{model_path}' loaded, backend={backend_url}, verifier={enable_verifier}")

        # cooldowns
        self.last_sent = {}
        self.cooldowns = {'illegal_construction': 60, 'overcrowding': 30}

        # lightweight buffers
        self.frame_history = deque(maxlen=16)
        self.bg_estimate = None
        self.bg_count = 0

        # optional verifier (Mask R-CNN) is loaded lazily
        self.verifier = None
        self._verifier_device = None
        if enable_verifier:
            try:
                import torch
                from torchvision.models.detection import maskrcnn_resnet50_fpn

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.verifier = maskrcnn_resnet50_fpn(pretrained=True).to(device).eval()
                self._verifier_device = device
                print(f"\U0001F50E Mask R-CNN verifier loaded on {device}")
            except Exception as e:
                print('WARNING: Failed to load verifier:', e)
                self.verifier = None

    def send_alert_to_backend(self, event_type, count, frame):
        try:
            ok, buf = cv2.imencode('.jpg', frame)
            if not ok:
                print('Failed to encode frame')
                return
            files = {'snapshot': ('snapshot.jpg', buf.tobytes(), 'image/jpeg')}
            payload = {'camera_id': self.camera_id, 'event': event_type, 'count': count}
            resp = requests.post(self.backend_url, data=payload, files=files, timeout=5)
            if resp.status_code == 200:
                try:
                    body = resp.json()
                except Exception:
                    body = resp.text
                print(f"Sent {event_type}, backend responded: {body}")
            else:
                print('Backend responded', resp.status_code)
        except Exception as e:
            print('Error sending alert:', e)

    def process_detections(self, frame, detections):
        counts = {}
        if detections and getattr(detections, 'boxes', None) is not None:
            for box in detections.boxes:
                try:
                    class_id = int(box.cls[0])
                except Exception:
                    continue
                name = self.class_names.get(class_id, 'unknown').lower()
                counts[name] = counts.get(name, 0) + 1

        # update bg estimate
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            gray = None

        if gray is not None:
            self.frame_history.append(gray)
            if self.bg_estimate is None:
                self.bg_estimate = gray.astype('float32')
                self.bg_count = 1
            elif self.bg_count < 120:
                cv2.accumulateWeighted(gray.astype('float32'), self.bg_estimate, 1.0 / (self.bg_count + 1))
                self.bg_count += 1

        def maybe_send(evt, cnt):
            now = time.time()
            last = self.last_sent.get(evt, 0)
            cd = self.cooldowns.get(evt, 30)
            if now - last < cd:
                return False
            self.last_sent[evt] = now
            self.send_alert_to_backend(evt, cnt, frame)
            return True

        # illegal construction check
        try:
            flagged, conf = self.detect_illegal_construction(frame, detections)
            if flagged and conf >= 0.5:
                equip_count = max(1, sum(1 for _ in (detections.boxes if detections and getattr(detections, 'boxes', None) is not None else [])))
                print(f"ILLEGAL CONSTRUCTION: conf={conf:.2f}")
                maybe_send('illegal_construction', equip_count)
        except Exception as e:
            print('Illegal detection error', e)

        # crowd detection basic
        person_ct = counts.get('person', 0)
        if person_ct == 0:
            for k, v in counts.items():
                if 'person' in k or 'human' in k or 'people' in k:
                    person_ct += v

        if person_ct > 1:
            print('CROWD:', person_ct)
            maybe_send('overcrowding', person_ct)

        # minimal logging
        try:
            log = {'t': time.time(), 'camera': self.camera_id, 'counts': counts}
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            log_path = os.path.join(project_root, 'logs', 'client_frame_log.jsonl')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'a') as f:
                f.write(json.dumps(log) + '\n')
        except Exception:
            pass

        return counts

    def detect_illegal_construction(self, frame, detections=None):
        """Lightweight multi-cue detector combining equipment, structural change, and motion.

        Returns: (bool_flagged, confidence_float)
        """
        weights = {'equipment': 0.5, 'structural': 0.25, 'motion': 0.25}

        # 1) Equipment detection score
        equipment_score = 0.0
        try:
            boxes = []
            if detections and getattr(detections, 'boxes', None) is not None:
                boxes = detections.boxes
            else:
                res = self.model(frame, conf=0.45, imgsz=640, verbose=False)[0]
                boxes = getattr(res, 'boxes', [])

            equipment_keywords = ['crane', 'truck', 'excavator', 'bulldozer', 'loader', 'backhoe', 'forklift', 'grader']
            matched = 0
            for b in boxes:
                cid = int(b.cls[0])
                cname = self.class_names.get(cid, '').lower()
                if any(k in cname for k in equipment_keywords):
                    matched += 1
            equipment_score = 1.0 - 2 ** (-matched)
        except Exception:
            equipment_score = 0.0

        # 2) Structural change score
        structural_score = 0.0
        try:
            if self.bg_estimate is not None:
                bg = cv2.convertScaleAbs(self.bg_estimate)
                cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                bgb = cv2.GaussianBlur(bg, (5, 5), 0)
                curb = cv2.GaussianBlur(cur, (5, 5), 0)
                diff = cv2.absdiff(bgb, curb)
                _, th = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
                k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k)
                change_ratio = (th > 0).sum() / float(th.size)
                structural_score = min(1.0, change_ratio * 5.0)
        except Exception:
            structural_score = 0.0

        # 3) Motion/activity score
        motion_score = 0.0
        try:
            if len(self.frame_history) >= 2:
                prev = self.frame_history[-2]
                cur = self.frame_history[-1]
                flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mean_mag = float(mag.mean())
                motion_score = 1.0 - 2 ** (-mean_mag * 10.0)
        except Exception:
            motion_score = 0.0

        confidence = (equipment_score * weights['equipment'] +
                      structural_score * weights['structural'] +
                      motion_score * weights['motion'])

        if not hasattr(self, '_conf_history'):
            self._conf_history = deque(maxlen=6)
        self._conf_history.append(confidence)
        smooth_conf = sum(self._conf_history) / len(self._conf_history)

        # verifier boost for ambiguous cases
        if self.verifier is not None and 0.4 <= smooth_conf < 0.8:
            try:
                vscore = self._run_verifier_on_frame(frame)
                confidence = max(confidence, vscore * 0.8 + confidence * 0.2)
                smooth_conf = (smooth_conf + confidence) / 2.0
            except Exception:
                pass

        return smooth_conf >= 0.4, float(smooth_conf)

    def _run_verifier_on_frame(self, frame):
        if self.verifier is None:
            return 0.0
        import torch
        device = self._verifier_device or torch.device('cpu')
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img / 255.0).permute(2, 0, 1).float().to(device)
        with torch.no_grad():
            out = self.verifier([img_t])[0]
        scores = out.get('scores', None)
        if scores is None:
            return 0.0
        top = 0.0
        for s in scores.cpu().numpy():
            if s > top:
                top = float(s)
        return float(top)

    def draw_interface(self, frame, detections):
        if detections and getattr(detections, 'boxes', None) is not None:
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, 'unknown')
                color = (0, 0, 255) if 'illegal' in class_name else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame


def resolve_model(model_arg):
    # If explicit path exists, use it
    if os.path.exists(model_arg):
        return model_arg
    aliases = {'yolov12-n': 'yolov12-n.pt', 'yolov12-s': 'yolov12-s.pt', 'yolov8n': 'yolov8n.pt'}
    if model_arg in aliases and os.path.exists(aliases[model_arg]):
        return aliases[model_arg]
    candidates = [
        os.path.join(os.path.dirname(__file__), '..', '..', 'yolov12-n.pt'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'yolov12-s.pt'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'yolov8n.pt'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'Edge', 'yolov8n.pt'),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return model_arg


def main(model_file, source, backend_url, camera_id, enable_verifier=False):
    resolved = resolve_model(model_file)
    client = UrbanMonitoringClient(resolved, backend_url, camera_id, enable_verifier=enable_verifier)

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    frame_buffer = deque(maxlen=4)
    stop_worker = False

    def inference_worker():
        nonlocal stop_worker
        while not stop_worker:
            if not frame_buffer:
                time.sleep(0.01)
                continue
            frame = frame_buffer.popleft()
            imgsz = 416
            try:
                detections = client.model(frame, conf=0.45, imgsz=imgsz, verbose=False)[0]
                client.process_detections(frame, detections)
            except Exception as e:
                print('Inference error', e)

    worker = Thread(target=inference_worker, daemon=True)
    worker.start()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    delay_ms = int(1000 / fps)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame_buffer) < frame_buffer.maxlen:
            frame_buffer.append(frame.copy())

        cv2.imshow(f"Urban Monitoring - {camera_id}", frame)
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break

    stop_worker = True
    worker.join(timeout=1.0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov12-n', help='Model path or alias (yolov12-n/yolov12-s/yolov8n)')
    parser.add_argument('--source', type=str, default='0', help='Camera index or video file path')
    parser.add_argument('--backend', type=str, default='http://127.0.0.1:8000/alerts', help='Backend alerts URL')
    parser.add_argument('--camera', type=str, default='CAM01', help='Camera id')
    parser.add_argument('--enable-verifier', action='store_true', help='Enable Mask R-CNN verifier (requires torchvision)')
    args = parser.parse_args()

    main(args.model, args.source, args.backend, args.camera, enable_verifier=args.enable_verifier)
