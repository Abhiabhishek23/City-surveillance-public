"""
edge/illegal_construction.py

Usage:
    python illegal_construction.py --video /Users/abhishekx/Desktop/City-surveillance/Assets/:path:to:land.mp4 --bg-image /Users/abhishekx/Desktop/City-surveillance/Assets/download (3).jpeg
    or
    python illegal_construction.py --video <rtsp-or-file>         # will compute median background (first N frames)

This script detects significant new structures by background subtraction and sends alert payloads
to the backend endpoint (http://localhost:8000/alerts by default). It uses Redis for cooldown.
"""

import cv2
import numpy as np
import argparse
import time
import os
import redis
import requests
import json

# ---------- CONFIG ----------
BACKEND_ALERT_URL = os.getenv("BACKEND_ALERT_URL", "http://localhost:8000/alerts")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
CAMERA_ID = os.getenv("CAMERA_ID", "CAM_01")
COOLDOWN_SECONDS = int(os.getenv("CONSTRUCTION_COOLDOWN", 24 * 60 * 60))  # default 24 hours
MIN_CONTOUR_AREA = int(os.getenv("MIN_CONTOUR_AREA", 5000))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 640))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 480))
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "./snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
# ----------------------------

rds = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

def compute_median_background(cap, n_frames=30, skip=2):
    frames = []
    read = 0
    while read < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frames.append(frame.astype(np.float32))
        read += 1
        for _ in range(skip):
            cap.grab()
    if not frames:
        return None
    median = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
    return median

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21,21), 0)
    return blur

def send_construction_alert(bbox, area, snapshot_path):
    payload = {
        "camera_id": CAMERA_ID,
        "event": "illegal_construction",
        "count": None,
        "bbox": [int(v) for v in bbox],  # [x,y,w,h]
        "snapshot": snapshot_path,
        "timestamp": int(time.time())
    }
    try:
        resp = requests.post(BACKEND_ALERT_URL, json=payload, timeout=5)
        print("[INFO] Backend response:", resp.status_code, resp.text)
    except Exception as e:
        print("[ERROR] sending alert:", e)

def main(args):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("[ERROR] video source cannot be opened:", args.video)
        return

    # Prepare background
    if args.bg_image and os.path.exists(args.bg_image):
        bg = cv2.imread(args.bg_image)
        bg = cv2.resize(bg, (FRAME_WIDTH, FRAME_HEIGHT))
        bg_pre = preprocess(bg)
        print("[INFO] Using provided background image:", args.bg_image)
    else:
        print("[INFO] Computing median background from first frames...")
        bg = compute_median_background(cap, n_frames=args.median_frames)
        if bg is None:
            print("[ERROR] could not compute background from source")
            return
        bg_pre = preprocess(bg)
        # after median compute, reset capture to start reading from 0 (if file)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cooldown_key = f"alert:{CAMERA_ID}:construction"

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream / cannot read frame. Exiting.")
            break
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = preprocess(frame)

        diff = cv2.absdiff(bg_pre, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA:
                continue
            x,y,w,h = cv2.boundingRect(c)
            rect_area = w*h
            if rect_area == 0:
                continue
            rect_ratio = area / float(rect_area)
            aspect = w / float(h) if h>0 else 0
            # heuristic filters: rectangular-ish, not extremely tall/skinny
            if rect_ratio > 0.3 and 0.2 < aspect < 5.0:
                candidates.append((x,y,w,h,area))

        # draw & handle candidates
        alert_sent = False
        for (x,y,w,h,area) in candidates:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(frame, f"NewStruct {int(area)}", (x,y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),2)

        # If any candidate found, and no cooldown, send alert
        if candidates:
            if not rds.exists(cooldown_key):
                # Save snapshot
                ts = int(time.time())
                fname = f"{CAMERA_ID}_{ts}.jpg"
                path = os.path.join(SNAPSHOT_DIR, fname)
                cv2.imwrite(path, frame)
                # choose largest candidate to send bbox
                largest = max(candidates, key=lambda t: t[4])
                x,y,w,h,area = largest
                print(f"[ALERT] Detected candidate bbox: {x,y,w,h} area={area}. Sending alert.")
                send_construction_alert([x,y,w,h], area, path)
                # set cooldown
                rds.setex(cooldown_key, COOLDOWN_SECONDS, "1")
                alert_sent = True
            else:
                print("[INFO] Construction alert suppressed due to cooldown.")

        # visualization
        cv2.imshow("Construction Monitor", frame)
        cv2.imshow("Diff Thresh", thresh)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="video file or rtsp url")
    parser.add_argument("--bg-image", required=False, help="reference empty land image (preferred)")
    parser.add_argument("--median-frames", type=int, default=30, help="frames to use when computing median bg")
    args = parser.parse_args()
    main(args)
