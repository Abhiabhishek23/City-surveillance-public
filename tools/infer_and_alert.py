#!/usr/bin/env python3
"""
Infer-and-Alert pipeline

Runs YOLO detections on a video/image/stream, applies Attribute Layer rules and (optional)
GIS zones to decide what is legal/illegal/conditional, and posts alerts with a snapshot
to the backend /alerts endpoint. Also logs per-frame detections to logs/client_frame_log.jsonl.

Usage (local example):
  python tools/infer_and_alert.py \
    --weights models/yolov12n_indian_urban/best.pt \
    --data Indian_Urban_Dataset_yolo/data.yaml \
    --source Assets/test_land.mp4 \
    --backend http://127.0.0.1:8000 \
    --camera-id CAM_TEST \
    --conf 0.25 --iou 0.45

Notes:
 - If zones.geojson is empty or missing, zone-based conditional checks are skipped.
 - Crowd alerts are computed from person detections and configurable thresholds.
 - Cooldowns are enforced per (camera_id, event) to avoid spamming the backend.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, time as dtime

import cv2
import numpy as np
import requests

try:
    from shapely.geometry import shape, Point, Polygon, MultiPolygon
    HAVE_SHAPELY = True
except Exception:
    HAVE_SHAPELY = False

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit("Ultralytics is required. Please 'pip install ultralytics' in your environment.")


# -------------------------------
# Config and rules
# -------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_json(path: str, default: Any = None) -> Any:
    if not path or not os.path.exists(path):
        return default
    with open(path, 'r') as f:
        return json.load(f)


@dataclass
class AttributeRules:
    per_class: Dict[str, Any]
    crowd: Dict[str, Any]
    min_confidence: Dict[str, Any]
    cooldown_seconds: Dict[str, Any]

    @staticmethod
    def from_path(path: str) -> 'AttributeRules':
        data = load_json(path, default={}) or {}
        return AttributeRules(
            per_class=data.get('per_class', {}),
            crowd=data.get('crowd', {"event": "overcrowding", "threshold_default": 20, "per_camera_overrides": {}}),
            min_confidence=data.get('min_confidence', {"default": 0.35, "per_class": {}}),
            cooldown_seconds=data.get('cooldown_seconds', {"default": 60, "per_event": {}}),
        )

    def get_min_conf(self, cls_name: str) -> float:
        return float(self.min_confidence.get('per_class', {}).get(cls_name, self.min_confidence.get('default', 0.35)))

    def get_crowd_threshold(self, camera_id: str) -> int:
        per = self.crowd.get('per_camera_overrides', {})
        if camera_id in per:
            return int(per[camera_id])
        return int(self.crowd.get('threshold_default', 20))

    def get_cooldown(self, event: str) -> int:
        per = self.cooldown_seconds.get('per_event', {})
        if event in per:
            return int(per[event])
        return int(self.cooldown_seconds.get('default', 60))


# -------------------------------
# GIS Zones
# -------------------------------

class ZonesIndex:
    """Zones index supporting:
    - Camera-specific pixel polygons: properties.camera_id = "CAM01" or omitted (applies to all)
    - Optional geo-calibrated zones: if calibration has a homography for the camera, (x,y) is mapped before hit-test
    Calibration file format (config/camera_calibration.json):
    {
      "CAM01": {"homography": [[h11,h12,h13],[h21,h22,h23],[h31,h32,h33]]}
    }
    """

    def __init__(self, geojson_path: Optional[str], calibration_path: Optional[str]):
        self.enabled = False
        self.features: List[Tuple[Any, Dict[str, Any]]] = []  # (geom, properties)
        self.calibration = load_json(calibration_path, default={}) if calibration_path and os.path.exists(calibration_path) else {}
        if not geojson_path or not os.path.exists(geojson_path) or not HAVE_SHAPELY:
            return
        try:
            gj = load_json(geojson_path, default={}) or {}
            feats = gj.get('features', [])
            for feat in feats:
                geom = feat.get('geometry')
                props = feat.get('properties', {})
                if not geom:
                    continue
                g = shape(geom)
                self.features.append((g, props))
            if self.features:
                self.enabled = True
        except Exception:
            # Non-fatal: operate without zones
            self.enabled = False

    def _apply_homography(self, camera_id: str, x: float, y: float) -> Tuple[float, float]:
        try:
            cam = self.calibration.get(camera_id)
            if not cam:
                return x, y
            H = cam.get('homography')
            if not H:
                return x, y
            H = np.array(H, dtype=float).reshape(3, 3)
            vec = np.array([x, y, 1.0])
            out = H @ vec
            if out[2] == 0:
                return x, y
            return float(out[0] / out[2]), float(out[1] / out[2])
        except Exception:
            return x, y

    def lookup(self, x_center: float, y_center: float, frame_w: int, frame_h: int, camera_id: str = "") -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        # If homography exists, map point first
        mx, my = self._apply_homography(camera_id, x_center, y_center)
        pt = Point(float(mx), float(my))
        hits = []
        for geom, props in self.features:
            # Filter by camera_id if provided in props
            zcam = props.get('camera_id')
            if zcam and camera_id and str(zcam) != str(camera_id):
                continue
            try:
                if geom.contains(pt) or geom.intersects(pt):
                    hits.append(props)
            except Exception:
                continue
        return hits


# -------------------------------
# Infer and decide violations
# -------------------------------

def draw_boxes(img: np.ndarray, boxes: np.ndarray, clsnames: List[str], confs: np.ndarray, cls_ids: np.ndarray) -> np.ndarray:
    for (x1, y1, x2, y2), c, ci in zip(boxes.astype(int), confs, cls_ids):
        name = clsnames[int(ci)] if 0 <= int(ci) < len(clsnames) else str(ci)
        color = (0, 255, 0)
        if name in {"garbage_pile", "open_fire", "banner_hoarding", "sand_pile"}:
            color = (0, 0, 255)
        elif name in {"hawker", "tent_pandal", "loudspeaker", "excavator"}:
            color = (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{name} {c:.2f}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def pick_event_for_class(cls_name: str, rules: AttributeRules) -> Optional[str]:
    pc = rules.per_class.get(cls_name, {})
    return pc.get('event')


def _now_local_time() -> dtime:
    try:
        return datetime.now().time()
    except Exception:
        return dtime(0, 0)


def _parse_hhmm(s: str) -> Optional[dtime]:
    try:
        hh, mm = s.strip().split(":")
        return dtime(int(hh), int(mm))
    except Exception:
        return None


def _is_time_in_window(now_t: dtime, start_s: str, end_s: str) -> bool:
    start = _parse_hhmm(start_s)
    end = _parse_hhmm(end_s)
    if not start or not end:
        return False
    if start <= end:
        # same-day window
        return start <= now_t <= end
    # overnight window (e.g., 23:00-06:00)
    return now_t >= start or now_t <= end


def _apply_time_windows(pc: Dict[str, Any], level: Optional[str]) -> Optional[str]:
    """Return 'illegal' or 'ok' or None based on time policy for this level.
    - If allowed windows exist for level and now is inside one -> 'ok'
    - If allowed windows exist for level and now is outside all -> 'illegal'
    - If disallowed windows exist for level and now is inside one -> 'illegal'
    - If no matching windows -> None
    """
    if not level:
        return None
    tw = pc.get('time_windows') or {}
    now_t = _now_local_time()

    # Allowed windows
    allowed = tw.get('allowed') or []
    any_allowed_match_level = False
    any_allowed_now_inside = False
    for w in allowed:
        levels = [str(x).lower() for x in (w.get('levels') or [])]
        if levels and level not in levels:
            continue
        any_allowed_match_level = True
        if _is_time_in_window(now_t, str(w.get('start', '00:00')), str(w.get('end', '23:59'))):
            any_allowed_now_inside = True
            break
    if any_allowed_match_level:
        return 'ok' if any_allowed_now_inside else 'illegal'

    # Disallowed windows
    disallowed = tw.get('disallowed') or []
    for w in disallowed:
        levels = [str(x).lower() for x in (w.get('levels') or [])]
        if levels and level not in levels:
            continue
        if _is_time_in_window(now_t, str(w.get('start', '00:00')), str(w.get('end', '23:59'))):
            return 'illegal'

    return None


def decide_is_illegal(cls_name: str, zones_props: List[Dict[str, Any]], rules: AttributeRules, include_conditional: bool = False) -> Tuple[bool, str]:
    """Return (illegal, reason_status). reason_status in {legal, illegal, conditional}.
    If include_conditional is True, treat conditional as illegal for alerting purposes.
    """
    pc = rules.per_class.get(cls_name, {})
    status = pc.get('status', 'legal')
    if status == 'illegal':
        return True, 'illegal'
    if status == 'conditional':
        # Zone-based policy: look for any property that marks disallowed area
        # Expect props like {"zone_type": "no_hawker_zone"}
        zone_policies = pc.get('zones', {})  # e.g., {"no_hawker_zone": "illegal", "vending_zone": "legal"}
        level_policies = pc.get('zone_level', {})  # e.g., {"red": "illegal", "orange": "conditional", ...}
        # First, check zone_level override if any of the zones provide a level (red/orange/yellow/green)
        zlevel_outcome = None
        zlevel_found = None
        for props in zones_props:
            zl = str(props.get('zone_level', '')).lower().strip()
            if zl and zl in level_policies:
                zlevel_outcome = level_policies[zl]
                zlevel_found = zl
                break
        if zlevel_outcome == 'illegal':
            return True, 'illegal'
        if zlevel_outcome == 'legal':
            return False, 'legal'
        # If zone-level is conditional or not provided, check time windows if any
        time_gate = _apply_time_windows(pc, zlevel_found)
        if time_gate == 'illegal':
            return True, 'illegal'
        if zlevel_outcome == 'conditional' and include_conditional:
            return True, 'conditional'
        outcome = None
        for props in zones_props:
            for k, v in props.items():
                if isinstance(v, str) and v in zone_policies:
                    outcome = zone_policies[v]
                    break
            if outcome:
                break
        # If outcome determined by zones
        if outcome == 'illegal':
            return True, 'illegal'
        if include_conditional:
            return True, 'conditional'
        return False, 'conditional'
    return False, 'legal'


def save_snapshot(img: np.ndarray, out_dir: str, camera_id: str, event: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{camera_id}_{event}_{int(time.time()*1000)}.jpg"
    path = os.path.join(out_dir, fname)
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return path


def post_alert(backend: str, camera_id: str, event: str, count: int, snapshot_path: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    url = backend.rstrip('/') + '/alerts'
    with open(snapshot_path, 'rb') as f:
        files = {'snapshot': (os.path.basename(snapshot_path), f, 'image/jpeg')}
        data = {'camera_id': camera_id, 'event': event, 'count': str(count)}
        try:
            resp = requests.post(url, files=files, data=data, timeout=timeout)
            try:
                return resp.json()
            except Exception:
                return {"status": str(resp.status_code), "text": resp.text}
        except Exception as e:
            print(f"! HTTP post to {url} failed: {e}")
            return None


def append_frame_log(repo_root: str, camera_id: str, dets: List[Dict[str, Any]]):
    log_dir = os.path.join(repo_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, 'client_frame_log.jsonl')
    rec = {
        'ts': time.time(),
        'camera_id': camera_id,
        'detections': dets,
    }
    try:
        with open(path, 'a') as f:
            f.write(json.dumps(rec) + '\n')
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description='Infer and send alerts to backend based on YOLO detections and legal/GIS rules')
    ap.add_argument('--weights', type=str, required=True, help='Path to YOLO weights (.pt)')
    ap.add_argument('--data', type=str, default='', help='Path to YOLO data.yaml (for class names if not embedded)')
    ap.add_argument('--source', type=str, required=True, help='Image/video path, folder, or stream (e.g. 0)')
    ap.add_argument('--backend', type=str, required=True, help='Backend base URL, e.g. http://127.0.0.1:8000')
    ap.add_argument('--camera-id', type=str, required=True, help='Camera ID to attach to alerts')
    ap.add_argument('--zones', type=str, default='config/zones.geojson', help='GeoJSON zones for policy checks (optional)')
    ap.add_argument('--calibration', type=str, default='config/camera_calibration.json', help='Camera calibration for geo zones (homography mapping)')
    ap.add_argument('--rules', type=str, default='config/attribute_rules.json', help='Attribute rules JSON')
    ap.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    ap.add_argument('--iou', type=float, default=0.45, help='IoU threshold')
    ap.add_argument('--device', type=str, default='', help='Device id (e.g., 0 for GPU, or "mps" on mac)')
    ap.add_argument('--save-dir', type=str, default='edge_uploads', help='Where to save annotated snapshots before posting')
    ap.add_argument('--include-conditional', action='store_true', help='Send alerts for conditional violations even without zone proof')
    ap.add_argument('--dry-run', action='store_true', help='Do not post to backend, just show/log detections')
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    zones_path = args.zones if os.path.isabs(args.zones) else os.path.join(repo_root, args.zones)
    calib_path = args.calibration if os.path.isabs(args.calibration) else os.path.join(repo_root, args.calibration)
    rules_path = args.rules if os.path.isabs(args.rules) else os.path.join(repo_root, args.rules)
    save_dir = args.save_dir if os.path.isabs(args.save_dir) else os.path.join(repo_root, args.save_dir)

    rules = AttributeRules.from_path(rules_path)
    zones = ZonesIndex(zones_path, calib_path)

    model = YOLO(args.weights)

    # Resolve class names
    clsnames: Dict[int, str] = model.model.names if hasattr(model, 'model') else {}
    if (not clsnames) and args.data and os.path.exists(args.data):
        y = load_yaml(args.data)
        names = y.get('names')
        if isinstance(names, dict):
            clsnames = {int(k): v for k, v in names.items()}
        elif isinstance(names, list):
            clsnames = {i: n for i, n in enumerate(names)}
    # fallback: ensure it's a list for fast indexing
    if isinstance(clsnames, dict):
        name_list = [clsnames[i] for i in sorted(clsnames.keys())]
    elif isinstance(clsnames, list):
        name_list = clsnames
    else:
        name_list = []

    cooldown_mem: Dict[Tuple[str, str], float] = {}

    # Run streaming prediction for video/webcam; for image/folder, Ultralytics also yields results
    results = model.predict(
        source=args.source,
        stream=True,
        conf=args.conf,
        iou=args.iou,
        device=args.device or None,
        imgsz=640,
        verbose=False,
        max_det=300,
    )

    for res in results:
        try:
            img = res.orig_img.copy() if hasattr(res, 'orig_img') else None
            if img is None:
                # Some backends may not return the raw image
                continue
            height, width = img.shape[:2]

            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                append_frame_log(repo_root, args.camera_id, [])
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            # Build detections for logging
            dets_for_log: List[Dict[str, Any]] = []
            person_count = 0
            for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids):
                n = name_list[cid] if 0 <= cid < len(name_list) else str(cid)
                dets_for_log.append({
                    'class': n,
                    'conf': float(c),
                    'box': [float(x1), float(y1), float(x2), float(y2)],
                })
                if n == 'person' and c >= rules.get_min_conf('person'):
                    person_count += 1

            append_frame_log(repo_root, args.camera_id, dets_for_log)

            # Decide crowd event
            crowd_event = rules.crowd.get('event', 'overcrowding')
            crowd_thr = rules.get_crowd_threshold(args.camera_id)
            fire_events: List[Tuple[str, int]] = []  # (event, count)
            if person_count >= crowd_thr:
                fire_events.append((crowd_event, int(person_count)))

            # Object-based events
            for (x1, y1, x2, y2), c, cid in zip(xyxy, confs, cls_ids):
                name = name_list[cid] if 0 <= cid < len(name_list) else str(cid)
                if c < rules.get_min_conf(name):
                    continue
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                zone_props = zones.lookup(cx, cy, width, height, camera_id=args.camera_id)
                illegal, status = decide_is_illegal(name, zone_props, rules, include_conditional=args.include_conditional)
                if not illegal:
                    continue
                event = pick_event_for_class(name, rules) or name
                fire_events.append((event, 1))

            if not fire_events:
                continue

            # Draw and save snapshot once per frame for the highest priority event
            annotated = draw_boxes(img.copy(), xyxy, name_list, confs, cls_ids)

            # Post events honoring cooldowns
            for event, count in fire_events:
                key = (args.camera_id, event)
                cooldown = rules.get_cooldown(event)
                now = time.time()
                last = cooldown_mem.get(key, 0)
                if last and (now - last) < cooldown:
                    continue
                snapshot_path = save_snapshot(annotated, save_dir, args.camera_id, event)
                if args.dry_run:
                    print(f"[DRY] Would POST event={event} count={count} snapshot={snapshot_path}")
                else:
                    resp = post_alert(args.backend, args.camera_id, event, count, snapshot_path)
                    print(f"POST /alerts -> {resp}")
                cooldown_mem[key] = now

        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Frame processing error:", e)
            continue


if __name__ == '__main__':
    main()
