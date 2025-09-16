"""
Lightweight YOLO inference utility for the backend.

Loads the trained Ultralytics YOLO model once and exposes a simple predict API.
Detects device (MPS on macOS if available, else CPU).

Environment overrides:
- YOLO_WEIGHTS: absolute or repo-relative path to .pt file

Defaults to the latest training run weights found under runs/local/auto_annot_hier_mps*/weights/best.pt.
"""
from __future__ import annotations

import os
from glob import glob
from pathlib import Path
from typing import Any, Dict, List

import torch
from ultralytics import YOLO


_detector_singleton = None


def _detect_device() -> str:
    try:
        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _find_default_weights(repo_root: Path) -> Path | None:
    # Prefer most recent auto_annot_hier_mps*/weights/best.pt
    candidates = sorted(
        glob(str(repo_root / "runs" / "local" / "auto_annot_hier_mps*" / "weights" / "best.pt")),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    # Fallback to root yolov12n.pt or yolov8n.pt
    for name in ("yolov12n.pt", "yolov8n.pt"):
        q = repo_root / name
        if q.exists():
            return q
    return None


class Detector:
    def __init__(self, weights: str | Path | None = None, device: str | None = None):
        repo_root = Path(__file__).resolve().parents[1]
        # Resolve weights from env or fallback discovery
        if weights is None:
            env_w = os.getenv("YOLO_WEIGHTS", "").strip().strip('"').strip("'")
            if env_w:
                # Unescape common shell-escaped spaces and expand vars/home
                env_w = env_w.replace("\\ ", " ")
                env_w = os.path.expandvars(os.path.expanduser(env_w))
                weights_path = Path(env_w)
            else:
                weights_path = _find_default_weights(repo_root)
        else:
            weights_path = Path(str(weights))
        if not weights_path or not Path(weights_path).exists():
            raise RuntimeError("No YOLO weights found. Set YOLO_WEIGHTS or place best.pt in runs/local/.../weights/")
        self.weights = str(weights_path)
        self.device = device or _detect_device()
        # Load model once
        print(f"[Detector] Loading weights: {self.weights} on device: {self.device}")
        self.model = YOLO(self.weights)
        # Model class names mapping
        self.names = self.model.names
        # Determine people-related class IDs using heuristics
        self.people_class_ids = set()
        try:
            # Normalize names into a dict {id: name}
            if isinstance(self.names, dict):
                id_to_name = {int(k): str(v).lower() for k, v in self.names.items()}
            else:
                id_to_name = {i: str(n).lower() for i, n in enumerate(self.names)}
            keywords = [
                'person','people','pedestrian','man','woman','boy','girl','child','kid','adult','elderly','human','crowd','security','police','guard'
            ]
            for cid, nm in id_to_name.items():
                if any(kw in nm for kw in keywords):
                    self.people_class_ids.add(cid)
            # Fallbacks
            if not self.people_class_ids:
                # Common COCO index
                if any('person' in n for n in id_to_name.values()):
                    self.people_class_ids.add(0)
            # If trained with 44 classes where first 10 are people-related
            if not self.people_class_ids and len(id_to_name) >= 44:
                self.people_class_ids = set(range(0, 10))
        except Exception:
            # Final safety fallback
            self.people_class_ids = set(range(0, 10))

    def predict(self, image_path: str | Path, conf: float = 0.25, iou: float = 0.45, imgsz: int = 640) -> Dict[str, Any]:
        """
        Run inference and return structured detections.
        Returns: {
            'detections': [{'class_id': int, 'class_name': str, 'conf': float, 'xyxy': [x1,y1,x2,y2]}],
            'counts_by_class': {<name>: count},
            'total_people': int,
        }
        """
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )
        dets: List[Dict[str, Any]] = []
        counts: Dict[str, int] = {}
        total_people = 0
        if not results:
            return {"detections": dets, "counts_by_class": counts, "total_people": 0}
        r = results[0]
        boxes = getattr(r, "boxes", None)
        if boxes is None or boxes.cls is None:
            return {"detections": dets, "counts_by_class": counts, "total_people": 0}
        try:
            cls_list = boxes.cls.cpu().numpy().astype(int).tolist()
            conf_list = boxes.conf.cpu().numpy().tolist() if boxes.conf is not None else [1.0] * len(cls_list)
            xyxy = boxes.xyxy.cpu().numpy().tolist()
        except Exception:
            return {"detections": dets, "counts_by_class": counts, "total_people": 0}
        for cid, cf, bb in zip(cls_list, conf_list, xyxy):
            name = self.names.get(cid, str(cid)) if isinstance(self.names, dict) else (self.names[cid] if cid < len(self.names) else str(cid))
            dets.append({"class_id": cid, "class_name": name, "conf": float(cf), "xyxy": [float(x) for x in bb]})
            counts[name] = counts.get(name, 0) + 1
            if cid in self.people_class_ids:
                total_people += 1
        return {"detections": dets, "counts_by_class": counts, "total_people": total_people}


def get_detector() -> Detector:
    global _detector_singleton
    if _detector_singleton is None:
        _detector_singleton = Detector()
    return _detector_singleton
