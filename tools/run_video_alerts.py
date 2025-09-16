import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import requests

# Ensure repo root in sys.path to import backend modules when running from project root
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
_sys.path.insert(0, str(REPO_ROOT / 'backend'))
try:
    from backend.utils.inference import get_detector
except Exception:
    from utils.inference import get_detector  # type: ignore

BACKEND = os.getenv('BACKEND_ORIGIN', 'http://127.0.0.1:8000')
UPLOAD_ENDPOINT = f"{BACKEND}/alerts"


def send_alert(image_path: Path, camera_id: str, event: str, count: int) -> Optional[int]:
    try:
        with open(image_path, 'rb') as f:
            files = { 'snapshot': (image_path.name, f, 'image/jpeg') }
            data = { 'camera_id': camera_id, 'event': event, 'count': str(count) }
            r = requests.post(UPLOAD_ENDPOINT, files=files, data=data, timeout=20)
        try:
            j = r.json()
        except Exception:
            print('Upload error:', r.status_code, r.text[:200])
            return None
        if r.ok and j.get('status') in ('ok','ignored'):
            print('Alert response:', j)
            return j.get('alert_id')
        print('Alert failed:', j)
    except Exception as e:
        print('send_alert error:', e)
    return None


def main():
    if len(sys.argv) < 2:
        print('Usage: python tools/run_video_alerts.py <video_path> [camera_id]')
        sys.exit(1)
    video_path = Path(sys.argv[1]).expanduser()
    if not video_path.exists():
        print('Video not found:', video_path)
        sys.exit(2)
    camera_id = sys.argv[2] if len(sys.argv) > 2 else video_path.stem

    # Primary detector (trained weights via backend utils)
    det = get_detector()
    # Optional: secondary fallback detector (e.g., pretrained COCO) if provided via env
    fallback_weights = os.getenv('VIDEO_FALLBACK_WEIGHTS', '').strip()
    coco_det = None
    if fallback_weights:
        try:
            from ultralytics import YOLO as _Y
            coco_det = _Y(fallback_weights)
            print(f"[Runner] Loaded fallback weights: {fallback_weights}")
        except Exception as e:
            print('[Runner] Fallback load failed:', e)
    # Use ffmpeg via ultralytics or manual frame extraction isn't available here; so sample frames with OpenCV
    try:
        import cv2
    except Exception:
        print('OpenCV not installed in backend venv; please install opencv-python to enable video processing')
        sys.exit(3)

    # Optional CLIP: heuristic scene scoring for illegal construction/encroachment
    use_clip = True
    get_clip_score = None
    # Extended prompt banks per violation type (scene-level signals)
    clip_prompt_sets: Dict[str, Dict[str, List[str]]] = {}
    try:
        import torch
        import clip  # openai-clip
        import PIL.Image as PILImage
        device = 'cpu'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_prompt_sets = {
            'illegal_construction': {
                'pos': [
                    "illegal construction on river bed",
                    "house built on riverbank",
                    "building too close to river",
                    "building on unstable slope",
                    "encroachment near river",
                ],
                'neg': [
                    "planned riverfront promenade",
                    "legal riverfront development",
                    "authorized riverfront infrastructure",
                ],
            },
            'vehicle_encroachment': {
                'pos': [
                    "vehicles parked blocking road",
                    "illegal parking both sides road",
                    "roadside encroachment hawker and vehicles",
                    "bullock cart blocking traffic",
                ],
                'neg': ["clear urban road", "organized traffic flow"],
            },
            'unauthorized_pandal': {
                'pos': [
                    "temporary festival pandal on river floodplain",
                    "unauthorized bamboo pandal near river",
                ],
                'neg': ["empty river bank", "authorized stage far from river"],
            },
            'idol_immersion': {
                'pos': ["idol immersion in river", "people immersing idol with waste"],
                'neg': ["clean river bank without immersion"],
            },
            'unauthorized_protest': {
                'pos': ["street protest crowd blocking traffic", "people with flags blocking road protest"],
                'neg': ["normal traffic intersection"],
            },
            'waste_mismanagement': {
                'pos': ["overflowing garbage on street", "garbage spilling onto road"],
                'neg': ["clean street no garbage"],
            },
            'open_fire_hazard': {
                'pos': ["burning waste on street", "open fire near people"],
                'neg': ["street no fire"],
            },
            'fireworks_violation': {
                'pos': ["people bursting fireworks near crowd", "fireworks sparks near stalls"],
                'neg': ["quiet night street"],
            },
            'open_defecation': {
                'pos': ["people defecating open near river", "open defecation by river bank"],
                'neg': ["clean riverbank no people"],
            },
            'disaster_encroachment': {
                'pos': ["flooded illegal huts near river", "rescue boats among flooded encroachments"],
                'neg': ["dry legal riverbank settlement"],
            },
        }
        def _clip_score(img_path: Path, pos: List[str], neg: List[str]) -> float:
            try:
                image = clip_preprocess(PILImage.open(str(img_path)).convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(image)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    def _score(prompts: List[str]):
                        text = clip.tokenize(prompts).to(device)
                        text_features = clip_model.encode_text(text)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                        return float(sims.max().item())
                    sil = _score(pos)
                    slg = _score(neg)
                    return sil - slg
            except Exception:
                return 0.0
        get_clip_score = _clip_score
    except Exception:
        use_clip = False

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print('Failed to open video:', video_path)
        sys.exit(4)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    # Sample ~4 frames per second for better chance to catch people
    interval = max(int(fps // 4) or 1, 1)
    frame_idx = 0
    total_alerts = 0
    outdir = REPO_ROOT / 'edge_uploads'
    outdir.mkdir(exist_ok=True)

    thr_default = int(os.getenv('CROWD_THRESHOLD', '3'))
    # Lower initial threshold slightly to encourage first detections; can be tuned upward after validation
    # Thresholds & cooldowns per violation (seconds)
    # Adjusted some under-firing categories (idol_immersion, fireworks_violation, open_defecation) slightly downward
    violation_specs: Dict[str, Dict[str, float]] = {
        'illegal_construction': {'thr': float(os.getenv('ILLEGAL_DELTA', '0.15')), 'cooldown': float(os.getenv('ILLEGAL_COOLDOWN', '60'))},
        'vehicle_encroachment': {'thr': 0.18, 'cooldown': 45},
        'unauthorized_pandal': {'thr': 0.20, 'cooldown': 120},
        'idol_immersion': {'thr': 0.17, 'cooldown': 90},
        'unauthorized_protest': {'thr': 0.15, 'cooldown': 60},
        'waste_mismanagement': {'thr': 0.13, 'cooldown': 90},
        'open_fire_hazard': {'thr': 0.14, 'cooldown': 75},
        'fireworks_violation': {'thr': 0.16, 'cooldown': 60},
        'open_defecation': {'thr': 0.15, 'cooldown': 120},
        'disaster_encroachment': {'thr': 0.14, 'cooldown': 120},
    }
    # --- Dynamic threshold overrides ---
    # Priority order:
    # 1. Environment variable VIOLATION_THRESHOLDS_JSON (JSON string)
    # 2. File config/violation_thresholds.json
    # Format example:
    # {"vehicle_encroachment": {"thr": 0.16}, "fireworks_violation": {"thr": 0.15, "cooldown": 45}}
    import json as _json
    overrides_raw = os.getenv('VIOLATION_THRESHOLDS_JSON', '').strip()
    overrides: Dict[str, Dict[str, float]] = {}
    if overrides_raw:
        try:
            overrides = _json.loads(overrides_raw)
        except Exception as e:
            print('[Thresholds] Failed to parse VIOLATION_THRESHOLDS_JSON:', e)
    else:
        cfg_file = REPO_ROOT / 'config' / 'violation_thresholds.json'
        if cfg_file.exists():
            try:
                overrides = _json.loads(cfg_file.read_text())
            except Exception as e:
                print('[Thresholds] Failed to read config/violation_thresholds.json:', e)
    if overrides:
        for k, v in overrides.items():
            if k in violation_specs:
                violation_specs[k].update({kk: vv for kk, vv in v.items() if kk in ('thr','cooldown') and isinstance(vv, (int,float))})
        print('[Thresholds] Applied overrides:', {k: violation_specs[k] for k in overrides.keys() if k in violation_specs})
    last_emit: Dict[str, float] = {k: 0.0 for k in violation_specs.keys()}
    # Rolling window of recent deltas (store last 3) and stats
    recent_deltas: Dict[str, List[float]] = {k: [] for k in violation_specs.keys()}
    max_delta: Dict[str, float] = {k: float('-inf') for k in violation_specs.keys()}
    sum_delta: Dict[str, float] = {k: 0.0 for k in violation_specs.keys()}
    count_delta: Dict[str, int] = {k: 0 for k in violation_specs.keys()}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval != 0:
            frame_idx += 1
            continue
        # write temp jpg
        ts = int(time.time()*1000)
        img_path = outdir / f"{camera_id}_frame_{ts}.jpg"
        ok = cv2.imwrite(str(img_path), frame)
        if not ok:
            frame_idx += 1
            continue
        # run detection
        conf = float(os.getenv('DEMO_CONF', '0.15'))
        iou = float(os.getenv('DEMO_IOU', '0.45'))
        res = det.predict(str(img_path), conf=conf, iou=iou)
        people = int(res.get('total_people', 0))
        if people == 0 and coco_det is not None:
            try:
                _res = coco_det.predict(source=str(img_path), conf=conf, iou=iou, verbose=False)
                _people = 0
                if _res:
                    r0 = _res[0]
                    boxes = getattr(r0, 'boxes', None)
                    if boxes is not None and boxes.cls is not None:
                        cls_list = boxes.cls.cpu().numpy().astype(int).tolist()
                        # COCO person id is 0
                        _people = sum(1 for c in cls_list if c == 0)
                people = max(people, _people)
            except Exception:
                pass
        if people >= thr_default:
            alert_id = send_alert(img_path, camera_id=camera_id, event='overcrowding', count=people)
            if alert_id:
                total_alerts += 1
        # Multi-violation CLIP heuristics
        if use_clip and get_clip_score is not None:
            now = time.time()
            for vtype, spec in violation_specs.items():
                if vtype not in clip_prompt_sets:
                    continue
                if now - last_emit[vtype] < spec['cooldown']:
                    continue
                prompts = clip_prompt_sets[vtype]
                delta = get_clip_score(img_path, prompts['pos'], prompts['neg'])
                # Stats update
                max_delta[vtype] = max(max_delta[vtype], delta)
                sum_delta[vtype] += delta
                count_delta[vtype] += 1
                rd = recent_deltas[vtype]
                rd.append(delta)
                if len(rd) > 3:
                    rd.pop(0)
                rolling_avg = sum(rd) / len(rd)
                # Print with rolling average context
                print(f"[CLIP:{vtype}] delta={delta:.4f} avg3={rolling_avg:.4f} thr={spec['thr']} frame={img_path.name}")
                # Trigger if instantaneous OR rolling average exceed threshold
                if delta >= spec['thr'] or rolling_avg >= (spec['thr'] * 1.02):  # small +2% cushion to avoid borderline spam
                    aid = send_alert(img_path, camera_id=camera_id, event=vtype, count=1)
                    if aid:
                        total_alerts += 1
                        last_emit[vtype] = now
                        print(f"[CLIP:{vtype}] alert sent (delta={delta:.4f}, avg3={rolling_avg:.4f})")
        frame_idx += 1

    cap.release()
    # End-of-run summary
    print(f'Done. Alerts sent: {total_alerts}')
    print('--- Violation Delta Summary ---')
    for vtype in violation_specs.keys():
        if count_delta[vtype] == 0:
            continue
        avg_all = sum_delta[vtype] / count_delta[vtype]
        max_d = max_delta[vtype]
        print(f"{vtype}: avg_all={avg_all:.4f} max={max_d:.4f} thr={violation_specs[vtype]['thr']}")


if __name__ == '__main__':
    main()
