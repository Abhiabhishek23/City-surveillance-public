# main.py (Backend Server)

import os
from dotenv import load_dotenv, find_dotenv
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Form, Depends, BackgroundTasks, HTTPException, Header, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import redis
import time
import json
import hmac
import hashlib
import requests
from pathlib import Path
import traceback
from starlette.responses import FileResponse
from glob import glob
from typing import Dict, List, Optional, Any
import asyncio
from collections import deque
from threading import Lock, Event, Thread

# Inference utility
try:
    from backend.utils.inference import get_detector
except Exception:
    # Fallback import path when running as module from within backend/
    try:
        from utils.inference import get_detector  # type: ignore
    except Exception:
        get_detector = None

# Import database and model definitions
from database import engine, Base, get_db, SessionLocal
from models import Alert as AlertModel, Personnel, NotificationLog

# --- SETUP ---
# Create database tables on startup
Base.metadata.create_all(bind=engine)

# Load .env if present (search from repo root and backend folder)
try:
    # Prefer backend/.env for backend-specific settings
    _backend_dir = os.path.dirname(__file__)
    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)
    else:
        # Try to locate a .env up the tree; fallback to explicit repo root path
        env_path = find_dotenv(usecwd=True)
        if not env_path:
            _repo_root = os.path.abspath(os.path.join(_backend_dir, '..'))
            env_path = os.path.join(_repo_root, '.env')
        load_dotenv(env_path)
except Exception:
    pass

# Environment variables for configuration
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "backend_uploads")
# IMPORTANT: Set FCM_KEY via environment (no hardcoded default). Support both FCM_KEY and FCM_SERVER_KEY.
FCM_KEY = os.getenv("FCM_KEY") or os.getenv("FCM_SERVER_KEY", "")
FCM_URL = "https://fcm.googleapis.com/fcm/send"
# Optional: VAPID public key for Web Push (from Firebase Console -> Cloud Messaging -> Web configuration)
FCM_VAPID_PUBLIC_KEY = os.getenv("FCM_VAPID_PUBLIC_KEY") or os.getenv("VAPID_PUBLIC_KEY", "")
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "")
AUTH_STRICT = int(os.getenv("AUTH_STRICT", "1"))
# Allow dev to bypass auth if ALLOW_DEV_NO_AUTH=true/1
if str(os.getenv("ALLOW_DEV_NO_AUTH", "")).lower() in ("1", "true", "yes", "on"):
    AUTH_STRICT = 0
ADMIN_USER_IDS = set([s.strip() for s in os.getenv("ADMIN_USER_IDS", "1").split(',') if s.strip()])
ADMIN_FIREBASE_UIDS = set([s.strip() for s in os.getenv("ADMIN_FIREBASE_UIDS", "").split(',') if s.strip()])
ADMIN_BYPASS_SECRET = os.getenv("ADMIN_BYPASS_SECRET", "")  # Dev-only header secret for admin bypass
SECRET_KEY = os.getenv("SECRET_KEY", "")
if not SECRET_KEY:
    # generate a runtime key (non-persistent) and warn; recommend setting SECRET_KEY in .env for stable signing
    SECRET_KEY = hashlib.sha256(os.urandom(32)).hexdigest()
    print("Warning: SECRET_KEY not set; generated volatile key for signed URLs.")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
CROWD_THRESHOLD_DEFAULT = int(os.getenv("CROWD_THRESHOLD", "3"))  # Default threshold
# Optional per-camera threshold overrides via JSON in env, e.g. '{"CAM01":1, "CAM02":5}'
import json as _json
_crowd_map_raw = os.getenv("CROWD_THRESHOLD_MAP", "{}")
try:
    CROWD_THRESHOLD_MAP = _json.loads(_crowd_map_raw)
except Exception:
    CROWD_THRESHOLD_MAP = {}

def get_crowd_threshold(camera_id: str):
    return int(CROWD_THRESHOLD_MAP.get(camera_id, CROWD_THRESHOLD_DEFAULT))

os.makedirs(UPLOAD_DIR, exist_ok=True)
# Directory to persist uploaded videos for playback in the GUI
VIDEO_UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'video_uploads'))
os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)

# Centralized violation threshold config (JSON file)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VIOLATION_CONFIG_PATH = os.getenv("VIOLATION_CONFIG_PATH", os.path.join(REPO_ROOT, 'config', 'violation_thresholds.json'))
_violation_cfg_cache: Dict[str, Any] = {}

def _default_violation_config() -> Dict[str, Any]:
    return {
        'illegal_construction': {'thr': 0.15, 'cooldown': 60},
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

def load_violation_config() -> Dict[str, Any]:
    global _violation_cfg_cache
    try:
        if os.path.exists(VIOLATION_CONFIG_PATH):
            with open(VIOLATION_CONFIG_PATH, 'r') as f:
                _violation_cfg_cache = json.load(f)
        else:
            cfg = _default_violation_config()
            os.makedirs(os.path.dirname(VIOLATION_CONFIG_PATH), exist_ok=True)
            with open(VIOLATION_CONFIG_PATH, 'w') as f:
                json.dump(cfg, f, indent=2)
            _violation_cfg_cache = cfg
    except Exception as e:
        print('Failed to load thresholds, using defaults:', e)
        _violation_cfg_cache = _default_violation_config()
    return _violation_cfg_cache

def save_violation_config(cfg: Dict[str, Any]):
    global _violation_cfg_cache
    os.makedirs(os.path.dirname(VIOLATION_CONFIG_PATH), exist_ok=True)
    with open(VIOLATION_CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2)
    _violation_cfg_cache = cfg

def resolve_violation_specs(camera_id: Optional[str]) -> Dict[str, Dict[str, float]]:
    """Merge base violation specs with optional per-camera overrides from config.
    Supports config shape:
    {
      "illegal_construction": {"thr": 0.15, "cooldown": 60},
      ...,
      "overrides": { "CAM_1": { "illegal_construction": {"thr": 0.2, "cooldown": 45} } }
    }
    """
    cfg = load_violation_config().copy()
    overrides = cfg.pop('overrides', {}) or {}
    base = {k: v for k, v in cfg.items() if isinstance(v, dict) and 'thr' in v}
    if camera_id and camera_id in overrides and isinstance(overrides[camera_id], dict):
        cam_over = overrides[camera_id]
        for k, ov in cam_over.items():
            try:
                merged = base.get(k, {}).copy()
                if 'thr' in ov: merged['thr'] = float(ov['thr'])
                if 'cooldown' in ov: merged['cooldown'] = float(ov['cooldown'])
                base[k] = merged
            except Exception:
                continue
    return base

# Redis client with graceful fallback to in-memory cache if Redis is unavailable
class _SimpleCache:
    def __init__(self):
        self._data = {}

    def exists(self, key: str) -> bool:
        now = time.time()
        v = self._data.get(key)
        if not v:
            return False
        expires_at, _ = v
        if expires_at and expires_at < now:
            # expired
            self._data.pop(key, None)
            return False
        return True

    def setex(self, key: str, ttl_seconds: int, value: str):
        self._data[key] = (time.time() + int(ttl_seconds), value)

    # Optional helper for compatibility
    def ping(self):
        return True


try:
    rds = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)
    # Validate connection early; fallback if it fails
    rds.ping()
    print(f"Redis connected at {REDIS_HOST}:6379")
except Exception as _e:
    print(f"Redis not available ({_e}); using in-memory fallback cache.")
    rds = _SimpleCache()

app = FastAPI(title="City Surveillance Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
from fastapi.staticfiles import StaticFiles
from jose import jwt
from cryptography import x509
import datetime

# Safe startup log (do not print the key). Also indicate presence of Firebase service account.
try:
    _cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH', '')
    fcm_modes = []
    if FCM_KEY:
        fcm_modes.append('HTTP_KEY')
    if _cred_path and os.path.exists(_cred_path):
        fcm_modes.append('ADMIN_SDK')
    if not fcm_modes:
        print('FCM configured: NO')
    else:
        print('FCM configured modes:', ','.join(fcm_modes))
except Exception:
    pass

# Optional server-side Sentry
SENTRY_DSN = os.getenv('SENTRY_DSN', '')
if SENTRY_DSN:
    try:
        import sentry_sdk
        sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=0.05)
        print('Sentry initialized')
    except Exception as e:
        print('Sentry init error', e)

# Normalize Firebase credential path env var if .env used FIREBASE_CREDENTIALS_FILE
if not os.getenv('FIREBASE_CREDENTIALS_PATH') and os.getenv('FIREBASE_CREDENTIALS_FILE'):
    os.environ['FIREBASE_CREDENTIALS_PATH'] = os.getenv('FIREBASE_CREDENTIALS_FILE') or ''

# Add this line to main.py to serve snapshot images
app.mount("/backend_uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
# Serve uploaded videos so the frontend can play them back
app.mount("/video_uploads", StaticFiles(directory=VIDEO_UPLOAD_DIR), name="video_uploads")
# Serve client UI (static) from Frontend folder under the same origin
_frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Frontend'))
if os.path.isdir(_frontend_dir):
    # html=True serves index.html for directory requests (so /client_ui works)
    app.mount("/client_ui", StaticFiles(directory=_frontend_dir, html=True), name="client_ui")

# Serve example assets for local presets
_assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Assets'))
if os.path.isdir(_assets_dir):
    app.mount("/assets", StaticFiles(directory=_assets_dir), name="assets")

_dashboard_build = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'build'))
if os.path.isdir(_dashboard_build):
    # Serve static assets under /admin/static to match CRA build structure
    static_dir = os.path.join(_dashboard_build, 'static')
    if os.path.isdir(static_dir):
        app.mount("/admin/static", StaticFiles(directory=static_dir), name="admin_static")
    # CRA index.html references assets at absolute /static; expose them as well
    app.mount("/static", StaticFiles(directory=static_dir), name="root_static")
    # Fallback routes to index.html for SPA routing
    @app.get("/admin")
    async def admin_index():
        idx = os.path.join(_dashboard_build, 'index.html')
        if os.path.exists(idx):
            return FileResponse(idx, media_type="text/html")
        raise HTTPException(status_code=404, detail="admin_not_found")

    @app.get("/admin/{path:path}")
    async def admin_catchall(path: str):
        idx = os.path.join(_dashboard_build, 'index.html')
        if os.path.exists(idx):
            return FileResponse(idx, media_type="text/html")
        raise HTTPException(status_code=404, detail="admin_not_found")

# --- WebSocket Alerts ---
class ConnectionManager:
    def __init__(self):
        self.active: List[Dict[str, Any]] = []  # {ws, camera_id}
        self._lock = Lock()

    async def connect(self, websocket: WebSocket, camera_id: Optional[str] = None):
        await websocket.accept()
        with self._lock:
            self.active.append({"ws": websocket, "camera_id": camera_id})

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            self.active = [c for c in self.active if c.get("ws") is not websocket]

    async def broadcast(self, message: Dict[str, Any]):
        remove_list = []
        for c in list(self.active):
            ws: WebSocket = c.get("ws")
            cam_filter = c.get("camera_id")
            try:
                if cam_filter and str(message.get('camera_id')) != str(cam_filter):
                    continue
                await ws.send_json({"type": "alert", "data": message})
            except Exception:
                remove_list.append(ws)
        for ws in remove_list:
            self.disconnect(ws)

manager = ConnectionManager()

_alert_queue = deque()
_alert_queue_lock = Lock()

def enqueue_alert(payload: Dict[str, Any]):
    with _alert_queue_lock:
        _alert_queue.append(payload)

# Background worker to broadcast queued alerts to connected websocket clients
_broadcast_task: Optional[asyncio.Task] = None

async def _alert_broadcast_worker():
    while True:
        try:
            item = None
            with _alert_queue_lock:
                if _alert_queue:
                    item = _alert_queue.popleft()
            if item is not None:
                await manager.broadcast(item)
            else:
                await asyncio.sleep(0.2)
        except Exception:
            await asyncio.sleep(0.5)

@app.on_event("startup")
async def _on_startup_ws():
    global _broadcast_task
    if _broadcast_task is None:
        _broadcast_task = asyncio.create_task(_alert_broadcast_worker())

@app.on_event("shutdown")
async def _on_shutdown_ws():
    global _broadcast_task
    try:
        if _broadcast_task:
            _broadcast_task.cancel()
    except Exception:
        pass
    _broadcast_task = None

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    try:
        # Extract optional camera_id from query string
        cam_filter = websocket.query_params.get('camera_id') if hasattr(websocket, 'query_params') else None
        await manager.connect(websocket, camera_id=cam_filter)
        while True:
            # Keep connection alive; we don't require client messages
            try:
                await websocket.receive_text()
            except Exception:
                await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

def _build_client_alert_payload(a: AlertModel) -> Dict[str, any]:
    backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port = os.getenv("BACKEND_PORT", os.getenv("PORT", "8000"))
    loc = a.camera.location if (a.camera and getattr(a.camera, 'location', None)) else a.camera_id
    image_url = None
    if a.snapshot_path and os.path.exists(a.snapshot_path):
        try:
            base_ts = int(a.created_at.timestamp()) if a.created_at else int(time.time())
        except Exception:
            base_ts = int(time.time())
        exp = base_ts + 7*24*3600
        sig = hmac.new(SECRET_KEY.encode(), f"{a.id}.{exp}".encode(), hashlib.sha256).hexdigest()
        image_rel = f"/alert_image/{a.id}?exp={exp}&sig={sig}"
        image_url = f"http://{backend_host}:{backend_port}{image_rel}"
    else:
        fb = _find_fallback_image(a.camera_id, a.event)
        if fb:
            image_url = f"http://{backend_host}:{backend_port}/backend_uploads/{os.path.basename(fb)}"
    return {
        "id": a.id,
        "event": a.event,
        "type": a.event,
        "camera_id": a.camera_id,
        "location": loc,
        "count": a.count,
        "timestamp": a.created_at.isoformat() if a.created_at else None,
        "snapshot_url": image_url,
    }

def verify_firebase_id_token(id_token: str):
    """Lightweight ID token parsing for dev.
    Returns claims without signature verification. In production, replace with proper verification.
    """
    if not id_token:
        return None
    try:
        claims = jwt.get_unverified_claims(id_token)
        return claims
    except Exception:
        return None

def _find_fallback_image(camera_id: str, event: str) -> str | None:
    """Return the newest matching snapshot in UPLOAD_DIR when an alert lacks a snapshot_path.
    Looks for files named like '{camera_id}_{event}_<uuid>.jpg'. Returns absolute path or None.
    """
    try:
        pattern = os.path.join(UPLOAD_DIR, f"{camera_id}_{event}_*.jpg")
        files = glob(pattern)
        if not files:
            return None
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return files[0]
    except Exception:
        return None
    


@app.get('/stats/aggregate')
def stats_aggregate():
    # Read logs/client_frame_log.jsonl from repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    log_path = os.path.join(repo_root, 'logs', 'client_frame_log.jsonl')
    if not os.path.exists(log_path):
        return {"error": "no_log"}
    stats = {}
    try:
        with open(log_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                cam = obj.get('camera_id')
                s = stats.setdefault(cam, {'frames':0,'detections':0,'by_class':{}})
                s['frames'] += 1
                dets = obj.get('detections', [])
                s['detections'] += len(dets)
                for d in dets:
                    cl = d.get('class')
                    s['by_class'][cl] = s['by_class'].get(cl, 0) + 1
    except Exception as e:
        return {"error": str(e)}
    return stats


@app.get('/admin_stats')
def admin_stats_page():
    html = Path(__file__).parent.joinpath('admin_stats.html').read_text()
    return HTMLResponse(content=html, status_code=200)

# --- Service Worker Route ---
# This is a critical new endpoint to serve the service worker file
@app.get("/firebase-messaging-sw.js")
async def get_service_worker():
    return FileResponse("firebase-messaging-sw.js", media_type="application/javascript")

# --- Web Client Route ---
# This new endpoint serves the HTML file for your test client
@app.get("/client")
async def get_client_html():
    return FileResponse("fcm_test_client.html", media_type="text/html")

# (No explicit /client_ui route needed; StaticFiles with html=True handles index)
@app.get("/client_ui")
async def client_ui_root():
    idx = os.path.join(_frontend_dir, 'index.html')
    if os.path.exists(idx):
        return FileResponse(idx, media_type="text/html")
    raise HTTPException(status_code=404, detail="client_ui_not_found")


# --- Inference Endpoint ---
@app.post("/infer")
async def infer_image(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    camera_id: str = Form("CAM_TEST"),
    create_alert: int = Form(0),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
    snapshot: UploadFile = File(...),
):
    """
    Run YOLO inference on an uploaded image.
    Params:
      - camera_id: optional label for the source camera
      - create_alert: set to 1 to create an overcrowding alert if threshold exceeded
      - conf, iou: detector thresholds

    Returns { detections, counts_by_class, total_people, alert_id? }
    """
    if get_detector is None:
        raise HTTPException(status_code=500, detail="inference_unavailable")

    # Save snapshot to uploads for reproducibility
    fname = f"{camera_id}_inference_{uuid.uuid4().hex}.jpg"
    snapshot_path = os.path.join(UPLOAD_DIR, fname)
    with open(snapshot_path, "wb") as f:
        shutil.copyfileobj(snapshot.file, f)

    # Run inference
    try:
        detector = get_detector()
        result = detector.predict(snapshot_path, conf=conf, iou=iou)
    except Exception as e:
        print("inference_error:", e)
        return {"status": "error", "error": str(e)}

    out = {"status": "ok", **result, "snapshot_path": snapshot_path}

    # Optionally create an overcrowding alert if threshold is exceeded
    if int(create_alert) == 1:
        total_people = int(result.get("total_people", 0))
        thr = get_crowd_threshold(camera_id)
        if total_people >= thr:
            alert_obj = AlertModel(camera_id=camera_id, event="overcrowding", count=total_people, snapshot_path=snapshot_path)
            db.add(alert_obj)
            db.commit()
            db.refresh(alert_obj)
            rds.setex(f"cooldown:{camera_id}:overcrowding", 60, "active")
            background_tasks.add_task(send_fcm_notifications, db, alert_obj)
            out["alert_id"] = alert_obj.id
        else:
            out["note"] = f"below_threshold {total_people}/{thr}"

    return out


# --- NOTIFICATION LOGIC ---
def send_fcm_notifications(db: Session, alert_obj: AlertModel):
    """Sends FCM push notifications to all active personnel."""
    # Gather recipients using a fresh DB session (avoid using request-scoped session in background tasks)
    try:
        session = SessionLocal()
        recips = session.query(Personnel).filter(Personnel.active == 1).all()
        tokens = [p.fcm_token for p in recips if p.fcm_token]
    except Exception as e:
        print('❌ Error loading recipients for FCM:', e)
        tokens = []

    # If no recipients, skip
    if not tokens:
        print("⚠️ No recipients with tokens. Skipping push notification.")
        try:
            session.close()
        except Exception:
            pass
        return

    # Build signed snapshot URL (absolute) and common metadata
    title = f"🚨 {alert_obj.event.replace('_', ' ').title()} Alert!"
    body = f"Violation detected by Camera {alert_obj.camera_id}."
    def _sign(alert_id: int, ttl_sec: int = 900):
        exp = int(time.time()) + int(ttl_sec)
        payload = f"{alert_id}.{exp}".encode()
        sig = hmac.new(SECRET_KEY.encode(), payload, hashlib.sha256).hexdigest()
        return exp, sig
    exp, sig = _sign(alert_obj.id, 900)
    snapshot_rel = f"/alert_image/{alert_obj.id}?exp={exp}&sig={sig}"
    backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port = os.getenv("BACKEND_PORT", os.getenv("PORT", "8000"))
    snapshot_url = f"http://{backend_host}:{backend_port}{snapshot_rel}"

    # Optional: lookup camera location if present
    location = alert_obj.camera_id
    try:
        cam = session.query(type(alert_obj).camera.property.mapper.class_).filter_by(camera_id=alert_obj.camera_id).first()
        if cam and getattr(cam, 'location', None):
            location = cam.location
    except Exception:
        pass

    common_data = {
        "alert_id": str(alert_obj.id),
        "snapshot_url": snapshot_url,
        "event": str(alert_obj.event),
        "camera_id": str(alert_obj.camera_id),
        "count": str(alert_obj.count if alert_obj.count is not None else ""),
        "location": str(location),
    }

    # Prefer server-side Firebase Admin SDK if credentials are available
    firebase_used = False
    firebase_resp = None
    cred_path = os.getenv('FIREBASE_CREDENTIALS_PATH', '')
    if cred_path:
        if not os.path.exists(cred_path):
            print(f"❌ FIREBASE_CREDENTIALS_PATH set but file not found: {cred_path}")
        else:
            print(f"ℹ️ Using Firebase Admin credentials at: {cred_path}")
        try:
            import firebase_admin
            from firebase_admin import credentials, messaging
            if not firebase_admin._apps:
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            # Prefer send_multicast if available, else fallback to send_all or per-token send
            firebase_used = True
            if hasattr(messaging, 'send_multicast'):
                message = messaging.MulticastMessage(
                    notification=messaging.Notification(title=title, body=body),
                    data=common_data,
                    tokens=tokens,
                )
                resp = messaging.send_multicast(message)
                firebase_resp = {'success_count': resp.success_count, 'failure_count': resp.failure_count}
                print('✅ Firebase Admin send_multicast resp:', firebase_resp)
            elif hasattr(messaging, 'send_all'):
                messages = [
                    messaging.Message(
                        notification=messaging.Notification(title=title, body=body),
                        data=common_data,
                        token=tok
                    ) for tok in tokens
                ]
                resp = messaging.send_all(messages)
                firebase_resp = {'success_count': resp.success_count, 'failure_count': resp.failure_count}
                print('✅ Firebase Admin send_all resp:', firebase_resp)
            else:
                ok = 0; fail = 0
                for tok in tokens:
                    try:
                        m = messaging.Message(
                            notification=messaging.Notification(title=title, body=body),
                            data=common_data,
                            token=tok
                        )
                        messaging.send(m)
                        ok += 1
                    except Exception:
                        fail += 1
                firebase_resp = {'success_count': ok, 'failure_count': fail}
                print('✅ Firebase Admin per-token send resp:', firebase_resp)
        except Exception as e:
            print('❌ Firebase Admin send error:', e)
            print(traceback.format_exc())

    payload = {
        "notification": {"title": title, "body": body},
        "data": common_data,
        "registration_ids": tokens,
    }
    headers = {"Authorization": f"key={FCM_KEY}", "Content-Type": "application/json"}

    if not firebase_used:
        # Fallback to legacy HTTP FCM using FCM_KEY
        if not FCM_KEY:
            print("⚠️ FCM HTTP key not configured. Skipping HTTP FCM send.")
            status = 'skipped'
            response_text = 'no_fcm_key'
        else:
            try:
                resp = requests.post(FCM_URL, headers=headers, json=payload, timeout=10)
                status = str(resp.status_code)
                response_text = resp.text
                print(f"✅ FCM HTTP Response: {status} - {response_text}")
            except Exception as e:
                status = "error"
                response_text = str(e)
                print(f"❌ FCM Sending Error: {e}")
    else:
        status = json.dumps(firebase_resp)
        response_text = 'firebase_admin_multicast'

    # Log the notification attempt using a fresh session
    try:
        log_session = SessionLocal()
        log = NotificationLog(alert_id=alert_obj.id, method="fcm", recipient=",".join(tokens), status=str(status), response=response_text)
        log_session.add(log)
        log_session.commit()
        log_session.close()
    except Exception as e:
        print('❌ Failed to write NotificationLog:', e)
    finally:
        try:
            session.close()
        except Exception:
            pass


# --- API ENDPOINTS ---
@app.get("/")
def health_check():
    return {"status": "ok", "timestamp": time.time()}

@app.get("/welcome")
def welcome_page():
        """Simple landing page with links to Client and Admin UIs."""
        host = os.getenv("BACKEND_HOST", "127.0.0.1")
        port = os.getenv("BACKEND_PORT", os.getenv("PORT", "8000"))
        html = f"""
        <html>
            <head>
                <meta charset='utf-8'/>
                <meta name='viewport' content='width=device-width,initial-scale=1'/>
                <title>City Surveillance</title>
                <style>
                    body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 40px; }}
                    .card {{ max-width: 720px; padding: 24px; border: 1px solid #eee; border-radius: 12px; box-shadow: 0 4px 16px rgba(0,0,0,0.05); }}
                    h1 {{ margin-top: 0; }}
                    a.button {{ display: inline-block; margin-right: 12px; margin-top: 8px; padding: 10px 14px; border-radius: 8px; background: #0ea5e9; color: white; text-decoration: none; }}
                    a.button.secondary {{ background: #6b7280; }}
                    code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 6px; }}
                </style>
            </head>
            <body>
                <div class="card">
                    <h1>✅ City Surveillance is running</h1>
                    <p>Pick an interface to continue:</p>
                    <p>
                        <a class="button" href="http://{host}:{port}/client_ui">Client UI</a>
                        <a class="button secondary" href="http://{host}:{port}/admin">Admin</a>
                    </p>
                    <p>Health: <code>GET /</code> • Realtime: <code>WS /ws/alerts</code></p>
                </div>
            </body>
        </html>
        """
        return HTMLResponse(content=html, status_code=200)


def _resolve_user_from_authorization(auth_header: str | None):
    """Parse Authorization: Bearer <id_token> and resolve to (user_id, firebase_uid, is_admin)."""
    # Dev-only: allow an explicit admin bypass header value
    if auth_header and ADMIN_BYPASS_SECRET and auth_header.strip() == f"Bypass {ADMIN_BYPASS_SECRET}":
        return (1, "admin-bypass", True)
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return (None, None, False)
    id_token = auth_header.split(" ", 1)[1].strip()
    payload = verify_firebase_id_token(id_token)
    if not payload:
        return (None, None, False)
    uid = payload.get("sub")
    # map to deterministic numeric id as earlier
    user_id = abs(hash(uid)) % 1000000000 or 1
    is_admin = (str(user_id) in ADMIN_USER_IDS) or (uid in ADMIN_FIREBASE_UIDS)
    return (user_id, uid, is_admin)


@app.get("/alerts")
def list_alerts(limit: int = 50, since_hours: int | None = None, db: Session = Depends(get_db), Authorization: str | None = Header(default=None)):
    """Return recent alerts as a JSON list for the dashboard.
    Each item contains id, image_url (public path), type, timestamp, camera_id and count.
    """
    # Admin-only endpoint
    _, _, is_admin = _resolve_user_from_authorization(Authorization)
    if not is_admin:
        raise HTTPException(status_code=403, detail="admin_required")
    q = db.query(AlertModel)
    if since_hours and since_hours > 0:
        try:
            cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=int(since_hours))
            q = q.filter(AlertModel.created_at >= cutoff)
        except Exception:
            pass
    alerts = q.order_by(AlertModel.created_at.desc()).limit(limit).all()
    result = []
    backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port = os.getenv("BACKEND_PORT", os.getenv("PORT", "8000"))
    for a in alerts:
        # Use signed URL for admin as well (longer validity)
        if a.snapshot_path and os.path.exists(a.snapshot_path):
            # Deterministic expiry based on created_at for stable URLs across polls
            try:
                base_ts = int(a.created_at.timestamp()) if a.created_at else int(time.time())
            except Exception:
                base_ts = int(time.time())
            # Use a long validity (7 days) so last-24h snapshots always render
            exp = base_ts + 7*24*3600
            sig = hmac.new(SECRET_KEY.encode(), f"{a.id}.{exp}".encode(), hashlib.sha256).hexdigest()
            image_rel = f"/alert_image/{a.id}?exp={exp}&sig={sig}"
        else:
            # Fallback to latest file in uploads for this camera/event
            fb = _find_fallback_image(a.camera_id, a.event)
            image_rel = f"/backend_uploads/{os.path.basename(fb)}" if fb else None
        image_url = f"http://{backend_host}:{backend_port}{image_rel}" if image_rel else None
        result.append({
            "id": a.id,
            "image_url": image_url,
            "type": a.event,
            "timestamp": a.created_at.isoformat() if a.created_at else None,
            "camera_id": a.camera_id,
            "count": a.count,
        })
    return result


@app.get('/client_alerts')
def client_alerts(limit: int = 50, since_hours: int | None = None, camera_id: str | None = None, db: Session = Depends(get_db), Authorization: str | None = Header(default=None)):
    """Client-facing recent alerts with limited fields; any valid Firebase user can access.
    Returns: [{id, event, camera_id, location, count, timestamp, snapshot_url}]
    """
    user_id, uid, is_admin = _resolve_user_from_authorization(Authorization)
    if AUTH_STRICT and not uid:
        raise HTTPException(status_code=401, detail="auth_required")
    q = db.query(AlertModel)
    if since_hours and since_hours > 0:
        try:
            cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=int(since_hours))
            q = q.filter(AlertModel.created_at >= cutoff)
        except Exception:
            pass
    if camera_id:
        q = q.filter(AlertModel.camera_id == camera_id)
    alerts = q.order_by(AlertModel.created_at.desc()).limit(limit).all()
    backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port = os.getenv("BACKEND_PORT", os.getenv("PORT", "8000"))
    out = []
    for a in alerts:
        # Lookup location from camera if available
        loc = a.camera.location if (a.camera and getattr(a.camera, 'location', None)) else a.camera_id
        # Short-lived signed URL for client (15 minutes)
        image_url = None
        if a.snapshot_path and os.path.exists(a.snapshot_path):
            # Deterministic expiry based on created_at for stable URLs across polls
            try:
                base_ts = int(a.created_at.timestamp()) if a.created_at else int(time.time())
            except Exception:
                base_ts = int(time.time())
            # Use a long validity (7 days) so last-24h snapshots always render
            exp = base_ts + 7*24*3600
            sig = hmac.new(SECRET_KEY.encode(), f"{a.id}.{exp}".encode(), hashlib.sha256).hexdigest()
            image_rel = f"/alert_image/{a.id}?exp={exp}&sig={sig}"
            image_url = f"http://{backend_host}:{backend_port}{image_rel}"
        else:
            fb = _find_fallback_image(a.camera_id, a.event)
            if fb:
                image_url = f"http://{backend_host}:{backend_port}/backend_uploads/{os.path.basename(fb)}"
        out.append({
            "id": a.id,
            "event": a.event,
            "type": a.event,
            "camera_id": a.camera_id,
            "location": loc,
            "count": a.count,
            "timestamp": a.created_at.isoformat() if a.created_at else None,
            "snapshot_url": image_url,
        })
    return out


# --- Video Upload and Background Processing ---
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = Lock()

# Live stream ingestion registry
STREAM_JOBS: Dict[str, Dict[str, Any]] = {}
STREAM_JOBS_LOCK = Lock()

@app.get('/job_status/{job_id}')
def job_status(job_id: str):
    with JOBS_LOCK:
        j = JOBS.get(job_id)
        if not j:
            raise HTTPException(status_code=404, detail='job_not_found')
        return j

@app.post('/process_video')
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    camera_id: str | None = Form(None),
    fps_target: int = Form(4),
    db: Session = Depends(get_db)
):
    """Upload a video and process it in background to generate alerts.
    Returns a playable URL and the camera_id to filter alerts on the client.
    """
    # Persist video to disk
    base_name = os.path.basename(video.filename or 'video.mp4')
    safe_name = base_name.replace(' ', '_')
    uid = uuid.uuid4().hex
    out_name = f"{uid}_{safe_name}"
    out_path = os.path.join(VIDEO_UPLOAD_DIR, out_name)
    with open(out_path, 'wb') as f:
        shutil.copyfileobj(video.file, f)

    # Resolve camera id
    cam_id = camera_id or f"UP_{int(time.time())}"
    # Ensure camera exists (optional; ignore errors if Camera model unavailable)
    try:
        from models import Camera as CameraModel
        sess = SessionLocal()
        cam = sess.query(CameraModel).filter(CameraModel.camera_id == cam_id).first()
        if not cam:
            cam = CameraModel(camera_id=cam_id, location=None, description="Uploaded video")
            sess.add(cam)
            sess.commit()
        sess.close()
    except Exception:
        pass

    # Setup job
    job = {"status": "queued", "camera_id": cam_id, "video_path": out_path, "processed": 0, "total": None, "started": time.time()}
    with JOBS_LOCK:
        JOBS[uid] = job
    # Schedule background processing
    background_tasks.add_task(_process_video_file, out_path, cam_id, int(max(1, fps_target)), uid)

    backend_host = os.getenv("BACKEND_HOST", "127.0.0.1")
    backend_port = os.getenv("BACKEND_PORT", os.getenv("PORT", "8000"))
    video_url = f"http://{backend_host}:{backend_port}/video_uploads/{out_name}"
    return {"status": "ok", "video_url": video_url, "camera_id": cam_id, "processing_id": uid}


def _process_video_file(video_path: str, camera_id: str, fps_target: int = 4, job_id: Optional[str] = None):
    """Background worker: sample frames from video, run detectors and CLIP heuristics, and insert alerts.
    Uses its own DB session. Applies per-type cooldown via Redis like the /alerts route.
    """
    print(f"[Processor] Start processing {video_path} for camera {camera_id} @ ~{fps_target} fps")
    # Detector
    try:
        det = get_detector()
    except Exception as e:
        print('[Processor] Detector init error:', e)
        return
    # Optional CLIP
    try:
        import torch
        import clip  # type: ignore
        import PIL.Image as PILImage  # type: ignore
        device = 'cpu'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        def _clip_score(img_path: str, pos: List[str], neg: List[str]) -> float:
            try:
                image = clip_preprocess(PILImage.open(img_path).convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(image)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    def _score(prompts: List[str]):
                        text = clip.tokenize(prompts).to(device)
                        text_features = clip_model.encode_text(text)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                        return float(sims.max().item())
                    sil = _score(pos); slg = _score(neg)
                    return sil - slg
            except Exception:
                return 0.0
        get_clip_score = _clip_score
        use_clip = True
    except Exception as e:
        print('[Processor] CLIP not available:', e)
        get_clip_score = None
        use_clip = False

    # Prompt sets and thresholds (aligned with tools/run_video_alerts.py intent)
    clip_prompt_sets: Dict[str, Dict[str, List[str]]] = {
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
    # Thresholds (centralized JSON)
    violation_specs: Dict[str, Dict[str, float]] = resolve_violation_specs(camera_id)
    last_emit: Dict[str, float] = {k: 0.0 for k in violation_specs.keys()}

    # Video loop
    try:
        import cv2  # type: ignore
    except Exception as e:
        print('[Processor] OpenCV not available:', e)
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print('[Processor] Failed to open video:', video_path)
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) else None
    except Exception:
        total_frames = None
    if job_id:
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]['status'] = 'running'
                JOBS[job_id]['total'] = total_frames
    interval = max(int(fps // max(1, fps_target)) or 1, 1)
    frame_idx = 0
    total_alerts = 0
    conf = float(os.getenv('DEMO_CONF', '0.15'))
    iou = float(os.getenv('DEMO_IOU', '0.45'))

    # Use a dedicated session for DB writes
    session = SessionLocal()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval != 0:
                frame_idx += 1
                continue
            ts = int(time.time()*1000)
            img_name = f"{camera_id}_frame_{ts}.jpg"
            img_path = os.path.join(UPLOAD_DIR, img_name)
            try:
                cv2.imwrite(img_path, frame)
            except Exception:
                frame_idx += 1
                continue

            # Crowd check (YOLO people count)
            try:
                res = det.predict(str(img_path), conf=conf, iou=iou)
                people = int(res.get('total_people', 0))
            except Exception:
                people = 0
            thr = get_crowd_threshold(camera_id)
            if people >= thr:
                # Create alert
                a = AlertModel(camera_id=camera_id, event='overcrowding', count=people, snapshot_path=img_path)
                session.add(a)
                session.commit()
                session.refresh(a)
                rds.setex(f"cooldown:{camera_id}:overcrowding", 60, "active")
                # Push immediately in the same worker (best-effort)
                try:
                    send_fcm_notifications(session, a)
                except Exception:
                    pass
                total_alerts += 1
                try:
                    enqueue_alert(_build_client_alert_payload(a))
                except Exception:
                    pass

            # CLIP heuristics
            if use_clip and get_clip_score is not None:
                now_ts = time.time()
                for vtype, spec in violation_specs.items():
                    if now_ts - last_emit[vtype] < spec['cooldown']:
                        continue
                    prompts = clip_prompt_sets.get(vtype)
                    if not prompts:
                        continue
                    delta = get_clip_score(img_path, prompts['pos'], prompts['neg'])
                    if delta >= spec['thr']:
                        a = AlertModel(camera_id=camera_id, event=vtype, count=1, snapshot_path=img_path)
                        session.add(a)
                        session.commit()
                        session.refresh(a)
                        cd = int(float(violation_specs.get(vtype, {}).get('cooldown', 60)))
                        rds.setex(f"cooldown:{camera_id}:{vtype}", cd, "active")
                        try:
                            send_fcm_notifications(session, a)
                        except Exception:
                            pass
                        total_alerts += 1
                        last_emit[vtype] = now_ts
                        try:
                            enqueue_alert(_build_client_alert_payload(a))
                        except Exception:
                            pass
            frame_idx += 1
            if job_id and frame_idx % interval == 0:
                with JOBS_LOCK:
                    if job_id in JOBS:
                        JOBS[job_id]['processed'] = frame_idx
    except Exception as e:
        print('[Processor] Error:', e)
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            session.close()
        except Exception:
            pass
    if job_id:
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]['status'] = 'done'
                JOBS[job_id]['alerts'] = total_alerts
    print(f"[Processor] Done {video_path}. Alerts sent: {total_alerts}")


def _process_stream(source_url: str, camera_id: str, fps_target: int = 4, job_id: Optional[str] = None):
    """Background worker to process a live RTSP/HTTP stream and emit alerts.
    Checks STREAM_JOBS[job_id]['stop'] Event to support graceful stop.
    """
    print(f"[Ingest] Starting stream {source_url} for camera {camera_id} @ ~{fps_target} fps")
    # Detector
    try:
        det = get_detector()
    except Exception as e:
        print('[Ingest] Detector init error:', e)
        return
    # Optional CLIP
    try:
        import torch
        import clip  # type: ignore
        import PIL.Image as PILImage  # type: ignore
        device = 'cpu'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        def _clip_score(img_path: str, pos: List[str], neg: List[str]) -> float:
            try:
                image = clip_preprocess(PILImage.open(img_path).convert('RGB')).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = clip_model.encode_image(image)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    def _score(prompts: List[str]):
                        text = clip.tokenize(prompts).to(device)
                        text_features = clip_model.encode_text(text)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                        return float(sims.max().item())
                    sil = _score(pos); slg = _score(neg)
                    return sil - slg
            except Exception:
                return 0.0
        get_clip_score = _clip_score
        use_clip = True
    except Exception as e:
        print('[Ingest] CLIP not available:', e)
        get_clip_score = None
        use_clip = False

    violation_specs: Dict[str, Dict[str, float]] = resolve_violation_specs(camera_id)
    last_emit: Dict[str, float] = {k: 0.0 for k in violation_specs.keys()}
    # Mirror prompt sets from _process_video_file for comprehensive coverage
    clip_prompt_sets: Dict[str, Dict[str, List[str]]] = {
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

    try:
        import cv2  # type: ignore
    except Exception as e:
        print('[Ingest] OpenCV not available:', e)
        return

    stop_event: Optional[Event] = None
    if job_id:
        with STREAM_JOBS_LOCK:
            j = STREAM_JOBS.get(job_id)
            stop_event = j.get('stop') if j else None
            if j:
                j['status'] = 'running'

    conf = float(os.getenv('DEMO_CONF', '0.15'))
    iou = float(os.getenv('DEMO_IOU', '0.45'))
    people_thr = get_crowd_threshold(camera_id)

    def should_stop() -> bool:
        return bool(stop_event.is_set()) if stop_event else False

    reconnect_delay = 2.0
    total_alerts = 0
    sample_interval = max(1.0 / max(1, fps_target), 0.1)

    while not should_stop():
        cap = cv2.VideoCapture(source_url)
        if not cap.isOpened():
            print(f"[Ingest] Failed to open stream: {source_url}. Retrying in {reconnect_delay}s")
            time.sleep(reconnect_delay)
            continue
        next_sample = time.time()
        try:
            while not should_stop():
                ok, frame = cap.read()
                if not ok or frame is None:
                    print('[Ingest] Frame read failed, reconnecting...')
                    break
                now = time.time()
                if now < next_sample:
                    # throttle
                    time.sleep(max(0.0, next_sample - now))
                next_sample = time.time() + sample_interval

                # Write a temp frame image
                ts = int(time.time()*1000)
                img_name = f"{camera_id}_rtsp_{ts}.jpg"
                img_path = os.path.join(UPLOAD_DIR, img_name)
                try:
                    cv2.imwrite(img_path, frame)
                except Exception:
                    continue

                # YOLO check
                try:
                    res = det.predict(str(img_path), conf=conf, iou=iou)
                    people = int(res.get('total_people', 0))
                except Exception:
                    people = 0
                if people >= people_thr:
                    from models import Alert as _Alert
                    s = SessionLocal()
                    try:
                        a = _Alert(camera_id=camera_id, event='overcrowding', count=people, snapshot_path=img_path)
                        s.add(a); s.commit(); s.refresh(a)
                        rds.setex(f"cooldown:{camera_id}:overcrowding", 60, "active")
                        try:
                            send_fcm_notifications(s, a)
                        except Exception:
                            pass
                        total_alerts += 1
                        enqueue_alert(_build_client_alert_payload(a))
                    except Exception as _e:
                        pass
                    finally:
                        try: s.close()
                        except Exception: pass

                # CLIP heuristics
                if use_clip and get_clip_score is not None:
                    now_ts = time.time()
                    for vtype, spec in violation_specs.items():
                        if now_ts - last_emit[vtype] < spec['cooldown']:
                            continue
                        prompts = clip_prompt_sets.get(vtype)
                        if not prompts:
                            continue
                        delta = get_clip_score(img_path, prompts['pos'], prompts['neg'])
                        if delta >= spec['thr']:
                            from models import Alert as _Alert
                            s = SessionLocal()
                            try:
                                a = _Alert(camera_id=camera_id, event=vtype, count=1, snapshot_path=img_path)
                                s.add(a); s.commit(); s.refresh(a)
                                cd = int(float(violation_specs.get(vtype, {}).get('cooldown', 60)))
                                rds.setex(f"cooldown:{camera_id}:{vtype}", cd, "active")
                                try:
                                    send_fcm_notifications(s, a)
                                except Exception:
                                    pass
                                total_alerts += 1
                                last_emit[vtype] = now_ts
                                enqueue_alert(_build_client_alert_payload(a))
                            except Exception:
                                pass
                            finally:
                                try: s.close()
                                except Exception: pass

                # Update job heartbeat
                if job_id:
                    with STREAM_JOBS_LOCK:
                        if job_id in STREAM_JOBS:
                            STREAM_JOBS[job_id]['processed'] = STREAM_JOBS[job_id].get('processed', 0) + 1
                            STREAM_JOBS[job_id]['alerts'] = total_alerts
        finally:
            try:
                cap.release()
            except Exception:
                pass

        # brief backoff before reconnect
        time.sleep(0.5)

    # Stopped
    print(f"[Ingest] Stopped stream for camera {camera_id}. Alerts sent: {total_alerts}")
    if job_id:
        with STREAM_JOBS_LOCK:
            if job_id in STREAM_JOBS:
                STREAM_JOBS[job_id]['status'] = 'stopped'


@app.get('/alert_image/{alert_id}')
def get_alert_image(alert_id: int, exp: int, sig: str, db: Session = Depends(get_db), Authorization: str | None = Header(default=None)):
    """Return snapshot image for an alert if signature is valid or caller is admin.
    Signature is HMAC-SHA256 over "{alert_id}.{exp}" with SECRET_KEY.
    """
    # Admin bypass
    _, _, is_admin = _resolve_user_from_authorization(Authorization)
    if not is_admin:
        # validate signature
        if int(exp) < int(time.time()):
            raise HTTPException(status_code=403, detail="url_expired")
        expected = hmac.new(SECRET_KEY.encode(), f"{alert_id}.{exp}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, sig):
            raise HTTPException(status_code=403, detail="bad_signature")
    a = db.query(AlertModel).filter(AlertModel.id == alert_id).first()
    if not a or not a.snapshot_path or not os.path.exists(a.snapshot_path):
        raise HTTPException(status_code=404, detail="not_found")
    st = os.stat(a.snapshot_path)
    etag = f'W/"{st.st_mtime_ns}-{st.st_size}"'
    last_mod = time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime(st.st_mtime))
    headers = {
        "Cache-Control": "public, max-age=3600, immutable",
        "ETag": etag,
        "Last-Modified": last_mod,
    }
    return FileResponse(a.snapshot_path, media_type="image/jpeg", headers=headers)


@app.get('/debug/personnel')
def debug_personnel(db: Session = Depends(get_db)):
    rows = db.query(Personnel).all()
    return [{'id': p.id, 'name': p.name, 'fcm_token': p.fcm_token, 'active': p.active} for p in rows]

@app.post("/register_fcm_token")
async def register_fcm_token(
    token: str = Form(...),
    user_id: int | None = Form(None),
    id_token: str | None = Form(None),
    db: Session = Depends(get_db)
):
    """Saves or updates a personnel's FCM token.
    Either provide user_id directly, or provide a Firebase ID token (id_token) from phone-auth/web-auth; we'll map to a Personnel row.
    """
    # Enforce auth via Firebase ID token by default (AUTH_STRICT=1)
    if AUTH_STRICT and not id_token:
        raise HTTPException(status_code=401, detail="id_token_required")
    resolved_user_id = user_id
    if not resolved_user_id and id_token:
        payload = verify_firebase_id_token(id_token)
        if payload and payload.get("sub"):
            # map Firebase uid to a deterministic integer namespace (last 9 digits) or create a new row
            uid = payload["sub"]
            # Simple mapping: hash to int range
            resolved_user_id = abs(hash(uid)) % 1000000000 or 1
            person = db.query(Personnel).filter(Personnel.id == resolved_user_id).first()
            if not person:
                person = Personnel(id=resolved_user_id, name=f"FirebaseUser:{uid[:8]}", phone="", active=1)
                db.add(person)
                db.commit()
        else:
            raise HTTPException(status_code=401, detail="Invalid id_token")

    if not resolved_user_id:
        raise HTTPException(status_code=400, detail="user_id_or_id_token_required")

    personnel = db.query(Personnel).filter(Personnel.id == resolved_user_id).first()
    if not personnel:
        # auto-create minimal record
        personnel = Personnel(id=resolved_user_id, name=f"User:{resolved_user_id}", phone="", active=1)
        db.add(personnel)
        db.commit()
    personnel.fcm_token = token
    db.commit()
    print(f"✅ FCM Token for user {resolved_user_id} updated.")
    return {"status": "ok", "message": "Token registered successfully.", "user_id": resolved_user_id}

@app.get('/auth/debug')
def auth_debug():
    return {"project": FIREBASE_PROJECT_ID, "vapid_set": bool(FCM_VAPID_PUBLIC_KEY)}
 
@app.get('/fcm_vapid_public_key')
def fcm_vapid_public_key():
    return {"vapidKey": FCM_VAPID_PUBLIC_KEY}
        
@app.post("/alerts")
async def receive_alert(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    camera_id: str = Form(...),
    event: str = Form(...),
    count: int = Form(...),
    snapshot: UploadFile = File(...)
):
    """Receives an alert, saves it, and dispatches notifications."""
    # Logic to check for the event and apply the crowd threshold
    if event == "overcrowding":
        thr = get_crowd_threshold(camera_id)
        if count < thr:
            print(f"✅ Crowd count below threshold ({count}/{thr}) for camera {camera_id}. Skipping alert.")
            return {"status": "ignored", "reason": "crowd_below_threshold"}

    # Cooldown check to prevent alert spam
    cooldown_key = f"cooldown:{camera_id}:{event}"
    if rds.exists(cooldown_key):
        return {"status": "ignored", "reason": "cooldown"}

    # Save snapshot image
    fname = f"{camera_id}_{event}_{uuid.uuid4().hex}.jpg"
    snapshot_path = os.path.join(UPLOAD_DIR, fname)
    with open(snapshot_path, "wb") as f:
        shutil.copyfileobj(snapshot.file, f)

    # Save alert to the database
    alert_obj = AlertModel(
        camera_id=camera_id,
        event=event,
        count=count,
        snapshot_path=snapshot_path,
    )
    db.add(alert_obj)
    db.commit()
    db.refresh(alert_obj)

    # Set a 60-second cooldown in Redis for this specific alert type
    rds.setex(cooldown_key, 60, "active")

    # Send notifications in the background so the AI client gets a fast response
    background_tasks.add_task(send_fcm_notifications, db, alert_obj)

    # Broadcast over websockets
    try:
        enqueue_alert(_build_client_alert_payload(alert_obj))
    except Exception:
        pass

    return {"status": "ok", "alert_id": alert_obj.id}

# --- Live Ingestion Endpoints (Admin) ---
@app.post('/ingest/start')
def ingest_start(body: Dict[str, Any], background_tasks: BackgroundTasks, Authorization: str | None = Header(default=None)):
    """Start processing a live stream URL (RTSP/HTTP) for a given camera_id."""
    _, _, is_admin = _resolve_user_from_authorization(Authorization)
    if AUTH_STRICT and not is_admin:
        raise HTTPException(status_code=403, detail="admin_required")
    src = (body or {}).get('source_url') or (body or {}).get('url')
    camera_id = (body or {}).get('camera_id')
    fps_target = int((body or {}).get('fps_target') or 4)
    if not src or not camera_id:
        raise HTTPException(status_code=400, detail='source_url_and_camera_id_required')
    job_id = uuid.uuid4().hex
    stop_ev = Event()
    with STREAM_JOBS_LOCK:
        STREAM_JOBS[job_id] = {"camera_id": camera_id, "source_url": src, "status": "starting", "started": time.time(), "stop": stop_ev, "processed": 0, "alerts": 0}
    # ensure camera exists
    try:
        from models import Camera as CameraModel
        sess = SessionLocal()
        cam = sess.query(CameraModel).filter(CameraModel.camera_id == camera_id).first()
        if not cam:
            cam = CameraModel(camera_id=camera_id, location=None, description=f"Ingest: {src}")
            sess.add(cam); sess.commit()
        sess.close()
    except Exception:
        pass
    # Start background worker
    background_tasks.add_task(_process_stream, src, camera_id, int(max(1, fps_target)), job_id)
    return {"status": "ok", "job_id": job_id, "camera_id": camera_id}

@app.post('/ingest/stop')
def ingest_stop(body: Dict[str, Any], Authorization: str | None = Header(default=None)):
    """Stop an active ingestion by job_id or camera_id."""
    _, _, is_admin = _resolve_user_from_authorization(Authorization)
    if AUTH_STRICT and not is_admin:
        raise HTTPException(status_code=403, detail="admin_required")
    job_id = (body or {}).get('job_id')
    camera_id = (body or {}).get('camera_id')
    stopped = []
    with STREAM_JOBS_LOCK:
        items = list(STREAM_JOBS.items())
    for jid, info in items:
        if job_id and jid != job_id:
            continue
        if camera_id and str(info.get('camera_id')) != str(camera_id):
            continue
        ev: Event = info.get('stop')
        if ev:
            ev.set()
            stopped.append(jid)
    return {"status": "ok", "stopped": stopped}

@app.get('/ingest/list')
def ingest_list(Authorization: str | None = Header(default=None)):
    """List active/known ingestion jobs (admin)."""
    _, _, is_admin = _resolve_user_from_authorization(Authorization)
    if AUTH_STRICT and not is_admin:
        raise HTTPException(status_code=403, detail="admin_required")
    with STREAM_JOBS_LOCK:
        return {jid: {k: v for k, v in info.items() if k != 'stop'} for jid, info in STREAM_JOBS.items()}

# --- Threshold Config Endpoints (Admin) ---
@app.get('/config/violation_thresholds')
def get_violation_thresholds(Authorization: str | None = Header(default=None)):
    _, _, is_admin = _resolve_user_from_authorization(Authorization)
    if not is_admin:
        raise HTTPException(status_code=403, detail="admin_required")
    return load_violation_config()

@app.put('/config/violation_thresholds')
def put_violation_thresholds(cfg: Dict[str, Dict[str, float]], Authorization: str | None = Header(default=None)):
    _, _, is_admin = _resolve_user_from_authorization(Authorization)
    if not is_admin:
        raise HTTPException(status_code=403, detail="admin_required")
    # Basic validation
    for k, v in cfg.items():
        if not isinstance(v, dict):
            raise HTTPException(status_code=400, detail=f"bad_spec_for_{k}")
        try:
            v['thr'] = float(v.get('thr', 0.0))
            v['cooldown'] = float(v.get('cooldown', 60.0))
        except Exception:
            raise HTTPException(status_code=400, detail=f"bad_values_for_{k}")
    save_violation_config(cfg)
    return {"status": "ok"}

# --- Zones GeoJSON ---
@app.get('/zones')
def get_zones():
    """Serve zones.geojson from the repo config directory."""
    try:
        cfg_path = os.path.abspath(os.path.join(REPO_ROOT, 'config', 'zones.geojson'))
        if not os.path.exists(cfg_path):
            raise HTTPException(status_code=404, detail='zones_not_found')
        with open(cfg_path, 'r') as f:
            data = json.load(f)
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/feeds')
def get_feeds():
    """Serve optional feed presets from config/feeds.json with shape {"local":[], "cctv":[], "drone":[]}"""
    try:
        cfg_path = os.path.abspath(os.path.join(REPO_ROOT, 'config', 'feeds.json'))
        if not os.path.exists(cfg_path):
            return {"local": [], "cctv": [], "drone": []}
        with open(cfg_path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"local": [], "cctv": [], "drone": []}
        return {
            "local": list(data.get('local', [])),
            "cctv": list(data.get('cctv', [])),
            "drone": list(data.get('drone', [])),
        }
    except Exception as e:
        return {"local": [], "cctv": [], "drone": [], "error": str(e)}
# --- Camera Location Endpoints ---
@app.get('/camera/{camera_id}')
def get_camera(camera_id: str, db: Session = Depends(get_db)):
    try:
        from models import Camera as CameraModel
        cam = db.query(CameraModel).filter(CameraModel.camera_id == camera_id).first()
        if not cam:
            return {"camera_id": camera_id, "location": None, "description": None, "latitude": None, "longitude": None}
        return {"camera_id": cam.camera_id, "location": cam.location, "description": cam.description, "latitude": cam.latitude, "longitude": cam.longitude}
    except Exception:
        # If Camera model missing, still return shape
        return {"camera_id": camera_id, "location": None, "description": None, "latitude": None, "longitude": None}

@app.post('/camera/{camera_id}/location')
def set_camera_location(camera_id: str, body: Dict[str, Any], db: Session = Depends(get_db), Authorization: str | None = Header(default=None)):
    _, _, is_admin = _resolve_user_from_authorization(Authorization)
    if AUTH_STRICT and not is_admin:
        raise HTTPException(status_code=403, detail="admin_required")
    loc = str((body or {}).get('location') or '').strip()
    lat = (body or {}).get('latitude')
    lng = (body or {}).get('longitude')
    try:
        lat = float(lat) if lat is not None and str(lat).strip() != '' else None
        lng = float(lng) if lng is not None and str(lng).strip() != '' else None
    except Exception:
        lat = None; lng = None
    if not loc:
        raise HTTPException(status_code=400, detail="location_required")
    try:
        from models import Camera as CameraModel
        cam = db.query(CameraModel).filter(CameraModel.camera_id == camera_id).first()
        if not cam:
            cam = CameraModel(camera_id=camera_id, location=loc, description="", latitude=lat, longitude=lng)
            db.add(cam)
        else:
            cam.location = loc
            cam.latitude = lat
            cam.longitude = lng
        db.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok"}
