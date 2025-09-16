import os
import json
import time
import requests
from .database import SessionLocal
from .models import Alert, NotificationLog, Personnel
from sqlalchemy.orm import Session
from fastapi import BackgroundTasks
import redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
rds = redis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

# Configure FCM (optional)
FCM_KEY = os.getenv("FCM_KEY", "")
FCM_URL = "https://fcm.googleapis.com/fcm/send"

def save_alert_to_db(db: Session, payload: dict, snapshot_path: str = None, clip_path: str = None):
    a = Alert(
        camera_id=payload.get("camera_id"),
        event=payload.get("event"),
        count=payload.get("count"),
        confidence=str(payload.get("confidence")) if payload.get("confidence") is not None else None,
        snapshot_path=snapshot_path,
        clip_path=clip_path,
        details=json.dumps(payload.get("details", {}))
    )
    db.add(a)
    db.commit()
    db.refresh(a)
    return a

def dispatch_notifications(db: Session, alert_obj: Alert):
    """
    Simple dispatcher:
      - send FCM to all active personnel with a token
      - fallback: print/log SMS action (integration with Twilio can be added)
    """
    # Find recipients
    recips = db.query(Personnel).filter(Personnel.active == 1).all()
    tokens = [p.fcm_token for p in recips if p.fcm_token]
    title = f"⚠ {alert_obj.event.upper()} @ {alert_obj.camera_id}"
    body = f"Count: {alert_obj.count}" if alert_obj.count is not None else alert_obj.event

    if tokens and FCM_KEY:
        payload = {
            "notification": {"title": title, "body": body},
            "data": {
                "alert_id": str(alert_obj.id),
                "camera_id": alert_obj.camera_id,
                "event": alert_obj.event,
                "snapshot": alert_obj.snapshot_path or ""
            },
            "registration_ids": tokens
        }
        headers = {"Authorization": f"key={FCM_KEY}", "Content-Type": "application/json"}
        try:
            resp = requests.post(FCM_URL, headers=headers, json=payload, timeout=5)
            log = NotificationLog(alert_id=alert_obj.id, method="fcm", recipient=",".join(tokens),
                                  status=str(resp.status_code), response=resp.text)
            db.add(log); db.commit()
        except Exception as e:
            log = NotificationLog(alert_id=alert_obj.id, method="fcm", recipient=",".join(tokens),
                                  status="error", response=str(e))
            db.add(log); db.commit()

    # Fallback: create SMS logs (real SMS integration not implemented here)
    for p in recips:
        if p.phone:
            # Here you would call Twilio or other provider
            log = NotificationLog(alert_id=alert_obj.id, method="sms", recipient=p.phone, status="queued", response="")
            db.add(log)
    db.commit()
