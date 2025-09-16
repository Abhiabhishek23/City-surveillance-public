# backend/enhanced_main.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import json
from backend.enhanced_alerts import EnhancedAlertSystem
from backend.database import get_db
from backend.models import Alert, ActionLog, ViolationPriority

app = FastAPI(title="Enhanced City Surveillance Backend")
alert_system = EnhancedAlertSystem()

@app.get("/")
def health_check():
    return {"status": "ok", "message": "🚀 Backend is running"}

@app.post("/alerts/enhanced")
async def receive_enhanced_alert(
    camera_id: str,
    event: str,
    count: int = None,
    confidence: str = None,
    details: str = None,
    snapshot_path: str = None,
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """Enhanced alert endpoint with priority processing"""
    try:
        result = alert_system.process_alert({
            "camera_id": camera_id,
            "event": event,
            "count": count,
            "confidence": confidence,
            "details": json.loads(details) if details else {}
        }, snapshot_path)
        
        if result["status"] == "alert_created":
            return {"status": "success", "alert_id": result["alert_id"], "priority": result["priority"]}
        else:
            return {"status": result["status"], "message": result["message"]}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/alerts/{alert_id}/action")
async def log_alert_action(
    alert_id: int,
    action_type: str,
    taken_by: str,
    notes: str = None,
    db: Session = Depends(get_db)
):
    """Log action taken on an alert"""
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    
    alert_system.log_action(alert_id, action_type, taken_by, notes)
    return {"status": "action_logged", "alert_id": alert_id}

@app.get("/alerts/stats")
async def get_alert_stats(
    hours: int = 24,
    db: Session = Depends(get_db)
):
    """Get alert statistics"""
    since_time = datetime.now() - timedelta(hours=hours)
    
    stats = {
        "total_alerts": db.query(Alert).filter(Alert.created_at >= since_time).count(),
        "by_type": {},
        "by_priority": {},
        "action_stats": {}
    }
    
    # Count by type
    for alert_type in ["overcrowding", "illegal_construction", "other"]:
        stats["by_type"][alert_type] = db.query(Alert).filter(
            Alert.event == alert_type,
            Alert.created_at >= since_time
        ).count()
    
    # Action statistics
    for action_type in ["acknowledged", "resolved", "dismissed", "escalated"]:
        stats["action_stats"][action_type] = db.query(ActionLog).filter(
            ActionLog.action_type == action_type,
            ActionLog.created_at >= since_time
        ).count()
    
    return stats
