# backend/enhanced_alerts.py
import os
import redis
import json
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from .models import Alert, ViolationPriority, ActionLog, NotificationLog, Personnel
from .database import SessionLocal

class EnhancedAlertSystem:
    def __init__(self):
        self.rds = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), 
                              port=6379, decode_responses=True)
    
    def check_violation_threshold(self, alert_data: dict) -> bool:
        """Check if violation breaches configured thresholds"""
        violation_type = alert_data.get("event")
        
        # Get threshold configuration from database
        db = SessionLocal()
        try:
            if violation_type == "overcrowding":
                threshold = int(os.getenv("OVERCROWD_THRESHOLD", "5"))
                return alert_data.get("count", 0) > threshold
            
            elif violation_type == "illegal_construction":
                confidence = float(alert_data.get("confidence", 0))
                confidence_threshold = float(os.getenv("CONSTRUCTION_CONFIDENCE_THRESHOLD", "0.1"))
                return confidence > confidence_threshold
            
            # Add other violation types here
            
        finally:
            db.close()
        
        return False
    
    def get_violation_priority(self, violation_type: str) -> dict:
        """Get priority settings for violation type"""
        db = SessionLocal()
        try:
            priority = db.query(ViolationPriority).filter(
                ViolationPriority.violation_type == violation_type
            ).first()
            
            if priority:
                return {
                    "level": priority.priority_level,
                    "cooldown": priority.cooldown_minutes,
                    "channels": json.loads(priority.notification_channels)
                }
            
            # Default settings
            return {
                "level": 2,
                "cooldown": 60,
                "channels": ["fcm", "sms"]
            }
            
        finally:
            db.close()
    
    def check_cooldown(self, camera_id: str, violation_type: str, priority_info: dict) -> bool:
        """Check if alert is in cooldown period"""
        cooldown_key = f"cooldown:{camera_id}:{violation_type}"
        if self.rds.exists(cooldown_key):
            return True
        
        # Set cooldown
        self.rds.setex(cooldown_key, priority_info["cooldown"] * 60, "1")
        return False
    
    def log_action(self, alert_id: int, action_type: str, taken_by: str, notes: str = None):
        """Log action taken on alert"""
        db = SessionLocal()
        try:
            action_log = ActionLog(
                alert_id=alert_id,
                action_type=action_type,
                taken_by=taken_by,
                notes=notes
            )
            db.add(action_log)
            db.commit()
        finally:
            db.close()
    
    def process_alert(self, alert_data: dict, snapshot_path: str = None) -> dict:
        """Enhanced alert processing with priority system"""
        # Check if violation breaches threshold
        if not self.check_violation_threshold(alert_data):
            return {"status": "below_threshold", "message": "Violation below configured threshold"}
        
        violation_type = alert_data.get("event")
        camera_id = alert_data.get("camera_id")
        
        # Get priority information
        priority_info = self.get_violation_priority(violation_type)
        
        # Check cooldown
        if self.check_cooldown(camera_id, violation_type, priority_info):
            return {"status": "cooldown", "message": "Alert in cooldown period"}
        
        # Save to database
        db = SessionLocal()
        try:
            alert_obj = save_alert_to_db(db, alert_data, snapshot_path)
            
            # Dispatch notifications based on priority
            if priority_info["level"] >= 3:  # Medium and high priority
                self.dispatch_notifications(db, alert_obj, priority_info)
            
            # Log the alert creation
            self.log_action(alert_obj.id, "detected", "system", 
                          f"Violation detected with priority {priority_info['level']}")
            
            return {"status": "alert_created", "alert_id": alert_obj.id, "priority": priority_info["level"]}
            
        finally:
            db.close()
    
    def dispatch_notifications(self, db: Session, alert_obj: Alert, priority_info: dict):
        """Enhanced notification system with priority channels"""
        channels = priority_info.get("channels", ["fcm"])
        
        # Get recipients based on priority
        if priority_info["level"] >= 4:  # High priority - notify all
            recips = db.query(Personnel).filter(Personnel.active == 1).all()
        else:  # Medium priority - notify specific roles
            recips = db.query(Personnel).filter(
                Personnel.active == 1, 
                Personnel.role.in_(["supervisor", "admin"])
            ).all()
        
        # Send via configured channels
        if "fcm" in channels:
            self.send_fcm_notifications(recips, alert_obj)
        if "sms" in channels:
            self.send_sms_notifications(recips, alert_obj)
        if "email" in channels:
            self.send_email_notifications(recips, alert_obj)
    