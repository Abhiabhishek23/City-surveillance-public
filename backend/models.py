from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base

# Note: Removed redundant 'from sqlalchemy import Boolean, Float' as it wasn't used.

class ViolationPriority(Base):
    __tablename__ = "violation_priorities"
    id = Column(Integer, primary_key=True, index=True)
    violation_type = Column(String, unique=True, index=True)
    priority_level = Column(Integer, default=1)  # 1-5, 5 being highest
    cooldown_minutes = Column(Integer, default=60)
    notification_channels = Column(String)  # JSON: ["fcm", "sms", "email"]

class ActionLog(Base):
    __tablename__ = "action_logs"
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, ForeignKey('alerts.id'))
    action_type = Column(String)  # "acknowledged", "resolved", "dismissed", "escalated"
    taken_by = Column(String)  # User ID or system
    notes = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    alert = relationship("Alert")

class SystemConfig(Base):
    __tablename__ = "system_config"
    id = Column(Integer, primary_key=True, index=True)
    config_key = Column(String, unique=True)
    config_value = Column(String)
    description = Column(Text)

class Camera(Base):
    __tablename__ = "cameras"
    id = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String, unique=True, index=True)
    location = Column(String, nullable=True)
    description = Column(String, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    alerts = relationship("Alert", back_populates="camera")

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    
    # --- THIS IS THE CORRECTED LINE ---
    camera_id = Column(String, ForeignKey("cameras.camera_id"), index=True)
    
    event = Column(String, index=True)         # overcrowd, construction, other
    count = Column(Integer, nullable=True)
    confidence = Column(String, nullable=True) # optional
    snapshot_path = Column(String, nullable=True)  # local path or S3 URL
    clip_path = Column(String, nullable=True)      # optional video clip path
    details = Column(Text, nullable=True)      # JSON string for additional metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Corrected the relationship to be simpler now that the ForeignKey is defined
    camera = relationship("Camera", back_populates="alerts")

class Personnel(Base):
    __tablename__ = "personnel"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    phone = Column(String)
    fcm_token = Column(String, nullable=True)
    active = Column(Integer, default=1)

class NotificationLog(Base):
    __tablename__ = "notification_logs"
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(Integer, nullable=True)
    method = Column(String)   # fcm, sms, email
    recipient = Column(String)
    status = Column(String)
    sent_at = Column(DateTime(timezone=True), server_default=func.now())
    response = Column(Text, nullable=True)