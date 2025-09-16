# setup_system.sh
#!/bin/bash

echo "Setting up Enhanced City Surveillance System..."

# Create directories
mkdir -p data/train/images data/train/annotations
mkdir -p data/val/images data/val/annotations
mkdir -p models uploads snapshots

# Install required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python albumentations geopy
pip install fastapi uvicorn sqlalchemy psycopg2-binary redis

# Download pretrained models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
wget https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth -O models/maskrcnn_pretrained.pth

# Initialize database
python -c "
from backend.database import engine, Base
from backend.models import Alert, Camera, Personnel, ViolationPriority, ActionLog, SystemConfig
Base.metadata.create_all(bind=engine)
print('Database tables created successfully!')
"

# Setup default priorities
python -c "
from backend.database import SessionLocal
from backend.models import ViolationPriority
import json

db = SessionLocal()

# Default priorities
priorities = [
    {'violation_type': 'overcrowding', 'priority_level': 3, 'cooldown_minutes': 30, 'notification_channels': '[\"fcm\", \"sms\"]'},
    {'violation_type': 'illegal_construction', 'priority_level': 4, 'cooldown_minutes': 1440, 'notification_channels': '[\"fcm\", \"sms\", \"email\"]'},
    {'violation_type': 'fire', 'priority_level': 5, 'cooldown_minutes': 5, 'notification_channels': '[\"fcm\", \"sms\", \"email\", \"siren\"]'}
]

for priority in priorities:
    if not db.query(ViolationPriority).filter_by(violation_type=priority['violation_type']).first():
        db.add(ViolationPriority(**priority))

db.commit()
print('Default priorities set up!')
db.close()
"

echo "System setup completed! Next steps:"
echo "1. Add your training images to data/train/images/"
echo "2. Annotate them using LabelMe or CVAT"
echo "3. Run: python train/complete_training.py"
echo "4. Start backend: uvicorn backend.enhanced_main:app --reload"
echo "5. Start edge detector: python edge/enhanced_detector_with_gps.py"