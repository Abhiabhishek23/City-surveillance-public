from pydantic import BaseModel
from typing import Optional, Any

class AlertIn(BaseModel):
    camera_id: str
    event: str
    count: Optional[int] = None
    confidence: Optional[float] = None
    details: Optional[Any] = None
