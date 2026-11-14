from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "SIFT Document Intelligence API"
    version: str = "0.1.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    environment: str = "development"
    documents_indexed: int = 0
    storage: Dict[str, Any] = Field(default_factory=dict)
