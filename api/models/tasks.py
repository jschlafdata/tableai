from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import logging 
from fastapi import FastAPI, HTTPException
import time

class TaskStatus(BaseModel):
    """Task status tracking"""
    id: str
    status: str
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[float] = None

class RateLimiter:
    """Rate limiting implementation"""

    def __init__(self, requests_per_minute: int = 60):
        self.requests = {}
        self.rate_limit = requests_per_minute
        self.window = 60
        self.logger = logging.getLogger("api_service.rate_limiter")

    async def check(self, client_id: str):
        now = time.time()
        if client_id in self.requests:
            requests = [req for req in self.requests[client_id] if now - req < self.window]
            if len(requests) >= self.rate_limit:
                self.logger.warning(f"Rate limit exceeded for client: {client_id}")
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            self.requests[client_id] = requests + [now]
        else:
            self.requests[client_id] = [now]