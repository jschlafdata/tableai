from sqlmodel import SQLModel, Field, create_engine, Session, select
from datetime import datetime 
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List
from dataclasses import dataclass

class SyncRecord(SQLModel, table=True):
    path_lower: str               = Field(primary_key=True)
    local_path: str
    size: int
    server_modified: datetime
    synced_at: datetime           = Field(default_factory=datetime.utcnow)
    metadata_json: str            = Field(default="{}")  # renamed from `metadata`


class SyncRequest(BaseModel):
    paths: List[str]

