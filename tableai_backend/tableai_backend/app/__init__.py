from __future__ import annotations
from sqlmodel import SQLModel, Field
from enum import Enum
from typing import Optional
from datetime import datetime, timezone

class Provider(str, Enum):
    DROPBOX = "dropbox"

class OAuthToken(SQLModel, table=True):
    __tablename__ = "oauth_token"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(index=True)
    provider: Provider = Field(index=True)
    access_token_enc: str
    refresh_token_enc: Optional[str] = None
    scope: Optional[str] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def is_expired(self) -> bool:
        return bool(self.expires_at and datetime.now(timezone.utc) >= self.expires_at)