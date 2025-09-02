from pydantic_settings import BaseSettings
from pydantic import AnyUrl, Field
from typing import List, Any, Annotated

from pydantic import (
    AnyUrl,
    BeforeValidator,
    EmailStr,
    HttpUrl,
    PostgresDsn,
    computed_field,
    model_validator,
)

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def parse_cors(v: Any) -> list[str] | str:
    if isinstance(v, str) and not v.startswith("["):
        return [i.strip() for i in v.split(",")]
    elif isinstance(v, list | str):
        return v
    raise ValueError(v)


class Settings(BaseSettings):
    # --- existing fields ---
    DATABASE_URL: str = "postgresql+psycopg://app:app@localhost:5432/app"
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    ALGORITHM: str = "HS256"
    CORS_ORIGINS: Annotated[list[AnyUrl] | str, BeforeValidator(parse_cors)] = Field(default_factory=lambda: ["http://localhost:5173"])
    FRONTEND_HOST: str = "http://localhost:5173"
    FIRST_SUPERUSER_EMAIL: str | None = None
    FIRST_SUPERUSER_PASSWORD: str | None = None

    # --- NEW: OAuth + S3 ---
    DROPBOX_CLIENT_ID: str | None = None
    DROPBOX_CLIENT_SECRET: str | None = None
    DROPBOX_REDIRECT_URI: str | None = None  # e.g. https://api.yourhost.com/oauth/dropbox/callback
    OAUTH_STATE_TTL_SECONDS: int = 600       # 10 minutes
    FERNET_SECRET_KEY: str | None = None     # optional; if set we encrypt tokens with Fernet

    AWS_S3_BUCKET: str | None = None         # required for sync
    AWS_REGION: str = "us-east-1"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def all_cors_origins(self) -> list[str]:
        return [str(origin).rstrip("/") for origin in self.CORS_ORIGINS] + [self.FRONTEND_HOST]

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
logging.info(f"Current settings: {settings.model_dump()}")
