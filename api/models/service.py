from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

class ServiceConfig(BaseSettings):
    """Service configuration with environment variable support"""
    cache_ttl_hours: int = Field(default=1, env="CACHE_TTL_HOURS")
    max_tasks: int = Field(default=100, env="MAX_TASKS")
    cleanup_interval: int = Field(default=3600, env="CLEANUP_INTERVAL")
    rate_limit_rpm: int = Field(default=60, env="RATE_LIMIT_RPM")
    port: int = Field(default=8000, env="PORT")
    force_restart: bool = Field(default=False, env="FORCE_RESTART")
    ollama_api_user: str = "api-user"
    dropbox_app_key: str
    dropbox_app_secret: str
    ollama_api_key: str
    db_connection_url: str
    
    class Config:
        env_file = ".env"


class DropboxAuthState(BaseModel):
    app_key: str = Field(..., description="Dropbox app key")
    app_secret: str = Field(..., description="Dropbox app secret")
    token_file: str = Field(default="~/.ssh/.dropbox_token.json", description="Path to token file")
    access_token: Optional[str] = Field(default=None, description="OAuth2 access token")
    refresh_token: Optional[str] = Field(default=None, description="OAuth2 refresh token")
    expires_at: Optional[str] = Field(default=None, description="Expiry time of access token")

    @classmethod
    def from_service_config(cls, config: "ServiceConfig", **kwargs):
        return cls(
            app_key=config.dropbox_app_key,
            app_secret=config.dropbox_app_secret,
            **kwargs
        )


class Settings:
    """File system settings and validation"""

    def __init__(self):
        self.logger = logging.getLogger("api_service.settings")
        self.SERVICE_DIR = Path(__file__).parent
        self.DATA_DIR = self.SERVICE_DIR.parent / 'data'
        self._validate_directories()

    @property
    def storage_path(self) -> Path:
        return self.DATA_DIR / 'storage'

    @property
    def cache_path(self) -> Path:
        return self.DATA_DIR / 'cache'

    def get_file_paths(self, pattern: str) -> List[Path]:
        import glob
        return [Path(p) for p in glob.glob(str(self.storage_path / pattern))]

    def _validate_directories(self):
        self.logger.info("Validating directories...")
        paths = {
            "DATA_DIR": self.DATA_DIR,
            "storage_path": self.storage_path,
            "cache_path": self.cache_path
        }
        
        for name, path in paths.items():
            self.logger.debug(f"Checking {name}: {path}")
            if not path.exists():
                self.logger.info(f"Creating {name}: {path}")
                path.mkdir(parents=True, exist_ok=True)