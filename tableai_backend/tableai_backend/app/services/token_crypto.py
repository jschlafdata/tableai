from __future__ import annotations
import base64
from typing import Optional
from cryptography.fernet import Fernet
from ..core.config import settings

class TokenCipher:
    """
    Encrypt/decrypt tokens; prefers Fernet if FERNET_SECRET_KEY is set.
    Otherwise falls back to base64 tagging.
    """
    def __init__(self):
        self._fernet: Optional[Fernet] = None
        if settings.FERNET_SECRET_KEY:
            key = settings.FERNET_SECRET_KEY
            if isinstance(key, str):
                key = key.encode()
            self._fernet = Fernet(key)

    def encrypt(self, s: str | None) -> str | None:
        if not s:
            return s
        if self._fernet:
            return self._fernet.encrypt(s.encode()).decode()
        return "b64:" + base64.b64encode(s.encode()).decode()

    def decrypt(self, s: str | None) -> str | None:
        if not s:
            return s
        if self._fernet:
            return self._fernet.decrypt(s.encode()).decode()
        if s.startswith("b64:"):
            return base64.b64decode(s[4:].encode()).decode()
        return s
