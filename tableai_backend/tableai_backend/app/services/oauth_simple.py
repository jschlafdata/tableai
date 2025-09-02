from __future__ import annotations
from typing import Tuple
from datetime import datetime, timedelta, timezone
from urllib.parse import quote
import hmac, time, requests
from hashlib import sha256
from sqlalchemy.orm import Session

from ..core.config import settings
from ..models import OAuthToken
from .token_crypto import TokenCipher

DROPBOX_AUTH_URL  = "https://www.dropbox.com/oauth2/authorize"
DROPBOX_TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"

DEFAULT_SCOPES = ["files.metadata.read", "files.content.read"]  # minimal read

def _pack_state(user_id: int, provider: str = "dropbox") -> str:
    ts = str(int(time.time()))
    payload = f"{user_id}:{provider}:{ts}"
    sig = hmac.new(settings.SECRET_KEY.encode(), payload.encode(), sha256).hexdigest()
    return f"{payload}:{sig}"

def _unpack_state(state: str) -> Tuple[int, str]:
    try:
        uid, provider, ts, sig = state.split(":")
        payload = f"{uid}:{provider}:{ts}"
        expected = hmac.new(settings.SECRET_KEY.encode(), payload.encode(), sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError("state signature mismatch")
        if (time.time() - int(ts)) > settings.OAUTH_STATE_TTL_SECONDS:
            raise ValueError("state expired")
        return int(uid), provider
    except Exception as e:
        raise ValueError(f"invalid state: {e}")

class SimpleOAuthDropbox:
    """Super‑simple OAuth helper bound to SQLAlchemy Session + settings."""

    def __init__(self, db: Session):
        self.db = db
        self.cipher = TokenCipher()
        # Validate required config up front
        if not settings.DROPBOX_CLIENT_ID or not settings.DROPBOX_CLIENT_SECRET or not settings.DROPBOX_REDIRECT_URI:
            raise RuntimeError("Dropbox OAuth not configured (check DROPBOX_* envs).")

    def start(self, user_id: int) -> str:
        state = _pack_state(user_id, "dropbox")
        scope = quote(" ".join(DEFAULT_SCOPES))
        return (
            f"{DROPBOX_AUTH_URL}"
            f"?client_id={settings.DROPBOX_CLIENT_ID}"
            f"&response_type=code"
            f"&redirect_uri={settings.DROPBOX_REDIRECT_URI}"
            f"&state={state}"
            f"&token_access_type=offline"
            f"&scope={scope}"
        )

    def finish(self, code: str, state: str) -> OAuthToken:
        user_id, provider = _unpack_state(state)
        if provider != "dropbox":
            raise ValueError("Unsupported provider")

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": settings.DROPBOX_REDIRECT_URI,
        }
        auth = (settings.DROPBOX_CLIENT_ID, settings.DROPBOX_CLIENT_SECRET)
        resp = requests.post(DROPBOX_TOKEN_URL, data=data, auth=auth, timeout=20)
        if resp.status_code != 200:
            raise RuntimeError(f"Token exchange failed: {resp.status_code} {resp.text}")

        tj = resp.json()
        access  = tj.get("access_token")
        refresh = tj.get("refresh_token")
        expires = tj.get("expires_in")
        scope   = tj.get("scope")

        if not access:
            raise RuntimeError("Missing access_token in response")

        expires_at = None
        if expires:
            expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(expires))

        # Upsert
        row = (
            self.db.query(OAuthToken)
            .filter(OAuthToken.user_id == user_id, OAuthToken.provider == "dropbox")
            .first()
        )
        enc_access  = self.cipher.encrypt(access)
        enc_refresh = self.cipher.encrypt(refresh) if refresh else None

        if row:
            row.access_token_enc  = enc_access
            row.refresh_token_enc = enc_refresh
            row.scope             = scope
            row.expires_at        = expires_at
            row.updated_at        = datetime.now(timezone.utc)
        else:
            row = OAuthToken(
                user_id=user_id,
                provider="dropbox",
                access_token_enc=enc_access,
                refresh_token_enc=enc_refresh,
                scope=scope,
                expires_at=expires_at,
            )
            self.db.add(row)

        self.db.commit()
        self.db.refresh(row)
        return row

    def status(self, user_id: int) -> dict:
        row = (
            self.db.query(OAuthToken)
            .filter(OAuthToken.user_id == user_id, OAuthToken.provider == "dropbox")
            .first()
        )
        if not row:
            return {"connected": False, "provider": "dropbox"}
        return {
            "connected": True,
            "provider": "dropbox",
            "has_refresh_token": bool(row.refresh_token_enc),
            "expires_at": row.expires_at.isoformat() if row.expires_at else None,
            "is_expired": row.is_expired() if row.expires_at else False,
            "scope": row.scope,
            "updated_at": row.updated_at.isoformat(),
        }

    def disconnect(self, user_id: int) -> bool:
        row = (
            self.db.query(OAuthToken)
            .filter(OAuthToken.user_id == user_id, OAuthToken.provider == "dropbox")
            .first()
        )
        if row:
            self.db.delete(row)
            self.db.commit()
        # (Optional) Call Dropbox revoke endpoint here.
        return True
