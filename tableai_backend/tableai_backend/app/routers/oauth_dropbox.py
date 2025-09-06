from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session

from ..database import get_db
from ..auth import get_current_user
from ..services.oauth_simple import SimpleOAuthDropbox

router = APIRouter(prefix="/oauth/dropbox", tags=["oauth"])

class OAuthStartRequest(BaseModel):
    redirect_uri: HttpUrl
    
class OAuthCallbackRequest(BaseModel):
    code: str
    state: str
    redirect_uri: HttpUrl

@router.post("/start")
def start(
    request: OAuthStartRequest,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    """
    Start OAuth flow with dynamic redirect URI.
    The frontend passes its current URL as the redirect_uri.
    """
    svc = SimpleOAuthDropbox(db)
    # Pass the redirect_uri to the service
    url = svc.start(user_id=user.id, redirect_uri=str(request.redirect_uri))
    return {"authorize_url": url}

@router.post("/callback")
def callback(
    request: OAuthCallbackRequest,
    db: Session = Depends(get_db),
):
    """
    Complete OAuth flow with the same redirect URI that was used to start.
    No auth required here since the state contains the user_id.
    """
    try:
        svc = SimpleOAuthDropbox(db)
        row = svc.finish(
            code=request.code, 
            state=request.state,
            redirect_uri=str(request.redirect_uri)
        )
        return {
            "connected": True,
            "provider": row.provider,
            "has_refresh_token": bool(row.refresh_token_enc),
            "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status")
def status(
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    """Get current OAuth connection status for the user."""
    svc = SimpleOAuthDropbox(db)
    return svc.status(user_id=user.id)

@router.post("/disconnect")
def disconnect(
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    """Disconnect OAuth integration for the user."""
    svc = SimpleOAuthDropbox(db)
    ok = svc.disconnect(user_id=user.id)
    return {"disconnected": ok}