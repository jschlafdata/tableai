from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..auth import get_current_user
from ..services.oauth_simple import SimpleOAuthDropbox

router = APIRouter(prefix="/oauth/dropbox", tags=["oauth"])

@router.get("/start")
def start(
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    svc = SimpleOAuthDropbox(db)
    url = svc.start(user_id=user.id)
    return {"authorize_url": url}

@router.get("/callback")
def callback(
    code: str = Query(...),
    state: str = Query(...),
    db: Session = Depends(get_db),
):
    try:
        svc = SimpleOAuthDropbox(db)
        row = svc.finish(code=code, state=state)
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
    svc = SimpleOAuthDropbox(db)
    return svc.status(user_id=user.id)

@router.post("/disconnect")
def disconnect(
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
):
    svc = SimpleOAuthDropbox(db)
    ok = svc.disconnect(user_id=user.id)
    return {"disconnected": ok}
