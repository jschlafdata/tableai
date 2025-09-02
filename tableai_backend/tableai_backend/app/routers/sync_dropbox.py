from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..auth import get_current_user
from ..services.dropbox_s3_sync import DropboxToS3Sync

router = APIRouter(prefix="/sync", tags=["sync"])

class DropboxSyncRequest(BaseModel):
    path: str = ""                         # "", "/", or "folder/sub"
    s3_prefix: str = "synced/dropbox"
    include_exts: Optional[List[str]] = None
    max_files: Optional[int] = None
    dry_run: bool = False

@router.post("/dropbox")
def sync_dropbox(
    body: DropboxSyncRequest,
    db: Session = Depends(get_db),
    user = Depends(get_current_user),
) -> Dict[str, Any]:
    try:
        syncer = DropboxToS3Sync(db, user.id)
        return syncer.sync(
            dropbox_path=body.path,
            s3_prefix=body.s3_prefix,
            include_exts=body.include_exts,
            max_files=body.max_files,
            dry_run=body.dry_run,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
