# backend/routers/integrations_dropbox.py
from __future__ import annotations

import os
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from ..services.task_manager import task_manager
from ..services.dropbox_s3_sync import sync_dropbox_pdfs_to_s3

router = APIRouter(prefix="/integrations/dropbox", tags=["integrations:dropbox"])

class SyncRequest(BaseModel):
    root_path: str = "/"               # Dropbox path to scan (e.g. "/Docs")
    bucket: str | None = None
    prefix: str | None = None
    force: bool = False

@router.post("/sync")
def sync_dropbox(req: SyncRequest, bg: BackgroundTasks):
    bucket = req.bucket or os.getenv("S3_BUCKET")
    prefix = req.prefix or os.getenv("S3_PREFIX", "").strip()
    if not bucket:
        raise HTTPException(400, "S3_BUCKET not configured and not provided")

    task_id = task_manager.create(kind="dropbox_sync", payload=req.model_dump())

    def _run():
        try:
            task_manager.update(task_id, status="running")
            res = sync_dropbox_pdfs_to_s3(req.root_path, bucket=bucket, prefix=prefix, force=req.force)
            task_manager.update(task_id, status="succeeded", result={
                "scanned": res.scanned,
                "uploaded": res.uploaded,
                "skipped": res.skipped,
                "errors": res.errors,
                "uploaded_keys": res.uploaded_keys[:100],  # cap
            })
        except Exception as e:
            task_manager.update(task_id, status="failed", error=str(e))

    bg.add_task(_run)
    return {"task_id": task_id}

@router.get("/sync/status")
def sync_status(task_id: str = Query(...)):
    st = task_manager.get(task_id)
    if not st:
        raise HTTPException(404, "task not found")
    return st
