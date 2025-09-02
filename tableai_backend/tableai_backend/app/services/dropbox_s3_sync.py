# backend/services/dropbox_sync.py
from __future__ import annotations
import os
from typing import List, Optional
from dataclasses import dataclass
from dropbox import Dropbox
from dropbox.files import ListFolderResult, FileMetadata

from .s3_utils import object_exists, head_object, upload_bytes

@dataclass
class SyncResult:
    scanned: int
    uploaded: int
    skipped: int
    errors: int
    uploaded_keys: List[str]

def _dbx() -> Dropbox:
    token = os.getenv("DROPBOX_TOKEN")
    if not token:
        raise RuntimeError("DROPBOX_TOKEN not set")
    return Dropbox(token)

def _iter_dropbox_files(root_path: str):
    """Yield FileMetadata under root_path (recursively)."""
    dbx = _dbx()
    res: ListFolderResult = dbx.files_list_folder(root_path, recursive=True)
    for ent in res.entries:
        if isinstance(ent, FileMetadata):
            yield ent
    while res.has_more:
        res = dbx.files_list_folder_continue(res.cursor)
        for ent in res.entries:
            if isinstance(ent, FileMetadata):
                yield ent

def sync_dropbox_pdfs_to_s3(
    root_path: str,
    bucket: str,
    prefix: str,
    *,
    force: bool = False,
) -> SyncResult:
    uploaded = 0
    skipped = 0
    errors = 0
    scanned = 0
    uploaded_keys: List[str] = []

    dbx = _dbx()
    for fm in _iter_dropbox_files(root_path):
        scanned += 1
        if not fm.name.lower().endswith(".pdf"):
            continue

        key = f"{prefix.rstrip('/')}/{fm.path_display.lstrip('/').replace(root_path.lstrip('/'), '').lstrip('/')}"
        key = key.replace("//", "/")
        try:
            if not force and object_exists(bucket, key):
                # Compare modified times; upload if Dropbox is newer
                s3_head = head_object(bucket, key)
                s3_last_mod = s3_head["LastModified"]  # datetime
                if fm.server_modified <= s3_last_mod:
                    skipped += 1
                    continue

            md, resp = dbx.files_download(fm.path_lower)
            content = resp.content
            upload_bytes(
                bucket,
                key,
                content,
                content_type="application/pdf",
                metadata={"dropbox_rev": fm.rev, "dropbox_client_modified": fm.client_modified.isoformat()},
            )
            uploaded += 1
            uploaded_keys.append(key)
        except Exception:
            errors += 1

    return SyncResult(scanned=scanned, uploaded=uploaded, skipped=skipped, errors=errors, uploaded_keys=uploaded_keys)