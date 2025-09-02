from __future__ import annotations
from typing import List, Optional, Dict, Any
import dropbox
from sqlalchemy.orm import Session

from ..core.config import settings
from ..models import OAuthToken
from .token_crypto import TokenCipher
from .boto_s3 import BotoS3

class DropboxToS3Sync:
    """
    Sequential, minimal mirror from a Dropbox path to s3://bucket/prefix.
    """
    def __init__(self, db: Session, user_id: int):
        self.db = db
        self.user_id = user_id

        if not settings.AWS_S3_BUCKET:
            raise RuntimeError("AWS_S3_BUCKET not configured")

        self.bucket = settings.AWS_S3_BUCKET

        # ---- NEW: use BotoS3 (profile/region aware) and ensure bucket exists
        self.boto = BotoS3(settings.AWS_PROFILE, settings.AWS_REGION, action="sync")
        if not self.boto.bucket_exists(self.bucket):
            if settings.AWS_S3_CREATE_IF_MISSING:
                self.boto.create_s3_bucket(self.bucket)
                if settings.AWS_S3_UPDATE_POLICIES_ON_CREATE:
                    # WARNING: may loosen public access protections; keep False unless you mean it
                    self.boto.update_bucket_policies(self.bucket)
            else:
                raise RuntimeError(
                    f"S3 bucket '{self.bucket}' does not exist and auto-create is disabled "
                    f"(AWS_S3_CREATE_IF_MISSING=False)."
                )

        # Use the same client/session for uploads
        self.s3 = self.boto.s3_client

        # ---- Dropbox client wiring (refresh token preferred)
        token_row = (
            db.query(OAuthToken)
            .filter(OAuthToken.user_id == user_id, OAuthToken.provider == "dropbox")
            .first()
        )
        if not token_row:
            raise RuntimeError("Dropbox is not connected for this user")

        cipher = TokenCipher()
        access  = cipher.decrypt(token_row.access_token_enc)
        refresh = cipher.decrypt(token_row.refresh_token_enc) if token_row.refresh_token_enc else None

        if refresh:
            self.dbx = dropbox.Dropbox(
                oauth2_refresh_token=refresh,
                app_key=settings.DROPBOX_CLIENT_ID,
                app_secret=settings.DROPBOX_CLIENT_SECRET,
            )
        else:
            self.dbx = dropbox.Dropbox(oauth2_access_token=access)

    def sync(
        self,
        dropbox_path: str = "",
        s3_prefix: str = "synced/dropbox",
        include_exts: Optional[List[str]] = None,
        max_files: Optional[int] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        # ... unchanged sync loop except using self.s3 ...
        root = "" if dropbox_path in ("", "/", "root") else "/" + dropbox_path.lstrip("/")

        uploaded = 0
        skipped = 0
        bytes_up = 0
        errors: List[str] = []

        res = self.dbx.files_list_folder(root, recursive=True)
        entries = list(res.entries)
        while res.has_more:
            res = self.dbx.files_list_folder_continue(res.cursor)
            entries.extend(res.entries)

        ext_filter = {e.lower().lstrip(".") for e in include_exts} if include_exts else None

        for e in entries:
            if isinstance(e, dropbox.files.FolderMetadata):
                continue
            if not isinstance(e, dropbox.files.FileMetadata):
                continue

            if ext_filter:
                name = (e.name or "").lower()
                ext = name.rsplit(".", 1)[-1] if "." in name else ""
                if ext not in ext_filter:
                    skipped += 1
                    continue

            s3_key = f"{s3_prefix.rstrip('/')}/{e.path_lower.lstrip('/')}"

            try:
                if dry_run:
                    uploaded += 1
                else:
                    _, resp = self.dbx.files_download(e.path_lower)
                    body = resp.content
                    self.s3.put_object(
                        Bucket=self.bucket,
                        Key=s3_key,
                        Body=body,
                        Metadata={"provider": "dropbox", "dropbox_id": e.id},
                    )
                    uploaded += 1
                    bytes_up += len(body)
            except Exception as ex:
                errors.append(f"{e.path_lower}: {ex}")

            if max_files and uploaded >= max_files:
                break

        return {
            "ok": len(errors) == 0,
            "uploaded": uploaded,
            "skipped": skipped,
            "bytes_uploaded": bytes_up,
            "errors": errors,
            "bucket": self.bucket,
            "prefix": s3_prefix,
            "source_path": root or "/",
        }
