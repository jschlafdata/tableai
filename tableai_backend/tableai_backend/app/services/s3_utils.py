# backend/services/s3_utils.py
from __future__ import annotations
import os
from typing import Iterable, List, Optional
from pathlib import Path
import boto3

def get_s3():
    return boto3.client("s3", region_name=os.getenv("AWS_REGION"))

def list_s3_pdfs(bucket: str, prefix: str) -> List[str]:
    s3 = get_s3()
    paginator = s3.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith(".pdf"):
                keys.append(key)
    return keys

def head_object(bucket: str, key: str):
    s3 = get_s3()
    return s3.head_object(Bucket=bucket, Key=key)

def object_exists(bucket: str, key: str) -> bool:
    s3 = get_s3()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False

def upload_bytes(bucket: str, key: str, content: bytes, content_type: str = "application/pdf", metadata: Optional[dict] = None):
    s3 = get_s3()
    extra = {"ContentType": content_type}
    if metadata:
        extra["Metadata"] = metadata
    s3.put_object(Bucket=bucket, Key=key, Body=content, **extra)

def download_to_path(bucket: str, key: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    s3 = get_s3()
    s3.download_file(bucket, key, str(dest))
    return dest

def upload_file(bucket: str, key: str, local_path: Path, content_type: str = "application/pdf", metadata: Optional[dict] = None):
    s3 = get_s3()
    extra = {"ContentType": content_type}
    if metadata:
        extra["Metadata"] = metadata
    with open(local_path, "rb") as f:
        s3.put_object(Bucket=bucket, Key=key, Body=f, **extra)
