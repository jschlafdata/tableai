# from __future__ import annotations

# import os
# import sys
# import uuid
# from pathlib import Path
# from typing import List

# import aiofiles
# from fastapi import APIRouter, UploadFile, HTTPException
# from fastapi.responses import FileResponse, JSONResponse
# from tqdm.auto import tqdm

# import os
# import sys
# import uuid
# from pathlib import Path
# from typing import List
# from fastapi import APIRouter, UploadFile, HTTPException, BackgroundTasks, Query
# from fastapi.responses import FileResponse, JSONResponse
# from tqdm.auto import tqdm

# from ..services.task_manager import task_manager
# from ..services.pdf_cluster import run_classification_local_dir, run_classification_from_s3, DATA_DIR
# from ..services.pdf_cluster import run_classification, ensure_data_dir, get_data_dir

# router = APIRouter(prefix="/classifier", tags=["classifier"])

# @router.post("/sync", summary="Stage PDFs in server stash")
# async def sync(files: List[UploadFile]):
#     if not files:
#         raise HTTPException(status_code=400, detail="No files uploaded")
#     DATA_DIR.mkdir(parents=True, exist_ok=True)
#     saved: list[str] = []
#     bar = tqdm(files, desc="saving", unit="pdf", file=sys.stdout, disable=not sys.stdout.isatty())
#     for up in bar:
#         fname = f"{uuid.uuid4()}_{up.filename}"
#         dest = DATA_DIR / fname
#         async with aiofiles.open(dest, "wb") as out:
#             while chunk := await up.read(64 << 10):
#                 await out.write(chunk)
#         await up.close()
#         saved.append(dest.name)
#     return JSONResponse({"saved": len(saved), "files": saved})

# @router.post("/classify", summary="Cluster all staged PDFs")
# def classify(min_cluster: int = 4):
#     pdfs = sorted(DATA_DIR.glob("*.pdf"))
#     if not pdfs:
#         raise HTTPException(400, "Nothing to classify – call /classifier/sync first")
#     yaml_path = run_classification_local_dir(DATA_DIR, min_cluster=min_cluster)
#     return FileResponse(yaml_path, media_type="text/yaml", filename="clusters.yaml")

# # -------- S3-based classifier (direct) --------
# @router.post("/classify-s3", summary="Download PDFs from S3 prefix and classify, returning clusters.yaml")
# def classify_s3(
#     bucket: str | None = None,
#     prefix: str | None = None,
#     min_cluster: int = 4,
#     upload_yaml_to_s3: bool = True,
# ):
#     bucket = bucket or os.getenv("S3_BUCKET")
#     prefix = prefix or os.getenv("S3_PREFIX", "")
#     if not bucket:
#         raise HTTPException(400, "S3_BUCKET not configured and not provided")
#     yaml_path, _ = run_classification_from_s3(
#         bucket=bucket, prefix=prefix, min_cluster=min_cluster, upload_yaml_to_s3=upload_yaml_to_s3
#     )
#     return FileResponse(yaml_path, media_type="text/yaml", filename="clusters.yaml")

# # -------- S3-based classifier (background task) --------
# @router.post("/classify-s3/async", summary="Async classification with task polling")
# def classify_s3_async(
#     bucket: str | None = None,
#     prefix: str | None = None,
#     min_cluster: int = 4,
#     upload_yaml_to_s3: bool = True,
#     bg: BackgroundTasks = None,
# ):
#     bucket = bucket or os.getenv("S3_BUCKET")
#     prefix = prefix or os.getenv("S3_PREFIX", "")
#     if not bucket:
#         raise HTTPException(400, "S3_BUCKET not configured and not provided")

#     task_id = task_manager.create(kind="classify_s3", payload={
#         "bucket": bucket, "prefix": prefix, "min_cluster": min_cluster, "upload_yaml_to_s3": upload_yaml_to_s3
#     })

#     def _run():
#         from ..services.pdf_cluster import run_classification_from_s3
#         try:
#             task_manager.update(task_id, status="running")
#             yaml_path, keys = run_classification_from_s3(
#                 bucket=bucket, prefix=prefix, min_cluster=min_cluster, upload_yaml_to_s3=upload_yaml_to_s3
#             )
#             task_manager.update(task_id, status="succeeded", result={
#                 "yaml_path": yaml_path,
#                 "count": len(keys),
#             })
#         except Exception as e:
#             task_manager.update(task_id, status="failed", error=str(e))

#     bg.add_task(_run)
#     return {"task_id": task_id}

# @router.get("/task/status")
# def task_status(task_id: str = Query(...)):
#     st = task_manager.get(task_id)
#     if not st:
#         raise HTTPException(404, "task not found")
#     return st