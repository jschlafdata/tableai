# app/main.py
import os, sys, uuid, tempfile, shutil
from pathlib import Path

import aiofiles                     # pip install aiofiles
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from tqdm.auto import tqdm          # auto picks text‑mode when no TTY

from .pdf_cluster_dit import cluster_dir

# ------------------------------------------------------------------+
#  Configuration                                                     |
# ------------------------------------------------------------------+
DATA_DIR = Path(os.getenv("DATA_DIR", "/data/pdfs")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)          # one shared stash

app = FastAPI(title="Table‑AI classifier", version="0.2.0")

# ------------------------------------------------------------------+
# 1. Sync endpoint                                                   |
# ------------------------------------------------------------------+
@app.post("/sync", summary="Stage PDFs in server stash")
async def sync(files: list[UploadFile]):
    if not files:
        raise HTTPException(400, "No files uploaded")

    saved: list[str] = []
    # tqdm only draws the bar if container was started with `-t`
    bar = tqdm(files, desc="saving", unit="pdf", file=sys.stdout,
               disable=not sys.stdout.isatty())

    for up in bar:
        fname = f"{uuid.uuid4()}_{up.filename}"
        dest  = DATA_DIR / fname

        async with aiofiles.open(dest, "wb") as out:
            while chunk := await up.read(64 << 10):   # 64 KiB
                await out.write(chunk)
        await up.close()
        saved.append(dest.name)

    return JSONResponse({"saved": len(saved), "files": saved})

# ------------------------------------------------------------------+
# 2. Classify endpoint                                               |
# ------------------------------------------------------------------+
@app.post("/classify", summary="Cluster all staged PDFs")
async def classify(min_cluster: int = 4):
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if not pdfs:
        raise HTTPException(400, "Nothing to classify – call /sync first")

    fd, yaml_path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)                   # close OS handle immediately
    try:
        cluster_dir(str(DATA_DIR), yaml_path, min_cluster=min_cluster)
        return FileResponse(
            yaml_path,
            media_type="text/yaml",
            filename="clusters.yaml",
        )
    finally:
        # leave PDFs in place (you can rm if you prefer)
        pass

# ------------------------------------------------------------------+
# 3. Health endpoint                                                 |
# ------------------------------------------------------------------+
@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok"}
