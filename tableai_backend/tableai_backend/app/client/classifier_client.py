from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple, List, Any, Callable

import httpx

# ---------- minimal async utils (drop-in) ----------
async def aislice(ait: AsyncIterator[Any], start: int, stop: Optional[int]) -> AsyncIterator[Any]:
    idx = 0
    async for x in ait:
        if idx >= start and (stop is None or idx < stop):
            yield x
        idx += 1
        if stop is not None and idx >= stop:
            break

async def amap(fn: Callable[[Any], Any], ait: AsyncIterator[Any], batch_size: int = 8) -> AsyncIterator[Any]:
    sem = asyncio.Semaphore(batch_size)
    async def run_one(item):
        async with sem:
            res = fn(item)
            if asyncio.iscoroutine(res):
                res = await res
            return res
    tasks: List[asyncio.Task] = []
    async for item in ait:
        tasks.append(asyncio.create_task(run_one(item)))
    for t in asyncio.as_completed(tasks):
        yield await t

def atee(ait: AsyncIterator[Any], count: int = 2):
    """
    Simple tee for async iterators; duplicates items into per-branch queues.
    """
    import asyncio
    queues = [asyncio.Queue() for _ in range(count)]
    done = object()

    async def produce():
        try:
            async for item in ait:
                for q in queues:
                    await q.put(item)
        finally:
            for q in queues:
                await q.put(done)

    async def consume(q):
        while True:
            item = await q.get()
            if item is done:
                break
            yield item

    # kick off producer
    asyncio.create_task(produce())
    return tuple(consume(q) for q in queues)

# ---------- normal helpers ----------
async def iter_pdf_paths(pdf_dir: str) -> AsyncIterator[Path]:
    p = Path(pdf_dir)
    for path in sorted(p.rglob("*.pdf")):
        yield path

async def to_formpart(pdf_path: Path) -> Tuple[str, Tuple[str, bytes, str]]:
    content = await asyncio.to_thread(pdf_path.read_bytes)
    return ('files', (pdf_path.name, content, 'application/pdf'))

# ---------- main client function ----------
async def upload_and_classify_async(
    pdf_dir: str,
    server_url: str = "http://localhost:8000",  # your backend base URL
    *,
    max_files: Optional[int] = None,
    read_concurrency: int = 8,
    log_progress: bool = True,
    min_cluster: int = 4,
) -> bytes:
    base_stream = iter_pdf_paths(pdf_dir)
    files_stream = aislice(base_stream, 0, max_files) if max_files is not None else base_stream

    if log_progress:
        read_stream, log_stream = atee(files_stream, count=2)
        async def logger():
            idx = 0
            async for p in log_stream:
                idx += 1
                print(f"[{idx}] staging: {p.name}")
        log_task = asyncio.create_task(logger())
    else:
        read_stream = files_stream
        log_task = None

    # Build the multipart form (concurrently reading the files)
    formparts: List[Tuple[str, Tuple[str, bytes, str]]] = []
    async for part in amap(to_formpart, read_stream, batch_size=read_concurrency):
        formparts.append(part)

    if log_task is not None:
        await log_task

    # POST to your backend router
    async with httpx.AsyncClient(timeout=300.0) as client:
        sync_resp = await client.post(f"{server_url}/classifier/sync", files=formparts)
        sync_resp.raise_for_status()
        saved = sync_resp.json().get("saved", 0)
        print(f"/classifier/sync accepted {saved} files (status {sync_resp.status_code})")

        cls_resp = await client.post(f"{server_url}/classifier/classify", params={"min_cluster": min_cluster})
        cls_resp.raise_for_status()

        Path("clusters.yaml").write_bytes(cls_resp.content)
        print("Classification saved to clusters.yaml")

        return cls_resp.content

# Convenience sync wrapper
def upload_and_classify(
    pdf_dir: str,
    server_url: str = "http://localhost:8000",
    **kwargs
) -> bytes:
    return asyncio.run(upload_and_classify_async(pdf_dir, server_url, **kwargs))
