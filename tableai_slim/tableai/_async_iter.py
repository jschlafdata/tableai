import asyncio
import inspect
from typing import AsyncIterator, Callable, Iterable, Optional, TypeVar, Awaitable

T = TypeVar("T")
U = TypeVar("U")

async def aislice(aiter: AsyncIterator[T], start: int, stop: Optional[int]) -> AsyncIterator[T]:
    """Async islice: yield items in [start, stop) from an async iterator."""
    i = 0
    async for item in aiter:
        if i >= start:
            if stop is not None and i >= stop:
                break
            yield item
        i += 1

async def amap(fn: Callable[[T], U | Awaitable[U]],
               aiter: AsyncIterator[T],
               batch_size: int = 0) -> AsyncIterator[U]:
    """
    Async map over an async iterator.
      - If fn is sync -> sequential.
      - If fn is async and batch_size > 1 -> gather in small batches.
    """
    is_async = inspect.iscoroutinefunction(fn)
    if not is_async or batch_size <= 1:
        async for x in aiter:
            res = fn(x)
            if inspect.isawaitable(res):
                res = await res
            yield res
        return

    # async fn + batching
    batch: list[Awaitable[U]] = []
    async for x in aiter:
        batch.append(fn(x))  # type: ignore[arg-type]
        if len(batch) >= batch_size:
            for r in await asyncio.gather(*batch):
                yield r
            batch.clear()
    if batch:
        for r in await asyncio.gather(*batch):
            yield r

async def atee(aiter: AsyncIterator[T], count: int = 2, *, maxsize: int = 0) -> tuple[AsyncIterator[T], ...]:
    """
    Async tee: split a single async stream into `count` branches.
    """
    queues = [asyncio.Queue(maxsize=maxsize) for _ in range(count)]
    SENTINEL = object()

    async def fanout():
        try:
            async for item in aiter:
                for q in queues:
                    await q.put(item)
        finally:
            for q in queues:
                await q.put(SENTINEL)

    asyncio.create_task(fanout())

    async def branch(q: asyncio.Queue):
        while True:
            item = await q.get()
            if item is SENTINEL:
                break
            yield item

    return tuple(branch(q) for q in queues)