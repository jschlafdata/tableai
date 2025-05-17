import functools
from typing import Optional, Union, List, Tuple
import uuid
from api.models.tasks import TaskStatus, RateLimiter
import asyncio
from fastapi import HTTPException

def task_runner(
    cache_prefix: Optional[str] = None,
    result_key: Optional[str] = None,
    force_refresh_arg: str = "force_refresh",
    eager: bool = True,
    post_run: Optional[Union[callable, List[callable], Tuple[callable, ...]]] = None
):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, task_id: Optional[str] = None, *args, **kwargs):
            task_id = task_id or str(uuid.uuid4())

            self.tasks[task_id] = TaskStatus(
                id=task_id,
                status="processing",
                progress=0.0,
                start_time=asyncio.get_event_loop().time()
            )

            if cache_prefix:
                force_refresh = kwargs.get(force_refresh_arg, True)
                if not force_refresh:
                    path = kwargs.get("path", "")
                    cache_key = f"{cache_prefix}:{path}"
                    cached_data = self._get_cached_data(cache_key)
                    if cached_data is not None:
                        result = {result_key: cached_data} if result_key else cached_data
                        self.tasks[task_id].result = result
                        self.tasks[task_id].status = "completed"
                        self.tasks[task_id].progress = 100.0
                        return result if eager else task_id

            try:
                data = await func(self, task_id, *args, **kwargs)

                result = {result_key: data} if result_key else data
                self.tasks[task_id].result = result
                self.tasks[task_id].status = "completed"
                self.tasks[task_id].progress = 100.0

                if cache_prefix and data is not None:
                    path = kwargs.get("path", "")
                    cache_key = f"{cache_prefix}:{path}"
                    self._cache_data(cache_key, data)

                # ✅ Run one or many post_run hooks
                if post_run:
                    hooks = post_run if isinstance(post_run, (list, tuple)) else [post_run]
                    for hook in hooks:
                        if asyncio.iscoroutinefunction(hook):
                            await hook(self.app, self.api)
                        else:
                            hook(self.app, self.api)

                return result if eager else task_id

            except Exception as e:
                self.tasks[task_id].status = "failed"
                self.tasks[task_id].error = str(e)
                self.logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)

                if cache_prefix:
                    fallback = self._get_cached_data(f"{cache_prefix}_partial")
                    if fallback:
                        self.logger.warning(f"Partial result returned from cache for {cache_prefix}_partial")
                        self.tasks[task_id].result = fallback

                if eager:
                    raise HTTPException(status_code=500, detail=str(e))
                else:
                    return task_id

        return wrapper
    return decorator