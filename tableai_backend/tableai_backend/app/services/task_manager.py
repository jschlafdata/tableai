# backend/services/task_manager.py
from __future__ import annotations
import uuid
import time
from typing import Optional, Dict, Any

class TaskManager:
    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}

    def create(self, kind: str, payload: Optional[dict] = None) -> str:
        task_id = str(uuid.uuid4())
        self._tasks[task_id] = {
            "id": task_id,
            "kind": kind,
            "status": "queued",
            "created_at": time.time(),
            "updated_at": time.time(),
            "payload": payload or {},
            "result": None,
            "error": None,
        }
        return task_id

    def update(self, task_id: str, **kw):
        t = self._tasks.get(task_id)
        if not t:
            return
        t.update(kw)
        t["updated_at"] = time.time()

    def get(self, task_id: str) -> Optional[dict]:
        return self._tasks.get(task_id)

task_manager = TaskManager()
