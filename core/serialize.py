import json
from datetime import datetime
from uuid import UUID
from core.time import UtcTimeUtil

def safe(obj):
    if isinstance(obj, (datetime, UUID)):
        return obj.isoformat()
    if isinstance(obj, UtcTimeUtil):
        return str(obj.now_utc())
    if not obj:
        return ""
    try:
        return str(obj)
    except Exception:
        return f"<Unserializable: {type(obj).__name__}>"

def safe_json(data: dict) -> str:
    return json.dumps(data, default=safe, indent=2)