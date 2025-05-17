import json
from datetime import datetime
from uuid import UUID
from tableai.core.time import UtcTimeUtil
from pathlib import Path

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

def serialize(obj):
    from datetime import datetime
    from uuid import UUID
    from pathlib import Path

    if isinstance(obj, (datetime, UUID)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, UtcTimeUtil):
        return str(obj.now_utc())
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(v) for v in obj]
    if hasattr(obj, "dict"):
        return serialize(obj.dict())
    return obj

def _serialize_field(value):
    if isinstance(value, (datetime, UUID)):
        return value.isoformat()
    if isinstance(value, UtcTimeUtil):
        return str(value.now_utc())
    if isinstance(value, Path):
        return str(value)
    return value

def _deserialize_field(field_name: str, value):
    if field_name in {"local_path", "input_dir", "output_dir"}:
        return Path(value) if value is not None else None
    if field_name.endswith("_directories") or field_name.endswith("_categories"):
        return json.loads(value or "{}")
    if field_name.endswith("_metadata"):
        return json.loads(value or "{}")
    if field_name.endswith("_stages") or field_name.endswith("_paths"):
        return json.loads(value or "{}")
    return value