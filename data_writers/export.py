import os
import json
import yaml
from datetime import datetime
from uuid import UUID
from typing import Dict, Literal
from core.serialize import safe_json
from core.serialize import safe
import uuid
from core.time import UtcTimeUtil

def safe(obj):
    if isinstance(obj, (datetime, UUID)):
        return obj.isoformat()
    if isinstance(obj, UtcTimeUtil):
        return str(obj.now_utc())
    try:
        return str(obj)
    except Exception:
        return f"<Unserializable: {type(obj).__name__}>"


class DictWriter:
    def __init__(self, 
                 data: Dict, 
                 output_dir: str, 
                 file_name: str = None, 
                 format: Literal["json", "yaml"] = "json"
        ):
        self.data = data
        self.output_dir = output_dir
        self.output_file = file_name if file_name else f"{str(uuid.uuid4())}.{format}"
        self.output_path = os.path.join(self.output_dir , self.output_file)
        self.format = format.lower()
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def write(self):
        with open(self.output_path, "w") as f:
            if self.format == "json":
                json.dump(self.data, f, default=safe, indent=2)
            elif self.format == "yaml":
                data = json.loads(safe_json(self.data))
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            else:
                raise ValueError("Unsupported format: choose 'json' or 'yaml'")