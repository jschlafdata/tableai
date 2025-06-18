import inspect
import ast
from textwrap import dedent
from pydantic import Field, BaseModel, ConfigDict
from typing import Optional, Any, List, Callable, Dict
import inspect
import json


class EnhancedTraceLog:
    """
    A stateful logger that captures a structured history of a flow's execution.
    It no longer runs the nodes itself, but records the data given by the Flow runner.
    """
    def __init__(self):
        self.history: List[Dict[str, Any]] = []

    def clear(self):
        self.history = []

    def log_node_execution(
        self,
        name: str,
        description: str,
        config: BaseModel,
        func_input: Any,
        sub_calls: List[Dict],
        output: Any,
        error: Optional[Dict] = None
    ):
        """Records a complete node execution event."""
        log_entry = {
            "name": name,
            "description": description,
            "config": _serialize_safely(config),
            "input": _serialize_safely(func_input),
            "sub_calls": sub_calls,
            "output": _serialize_safely(output),
            "error": error
        }
        self.history.append(log_entry)
        
    def get_log(self) -> Dict[str, Any]:
        """Returns the complete execution history."""
        return {f"{i}_{entry['name']}": entry for i, entry in enumerate(self.history)}

def _serialize_safely(obj: Any) -> Any:
    """
    A robust serializer that handles Pydantic models, custom list-like objects,
    and other common types for clean logging.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool, dict, list, tuple)):
        return obj
    # For Pydantic models (like ParamContext, GraphContext, NodeInput, etc.)
    if hasattr(obj, 'model_dump'):
        try:
            # Use a custom default to handle non-serializable types gracefully
            return json.loads(obj.model_dump_json())
        except Exception:
            # Fallback for complex cases
            return obj.model_dump()
    # For our custom list-like containers (ResultSet, Chainable)
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()
    # Fallback for any other object
    return repr(obj)