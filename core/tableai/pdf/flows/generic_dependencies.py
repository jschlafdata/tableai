from pydantic import BaseModel, Field, computed_field, ConfigDict
from typing import TypeVar, Dict, Any
from tableai.pdf.flows.trace_log import EnhancedTraceLog

D = TypeVar("D", bound="FlowDependencies")

class FlowDependencies(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pdf_model: Any
    trace: EnhancedTraceLog