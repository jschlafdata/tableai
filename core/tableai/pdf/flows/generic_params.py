
from tableai.pdf.flows.generic_dependencies import FlowDependencies
from tableai.pdf.flows.generic_results import FlowResult
from pydantic import BaseModel, Field, ConfigDict, create_model, PrivateAttr
from typing import Dict, List, Any, Optional, Union, get_type_hints, Type

class FlowParams(BaseModel):
    """A stateless configuration object describing a Flow's metadata and types."""
    deps_type: Type[FlowDependencies]
    result_type: Type[FlowResult]
    overview: str
    goal: str
    analysis_exclude_modules: Optional[List] = None