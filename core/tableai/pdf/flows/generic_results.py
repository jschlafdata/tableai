from pydantic import BaseModel, Field, computed_field, ConfigDict
from typing import TypeVar, Dict, Any

R = TypeVar("R", bound="FlowResult")

class FlowResult(BaseModel):
    overview: str
    goal: str
    flow_results: Dict[str, Any] = Field(default_factory=dict)

class FlowResultStage(BaseModel):
    """
    A simple, flexible container used to pass partially-built results
    between the final nodes of a flow. It acts as a "data bucket"
    that can hold any attribute.
    """
    # This is the magic: it tells Pydantic to accept any keyword
    # arguments and turn them into attributes.
    model_config = ConfigDict(extra='allow')

# Your helper function is perfect.
def section_field(section: str, description: str = "", **field_kwargs):
    return Field(description=description, json_schema_extra={"section": section}, **field_kwargs)
