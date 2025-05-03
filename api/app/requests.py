from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class FilterRequest(BaseModel):
    """Filter configuration"""
    version: Optional[int] = Field(default=1, description="Version number to use")
    include_categories: Optional[List[str]] = Field(default=None, description="List of specific categories to include")
    exclude_categories: Optional[List[str]] = Field(default=None, description="List of categories to exclude")
    custom_filter: Optional[bool] = Field(default=False, description="Whether to apply custom filtering logic")
    include_metadata: Optional[bool] = Field(default=False, description="Whether to include additional metadata")
    category_filters: Optional[Dict[str, Any]] = Field(default=None, description="Filters to apply to category selection")
    composite_filters: Optional[List[str]] = Field(default=None, description="List of strings to filter by composite values")
    priority_order: Optional[List[int]] = Field(default=None, description="Filter by priority order")

class DataRequest(BaseModel):
    """Data request parameters"""
    category: str
    resource_id: str
    force_refresh: Optional[bool] = False
    filters: Optional[FilterRequest] = None
    include_metadata: Optional[bool] = False