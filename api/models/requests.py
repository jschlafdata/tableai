from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMTableStructureRequest(BaseModel):
    file_id: str
    stage: int
    classification_label: str

class LLMInferenceResultRequest(BaseModel):
    file_id: str
    stage: int
    classification_label: str
    prompt_version: Optional[int] = None
    prompt_name: Optional[str] = None

class PDFExtractRequest(BaseModel):
    file_id: str

class DropboxSyncRequest(BaseModel):
    paths: List[str]
    force_refresh: Optional[bool] = False
    file_types: Optional[List[str]] = None  # e.g., ["pdf", "xlsx"]
    ignore: Optional[str] = None
    path_categories: Optional[str] = None
    auto_label: Optional[str] = None

class DropboxRegisterRequest(BaseModel):
    """
    Request body for registering synced Dropbox items into the local DB.
    """
    file_ids: Optional[List[str]] = None       # e.g. ["id:q96KjlOoc_kAAAAAAAAPtg", ...]
    directories: Optional[List[str]] = None    # e.g. ["allianz/automation/reports/8_24", ...]
    force_refresh: bool = False
    stage: int = 0

class DropboxProcessRequest(BaseModel):
    """
    Request body for registering synced Dropbox items into the local DB.
    """
    file_ids: Optional[List[str]] = None       # e.g. ["id:q96KjlOoc_kAAAAAAAAPtg", ...]
    force_refresh: bool = False
    stage: int = 0

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