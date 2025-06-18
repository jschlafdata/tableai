from collections import UserList
from typing import Optional, List, Tuple, Union, Dict, Any, Callable, TypeVar, Generic
from pydantic import BaseModel, Field
from typing import Generic, TypeVar, List, Dict, Any, Callable, Optional, Tuple, Union

T = TypeVar('T')
class ChainResult(Generic[T], UserList[T]):
    """
    A generic, list-like container for the results of chained operations.
    """
    def __repr__(self) -> str:
        if not self.data: return "ChainResult(count=0)"
        return f"ChainResult(count={len(self)}, type={type(self.data[0]).__name__})"
    def apply(self, func: Callable[[T], Any]) -> list:
        return [func(item) for item in self.data]
    def pluck(self, *keys: str) -> Union[list, list[tuple]]:
        if not all(isinstance(item, dict) for item in self.data):
            raise TypeError(".pluck() is only supported for collections of dictionaries.")
        if len(keys) == 1: return [item.get(keys[0]) for item in self.data]
        return [tuple(item.get(key) for key in keys) for item in self.data]
    def to_dict(self) -> List[Dict[str, Any]]:
        return self.data

# Make ResultSet a generic class that inherits from UserList.
# UserList is itself generic since Python 3.9, so UserList[T] is the proper way.
class ResultSet(Generic[T], UserList[T]):
    """
    A generic, list-like container for query results with fluent, chainable
    methods and type-aware accessors for discoverability.
    """
    
    def __repr__(self) -> str:
        """Provides a helpful representation in notebooks."""
        if not self.data:
            return "ResultSet(count=0)"
        item_type = type(self.data[0]).__name__
        return f"ResultSet(count={len(self)}, type={item_type})"

    def apply(self, func: Callable[[T], Any]) -> list:
        """
        Applies a function to each item in the ResultSet.
        Thanks to generics, your IDE will know the type of the item in the lambda.
        
        Example:
            >>> results.apply(lambda item: item.x0 * 2) 
        """
        return [func(item) for item in self.data]

    def pluck(self, *attrs: str) -> Union[list, list[tuple]]:
        """
        Extract one or more attributes from each item.
        """
        if len(attrs) == 1:
            return [getattr(item, attrs[0], None) for item in self.data]
        else:
            return [tuple(getattr(item, attr, None) for attr in attrs) for item in self.data]

    def to_dict(self) -> List[Dict[str, Any]]:
        """
        Converts the ResultSet into a list of dictionaries.

        For each item in the set, it calls the item's own .dict() method
        if it exists. Otherwise, it returns the item as is.
        This is useful for serialization (e.g., to JSON).
        """
        if not self.data:
            return []

        # Use a list comprehension to call .dict() on each item that has it
        return [
            item.dict() if hasattr(item, 'dict') and callable(getattr(item, 'dict')) else item
            for item in self.data
        ]


class BaseQueryResult(BaseModel):
    """
    A base model containing fields common to all query result types.
    This provides a single source of truth for shared metadata.
    """
    class Config:
        # This ensures compatibility with your custom __init__ in GroupbyQueryResult
        extra = 'allow'

    # --- Consistent Fields ---
    page: Optional[int] = Field(default=None, description="The VIRTUAL page number in the combined document.")
    index: Optional[int] = Field(default=None, description="A unique, sequential ID for this row within the entire LineTextIndex. For groups, this is the index of the first member.")
    physical_page: Optional[int] = Field(default=None, description="The ORIGINAL page number from the source multi-page PDF document.")
    region: Optional[str] = Field(default=None, description="A simple geometric classification based on position ('header' or 'footer').")
    
    # --- Metadata ---
    meta: Dict[str, Any] = Field(default_factory=dict, description="A dictionary for extra metadata, often containing virtual page bounds.")
    query_label: Optional[str] = Field(default=None, description="A label passed down from the query that generated this result.")
    physical_page_bounds: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="A dictionary {'width': ..., 'height': ...} for the ORIGINAL source page."
    )

class DefaultQueryResult(BaseQueryResult): # <-- Inherits from BaseQueryResult
    """A single, atomic result from the LineTextIndex, enriched with metadata."""
    
    # --- Fields unique to a single result ---
    block: int = Field(description="The block number from the original Fitz document structure.")
    line: int = Field(description="The line number within the block from the original Fitz document structure.")
    span: Optional[int] = Field(default=None, description="The span number within the line. Often None, as the index is line-based.")
    path: Optional[str] = Field(default=None, description="The full dictionary path to this element. None for synthetic rows.")
    key: str = Field(description="The type of this row (e.g., 'text', 'normalized_text').")
    value: Any = Field(description="The actual data value for this row.")
    gap: Optional[float] = Field(default=None, description="For whitespace rows, the height of the gap.")
    font_meta: Optional[Dict[str, Any]] = Field(default=None, description="A dictionary of font properties.")
    normalized_value: Optional[str] = Field(default=None, description="A lowercased, whitespace-stripped version of the 'value'.")
    bbox: Optional[Tuple[float, ...]] = Field(default=None, description="The absolute bounding box (x0, y0, x1, y1).")
    x0: Optional[float] = Field(default=None, description="Absolute x0 coordinate.")
    y0: Optional[float] = Field(default=None, description="Absolute y0 coordinate.")
    x1: Optional[float] = Field(default=None, description="Absolute x1 coordinate.")
    y1: Optional[float] = Field(default=None, description="Absolute y1 coordinate.")
    bbox_rel: Optional[Tuple[float, ...]] = Field(default=None, description="The bounding box relative to its VIRTUAL page.")
    x0_rel: Optional[float] = Field(default=None, description="Relative x0 coordinate.")
    y0_rel: Optional[float] = Field(default=None, description="Relative y0 coordinate.")
    x1_rel: Optional[float] = Field(default=None, description="Relative x1 coordinate.")
    y1_rel: Optional[float] = Field(default=None, description="Relative y1 coordinate.")
    page_height_rel: Optional[float] = Field(default=None, description="The height of the VIRTUAL page.")
    page_width_rel: Optional[float] = Field(default=None, description="The width of the VIRTUAL page.")
    x_span: Optional[float] = Field(default=None, description="Calculated width of the bounding box.")
    y_span: Optional[float] = Field(default=None, description="Calculated height of the bounding box.")

    # --- Methods remain unchanged and work perfectly with inheritance ---
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DefaultQueryResult":
        key_map = {"x0(rel)": "x0_rel", "y0(rel)": "y0_rel", "x1(rel)": "x1_rel", "y1(rel)": "y1_rel", "bbox(rel)": "bbox_rel", "page_height(rel)": "page_height_rel", "page_width(rel)": "page_width_rel"}
        init_kwargs = {key_map.get(k, k): v for k, v in data.items()}
        return cls(**init_kwargs)

    def dict(self):
        return self.model_dump()


class GroupbyQueryResult(BaseQueryResult): # <-- Inherits from BaseQueryResult
    """A collection of DefaultQueryResult rows, grouped by a common key."""
    
    # --- Fields unique to a grouped result ---
    group_id: Tuple = Field(description="A tuple of the values that uniquely identify this group.")
    groupby_keys: Tuple[str, ...] = Field(description="A tuple of the key names that were used to create this group.")
    member_count: int = Field(description="The total number of original rows contained in this group.")
    group_bboxes: List[Tuple[float, ...]] = Field(default_factory=list, description="A list of all absolute bounding boxes from every member.")
    group_bboxes_rel: Optional[List[Tuple[float, ...]]] = Field(default_factory=list, description="A list of all relative bounding boxes from every member.")
    group_paths: List[str] = Field(default_factory=list, description="A list of all dictionary paths from every member.")
    group_text: List[str] = Field(default_factory=list, description="A list of all 'value' strings from every member.")

    # --- Methods remain unchanged and work perfectly with inheritance ---
    def __init__(self, **kwargs):
        # This correctly passes only the known fields (from this class AND the base class)
        # to the Pydantic constructor, preserving compatibility with your groupby transform.
        super().__init__(**{k: v for k, v in kwargs.items() if k in self.model_fields})
    
    def dict(self):
        return self.model_dump()
