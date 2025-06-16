from collections import UserList
from typing import Optional, List, Tuple, Union, Dict, Any, Callable, TYPE_CHECKING, TypeVar, Generic, Type
from pydantic import BaseModel, field_validator, model_validator, ValidationError, Field, create_model, field_serializer
import re
from tableai.pdf.coordinates import (
    Geometry,
    CoordinateMapping
)

class GenericFunctionParams(BaseModel):
    """
    A base model for query parameters. All dynamically created parameter
    models will inherit from this class. You can place truly universal
    parameters here.
    """
    # Example of a truly universal parameter that all models will inherit
    query_label: Optional[str] = Field(
        default=None,
        description="An optional label to attach to the query results for tracking."
    )

    @classmethod
    def create_custom_model(
        cls, 
        model_name: str, 
        custom_fields: Dict[str, Dict[str, Any]]
    ) -> Type[BaseModel]:
        """
        Dynamically creates a new Pydantic model that inherits from this base class.

        Args:
            model_name (str): The name for the new Pydantic model class (e.g., "HorizontalWhitespaceParams").
            custom_fields (Dict): A dictionary defining the custom fields for the new model.
                The format is:
                {
                    "field_name": {
                        "type": field_type (e.g., int, str),
                        "default": default_value,
                        "description": "A helpful description."
                    },
                    ...
                }

        Returns:
            A new Pydantic model class, ready to be instantiated.
        """
        # Prepare a dictionary of field definitions in the format Pydantic's create_model expects
        pydantic_fields: Dict[str, Any] = {}
        
        for field_name, config in custom_fields.items():
            field_type = config.get("type", Any)
            default_value = config.get("default", ...) # ... means the field is required if no default
            description = config.get("description", None)
            
            # The value in the dict must be a tuple: (type, Field_instance)
            pydantic_fields[field_name] = (
                field_type,
                Field(default=default_value, description=description)
            )

        # Use Pydantic's built-in function to create the new model class
        # __base__=cls ensures it inherits from GenericQueryParams
        new_model_class = create_model(
            model_name,
            __base__=cls,
            **pydantic_fields
        )
        
        return new_model_class


# For horizontal_whitespace
HorizontalWhitespaceParams = GenericFunctionParams.create_custom_model(
    "HorizontalWhitespaceParams", {
        'page_number': { 'type': Optional[int], 'default': None, 'description': "Optional page number to search within." },
        'y_tolerance': { 'type': int, 'default': 10, 'description': "Minimum vertical gap to be considered whitespace." }
    }
)

# For group_vertically_touching_bboxes
GroupTouchingBoxesParams = GenericFunctionParams.create_custom_model(
    "GroupTouchingBoxesParams", {
        'y_tolerance': { 'type': float, 'default': 2.0, 'description': "Max vertical distance between boxes to be considered 'touching'." }
    }
)

# For paragraphs -> find_paragraph_blocks
ParagraphsParams = GenericFunctionParams.create_custom_model(
    "ParagraphsParams", {
        'width_threshold': { 'type': float, 'default': 0.5, 'description': "Minimum relative width (0.0-1.0) for a line to be a paragraph seed." },
        'x0_tol': { 'type': float, 'default': 2.0, 'description': "Tolerance for x0 alignment between paragraph lines." },
        'font_size_tol': { 'type': float, 'default': 0.2, 'description': "Tolerance for font size similarity between lines." },
        'y_gap_max': { 'type': float, 'default': 7.0, 'description': "Maximum vertical gap allowed between lines in a paragraph." }
    }
)

class TextNormalizer:
    """A callable object that normalizes text based on a set of regex patterns."""
    def __init__(self, patterns: Dict[str, str], description: Optional[str] = None, output_key: Optional[str]='normalized_text'):
        self.patterns = patterns
        self.output_key = output_key or 'normalized_text'
        self.description = description or f"Normalizes text using a set of regex substitutions. Used to build index items with the key=[{self.output_key}]."

    def __call__(self, text: str) -> str:
        """Makes the object callable to perform the normalization."""
        t = text.lower().strip()
        for pattern, replacement in self.patterns.items():
            t = re.sub(pattern, replacement, t)
        return t

    def to_dict(self) -> dict:
        """Creates a human-readable dictionary for logging."""
        return {
            "type": "TextNormalizer",
            "patterns": self.patterns,
            "description": self.description
        }

class WhitespaceGenerator:
    """A callable object that computes full-width vertical whitespace."""
    def __init__(self, min_gap: float = 5.0, description: Optional[str] = None, output_key: Optional[str]='full_width_v_whitespace'):
        self.min_gap = min_gap
        self.output_key = output_key or 'full_width_v_whitespace'
        self.description = description or f"Detects vertical whitespace regions spanning the page width. Used to build index items with the key=[{self.output_key}]."

    def __call__(self, by_page: Dict, page_metadata: Dict) -> List[Dict[str, Any]]:
        """Makes the object callable to perform the calculation."""
        # The entire logic from your old compute_full_width_v_whitespace function goes here.
        # ... just use self.min_gap instead of the hardcoded value.
        results = []
        for page_num, rows in by_page.items():
            spans = [r for r in rows if r.get("key") == "text" and r.get("bbox")]
            spans = sorted(spans, key=lambda r: r["y0"])
            page_width = page_metadata.get(page_num, {}).get("width", 612.0)  # fallback default A4 width
    
            for i in range(len(spans) - 1):
                a, b = spans[i], spans[i + 1]
                gap = b["y0"] - a["y1"]
                if gap >= self.min_gap:
                    y0 = a["y1"]
                    y1 = b["y0"]
                    results.append({
                        "page": page_num,
                        "block": -1,
                        "line": -1,
                        "span": -1,
                        "index": -1,
                        "key": "full_width_v_whitespace",
                        "value": "",
                        "path": None,
                        "gap": gap,
                        "bbox": (0.0, y0, page_width, y1),
                        "x0": 0.0,
                        "y0": y0,
                        "x1": page_width,
                        "y1": y1,
                        "x_span": page_width,
                        "y_span": y1 - y0,
                        "meta": {"gap_class": "large" if gap > 20 else "small"}
                    })
        return results

    def to_dict(self) -> dict:
        """Creates a human-readable dictionary for logging."""
        return {
            "type": "WhitespaceGenerator",
            "min_gap": self.min_gap,
            "description": self.description
        }

T = TypeVar('T')

class BaseAccessor(Generic[T]):
    """Base class for type-aware accessors."""
    def __init__(self, data: List[T]):
        self._data = data
        self._validate()

    def _validate(self):
        # Optional: Add runtime validation
        pass

    def apply(self, func: Callable[[T], any]) -> list:
        """Applies a function to each item."""
        return [func(item) for item in self._data]

class GroupOps:
    """A collection of reusable static methods for processing GroupbyQueryResult objects."""
    
    @staticmethod
    def merge_bboxes(group: 'GroupbyQueryResult', key: str = 'group_bboxes') -> Optional[Tuple[float, ...]]:
        """
        Takes a group and merges a specified list of bboxes within it.

        Args:
            group: The GroupbyQueryResult object to process.
            key: The name of the attribute on the group object that holds the
                 list of bboxes (e.g., 'group_bboxes', 'group_bboxes_rel').
                 Defaults to 'group_bboxes'.
        
        Returns:
            A single merged bounding box tuple, or None if the key doesn't
            exist or the list is empty.
        """
        # Safely get the list of bboxes using the specified key
        bboxes_to_merge = getattr(group, key, None)
        
        if not bboxes_to_merge:
            return None
            
        return Geometry.merge_all_boxes(bboxes_to_merge)

    @staticmethod
    def concat_text(group: 'GroupbyQueryResult', delimiter: str = '|') -> str:
        """Takes a group and returns its concatenated text string."""
        return delimiter.join(group.group_text)


# --- Accessor for GroupbyQueryResult ---
class GroupAccessor(BaseAccessor['GroupbyQueryResult']):
    def process(
        self,
        aggregations: Optional[Dict[str, Callable]] = None,
        filters: Optional[List[Callable]] = None,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Processes each group in the ResultSet, returning a filtered list of structured dictionaries.
        
        Args:
            aggregations: Dictionary of {field_name: aggregation_function} to compute new fields
            filters: List of filter functions that receive the full output dict and return bool
            include: List of original field names to include from the group object
            **kwargs: Additional parameters (for future extensibility)
        
        Returns:
            List of dictionaries with processed group data
        """
        # Use provided parameters or defaults
        aggregations = aggregations or {}
        filters = filters or []
        include = include or []

        results = []
        for group in self._data:
            # Build the initial dictionary for this group
            output_dict = {'group_id': group.group_id}

            # Add requested fields from the group object
            for field_name in include:
                if hasattr(group, field_name):
                    output_dict[field_name] = getattr(group, field_name)

            # Apply aggregation functions to compute new fields
            for new_field_name, agg_func in aggregations.items():
                output_dict[new_field_name] = agg_func(group)

            # Apply all filter functions - keep group only if all return True
            keep_group = all(filter_func(output_dict) for filter_func in filters)
            
            if keep_group:
                results.append(output_dict)
            
        return results

    # --- Legacy methods kept for backward compatibility ---
    def merge_all_bboxes(self) -> list[tuple]:
        """Merges the bboxes of each group into a single bbox per group."""
        return self.apply(GroupOps.merge_bboxes)
        
    def get_text(self) -> list[list[str]]:
        """Returns a list of the text lists for each group."""
        return self.apply(lambda item: item.group_text)


# --- Accessor for DefaultQueryResult ---
class DefaultAccessor(BaseAccessor['DefaultQueryResult']):
    """

    Provides type-aware methods for a ResultSet of DefaultQueryResult objects.
    """
    def get_values(self) -> list:
        """Extracts the 'value' from each result."""
        return self.apply(lambda item: item.value)

    def get_bboxes(self) -> list[tuple | None]:
        """Extracts the 'bbox' from each result."""
        return self.apply(lambda item: item.bbox)


# Make ResultSet a generic class that inherits from UserList.
# UserList is itself generic since Python 3.9, so UserList[T] is the proper way.
class ResultSet(Generic[T], UserList[T]):
    """
    A generic, list-like container for query results with fluent, chainable
    methods and type-aware accessors for discoverability.
    """

    @property
    def group(self) -> GroupAccessor:
        """
        Accessor for operations on a ResultSet of GroupbyQueryResult objects.
        Provides autocompletion and type safety.
        
        Example:
            >>> header_groups.group.merge_all_bboxes()
        """
        if not self.data or not isinstance(self.data[0], GroupbyQueryResult):
            raise AttributeError(
                "'.group' accessor is only available for ResultSets containing GroupbyQueryResult objects."
            )
        return GroupAccessor(self.data)

    @property
    def default(self) -> DefaultAccessor:
        """
        Accessor for operations on a ResultSet of DefaultQueryResult objects.
        
        Example:
            >>> text_results.default.get_values()
        """
        if not self.data or not isinstance(self.data[0], DefaultQueryResult):
            raise AttributeError(
                "'.default' accessor is only available for ResultSets containing DefaultQueryResult objects."
            )
        return DefaultAccessor(self.data)
    
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
