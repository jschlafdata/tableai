from collections import UserList
from typing import Optional, List, Tuple, Union, Dict, Any, Callable, TYPE_CHECKING, TypeVar, Generic, Type
from pydantic import BaseModel, field_validator, model_validator, ValidationError, Field, create_model, field_serializer
import re
from tableai.pdf.coordinates import (
    Geometry,
    CoordinateMapping
)
from typing import Generic, TypeVar, List, Dict, Any, Callable, Optional, Tuple, Union, Protocol
from copy import deepcopy

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

# Type definitions for better type safety
BBox = Tuple[float, float, float, float]
BBoxList = List[BBox]
GroupFunction = Callable[['GroupbyQueryResult'], Any]

class HasBBoxField(Protocol):
    """Protocol for objects that have bbox fields."""
    def __getattribute__(self, name: str) -> Any: ...

# Field accessor functions that return aggregation functions with type safety
def merge_all_bboxes(field_name: str) -> GroupFunction:
    """
    Returns a function that merges all bboxes from the specified field.
    
    Args:
        field_name: Name of the field containing the list of bboxes
                   (e.g., 'group_bboxes', 'group_bboxes_rel')
    
    Returns:
        Function that takes a group and returns merged bbox or None
    """
    def _merge_bboxes(group: 'GroupbyQueryResult') -> Optional[BBox]:
        bboxes_to_merge: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes_to_merge:
            return None
        return Geometry.merge_all_boxes(bboxes_to_merge)
    
    return _merge_bboxes

def merge_overlapping_bboxes(field_name: str) -> GroupFunction:
    """
    Returns a function that merges overlapping bboxes from the specified field.
    
    Args:
        field_name: Name of the field containing the list of bboxes
    
    Returns:
        Function that takes a group and returns list of merged bboxes
    """
    def _merge_overlapping(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return []
        return Geometry.merge_overlapping_boxes(bboxes)
    
    return _merge_overlapping

def concat_text(field_name: str, delimiter: str = '|') -> GroupFunction:
    """
    Returns a function that concatenates text from the specified field.
    
    Args:
        field_name: Name of the field containing the list of text strings
        delimiter: String to join the text with
    
    Returns:
        Function that takes a group and returns concatenated text
    """
    def _concat_text(group: 'GroupbyQueryResult') -> str:
        text_list: List[str] = getattr(group, field_name, [])
        return delimiter.join(text_list)
    
    return _concat_text

def get_field(field_name: str) -> GroupFunction:
    """
    Returns a function that extracts a specific field from the group.
    
    Args:
        field_name: Name of the field to extract
    
    Returns:
        Function that takes a group and returns the field value
    """
    def _get_field(group: 'GroupbyQueryResult') -> Any:
        return getattr(group, field_name, None)
    
    return _get_field

def count_items(field_name: str) -> GroupFunction:
    """
    Returns a function that counts items in a list field.
    
    Args:
        field_name: Name of the field containing a list
    
    Returns:
        Function that takes a group and returns the count
    """
    def _count_items(group: 'GroupbyQueryResult') -> int:
        items = getattr(group, field_name, [])
        return len(items) if items else 0
    
    return _count_items

def check_overlap_with_bbox(field_name: str, target_bbox: BBox) -> GroupFunction:
    """
    Returns a function that checks if any bbox in the field overlaps with the target bbox.
    
    Args:
        field_name: Name of the field containing bboxes
        target_bbox: The bbox to check overlap against
    
    Returns:
        Function that takes a group and returns True if any bbox overlaps
    """
    def _check_overlap(group: 'GroupbyQueryResult') -> bool:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return False
        
        for bbox in bboxes:
            if Geometry.bbox_overlaps(bbox, target_bbox):
                return True
        return False
    
    return _check_overlap

def check_x_overlap_with_bbox(field_name: str, target_bbox: BBox) -> GroupFunction:
    """
    Returns a function that checks if any bbox in the field has x-overlap with the target bbox.
    
    Args:
        field_name: Name of the field containing bboxes
        target_bbox: The bbox to check x-overlap against
    
    Returns:
        Function that takes a group and returns True if any bbox has x-overlap
    """
    def _check_x_overlap(group: 'GroupbyQueryResult') -> bool:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return False
        
        for bbox in bboxes:
            if Geometry.is_x_overlapping(bbox, target_bbox):
                return True
        return False
    
    return _check_x_overlap

def is_fully_contained_in(field_name: str, outer_bbox: BBox, index: int = 0) -> GroupFunction:
    """
    Returns a function that checks if a bbox is fully contained within the outer bbox.
    
    Args:
        field_name: Name of the field containing bboxes
        outer_bbox: The containing bbox
        index: Index of the bbox to check (default: 0)
    
    Returns:
        Function that takes a group and returns True if bbox is fully contained
    """
    def _is_contained(group: 'GroupbyQueryResult') -> bool:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes or len(bboxes) <= index:
            return False
        
        return Geometry.is_fully_contained(bboxes[index], outer_bbox)
    
    return _is_contained

def percent_contained_in(field_name: str, outer_bbox: BBox, index: int = 0) -> GroupFunction:
    """
    Returns a function that calculates what percentage of a bbox is contained in the outer bbox.
    
    Args:
        field_name: Name of the field containing bboxes
        outer_bbox: The containing bbox
        index: Index of the bbox to check (default: 0)
    
    Returns:
        Function that takes a group and returns percentage contained (0.0 to 1.0)
    """
    def _percent_contained(group: 'GroupbyQueryResult') -> float:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes or len(bboxes) <= index:
            return 0.0
        
        return Geometry.percent_contained(bboxes[index], outer_bbox)
    
    return _percent_contained

def scale_bboxes_y(field_name: str, y_offset: float) -> GroupFunction:
    """
    Returns a function that scales all bboxes in a field by a y-offset.
    
    Args:
        field_name: Name of the field containing bboxes
        y_offset: Offset to apply to y coordinates
    
    Returns:
        Function that takes a group and returns scaled bboxes
    """
    def _scale_y(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return []
        
        return [Geometry.scale_y(bbox, y_offset) for bbox in bboxes]
    
    return _scale_y

def transform_bboxes(field_name: str, transform_func: Callable[[BBox], BBox]) -> GroupFunction:
    """
    Returns a function that applies a custom transformation to all bboxes in a field.
    
    Args:
        field_name: Name of the field containing bboxes
        transform_func: Function that takes a bbox and returns a transformed bbox
    
    Returns:
        Function that takes a group and returns transformed bboxes
    """
    def _transform(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return []
        
        return [transform_func(bbox) for bbox in bboxes]
    
    return _transform


def expand_bboxes(field_name: str, margin: float) -> GroupFunction:
    """
    Returns a function that expands all bboxes in a field by a margin.
    
    Args:
        field_name: Name of the field containing bboxes
        margin: Margin to expand in all directions
    
    Returns:
        Function that takes a group and returns expanded bboxes
    """
    def _expand(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return []
        
        return [Geometry.expand_bbox(bbox, margin) for bbox in bboxes]
    
    return _expand

def contract_bboxes(field_name: str, margin: float) -> GroupFunction:
    """
    Returns a function that contracts all bboxes in a field by a margin.
    
    Args:
        field_name: Name of the field containing bboxes
        margin: Margin to contract in all directions
    
    Returns:
        Function that takes a group and returns contracted bboxes
    """
    def _contract(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return []
        
        return [Geometry.contract_bbox(bbox, margin) for bbox in bboxes]
    
    return _contract

def bbox_centers(field_name: str) -> GroupFunction:
    """
    Returns a function that calculates center points for all bboxes in a field.
    
    Args:
        field_name: Name of the field containing bboxes
    
    Returns:
        Function that takes a group and returns list of center points
    """
    def _centers(group: 'GroupbyQueryResult') -> List[Tuple[float, float]]:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return []
        
        return [Geometry.bbox_center(bbox) for bbox in bboxes]
    
    return _centers


def sort_bboxes(field_name: str, sort_by: str = 'top_left') -> GroupFunction:
    """
    Returns a function that sorts bboxes by position or size.
    
    Args:
        field_name: Name of the field containing bboxes
        sort_by: Sorting method - 'top_left', 'center', 'area', 'width', 'height'
    
    Returns:
        Function that takes a group and returns sorted bboxes
    """
    def _sort(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return []
        
        return Geometry.sort_bboxes_by_position(bboxes, sort_by)
    
    return _sort

def absolute_to_relative_bboxes(field_name: str, page_bounds_field: str) -> GroupFunction:
    """
    Returns a function that converts absolute bboxes to page-relative coordinates.
    
    Args:
        field_name: Name of the field containing bboxes
        page_bounds_field: Name of the field containing page bounds object
    
    Returns:
        Function that takes a group and returns relative bboxes
    """
    def _to_relative(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        page_bounds = getattr(group, page_bounds_field, None)
        
        if not bboxes or not page_bounds:
            return []
        
        return [CoordinateMapping.absolute_to_relative(bbox, page_bounds) for bbox in bboxes]
    
    return _to_relative

def relative_to_absolute_bboxes(field_name: str, page_bounds_field: str) -> GroupFunction:
    """
    Returns a function that converts relative bboxes to absolute coordinates.
    
    Args:
        field_name: Name of the field containing bboxes
        page_bounds_field: Name of the field containing page bounds object
    
    Returns:
        Function that takes a group and returns absolute bboxes
    """
    def _to_absolute(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        page_bounds = getattr(group, page_bounds_field, None)
        
        if not bboxes or not page_bounds:
            return []
        
        return [CoordinateMapping.relative_to_absolute(bbox, page_bounds) for bbox in bboxes]
    
    return _to_absolute

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

class BaseChain:
    """
    An empty base class to mark an object as a 'chain' that needs
    to be executed to produce a final result.
    """
    # The abstract method now correctly returns a ChainResult
    async def as_chain_result(self) -> ChainResult:
        raise NotImplementedError

class GroupChain(BaseChain):
    """Chainable processor for GroupbyQueryResult objects with pandas-style API."""
    
    def __init__(self, data: List['GroupbyQueryResult']):
        self._data = data
        self._result_data = None  # Will hold processed dictionaries
        
    def include(self, fields: Union[str, List[str]]) -> 'GroupChain':
        """
        Include fields from the original group objects.
        
        Args:
            fields: Field name(s) to include from group objects
            
        Returns:
            New GroupChain instance for continued chaining
        """
        if isinstance(fields, str):
            fields = [fields]
            
        # Initialize result data if not already done
        if self._result_data is None:
            self._result_data = [{'group_id': group.group_id} for group in self._data]
        
        # Create new chain with copied data
        new_chain = GroupChain(self._data)
        new_chain._result_data = deepcopy(self._result_data)
        
        # Add requested fields
        for i, group in enumerate(self._data):
            for field_name in fields:
                if hasattr(group, field_name):
                    new_chain._result_data[i][field_name] = getattr(group, field_name)
                    
        return new_chain
    
    def agg(self, aggregations: Dict[str, Callable]) -> 'GroupChain':
        """
        Apply aggregation functions to compute new fields.
        
        Args:
            aggregations: Dictionary of {field_name: aggregation_function}
            
        Returns:
            New GroupChain instance for continued chaining
        """
        # Initialize result data if not already done
        if self._result_data is None:
            self._result_data = [{'group_id': group.group_id} for group in self._data]
            
        # Create new chain with copied data
        new_chain = GroupChain(self._data)
        new_chain._result_data = deepcopy(self._result_data)
        
        # Apply aggregations
        for i, group in enumerate(self._data):
            for field_name, agg_func in aggregations.items():
                new_chain._result_data[i][field_name] = agg_func(group)
                
        return new_chain
    
    def filter(self, condition: Callable[[Dict[str, Any]], bool]) -> 'GroupChain':
        """
        Filter groups based on a condition function.
        
        Args:
            condition: Function that takes a result dict and returns bool
            
        Returns:
            New GroupChain instance for continued chaining
        """
        if self._result_data is None:
            raise ValueError("Must call include() or agg() before filter()")
            
        # Create new chain with filtered data
        new_chain = GroupChain([])
        new_chain._result_data = []
        
        filtered_groups = []
        for i, result_dict in enumerate(self._result_data):
            if condition(result_dict):
                new_chain._result_data.append(deepcopy(result_dict))
                filtered_groups.append(self._data[i])
                
        new_chain._data = filtered_groups
        return new_chain
    
    def query(self, condition: str) -> 'GroupChain':
        """
        Filter using a pandas-style query string (simplified version).
        
        Args:
            condition: Query condition as string (e.g., "merged_bbox_rel_y < 100")
            
        Returns:
            New GroupChain instance for continued chaining
        """
        # This is a simplified implementation - you could use pandas.eval for full functionality
        def eval_condition(row_dict):
            # Create a safe evaluation context with the row data
            local_vars = row_dict.copy()
            try:
                return eval(condition, {"__builtins__": {}}, local_vars)
            except:
                return False
                
        return self.filter(eval_condition)
    
    def assign(self, **kwargs) -> 'GroupChain':
        """
        Create new columns using keyword arguments (pandas-style).
        
        Args:
            **kwargs: Column_name=function pairs
            
        Returns:
            New GroupChain instance for continued chaining
        """
        return self.agg(kwargs)
    
    def to_list(self) -> List[Dict[str, Any]]:
        """
        Return the processed data as a list of dictionaries.
        
        Returns:
            List of processed group dictionaries
        """
        if self._result_data is None:
            # If no processing was done, return basic group info
            return [{'group_id': group.group_id} for group in self._data]
        return self._result_data.copy()
    
    def to_dict(self, orient: str = 'records') -> Union[List[Dict], Dict[str, List]]:
        """
        Convert to dictionary in various formats (pandas-style).
        
        Args:
            orient: 'records' (list of dicts) or 'dict' (dict of lists)
            
        Returns:
            Data in requested format
        """
        data = self.to_list()
        
        if orient == 'records':
            return data
        elif orient == 'dict':
            if not data:
                return {}
            result = {}
            for key in data[0].keys():
                result[key] = [row.get(key) for row in data]
            return result
        else:
            raise ValueError(f"Unsupported orient: {orient}")
    
    def head(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return first n results (pandas-style)."""
        return self.to_list()[:n]

    async def as_chain_result(self) -> ChainResult[Dict[str, Any]]:
        """
        Executes the chain and wraps the resulting list of dictionaries
        in a Chainable container, preserving the fluent interface.
        """
        result_list = self.to_list()
        return ChainResult(result_list)
    
    def __len__(self) -> int:
        """Return number of groups after filtering."""
        if self._result_data is None:
            return len(self._data)
        return len(self._result_data)
    
    def __repr__(self) -> str:
        """String representation showing current state."""
        return f"GroupChain({len(self)} groups)"


# --- Updated GroupAccessor with chaining support ---
class GroupAccessor(BaseAccessor['GroupbyQueryResult']):
    
    @property
    def chain(self) -> GroupChain:
        """Access the chainable interface."""
        return GroupChain(self._data)
    
    def process(
        self,
        aggregations: Optional[Dict[str, Callable]] = None,
        filters: Optional[List[Callable]] = None,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Legacy method for backward compatibility.
        Consider using the chain interface instead.
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
        self.description = description or f"Normalizes text using a set of regex substitutions. Used to build index items with the key=[{output_key}]."

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
        self.description = description or f"Detects vertical whitespace regions spanning the page width. Used to build index items with the key=[{output_key}]."

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
