import itertools
from collections import defaultdict, UserList
from typing import List, Tuple, Any, Optional, Tuple, Dict, Union, Callable
import re
import fnmatch
import json
import fitz
from tableai.pdf.coordinates import Map
from typing import Optional, List, Tuple, Union, Dict, Any, Callable, TYPE_CHECKING, TypeVar, Generic, Type
from pydantic import BaseModel, field_validator, model_validator, ValidationError, Field, create_model, field_serializer


# Define the generic Type Variable. 'T' will stand for whatever type
# the ResultSet holds (e.g., DefaultQueryResult).
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
            
        return Map.merge_all_boxes(bboxes_to_merge)

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

class LineTextIndex:
    """Improved LineTextIndex with proper virtual page mapping."""
    
    FONT_ATTRS = ['size', 'flags', 'bidi', 'char_flags', 'font', 'color', 'alpha']

    def __init__(self, data: List[Tuple[int, str, Any, Dict[str, Any]]], 
                 page_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
                 virtual_page_metadata=None, **kwargs):
        """Initialize with proper virtual page support."""
        # Convert input format to internal format
        self.raw = [(row[0], i, row[1], row[2], row[3]) for i, row in enumerate(data)]
        
        self.page_metadata = page_metadata or {}
        self.virtual_page_metadata = virtual_page_metadata or {}
        
        # Build virtual page lookup for efficient y-coordinate to page mapping
        self._virtual_breaks = []
        if self.virtual_page_metadata and "page_breaks" in self.virtual_page_metadata:
            self._virtual_breaks = sorted(self.virtual_page_metadata["page_breaks"])

        self.text_normalizer = kwargs.get('text_normalizer', 
            TextNormalizer(patterns={
                r'page\s*\d+\s*of\s*\d+': 'page xx of xx',
                r'page\s*\d+': 'page xx'
            }, 
            description='Normalizes text using a set of regex substitutions. This is used in the index to create index items with the key=[]')
        )
        self.whitespace_generator = kwargs.get('whitespace_generator', 
            WhitespaceGenerator(min_gap=5.0)
        )
        
        # Initialize data structures
        self.index: List[Dict[str, Any]] = []
        self.structured_index: Dict[int, Dict[int, Dict[int, Dict[int, Dict]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )
        self.by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.page_text_bounds = defaultdict(lambda: {
            "min_x": float("inf"), "min_y": float("inf"),
            "max_x": float("-inf"), "max_y": float("-inf")
        })
        self.restriction_store: Dict[str, List[Tuple[float, ...]]] = {}
        self._build_index()

    def _get_virtual_page_num(self, y0: float) -> int:
        """Efficiently determines virtual page number for a y-coordinate."""
        if not self._virtual_breaks:
            return 0
        
        # Binary search for the correct virtual page
        left, right = 0, len(self._virtual_breaks) - 1
        result_page = 0
        
        while left <= right:
            mid = (left + right) // 2
            y_start, page_num = self._virtual_breaks[mid]
            
            if y0 >= y_start:
                result_page = page_num
                left = mid + 1
            else:
                right = mid - 1
        
        return result_page

    def get_virtual_page_coords(self, bbox):
        y0 = bbox[1]
        virtual_page = self._get_virtual_page_num(y0)
        x0, y0, page_width, y1 = self.virtual_page_metadata['page_bounds'][virtual_page]
        bounds = [x0, y0, page_width, y1]
        return bounds, virtual_page

    def get_virtual_page_wh(self, bbox):
        bounds, virtual_page = self.get_virtual_page_coords(bbox)
        x0, y0, page_width, y1 = bounds
        page_height = y1-y0
        return {'page_number':virtual_page, 'page_width': page_width, 'page_height': page_height}
    
    @staticmethod
    def flatten_fitz_dict(data, page_num: int, parent_key='', sep='.', result=None, 
                         parent_dict=None, inherited_font_meta=None):
        """Flattens a fitz text dict with font metadata."""
        FONT_ATTRS = LineTextIndex.FONT_ATTRS
        if result is None:
            result = []
        
        font_meta = None
        if isinstance(data, dict):
            if set(FONT_ATTRS).issubset(data.keys()):
                font_meta = {attr: data.get(attr) for attr in FONT_ATTRS}
            else:
                font_meta = inherited_font_meta
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                LineTextIndex.flatten_fitz_dict(
                    v, page_num, new_key, sep, result, 
                    parent_dict=data, inherited_font_meta=font_meta
                )
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_key = f"{parent_key}[{i}]"
                LineTextIndex.flatten_fitz_dict(
                    item, page_num, new_key, sep, result,
                    parent_dict=parent_dict, inherited_font_meta=inherited_font_meta
                )
        else:
            result.append((page_num, parent_key, data, {'font_meta': inherited_font_meta}))
        return result
    
    @classmethod
    def from_document(cls, doc: fitz.Document, virtual_page_metadata: Optional[dict] = None, **kwargs):
        """Creates a LineTextIndex from a combined document with virtual page awareness."""
        if len(doc) != 1:
            raise ValueError(f"Expected combined document with 1 page, got {len(doc)} pages")
            
        page = doc[0]
        text_dict = page.get_text("dict")
        flattened_data = cls.flatten_fitz_dict(text_dict, page_num=0)
        
        page_metadata = {0: {"width": page.rect.width, "height": page.rect.height}}
            
        return cls(
            data=flattened_data,
            page_metadata=page_metadata,
            virtual_page_metadata=virtual_page_metadata, 
            **kwargs
        )

    def _build_index(self):
        """
        Builds the indexes, adding virtual page information and calculating
        region based on VIRTUAL page bounds. Now includes physical_page_bounds
        for easier header/footer detection.
        """
        # --- Pass 1: Build hierarchical index ---
        for page_num, idx, path, value, font_meta_dict in self.raw:
            parsed_path = self._parse_path_for_build(path)
            if not parsed_path: 
                continue
            block, line, span, key = parsed_path
            span_data = self.structured_index[page_num][block][line].setdefault(span, {})
            span_data[key] = value
            if 'font_meta' not in span_data:
                span_data['font_meta'] = font_meta_dict.get("font_meta")
    
        # --- Pass 2: Build flat index with virtual page mapping and physical bounds ---
        all_flat_rows = []
        unique_idx_counter = 0
        for physical_page_num, blocks in self.structured_index.items():
            for block_num, lines in blocks.items():
                for line_num, spans in lines.items():
                    for span_num, span_data in spans.items():
                        base_row = {
                            "physical_page": physical_page_num,
                            "block": block_num, 
                            "line": line_num, 
                            "span": span_num,
                            "font_meta": span_data.get("font_meta"),
                            "index": unique_idx_counter
                        }
                        unique_idx_counter += 1
                        # Handle bbox and virtual page assignment
                        if "bbox" in span_data:
                            bbox = tuple(span_data["bbox"])
                            virtual_page = self._get_virtual_page_num(bbox[1])
                            
                            base_row.update({
                                "page": virtual_page,  # Virtual page number
                                "physical_page": physical_page_num,
                                "bbox": bbox, 
                                "x0": bbox[0], "y0": bbox[1], 
                                "x1": bbox[2], "y1": bbox[3],
                                "x_span": bbox[2] - bbox[0], 
                                "y_span": bbox[3] - bbox[1]
                            })
                            
                            # Calculate virtual page bounds and physical page bounds
                            if self.virtual_page_metadata:
                                vp_bounds = self.virtual_page_metadata["page_bounds"].get(virtual_page)
                                if vp_bounds:

                                    original_dims = self.virtual_page_metadata.get("original_page_dims")
                                    physical_page_bounds=None
                                    if original_dims and physical_page_num < len(original_dims):
                                        physical_page_bounds = original_dims[physical_page_num]
                                    
                                    vp_x0, vp_y0, vp_x1, vp_y1 = vp_bounds
                                    vp_height = vp_y1 - vp_y0
                                    
                                    # Calculate midpoint relative to virtual page for region assignment
                                    mid_y_absolute = (bbox[1] + bbox[3]) / 2.0
                                    mid_y_relative_to_vp = mid_y_absolute - vp_y0
                                    
                                    # Assign region based on position within virtual page
                                    base_row["region"] = "header" if mid_y_relative_to_vp < (vp_height / 2) else "footer"
                                    x0_rel = bbox[0] - vp_x0 # Relative to virtual page left edge
                                    y0_rel = bbox[1] - vp_y0 # Relative to virtual page top edge 
                                    x1_rel = bbox[2] - vp_x0 # Relative to virtual page left edge
                                    y1_rel = bbox[3] - vp_y0 # Relative to virtual page top edge

                                    base_row.update({
                                        "x0(rel)": x0_rel,
                                        "y0(rel)": y0_rel, 
                                        "x1(rel)": x1_rel,
                                        "y1(rel)": y1_rel,
                                        "bbox(rel)": (x0_rel, y0_rel, x1_rel, y1_rel), 
                                        "page_height(rel)": vp_height,
                                        "page_width(rel)": vp_x1 - vp_x0,
                                        "physical_page_bounds": physical_page_bounds
                                    })
                        else:
                            # For items without bbox, assign to virtual page 0
                            base_row["page"] = 0
    
                        # Handle text normalization
                        if "text" in span_data and isinstance(span_data["text"], str):
                            base_row["normalized_value"] = self.text_normalizer(span_data["text"])

                        # Create flat rows for each key-value pair
                        for key, value in span_data.items():
                            if key == 'font_meta': 
                                continue
                            flat_row = base_row.copy()
                            flat_row['key'] = key
                            flat_row['value'] = value
                            flat_row['path'] = f"blocks[{block_num}].lines[{line_num}].spans[{span_num}].{key}"
                            all_flat_rows.append(flat_row)

                        
                        normalized_row = base_row.copy()
                        if normalized_row.get('normalized_value', None):
                            unique_idx_counter += 1
                            normalized_row.update({
                                "key": "normalized_text", 
                                "value": base_row['normalized_value'],
                                "normalized_value": base_row['normalized_value'],
                                "path": f"blocks[{block_num}].lines[{line_num}]",
                                "index": unique_idx_counter
                            })
                            all_flat_rows.append(normalized_row)
    
        self.index = all_flat_rows
        
        # Group by virtual page number
        for row in self.index:
            self.by_page[row['page']].append(row)
    
        # Add whitespace gaps
        fw_gaps = self.whitespace_generator(self.by_page, self.page_metadata)
        for gap in fw_gaps:
            gap.setdefault("key", "full_width_v_whitespace")
            gap.setdefault("font_meta", None)
            gap.setdefault("physical_page_bounds", None)  # Add this for consistency
            self.index.append(gap)
            self.by_page[gap["page"]].append(gap)

    def _parse_path_for_build(self, path: str) -> Optional[Tuple[int, int, int, str]]:
        """Parse path for building the hierarchical index."""
        match = re.match(r"blocks\[(\d+)\]\.lines\[(\d+)\]\.spans\[(\d+)\]\.(\w+)", path)
        if not match:
            return None
        groups = match.groups()
        return (int(groups[0]), int(groups[1]), int(groups[2]), groups[3])

    def query(
        self, 
        params: Optional[QueryParams] = None, 
        **kwargs
    ) -> 'ResultSet':
        """
        Executes a query against the index using a structured QueryParams object.
        """
        # 1. Self-initialize parameters: Start with the provided object or a default,
        #    then apply any kwargs as convenient overrides.
        p = params or QueryParams()
        final_params = p.model_copy(update=kwargs)
    
        # 2. Get exclusion zones if a key is provided
        exclusion_bboxes = []
        if final_params.exclude_bounds:
            exclusion_bboxes = self.get_bound_restriction(final_params.exclude_bounds)
            if exclusion_bboxes is None:
                raise KeyError(f"Exclusion bounds key '{final_params.exclude_bounds}' not found in the store.")
    
        # 3. Filter the data source
        source = self.by_page.get(final_params.page, self.index) if final_params.page is not None else self.index
        result = []
        for row in source:
            # Apply filters from the params object
            if final_params.key is not None and row.get("key") != final_params.key:
                continue
            if final_params.line is not None and row.get("line") != final_params.line:
                continue
            
            if exclusion_bboxes:
                bbox = row.get("bbox")
                if bbox and any(Map.is_overlapping(bbox, ex_box) for ex_box in exclusion_bboxes):
                    continue
    
            if final_params.bounds_filter and not final_params.bounds_filter(row):
                continue
    
            result_row = dict(row)
            if final_params.query_label:
                result_row["query_label"] = final_params.query_label
            result.append(result_row)

        processed_result = result
        if final_params.groupby:
            processed_result = final_params.groupby(result)
    
        if final_params.transform:
            processed_result = final_params.transform(processed_result)
    
        output_dicts = processed_result
        if not output_dicts:
            return ResultSet()
        
        first_item = output_dicts[0]
        if isinstance(first_item, dict) and "group_id" in first_item:
            items = [GroupbyQueryResult(**item) for item in output_dicts]
            return ResultSet[GroupbyQueryResult](items)
        elif isinstance(first_item, dict):
            items = [DefaultQueryResult.from_dict(item) for item in output_dicts]
            return ResultSet[DefaultQueryResult](items)
        else:
            return ResultSet(output_dicts)
    
    def add_bound_restriction(self, key: str, bounds: List[Tuple[float, ...]]):
        """
        Adds a named list of bounding boxes to the restriction store.
        These can be used later in queries.

        Args:
            key: A unique name for this set of restrictions (e.g., 'header_footer_bounds').
            bounds: A list of bounding box tuples [(x0,y0,x1,y1), ...].
        """
        if not isinstance(bounds, list):
            raise TypeError(f"Expected 'bounds' to be a list, but got {type(bounds)}.")
        self.restriction_store[key] = bounds

    def get_bound_restriction(self, key: str) -> Optional[List[Tuple[float, ...]]]:
        """Retrieves a named list of bounding boxes from the store."""
        return self.restriction_store.get(key)

    def clear_all_restrictions(self):
        """Clears the entire restriction store."""
        self.restriction_store = {}
        
    def remove_restriction(self, key: str):
        """Removes a single named restriction from the store."""
        if key in self.restriction_store:
            del self.restriction_store[key]


# Add this new class to your file
class GroupbyTransform:
    """A callable object that encapsulates the logic and parameters of a groupby operation."""
    def __init__(self, *keys, filterby=None, include=None, query_label=None, description=None):
        self.keys = keys
        self.group_id_field: str = "group_id"
        self.filterby = filterby or (lambda g: True)
        self.include = include if include is not None else ["bbox", "path", "text"]
        self.query_label = query_label
        self.description = description or f"Groups data by the following keys: {keys}"
        
        # --- Pre-define the mappings here for clarity ---
        self.AGGREGATE_MAPPING = {
            "page": "group_page",
            "block": "group_block",
            "line": "group_line",
            "span": "group_span",
            "font_meta": "group_font_meta",
            "bbox": "group_bboxes",
            "x_span": "group_x_spans",
            "y_span": "group_y_spans",
            "region": "group_regions",
            "normalized_value": "group_normalized_values",
            "path": "group_paths",
            "bbox(rel)": "group_bboxes_rel"
        }
        self.CONSISTENT_FIELDS = [
            "page",
            "index", 
            "region",
            "physical_page",
            "physical_page_bounds",
            "meta",
        ]

    def to_dict(self) -> dict:
        """Creates a human-readable dictionary representation for logging."""
        return {
            "type": "groupby",
            "keys": self.keys,
            "include": self.include,
            "filterby": getattr(self.filterby, '__name__', '<lambda>'),
            "description": self.description
        }

    def __call__(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """This makes instances of the class callable, just like a function."""
        # --- The entire logic from your old _transform function goes here ---
        grouped = defaultdict(list)
        for row in rows:
            key_tuple = tuple(row.get(k) for k in self.keys)
            grouped[key_tuple].append(row)
            
        output_summaries = []
        for group_key, group_rows in grouped.items():
            if not self.filterby(group_rows):
                continue
            
            first_member = group_rows[0]

            # 3. Create the base summary dictionary with consistent fields
            summary = {
                self.group_id_field: group_key,
                "groupby_keys": self.keys,
                "member_count": len(group_rows),
            }
            if self.query_label:
                summary['query_label'] = self.query_label
            
            # Copy over the consistent fields from the first member of the group
            for field in self.CONSISTENT_FIELDS:
                if field in first_member:
                    summary[field] = first_member[field]
            
            # Dynamically add the key-value pairs used for grouping (e.g., summary['region'] = 'header')
            for i, key_name in enumerate(self.keys):
                summary[key_name] = group_key[i]

            # 4. Initialize and populate aggregated lists
            
            # Initialize lists for all fields that will be aggregated
            for item_key in self.include:
                if item_key in self.AGGREGATE_MAPPING:
                    summary[self.AGGREGATE_MAPPING[item_key]] = []
                # Special handling for "text"
                elif item_key in ["text", "normalized_text"]:
                    summary["group_text"] = []
            
            # Iterate through rows once to populate the lists
            for row in group_rows:
                # Handle special "text" case
                if "text" in self.include and row.get("key") == "text":
                    summary["group_text"].append(row.get("value"))
                if "normalized_text" in self.include and row.get("key") == "normalized_text":
                    summary["group_text"].append(row.get("value"))
                
                # Handle standard aggregations
                for item_key in self.include:
                    if item_key in self.AGGREGATE_MAPPING and item_key in row:
                        dest_key = self.AGGREGATE_MAPPING[item_key]
                        summary[dest_key].append(row[item_key])
            
            output_summaries.append(summary)
            
        return output_summaries

def groupby(*keys, **kwargs) -> GroupbyTransform:
    """
    Factory function that creates a self-describing, callable GroupbyTransform object.
    
    All arguments (filterby, include, query_label, description) are passed
    directly to the GroupbyTransform constructor.
    """
    return GroupbyTransform(*keys, **kwargs)


def regroup_by_key(
    data: List[Dict[str, Any]], 
    key: str, 
    min_count: int,
    return_list: Optional[bool] = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Regroups a list of dictionaries by a specified key and filters for groups of a minimum size.

    This is useful for performing a second-level aggregation on data that has already
    been processed once.

    Args:
        data: A list of dictionaries to process (e.g., your 'processed_data').
        key: The dictionary key to group by (e.g., 'full_text').
        min_count: The minimum number of items a group must have to be included
                   in the final result.

    Returns:
        A dictionary where keys are the unique values from the grouping 'key',
        and values are lists of the original dictionaries that belong to that group.
    """
    # Step 1: Group all items by the specified key.
    # The defaultdict makes this easy: if a key doesn't exist, it creates a new list for it.
    groups = defaultdict(list)
    for item in data:
        if key in item:
            group_key = item[key]
            groups[group_key].append(item)

    # Step 2: Filter the groups, keeping only those with enough members.
    # A dictionary comprehension is a clean way to build the final result.
    if return_list:
        filtered_groups = [
            items for group_key, items in groups.items()
            if len(items) >= min_count
        ]
        if filtered_groups:
            return list(itertools.chain(*filtered_groups))
        else:
            return []
    else:
        # Return a dictionary containing only the groups that meet the min_count
        return {
            group_key: items
            for group_key, items in groups.items()
            if len(items) >= min_count
        }

# class FitzTextIndex:
#     FONT_ATTRS = ['size', 'flags', 'bidi', 'char_flags', 'font', 'color', 'alpha']

#     def __init__(self, flattened_data: List[Tuple[str, Any]]):
#         """
#         Initialize the index with a list of (page_num, path, value, font_meta_dict).
#         """
#         self.flattened_data = flattened_data
#         self.page_metadata = {}
#         self.unique_pages = set()
#         self._build_indices()

#     @staticmethod
#     def make_hashable(val):
#         if isinstance(val, (list, dict)):
#             return json.dumps(val, sort_keys=True)
#         return val

#     @staticmethod
#     def flatten_fitz_dict(data, page_num: int, parent_key='', sep='.', result=None, parent_dict=None, inherited_font_meta=None):
#         """
#         Flattens a fitz text dict and always attaches font_meta to each entry.
#         """
#         FONT_ATTRS = FitzTextIndex.FONT_ATTRS
#         if result is None:
#             result = []
#         # If this is a span dict, build font_meta once and inherit for all children
#         font_meta = None
#         if isinstance(data, dict):
#             if set(FONT_ATTRS).issubset(data.keys()):
#                 font_meta = {attr: data.get(attr) for attr in FONT_ATTRS}
#             else:
#                 font_meta = inherited_font_meta  # inherit from parent span, if any
#             for k, v in data.items():
#                 new_key = f"{parent_key}{sep}{k}" if parent_key else k
#                 FitzTextIndex.flatten_fitz_dict(
#                     v,
#                     page_num,
#                     new_key,
#                     sep,
#                     result,
#                     parent_dict=data,
#                     inherited_font_meta=font_meta
#                 )
#         elif isinstance(data, list):
#             for i, item in enumerate(data):
#                 new_key = f"{parent_key}[{i}]"
#                 FitzTextIndex.flatten_fitz_dict(
#                     item,
#                     page_num,
#                     new_key,
#                     sep,
#                     result,
#                     parent_dict=parent_dict,
#                     inherited_font_meta=inherited_font_meta
#                 )
#         else:
#             # Always attach font_meta, even if None
#             result.append((page_num, parent_key, data, {'font_meta': inherited_font_meta}))
#         return result

#     @classmethod
#     def from_document(cls, doc):
#         """
#         Build a FitzTextIndex from all pages in a fitz.Document.
#         """
#         instance = cls(flattened_data=[])
#         all_flattened = []
#         for page_num in range(len(doc)):
#             page = doc[page_num]
#             text_dict = page.get_text("dict")
#             flattened = cls.flatten_fitz_dict(text_dict, page_num)
#             all_flattened.extend(flattened)
#             instance.page_metadata[page_num] = {
#                 "width": page.rect.width,
#                 "height": page.rect.height
#             }
#             instance.unique_pages.add(page_num)
#         instance.flattened_data = all_flattened
#         instance._build_indices()
#         return instance

#     def _build_indices(self) -> None:
#         PAGE_STRIDE = 10_000
#         self.indices = defaultdict(lambda: defaultdict(set))
#         self.field_names = {}
#         self.parent_map = {}
#         self.grouped_by_parent = defaultdict(set)
#         self.page_map = {}
#         self.index_map = {}

#         for local_idx, row in enumerate(self.flattened_data):
#             if len(row) == 4:
#                 page_num, path, value, font_meta_dict = row
#             elif len(row) == 3:
#                 page_num, path, value = row
#                 font_meta_dict = {'font_meta': None}
#             else:
#                 raise ValueError("Unexpected tuple length in flattened_data")
#             global_idx = page_num * PAGE_STRIDE + local_idx
#             wildcard_path = re.sub(r'\[\d+\]', '[*]', path)
#             hashable_value = FitzTextIndex.make_hashable(value)
#             self.indices[path][hashable_value].add(global_idx)
#             self.indices[wildcard_path][hashable_value].add(global_idx)
#             self.page_map[global_idx] = page_num
#             self.index_map[global_idx] = (page_num, path, value, font_meta_dict)
#             self.field_names[global_idx] = path.split(".")[-1]
#             parent = ".".join(path.split(".")[:-1])
#             self.parent_map[global_idx] = parent
#             self.grouped_by_parent[parent].add(global_idx)
#             self.unique_pages.add(page_num)

#     def query(
#         self,
#         restrict: Optional[Union[str, List[str]]] = None,
#         page: Optional[Union[int, List[int]]] = None,
#         **kwargs
#     ) -> List[Tuple[int, int, str, Any]]:
#         if not restrict:
#             restrict=["*"]
#         elif restrict and "bbox" not in restrict:
#             if isinstance(restrict, str):
#                 restrict = [restrict] + ["bbox"]
#             else:
#                 restrict = restrict + ["bbox"]
#         multi_page_results=[]
#         if page in self.unique_pages:
#             page_search = self._query(restrict=restrict, page=page, **kwargs)
#             if page_search:
#                 multi_page_results.append(page_search)
#         else: 
#             multi_page_results=[]
#             for page in self.unique_pages:
#                 page_search = self._query(restrict=restrict, page=page, **kwargs)
#                 if page_search:
#                     multi_page_results.append(page_search)
#         if multi_page_results:
#             return list(itertools.chain.from_iterable(multi_page_results))
#         else:
#             return None

#     def _query(
#         self,
#         restrict: Optional[Union[str, List[str]]] = None,
#         page: Optional[Union[int, List[int]]] = None,
#         **kwargs
#     ) -> List[Tuple[int, int, str, Any]]:
#         if restrict:
#             restrict = [restrict] if isinstance(restrict, str) else restrict
#         page_filter = None
#         if page is not None:
#             page_filter = [page] if isinstance(page, int) else page
#         # Empty query returns all values, optionally filtered
#         if not kwargs:
#             return [
#                 (self.page_map[idx], idx, path, value, font_meta_dict)
#                 for idx, (pg, path, value, font_meta_dict) in self.index_map.items()
#                 if (restrict is None or self.field_names[idx] in restrict)
#                 and (page_filter is None or self.page_map[idx] in page_filter)
#             ]
#         matched_parent_sets: List[Set[str]] = []
#         for path_pattern, expected_value in kwargs.items():
#             matched_indices = set()
#             value_map = self.indices.get(path_pattern, {})
#             for val, idx_set in value_map.items():
#                 if expected_value == "*" or str(val) == str(expected_value):
#                     matched_indices.update(idx_set)
#             # Apply page filter before grouping
#             if page_filter:
#                 matched_indices = {i for i in matched_indices if self.page_map[i] in page_filter}
#             matched_parents = {self.parent_map[i] for i in matched_indices}
#             matched_parent_sets.append(matched_parents)
#         if not matched_parent_sets:
#             return []
#         # Intersect across all query conditions
#         common_parents = set.intersection(*matched_parent_sets)
#         final_indices = set()
#         for parent in common_parents:
#             final_indices.update(self.grouped_by_parent[parent])
#         # Apply restrict
#         if restrict:
#             final_indices = {
#                 i for i in final_indices
#                 if any(fnmatch.fnmatch(self.field_names[i], pattern) for pattern in restrict)
#             }
#         final_indices = {i for i in final_indices if self.page_map[i] in page_filter}
#         return [(self.page_map[i], i, self.index_map[i][1], self.index_map[i][2], self.index_map[i][3]) for i in sorted(final_indices)]

# # --- LineTextIndex ---

# class LineTextIndex:
#     def __init__(
#         self,
#         data: List[Tuple[int, int, str, Any, Dict[str, Any]]],
#         page_metadata: Optional[Dict[int, Dict[str, Any]]] = None
#     ):
#         """
#         Accepts a list of (page_num, index, path, value, font_meta_dict) and builds a searchable index.
#         Optionally accepts page_metadata: {page_num: {"width": ..., "height": ...}}
#         """
#         self.raw = data
#         self.page_metadata = page_metadata or {}
#         self.index: List[Dict[str, Any]] = []
#         self.by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
#         self.page_text_bounds = defaultdict(lambda: {
#             "min_x": float("inf"),
#             "min_y": float("inf"),
#             "max_x": float("-inf"),
#             "max_y": float("-inf")
#         })
#         self._bound_restrictions = {}
#         self._build_index()

#     def set_search_result_bound_restrictions(self, result):
#         restricted_bounds=defaultdict(list)
#         for pg_num, items in result['results']['pages'].items():
#             for res in items:
#                 bbox = res['bbox']
#                 restricted_bounds[pg_num].append(bbox)
#         self._set_bound_restrictions(restricted_bounds)
    
#     def _set_bound_restrictions(self, restrictions: Optional[Dict[int, List[List[float]]]]):
#         """
#         Sets bounding box restrictions by page.
#         Any query result overlapping a restriction will be filtered out.
#         """
#         self._bound_restrictions = restrictions or {}

#     def clear_bound_restrictions(self):
#         """Clears all bound restrictions."""
#         self._bound_restrictions = {}

#     def _build_index(self):
#         # Collect all bboxes (first pass)
#         text_bbox_map = {}
#         for row_tuple in self.raw:
#             page_num, idx, path, value, font_meta_dict = row_tuple
#             parsed = self._parse_path(path)
#             if parsed is None:
#                 continue
#             block, line, span, key = parsed
#             if key == "bbox" and isinstance(value, (list, tuple)) and len(value) == 4:
#                 group_key = (page_num, block, line, span)
#                 text_bbox_map[group_key] = value
#         # Second pass - build indexed rows (text, font, bbox, etc.)
#         for row_tuple in self.raw:
#             page_num, idx, path, value, font_meta_dict = row_tuple
#             parsed = self._parse_path(path)
#             if parsed is None:
#                 continue
#             block, line, span, key = parsed
#             group_key = (page_num, block, line, span)
#             row = {
#                 "page": page_num,
#                 "index": idx,
#                 "block": block,
#                 "line": line,
#                 "span": span,
#                 "key": key,
#                 "value": value,
#                 "path": path,
#                 "font_meta": font_meta_dict.get("font_meta") if font_meta_dict else None
#             }
#             if key == "text":
#                 bbox = text_bbox_map.get(group_key)
#                 if bbox:
#                     row["bbox"] = tuple(bbox)
#                     row["x0"], row["y0"], row["x1"], row["y1"] = bbox
#                     row["x_span"] = row["x1"] - row["x0"]
#                     row["y_span"] = row["y1"] - row["y0"]
#                     bounds = self.page_text_bounds[page_num]
#                     bounds["min_x"] = min(bounds["min_x"], row["x0"])
#                     bounds["min_y"] = min(bounds["min_y"], row["y0"])
#                     bounds["max_x"] = max(bounds["max_x"], row["x1"])
#                     bounds["max_y"] = max(bounds["max_y"], row["y1"])
#                 else:
#                     row["bbox"] = None
#                     row["x0"] = row["y0"] = row["x1"] = row["y1"] = None
#                 row["normalized_value"] = normalize_recurring_text(value) if isinstance(value, str) else None

#                 height = self.page_metadata.get(page_num, {}).get("height")
#                 if height is not None and row["y0"] is not None and row["y1"] is not None:
#                     mid_y = (row["y0"] + row["y1"]) / 2.0
#                     if mid_y < (height / 2):
#                         row["region"] = "header"
#                     else:
#                         row["region"] = "footer"
#                 else:
#                     row["region"] = None
            
#             self.index.append(row)
#             self.by_page[page_num].append(row)
#         # Add whitespace rows (must include "key" for querying)
#         fw_gaps = compute_full_width_v_whitespace(self.by_page, self.page_metadata, min_gap=5.0)
#         for gap in fw_gaps:
#             gap.setdefault("key", "full_width_v_whitespace")
#             gap.setdefault("font_meta", None)
#             self.index.append(gap)
#             self.by_page[gap["page"]].append(gap)

#     def _parse_path(self, path: str) -> Optional[Tuple[int, int, int, str]]:
#         """
#         Parse a path like: blocks[0].lines[2].spans[0].text → (0, 2, 0, 'text')
#         """
#         match = re.match(
#             r"blocks\[(\d+)\]\.lines\[(\d+)\]\.spans\[(\d+)\]\.(\w+)",
#             path
#         )
#         if not match:
#             return None
#         return tuple(map(lambda x: int(x) if x.isdigit() else x, match.groups()))

#     def query(
#         self,
#         page: Optional[int] = None,
#         line: Optional[int] = None,
#         key: Optional[str] = None,
#         func: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
#         bounds_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
#         transform: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
#         description: Optional[str] = None,
#         query_label: Optional[str] = None,
#     ) -> Union[List[Dict[str, Any]], Any]:
#         """
#         Query by page, line, key (e.g. 'text'), with optional function and bounding-box filtering.
#         Each result row will include the query_label if provided.
#         """
#         if description:
#             self._last_query_description = description

#         source = self.by_page.get(page, self.index) if page is not None else self.index

#         result = []
#         for row in source:
#             if line is not None and row["line"] != line:
#                 continue
#             if key is not None and row["key"] != key:
#                 continue

#             # Inject metadata
#             if self.page_metadata and row["page"] in self.page_metadata:
#                 row["meta"] = self.page_metadata[row["page"]]

#             if bounds_filter and not bounds_filter(row):
#                 continue

#             # FIXED: Bounds restriction logic
#             if self._bound_restrictions:
#                 # Convert page number to string for lookup (your data uses string keys)
#                 page_key = str(row["page"])
#                 page_restrictions = self._bound_restrictions.get(page_key, [])
                
#                 if page_restrictions:  # If there are restrictions for this page
#                     bbox = row.get("bbox")
#                     if bbox:
#                         # Check if this text overlaps with ANY of the restricted bounds
#                         overlaps_any_restriction = False
#                         for restriction in page_restrictions:
#                             if Map.is_overlapping(bbox, restriction):
#                                 overlaps_any_restriction = True
#                                 break
                        
#                         # SKIP if it does NOT overlap any restriction
#                         # (i.e., only keep text that IS within the restricted bounds)
#                         if not overlaps_any_restriction:
#                             continue
#                     else:
#                         # No bbox means we can't determine overlap, so exclude it
#                         continue
            
#             # Create a shallow copy to avoid modifying original index
#             result_row = dict(row)
#             if query_label:
#                 result_row["query_label"] = query_label

#             result.append(result_row)

#         # If transform is applied, try to propagate query_label
#         if transform:
#             output = transform(result)
#             if query_label and isinstance(output, list):
#                 for group in output:
#                     if isinstance(group, list):  # grouped result
#                         for item in group:
#                             item["query_label"] = query_label
#                     elif isinstance(group, dict):
#                         group["query_label"] = query_label
#             return output

#         if func:
#             return func(result)

#         return result
    

# def groupby(
#     *keys: str,
#     filterby: Callable[[List[Dict[str, Any]]], bool] = lambda g: True,
#     group_id_field: str = "group_id"
# ):
#     """
#     Groups rows by given keys, filters, and injects unique group_id (tuple of key values).
#     """
#     def _transform(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
#         from collections import defaultdict
#         grouped = defaultdict(list)
#         for row in rows:
#             key = tuple(row.get(k) for k in keys)
#             grouped[key].append(row)
#         output = []
#         for group_key, group in grouped.items():
#             if filterby(group):
#                 for row in group:
#                     row[group_id_field] = group_key  # Now a tuple: (region, page)
#                 output.append(group)
#         return output
#     return _transform


# def filterby(
#     func: Callable[[str], Any],
#     field: str = "value",
#     test: Callable[[Any], bool] = bool,   # Default to bool (truthy)
# ) -> Callable[[list], list]:
#     """
#     Generic filter for rows, using func to process the field and test to evaluate pass.
#     """
#     def _transform(rows: list) -> list:
#         return [
#             row for row in rows
#             if isinstance(row.get(field), str) and test(func(row[field]))
#         ]
#     return _transform


# def chain_transform(*funcs):
#     def _chained(rows):
#         result = rows
#         for f in funcs:
#             result = f(result)
#         return result
#     return _chained


