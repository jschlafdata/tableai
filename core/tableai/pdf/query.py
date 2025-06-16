import itertools
from collections import defaultdict, UserList
from typing import List, Tuple, Any, Optional, Tuple, Dict, Union, Callable
import re
import fnmatch
import json
import fitz
from tableai.pdf.coordinates import (
    Geometry,
    CoordinateMapping
)
from typing import Optional, List, Tuple, Union, Dict, Any, Callable, TYPE_CHECKING, TypeVar, Generic, Type
from pydantic import BaseModel, field_validator, model_validator, ValidationError, Field, create_model, field_serializer

from tableai.pdf.generic_models import (
    DefaultQueryResult, 
    GroupbyQueryResult, 
    ResultSet
)

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

class QueryParams(BaseModel):
    """
    A comprehensive, self-documenting model for all parameters
    used by the LineTextIndex.query method.
    """
    class Config:
        # Pydantic needs this to allow non-serializable types like functions.
        arbitrary_types_allowed = True

    # --- Basic Filters ---
    page: Optional[int] = Field(
        default=None,
        description="Filter results to a single virtual page number."
    )
    line: Optional[int] = Field(
        default=None,
        description="Filter results to a single line number within a block."
    )
    key: Optional[str] = Field(
        default=None,
        description="Filter for rows with this specific key. Common keys are 'text', 'normalized_text', and 'full_width_v_whitespace'."
    )

    # --- Spatial & Custom Filters ---
    exclude_bounds: Optional[str] = Field(
        default=None,
        description="The string key of a pre-defined boundary set in the LineTextIndex's restriction_store. Any row whose bbox overlaps with these zones will be EXCLUDED from the results."
    )
    bounds_filter: Optional[Callable[[dict], bool]] = Field(
        default=None,
        description="A custom, dynamic filter function (e.g., a lambda) that receives a complete row dictionary and must return True to keep it. This is applied AFTER the 'exclude_bounds' filter, making it ideal for fine-grained spatial logic (e.g., checking x_span, y0_rel, etc.)."
    )

    groupby: Optional[GroupbyTransform] = Field(
        default=None,
        description="A GroupbyTransform object that defines how to group the filtered results."
    )

    # --- Post-Processing & Metadata ---
    transform: Optional[Callable[[list], list]] = Field(
        default=None,
        description="A function that takes the entire list of filtered results and reshapes it. Primarily used for grouping operations like the `groupby()` transform."
    )
    query_label: Optional[str] = Field(
        default=None,
        description="An optional label to attach to each result object for tracking and identification purposes."
    )
    
    @field_serializer('groupby', 'transform', 'bounds_filter', when_used='json-unless-none')
    def serialize_special_types(self, value: Any) -> Any:
        """Intelligently serializes special types for readable logs."""
        if isinstance(value, GroupbyTransform):
            return value.to_dict() # Use our new descriptive method
        
        if callable(value):
            func_name = getattr(value, '__name__', '<lambda>')
            if hasattr(value, 'func'):
                func_name = f"partial({getattr(value.func, '__name__', 'unknown')})"
            return f"<function: {func_name}>"
            
        return value

class LineTextIndex:
    """Improved LineTextIndex that leverages VirtualPageManager for coordinate handling."""
    
    FONT_ATTRS = ['size', 'flags', 'bidi', 'char_flags', 'font', 'color', 'alpha']

    def __init__(self, 
                 data: List[Tuple[int, str, Any, Dict[str, Any]]], 
                 page_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
                 vpm: Optional['VirtualPageManager'] = None,
                 text_normalizer: Optional['TextNormalizer'] = None, 
                 whitespace_generator: Optional['WhitespaceGenerator'] = None,
                 **kwargs):
        """Initialize with VirtualPageManager for coordinate operations."""
        
        # Convert input format to internal format
        self.raw = [(row[0], i, row[1], row[2], row[3]) for i, row in enumerate(data)]
        self.page_metadata = page_metadata or {}
        
        # NEW: Use the provided VirtualPageManager instead of recreating logic
        self.vpm = vpm
        if not self.vpm:
            raise ValueError("VirtualPageManager is required for proper coordinate handling")
        
        # # Initialize text processing components
        self.text_normalizer = text_normalizer
        self.whitespace_generator = whitespace_generator
        
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
        
        # Build the index
        self._build_index()

    @classmethod
    def from_pdf_model(
            cls, 
            pdf_model: 'PDFModel', 
            text_normalizer: 'TextNormalizer', 
            whitespace_generator: 'WhitespaceGenerator', 
            **kwargs
        ):
        """
        Create LineTextIndex from a PDFModel, reusing its VirtualPageManager.
        
        This is the preferred initialization method as it avoids duplicating
        virtual page logic and ensures consistency with the PDF model.
        """
        if not pdf_model.doc:
            raise ValueError("PDFModel document not loaded")
        
        if len(pdf_model.doc) != 1:
            raise ValueError(f"Expected combined document with 1 page, got {len(pdf_model.doc)} pages")
        
        if not pdf_model.vpm:
            raise ValueError("PDFModel must have initialized VirtualPageManager")
            
        page = pdf_model.doc[0]
        text_dict = page.get_text("dict")
        flattened_data = cls.flatten_fitz_dict(text_dict, page_num=0)
        
        page_metadata = {0: {"width": page.rect.width, "height": page.rect.height}}
            
        return cls(
            data=flattened_data,
            page_metadata=page_metadata,
            vpm=pdf_model.vpm,
            text_normalizer=text_normalizer, 
            whitespace_generator=whitespace_generator,
            **kwargs
        )

    # LEGACY: Keep old method name for backward compatibility  
    def _get_virtual_page_num(self, y0: float) -> int:
        """Legacy method - use self.vpm.get_virtual_page_number() instead."""
        return self.vpm.get_virtual_page_number(y0)

    def get_virtual_page_coords(self, bbox: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float, float, float], int]:
        """Legacy method - use self.vpm.bbox_to_virtual_page_coords() instead."""
        page_bounds, relative_coords = self.vpm.bbox_to_virtual_page_coords(bbox)
        return page_bounds.tuple, page_bounds.page_number

    def get_virtual_page_wh(self, bbox: Tuple[float, float, float, float]) -> Dict[str, Union[int, float]]:
        """Legacy method - use self.vpm directly instead.""" 
        page_bounds, _ = self.vpm.bbox_to_virtual_page_coords(bbox)
        return {
            'page_number': page_bounds.page_number, 
            'page_width': page_bounds.width, 
            'page_height': page_bounds.height
        }

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

    def _build_index(self):
        """
        Build indexes using VirtualPageManager for all coordinate operations.
        Much cleaner now that coordinate logic is centralized.
        """
        # Pass 1: Build hierarchical index (unchanged)
        for page_num, idx, path, value, font_meta_dict in self.raw:
            parsed_path = self._parse_path_for_build(path)
            if not parsed_path: 
                continue
            block, line, span, key = parsed_path
            span_data = self.structured_index[page_num][block][line].setdefault(span, {})
            span_data[key] = value
            if 'font_meta' not in span_data:
                span_data['font_meta'] = font_meta_dict.get("font_meta")
    
        # Pass 2: Build flat index using VirtualPageManager
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
                        
                        # UPDATED: Handle bbox using VirtualPageManager and CoordinateMapping
                        if "bbox" in span_data:
                            bbox = tuple(span_data["bbox"])
                            
                            # Use VPM directly for virtual page number
                            virtual_page = self.vpm.get_virtual_page_number(bbox[1])
                            page_bounds = self.vpm.get_page_bounds(virtual_page)
                            
                            base_row.update({
                                "page": virtual_page,
                                "physical_page": physical_page_num,
                                "bbox": bbox, 
                                "x0": bbox[0], "y0": bbox[1], 
                                "x1": bbox[2], "y1": bbox[3],
                                "x_span": bbox[2] - bbox[0], 
                                "y_span": bbox[3] - bbox[1],
                                # Use VPM directly for region calculation
                                "region": self.vpm.get_region_in_page(bbox)
                            })
                            
                            # UPDATED: Use CoordinateMapping for relative coordinates
                            if page_bounds:
                                relative_coords = CoordinateMapping.absolute_to_relative(bbox, page_bounds)
                                
                                # Get original page dimensions if available
                                original_dims = self.vpm.metadata.get("original_page_dims")
                                physical_page_bounds = None
                                if original_dims and physical_page_num < len(original_dims):
                                    physical_page_bounds = original_dims[physical_page_num]
                                
                                base_row.update({
                                    "x0(rel)": relative_coords[0],
                                    "y0(rel)": relative_coords[1], 
                                    "x1(rel)": relative_coords[2],
                                    "y1(rel)": relative_coords[3],
                                    "bbox(rel)": relative_coords, 
                                    "page_height(rel)": page_bounds.height,
                                    "page_width(rel)": page_bounds.width,
                                    "physical_page_bounds": physical_page_bounds
                                })
                        else:
                            # For items without bbox, assign to virtual page 0
                            base_row["page"] = 0
    
                        # Handle text normalization (unchanged)
                        if "text" in span_data and isinstance(span_data["text"], str):
                            base_row["normalized_value"] = self.text_normalizer(span_data["text"])

                        # Create flat rows for each key-value pair (unchanged)
                        for key, value in span_data.items():
                            if key == 'font_meta': 
                                continue
                            flat_row = base_row.copy()
                            flat_row['key'] = key
                            flat_row['value'] = value
                            flat_row['path'] = f"blocks[{block_num}].lines[{line_num}].spans[{span_num}].{key}"
                            all_flat_rows.append(flat_row)

                        # Add normalized text row (unchanged)
                        if base_row.get('normalized_value'):
                            unique_idx_counter += 1
                            normalized_row = base_row.copy()
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
    
        # Add whitespace gaps (unchanged)
        fw_gaps = self.whitespace_generator(self.by_page, self.page_metadata)
        for gap in fw_gaps:
            gap.setdefault("key", "full_width_v_whitespace")
            gap.setdefault("font_meta", None)
            gap.setdefault("physical_page_bounds", None)
            self.index.append(gap)
            self.by_page[gap["page"]].append(gap)

    def _parse_path_for_build(self, path: str) -> Optional[Tuple[int, int, int, str]]:
        """Parse path for building the hierarchical index."""
        match = re.match(r"blocks\[(\d+)\]\.lines\[(\d+)\]\.spans\[(\d+)\]\.(\w+)", path)
        if not match:
            return None
        groups = match.groups()
        return (int(groups[0]), int(groups[1]), int(groups[2]), groups[3])

    # UPDATED: Enhanced query method with VPM-based filtering
    def query(self, 
              params: Optional['QueryParams'] = None, 
              **kwargs) -> 'ResultSet':
        """Execute a query with enhanced virtual page filtering capabilities."""
        
        p = params or QueryParams()
        final_params = p.model_copy(update=kwargs)
    
        # Get exclusion zones if a key is provided
        exclusion_bboxes = []
        if final_params.exclude_bounds:
            exclusion_bboxes = self.get_bound_restriction(final_params.exclude_bounds)
            if exclusion_bboxes is None:
                raise KeyError(f"Exclusion bounds key '{final_params.exclude_bounds}' not found.")
    
        # UPDATED: Add page limit filtering using VPM directly
        if hasattr(final_params, 'page_limit') and final_params.page_limit is not None:
            source = []
            for row in (self.by_page.get(final_params.page, self.index) if final_params.page is not None else self.index):
                if 'bbox' in row and row['bbox']:
                    if self.vpm.get_virtual_page_number(row['bbox'][1]) > final_params.page_limit:
                        continue
                source.append(row)
        else:
            source = self.by_page.get(final_params.page, self.index) if final_params.page is not None else self.index
        
        result = []
        for row in source:
            # Apply standard filters
            if final_params.key is not None and row.get("key") != final_params.key:
                continue
            if final_params.line is not None and row.get("line") != final_params.line:
                continue
            
            # Apply exclusion zones
            if exclusion_bboxes:
                bbox = row.get("bbox")
                if bbox and any(Geometry.is_overlapping(bbox, ex_box) for ex_box in exclusion_bboxes):
                    continue
    
            # Apply custom bounds filter
            if final_params.bounds_filter and not final_params.bounds_filter(row):
                continue
    
            result_row = dict(row)
            if final_params.query_label:
                result_row["query_label"] = final_params.query_label
            result.append(result_row)

        # Apply groupby and transform (unchanged)
        processed_result = result
        if final_params.groupby:
            processed_result = final_params.groupby(result)
    
        if final_params.transform:
            processed_result = final_params.transform(processed_result)
    
        # Return results (unchanged)
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
