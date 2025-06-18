from collections import defaultdict
from typing import List, Tuple, Any, Optional, Tuple, Dict, Union, Callable
import re
from tableai.pdf.coordinates import (
    Geometry,
    CoordinateMapping
)
from typing import Optional, List, Tuple, Union, Dict, Any, Callable, TYPE_CHECKING, TypeVar, Generic, Type
from tableai.pdf.generic_results import (
    DefaultQueryResult, 
    GroupbyQueryResult, 
    ResultSet
)
from tableai.pdf.generic_params import (
    TextNormalizer, 
    WhitespaceGenerator, 
    QueryParams
)
from tableai.pdf.pdf_page import VirtualPageManager

class FitzSearchIndex:
    """Improved FitzSearchIndex that leverages VirtualPageManager for coordinate handling."""
    
    FONT_ATTRS = ['size', 'flags', 'bidi', 'char_flags', 'font', 'color', 'alpha']

    def __init__(self, 
                 data: List[Tuple[int, str, Any, Dict[str, Any]]], 
                 page_metadata: Optional[Dict[int, Dict[str, Any]]] = None,
                 vpm: Optional[VirtualPageManager] = None,
                 text_normalizer: Optional[TextNormalizer] = None, 
                 whitespace_generator: Optional[WhitespaceGenerator] = None,
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
            pdf_model, 
            text_normalizer: TextNormalizer, 
            whitespace_generator: WhitespaceGenerator, 
            **kwargs
        ):
        """
        Create FitzSearchIndex from a PDFModel, reusing its VirtualPageManager.
        
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
        FONT_ATTRS = FitzSearchIndex.FONT_ATTRS
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
                FitzSearchIndex.flatten_fitz_dict(
                    v, page_num, new_key, sep, result, 
                    parent_dict=data, inherited_font_meta=font_meta
                )
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_key = f"{parent_key}[{i}]"
                FitzSearchIndex.flatten_fitz_dict(
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
              params: Optional[QueryParams] = None, 
              **kwargs) -> ResultSet:
        """Execute a query with enhanced virtual page filtering capabilities."""
        
        p = params or QueryParams()
        final_params = p.model_copy(update=kwargs)
        print(f"final_params: {final_params}")
    
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