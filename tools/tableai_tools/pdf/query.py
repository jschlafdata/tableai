import itertools
from collections import defaultdict
from typing import List, Tuple, Any, Optional, Tuple, Dict, Union, Callable
import re
import fnmatch
import json
import fitz
from tableai_tools.pdf.coordinates import Map
from tableai_tools.pdf.query_funcs import (
    compute_full_width_v_whitespace, 
    normalize_recurring_text
)

class FitzTextIndex:
    FONT_ATTRS = ['size', 'flags', 'bidi', 'char_flags', 'font', 'color', 'alpha']

    def __init__(self, flattened_data: List[Tuple[str, Any]]):
        """
        Initialize the index with a list of (page_num, path, value, font_meta_dict).
        """
        self.flattened_data = flattened_data
        self.page_metadata = {}
        self.unique_pages = set()
        self._build_indices()

    @staticmethod
    def make_hashable(val):
        if isinstance(val, (list, dict)):
            return json.dumps(val, sort_keys=True)
        return val

    @staticmethod
    def flatten_fitz_dict(data, page_num: int, parent_key='', sep='.', result=None, parent_dict=None, inherited_font_meta=None):
        """
        Flattens a fitz text dict and always attaches font_meta to each entry.
        """
        FONT_ATTRS = FitzTextIndex.FONT_ATTRS
        if result is None:
            result = []
        # If this is a span dict, build font_meta once and inherit for all children
        font_meta = None
        if isinstance(data, dict):
            if set(FONT_ATTRS).issubset(data.keys()):
                font_meta = {attr: data.get(attr) for attr in FONT_ATTRS}
            else:
                font_meta = inherited_font_meta  # inherit from parent span, if any
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                FitzTextIndex.flatten_fitz_dict(
                    v,
                    page_num,
                    new_key,
                    sep,
                    result,
                    parent_dict=data,
                    inherited_font_meta=font_meta
                )
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_key = f"{parent_key}[{i}]"
                FitzTextIndex.flatten_fitz_dict(
                    item,
                    page_num,
                    new_key,
                    sep,
                    result,
                    parent_dict=parent_dict,
                    inherited_font_meta=inherited_font_meta
                )
        else:
            # Always attach font_meta, even if None
            result.append((page_num, parent_key, data, {'font_meta': inherited_font_meta}))
        return result

    @classmethod
    def from_document(cls, doc):
        """
        Build a FitzTextIndex from all pages in a fitz.Document.
        """
        instance = cls(flattened_data=[])
        all_flattened = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            flattened = cls.flatten_fitz_dict(text_dict, page_num)
            all_flattened.extend(flattened)
            instance.page_metadata[page_num] = {
                "width": page.rect.width,
                "height": page.rect.height
            }
            instance.unique_pages.add(page_num)
        instance.flattened_data = all_flattened
        instance._build_indices()
        return instance

    def _build_indices(self) -> None:
        PAGE_STRIDE = 10_000
        self.indices = defaultdict(lambda: defaultdict(set))
        self.field_names = {}
        self.parent_map = {}
        self.grouped_by_parent = defaultdict(set)
        self.page_map = {}
        self.index_map = {}

        for local_idx, row in enumerate(self.flattened_data):
            if len(row) == 4:
                page_num, path, value, font_meta_dict = row
            elif len(row) == 3:
                page_num, path, value = row
                font_meta_dict = {'font_meta': None}
            else:
                raise ValueError("Unexpected tuple length in flattened_data")
            global_idx = page_num * PAGE_STRIDE + local_idx
            wildcard_path = re.sub(r'\[\d+\]', '[*]', path)
            hashable_value = FitzTextIndex.make_hashable(value)
            self.indices[path][hashable_value].add(global_idx)
            self.indices[wildcard_path][hashable_value].add(global_idx)
            self.page_map[global_idx] = page_num
            self.index_map[global_idx] = (page_num, path, value, font_meta_dict)
            self.field_names[global_idx] = path.split(".")[-1]
            parent = ".".join(path.split(".")[:-1])
            self.parent_map[global_idx] = parent
            self.grouped_by_parent[parent].add(global_idx)
            self.unique_pages.add(page_num)

    def query(
        self,
        restrict: Optional[Union[str, List[str]]] = None,
        page: Optional[Union[int, List[int]]] = None,
        **kwargs
    ) -> List[Tuple[int, int, str, Any]]:
        if not restrict:
            restrict=["*"]
        elif restrict and "bbox" not in restrict:
            if isinstance(restrict, str):
                restrict = [restrict] + ["bbox"]
            else:
                restrict = restrict + ["bbox"]
        multi_page_results=[]
        if page in self.unique_pages:
            page_search = self._query(restrict=restrict, page=page, **kwargs)
            if page_search:
                multi_page_results.append(page_search)
        else: 
            multi_page_results=[]
            for page in self.unique_pages:
                page_search = self._query(restrict=restrict, page=page, **kwargs)
                if page_search:
                    multi_page_results.append(page_search)
        if multi_page_results:
            return list(itertools.chain.from_iterable(multi_page_results))
        else:
            return None

    def _query(
        self,
        restrict: Optional[Union[str, List[str]]] = None,
        page: Optional[Union[int, List[int]]] = None,
        **kwargs
    ) -> List[Tuple[int, int, str, Any]]:
        if restrict:
            restrict = [restrict] if isinstance(restrict, str) else restrict
        page_filter = None
        if page is not None:
            page_filter = [page] if isinstance(page, int) else page
        # Empty query returns all values, optionally filtered
        if not kwargs:
            return [
                (self.page_map[idx], idx, path, value, font_meta_dict)
                for idx, (pg, path, value, font_meta_dict) in self.index_map.items()
                if (restrict is None or self.field_names[idx] in restrict)
                and (page_filter is None or self.page_map[idx] in page_filter)
            ]
        matched_parent_sets: List[Set[str]] = []
        for path_pattern, expected_value in kwargs.items():
            matched_indices = set()
            value_map = self.indices.get(path_pattern, {})
            for val, idx_set in value_map.items():
                if expected_value == "*" or str(val) == str(expected_value):
                    matched_indices.update(idx_set)
            # Apply page filter before grouping
            if page_filter:
                matched_indices = {i for i in matched_indices if self.page_map[i] in page_filter}
            matched_parents = {self.parent_map[i] for i in matched_indices}
            matched_parent_sets.append(matched_parents)
        if not matched_parent_sets:
            return []
        # Intersect across all query conditions
        common_parents = set.intersection(*matched_parent_sets)
        final_indices = set()
        for parent in common_parents:
            final_indices.update(self.grouped_by_parent[parent])
        # Apply restrict
        if restrict:
            final_indices = {
                i for i in final_indices
                if any(fnmatch.fnmatch(self.field_names[i], pattern) for pattern in restrict)
            }
        final_indices = {i for i in final_indices if self.page_map[i] in page_filter}
        return [(self.page_map[i], i, self.index_map[i][1], self.index_map[i][2], self.index_map[i][3]) for i in sorted(final_indices)]

# --- LineTextIndex ---

class LineTextIndex:
    def __init__(
        self,
        data: List[Tuple[int, int, str, Any, Dict[str, Any]]],
        page_metadata: Optional[Dict[int, Dict[str, Any]]] = None
    ):
        """
        Accepts a list of (page_num, index, path, value, font_meta_dict) and builds a searchable index.
        Optionally accepts page_metadata: {page_num: {"width": ..., "height": ...}}
        """
        self.raw = data
        self.page_metadata = page_metadata or {}
        self.index: List[Dict[str, Any]] = []
        self.by_page: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self.page_text_bounds = defaultdict(lambda: {
            "min_x": float("inf"),
            "min_y": float("inf"),
            "max_x": float("-inf"),
            "max_y": float("-inf")
        })
        self._bound_restrictions = {}
        self._build_index()

    def set_search_result_bound_restrictions(self, result):
        restricted_bounds=defaultdict(list)
        for pg_num, items in result['results']['pages'].items():
            for res in items:
                bbox = res['bbox']
                restricted_bounds[pg_num].append(bbox)
        self._set_bound_restrictions(restricted_bounds)
    
    def _set_bound_restrictions(self, restrictions: Optional[Dict[int, List[List[float]]]]):
        """
        Sets bounding box restrictions by page.
        Any query result overlapping a restriction will be filtered out.
        """
        self._bound_restrictions = restrictions or {}

    def clear_bound_restrictions(self):
        """Clears all bound restrictions."""
        self._bound_restrictions = {}

    def _build_index(self):
        # Collect all bboxes (first pass)
        text_bbox_map = {}
        for row_tuple in self.raw:
            page_num, idx, path, value, font_meta_dict = row_tuple
            parsed = self._parse_path(path)
            if parsed is None:
                continue
            block, line, span, key = parsed
            if key == "bbox" and isinstance(value, (list, tuple)) and len(value) == 4:
                group_key = (page_num, block, line, span)
                text_bbox_map[group_key] = value
        # Second pass - build indexed rows (text, font, bbox, etc.)
        for row_tuple in self.raw:
            page_num, idx, path, value, font_meta_dict = row_tuple
            parsed = self._parse_path(path)
            if parsed is None:
                continue
            block, line, span, key = parsed
            group_key = (page_num, block, line, span)
            row = {
                "page": page_num,
                "index": idx,
                "block": block,
                "line": line,
                "span": span,
                "key": key,
                "value": value,
                "path": path,
                "font_meta": font_meta_dict.get("font_meta") if font_meta_dict else None
            }
            if key == "text":
                bbox = text_bbox_map.get(group_key)
                if bbox:
                    row["bbox"] = tuple(bbox)
                    row["x0"], row["y0"], row["x1"], row["y1"] = bbox
                    row["x_span"] = row["x1"] - row["x0"]
                    row["y_span"] = row["y1"] - row["y0"]
                    bounds = self.page_text_bounds[page_num]
                    bounds["min_x"] = min(bounds["min_x"], row["x0"])
                    bounds["min_y"] = min(bounds["min_y"], row["y0"])
                    bounds["max_x"] = max(bounds["max_x"], row["x1"])
                    bounds["max_y"] = max(bounds["max_y"], row["y1"])
                else:
                    row["bbox"] = None
                    row["x0"] = row["y0"] = row["x1"] = row["y1"] = None
                row["normalized_value"] = normalize_recurring_text(value) if isinstance(value, str) else None

                height = self.page_metadata.get(page_num, {}).get("height")
                if height is not None and row["y0"] is not None and row["y1"] is not None:
                    mid_y = (row["y0"] + row["y1"]) / 2.0
                    if mid_y < (height / 2):
                        row["region"] = "header"
                    else:
                        row["region"] = "footer"
                else:
                    row["region"] = None
            
            self.index.append(row)
            self.by_page[page_num].append(row)
        # Add whitespace rows (must include "key" for querying)
        fw_gaps = compute_full_width_v_whitespace(self.by_page, self.page_metadata, min_gap=5.0)
        for gap in fw_gaps:
            gap.setdefault("key", "full_width_v_whitespace")
            gap.setdefault("font_meta", None)
            self.index.append(gap)
            self.by_page[gap["page"]].append(gap)

    def _parse_path(self, path: str) -> Optional[Tuple[int, int, int, str]]:
        """
        Parse a path like: blocks[0].lines[2].spans[0].text → (0, 2, 0, 'text')
        """
        match = re.match(
            r"blocks\[(\d+)\]\.lines\[(\d+)\]\.spans\[(\d+)\]\.(\w+)",
            path
        )
        if not match:
            return None
        return tuple(map(lambda x: int(x) if x.isdigit() else x, match.groups()))

    def query(
        self,
        page: Optional[int] = None,
        line: Optional[int] = None,
        key: Optional[str] = None,
        func: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
        bounds_filter: Optional[Callable[[Dict[str, Any]], bool]] = None,
        transform: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
        description: Optional[str] = None,
        query_label: Optional[str] = None,
    ) -> Union[List[Dict[str, Any]], Any]:
        """
        Query by page, line, key (e.g. 'text'), with optional function and bounding-box filtering.
        Each result row will include the query_label if provided.
        """
        if description:
            self._last_query_description = description

        source = self.by_page.get(page, self.index) if page is not None else self.index

        result = []
        for row in source:
            if line is not None and row["line"] != line:
                continue
            if key is not None and row["key"] != key:
                continue

            # Inject metadata
            if self.page_metadata and row["page"] in self.page_metadata:
                row["meta"] = self.page_metadata[row["page"]]

            if bounds_filter and not bounds_filter(row):
                continue

            # FIXED: Bounds restriction logic
            if self._bound_restrictions:
                # Convert page number to string for lookup (your data uses string keys)
                page_key = str(row["page"])
                page_restrictions = self._bound_restrictions.get(page_key, [])
                
                if page_restrictions:  # If there are restrictions for this page
                    bbox = row.get("bbox")
                    if bbox:
                        # Check if this text overlaps with ANY of the restricted bounds
                        overlaps_any_restriction = False
                        for restriction in page_restrictions:
                            if Map.is_overlapping(bbox, restriction):
                                overlaps_any_restriction = True
                                break
                        
                        # SKIP if it does NOT overlap any restriction
                        # (i.e., only keep text that IS within the restricted bounds)
                        if not overlaps_any_restriction:
                            continue
                    else:
                        # No bbox means we can't determine overlap, so exclude it
                        continue
            
            # Create a shallow copy to avoid modifying original index
            result_row = dict(row)
            if query_label:
                result_row["query_label"] = query_label

            result.append(result_row)

        # If transform is applied, try to propagate query_label
        if transform:
            output = transform(result)
            if query_label and isinstance(output, list):
                for group in output:
                    if isinstance(group, list):  # grouped result
                        for item in group:
                            item["query_label"] = query_label
                    elif isinstance(group, dict):
                        group["query_label"] = query_label
            return output

        if func:
            return func(result)

        return result
    

def groupby(
    *keys: str,
    filterby: Callable[[List[Dict[str, Any]]], bool] = lambda g: True,
    group_id_field: str = "group_id"
):
    """
    Groups rows by given keys, filters, and injects unique group_id (tuple of key values).
    """
    def _transform(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        from collections import defaultdict
        grouped = defaultdict(list)
        for row in rows:
            key = tuple(row.get(k) for k in keys)
            grouped[key].append(row)
        output = []
        for group_key, group in grouped.items():
            if filterby(group):
                for row in group:
                    row[group_id_field] = group_key  # Now a tuple: (region, page)
                output.append(group)
        return output
    return _transform


def filterby(
    func: Callable[[str], Any],
    field: str = "value",
    test: Callable[[Any], bool] = bool,   # Default to bool (truthy)
) -> Callable[[list], list]:
    """
    Generic filter for rows, using func to process the field and test to evaluate pass.
    """
    def _transform(rows: list) -> list:
        return [
            row for row in rows
            if isinstance(row.get(field), str) and test(func(row[field]))
        ]
    return _transform


def chain_transform(*funcs):
    def _chained(rows):
        result = rows
        for f in funcs:
            result = f(result)
        return result
    return _chained


