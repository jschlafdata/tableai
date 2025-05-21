from collections import defaultdict
from typing import Any, Callable, Dict, List, Set, Tuple
import fnmatch
import re
import itertools
import re
from typing import Any, Dict, List, Tuple, Optional, Union, Literal
from collections import defaultdict
import unicodedata
import fitz
import dateparser
from datetime import datetime
from tableai.extract.helpers import Map

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
        skip=False
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

            if self._bound_restrictions:
                page_restrictions = self._bound_restrictions.get(str(row["page"]), [])
                bbox = row.get("bbox")
                if bbox:
                    for restriction in page_restrictions:
                        if Map.is_overlapping(bbox, restriction):
                            skip = True
                            break
            if skip:
                skip=False
                continue  # SKIP this row
            
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


def render_api(
    query_label: str,
    description: Optional[str],
    pdf_metadata: Dict[str, Any],
    include: Optional[List] = None
) -> Callable[[List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Custom transform function to format query results for API rendering.

    Args:
        query_label: A label describing the type of query.
        description: Optional description of the query.
        pdf_metadata: Metadata about the PDF document (e.g., file path).

    Returns:
        A transform function for LineTextIndex.query
    """
    def _transform(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        pages = defaultdict(list)
        unique_meta_by_page={}
        for row in rows:
            page = row['page']
            meta = row.get('meta', None)
            if meta:
                unique_meta_by_page[page] = meta
            if include:
                row = {k:v for k,v in row.items() if k in list(set(['page'] + include))}
            pages[str(row["page"])].append(row)

        return {
            "query_label": query_label,
            "description": description,
            "pdf_metadata": {**pdf_metadata, **unique_meta_by_page},
            "results": {
                "pages": dict(pages)
            }
        }

    return _transform


def chain_transform(*funcs):
    def _chained(rows):
        result = rows
        for f in funcs:
            result = f(result)
        return result
    return _chained


def merge_result_group_bounds(query_label=None):
    def _transform(groups):
        out = []
        for group in groups:
            if not group:
                continue
            group_id = group[0].get("group_id")
            page = group[0].get("page")
            region = group[0].get("region")
            meta = group[0].get("meta", {})  # <<-- Make sure meta comes along!
            bbox = Map.merge_all_boxes([row['bbox'] for row in group])
            summary = {
                "group_id": group_id,
                "page": page,
                "region": region,
                "meta": meta,  # <<-- Carry it to the summary
                "bbox": bbox,
                "source_grouped_metadata": group
            }
            if query_label:
                summary["query_label"] = query_label
            out.append(summary)
        return out
    return _transform


def expand_bounds(
    x_mode: str = "full",
    y_mode: str = "auto",
    region_key: str = "region",
    page_meta_key: str = "meta",
    bbox_key: str = "bbox"
):
    """
    Expands the bbox for header/footer:
    - x0 = 0, x1 = width
    - header:   y0 = 0,     y1 = group's y1
    - footer:   y0 = group's y0, y1 = height
    """
    def _transform(groups):
        for group in groups:
            meta = group.get(page_meta_key, {})
            width = meta.get("width")
            height = meta.get("height")
            bbox = list(group.get(bbox_key, [None, None, None, None]))
            region = group.get(region_key)

            if bbox and None not in bbox and width and height:
                # Set x0, x1 always
                bbox[0] = 0
                bbox[2] = width
                # Header region
                if y_mode == "auto" and region == "header":
                    bbox[1] = 0           # y0 = 0 (top)
                    # y1 stays as is (group's detected y1)
                # Footer region
                elif y_mode == "auto" and region == "footer":
                    # y0 stays as is (group's detected y0)
                    bbox[3] = height      # y1 = height (bottom)
                group[bbox_key] = tuple(bbox)
        return groups
    return _transform


def compute_full_width_v_whitespace(by_page, page_metadata, min_gap: float = 5.0) -> List[Dict[str, Any]]:
    """
    Detect vertical whitespace regions that span the full width of the page.

    Args:
        by_page: the indexed content by page
        page_metadata: metadata with width/height per page
        min_gap: minimum vertical gap (in pts) to consider

    Returns:
        List of whitespace row dicts to be injected into LineTextIndex
    """
    results = []

    for page_num, rows in by_page.items():
        spans = [r for r in rows if r.get("key") == "text" and r.get("bbox")]
        spans = sorted(spans, key=lambda r: r["y0"])
        page_width = page_metadata.get(page_num, {}).get("width", 612.0)  # fallback default A4 width

        for i in range(len(spans) - 1):
            a, b = spans[i], spans[i + 1]
            gap = b["y0"] - a["y1"]
            if gap >= min_gap:
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



def search_normalized_text(
    rows: List[Dict[str, Any]],
    target: str,
    mode: Literal["word", "block"] = "word",
    best_match: bool = False
) -> List[Dict[str, Any]]:
    normalized_target = normalize_word(target)
    matches = []

    if mode == "block":
        for row in rows:
            if row["key"] != "text" or not isinstance(row["value"], str):
                continue
            norm_val = normalize_word(row["value"])
            if normalized_target in norm_val:
                matches.append({
                    "bboxes": [row["bbox"]],
                    "spans": [row],
                    "matched_text": row["value"],
                    "match_score": abs(len(norm_val) - len(normalized_target))
                })

    elif mode == "word":
        pdf_words = [
            {
                "index": i,
                "text": row["value"],
                "norm": normalize_word(row["value"]),
                "bbox": row["bbox"],
                "row": row
            }
            for i, row in enumerate(rows)
            if row["key"] == "text" and isinstance(row["value"], str)
        ]

        target_len = len(normalized_target)

        for start_index in range(len(pdf_words)):
            matched_length = 0
            match_bboxes = []
            match_rows = []

            w = start_index
            while w < len(pdf_words) and matched_length < target_len:
                word = pdf_words[w]
                norm = word["norm"]
                bbox = word["bbox"]

                local_index = 0
                while local_index < len(norm) and matched_length < target_len:
                    if norm[local_index] == normalized_target[matched_length]:
                        if not match_bboxes or match_bboxes[-1] != bbox:
                            match_bboxes.append(bbox)
                            match_rows.append(word["row"])
                        matched_length += 1
                        local_index += 1
                    else:
                        break

                if local_index < len(norm) and matched_length < target_len:
                    break

                w += 1

            if matched_length == target_len:
                matched_text = ' '.join(r["value"] for r in match_rows)
                matches.append({
                    "bboxes": match_bboxes,
                    "spans": match_rows,
                    "matched_text": matched_text,
                    "start_index": start_index,
                    "match_score": abs(len(normalize_word(matched_text)) - len(normalized_target))
                })

    if best_match and matches:
        matches.sort(key=lambda m: m["match_score"])
        return [matches[0]]

    return matches


def normalize_word(text: str) -> str:
    """Lowercase and remove non-spacing marks, punctuation, etc."""
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text.lower())
        if not unicodedata.combining(c) and c.isalnum()
    )

def normalize_recurring_text(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r'page\s*\d+\s*of\s*\d+', 'page xx of xx', t)
    t = re.sub(r'page\s*\d+', 'page xx', t)
    return t

def patterns(text: str, pattern_name: str = "toll_free") -> list:
    """
    General-purpose pattern matcher that identifies matches from predefined named pattern groups.

    Args:
        text: The string to search within.
        pattern_name: One of the keys from `named_patterns`.

    Returns:
        List of dicts with match metadata.
    """
    TOLL_FREE_PREFIXES = ["800", "888", "877", "866", "855", "844", "833", "822"]
    TOLL_FREE_PATTERN = f"({'|'.join(TOLL_FREE_PREFIXES)})"

    named_patterns = {
        "toll_free": {
            "patterns": [
                fr'1-{TOLL_FREE_PATTERN}-\d{{3}}-\d{{4}}',
                fr'1{TOLL_FREE_PATTERN}\d{{7}}',
                fr'{TOLL_FREE_PATTERN}\d{{7}}'
            ],
            "formats": [
                "hyphenated with country code",
                "continuous with country code",
                "continuous without country code"
            ]
        },
        "currency": {
            "patterns": [
                r'[$€£¥₹₽₩₺₴₸R\฿\u20A0-\u20CF]',        
            ],
            "formats": [
                "Common currency prefixes",
            ]
        }
    }

    # Validate pattern group
    if pattern_name not in named_patterns:
        raise ValueError(f"Unknown pattern group: {pattern_name}")

    pattern_set = named_patterns[pattern_name]
    regexes = pattern_set["patterns"]
    descriptions = pattern_set["formats"]

    results = []
    for idx, regex in enumerate(regexes):
        for match in re.finditer(regex, text, re.IGNORECASE):
            results.append({
                "match": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "format": descriptions[idx]
            })

    return results


def try_convert_float(value: str) -> float:
    """
    Attempts to clean and convert a string to a float.
    Handles:
        - Leading/trailing currency symbols (e.g., $123, 123€, ₹1,000.00)
        - Negative/positive signs before or after symbol
        - Commas as thousand separators
        - Graceful fallback to None if conversion fails

    Args:
        value (str): String potentially representing a currency or number

    Returns:
        float or None
    """
    if not isinstance(value, str):
        return None

    value = value.strip()

    # Regex for optional sign, optional currency, then number
    # Handles: -$1,234.56, +€123, 123.45, $-123.45, etc.
    pattern = re.compile(
        r"""^
        \s*            # Optional leading whitespace
        (?P<sign>[-+])?  # Optional leading sign
        \s*            # Optional whitespace
        (?P<currency>[€£¥₹₽₩₺₴₸฿$])? # Optional currency symbol
        \s*            # Optional whitespace
        (?P<number>[\d,]*\.?\d+)
        \s*            # Optional trailing whitespace
        (?P<currency2>[€£¥₹₽₩₺₴₸฿$])? # Optional trailing currency symbol
        \s*            # Optional trailing whitespace
        $""",
        re.VERBOSE
    )

    match = pattern.match(value)
    if match:
        number = match.group("number").replace(',', '')
        sign = match.group("sign") or ''
        try:
            return float(f"{sign}{number}")
        except ValueError:
            return None

    # Fallback: try plain float conversion
    try:
        return float(value.replace(',', ''))
    except ValueError:
        return None



def identify_currency_symbols(text: str) -> list:
    """
    Identifies common currency expressions (symbol + number or number + symbol).
    Matches things like: $123, 123€, € 9.99, ₹1,000.00

    Returns a list of match metadata.
    """
    pattern = r"""
        (?:                             # non-capturing group for the two directions
            [€£¥₹₽₩₺₴₸฿$]               # symbol first
            \s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?
        |
            \d{1,3}(?:,\d{3})*(?:\.\d{2})?  # number first
            \s?[€£¥₹₽₩₺₴₸฿$]
        )
    """

    return [
        {
            "match": m.group(0),
            "start": m.start(),
            "end": m.end(),
            "format": "symbol-attached currency"
        }
        for m in re.finditer(pattern, text, flags=re.VERBOSE)
    ]

def try_convert_percent(value: str) -> Optional[float]:
    """
    Attempts to clean and convert a string to a float percentage.
    Handles:
        - Leading/trailing percent sign: 60%, %60, -60%, +60%, -%60.5, 60.5%, etc.
        - Optional sign (+/-)
        - Graceful fallback to None if conversion fails

    Args:
        value (str): String potentially representing a percentage

    Returns:
        float: As a fraction (e.g. 50% => 0.5, -0.5 for -50%)
        or None if not a percentage
    """
    if not isinstance(value, str):
        return None

    value = value.strip()

    if try_convert_float(value) is not None:
        return None

    # Regex for percent patterns
    pattern = re.compile(
        r"""^
            \s*
            (?P<sign>[-+])?           # Optional sign
            \s*
            (%\s*)?                   # Optional leading percent
            (?P<number>\d*\.?\d+)
            \s*
            (%\s*)?                   # Optional trailing percent
            \s*$
        """, re.VERBOSE
    )

    match = pattern.match(value)
    if match:
        number = match.group("number")
        sign = match.group("sign") or ''
        try:
            val = float(f"{sign}{number}")
            return val / 100
        except ValueError:
            return None

    return None


def bbox_proximity(
    bbox1: Tuple[float, float, float, float],
    bbox2: Tuple[float, float, float, float],
    axis: Literal["x", "y"] = "x",
    direction: Literal["forward", "reverse"] = "forward"
) -> float:
    """
    Compute the proximity between two bounding boxes along a given axis.

    Args:
        bbox1: First bounding box (x0, y0, x1, y1)
        bbox2: Second bounding box (x0, y0, x1, y1)
        axis: "x" for horizontal gap, "y" for vertical gap
        direction: 
            - "forward" (default): measures end of bbox1 to start of bbox2
            - "reverse": measures end of bbox2 to start of bbox1

    Returns:
        A float representing the distance between the two boxes along the chosen axis.
        Negative if overlapping.
    """
    if axis == "x":
        if direction == "forward":
            return bbox2[0] - bbox1[2]  # x0 of 2 - x1 of 1
        else:
            return bbox1[0] - bbox2[2]  # x0 of 1 - x1 of 2
    elif axis == "y":
        if direction == "forward":
            return bbox2[1] - bbox1[3]  # y0 of 2 - y1 of 1 (top-down)
        else:
            return bbox1[1] - bbox2[3]  # y0 of 1 - y1 of 2
    else:
        raise ValueError("axis must be 'x' or 'y'")


def proximity_search(
    axis: Literal["x", "y"] = "x",
    direction: Literal["forward", "reverse"] = "forward",
    min_distance: float = 0.0,
    max_distance: float = 1.0,
    pair_filter: Optional[Callable[[Dict[str, Any], Dict[str, Any]], bool]] = None
) -> Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Returns a list of dicts: {
        'row_a': ...,
        'row_b': ...,
        'distance': float
    }
    """
    def _transform(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []

        for i, row_a in enumerate(rows):
            for j, row_b in enumerate(rows):
                if i == j or row_a["page"] != row_b["page"]:
                    continue

                if axis == "x":
                    a_edge = row_a["x1"] if direction == "forward" else row_b["x1"]
                    b_edge = row_b["x0"] if direction == "forward" else row_a["x0"]
                    y_diff = abs(row_a["y0"] - row_b["y0"])
                else:
                    a_edge = row_a["y1"] if direction == "forward" else row_b["y1"]
                    b_edge = row_b["y0"] if direction == "forward" else row_a["y0"]
                    y_diff = abs(row_a["x0"] - row_b["x0"])

                distance = b_edge - a_edge

                if min_distance <= distance <= max_distance:
                    if pair_filter is None or pair_filter(row_a, row_b):
                        results.append({
                            "row_a": row_a,
                            "row_b": row_b,
                            "distance": distance,
                            "y_diff": y_diff
                        })

        return results

    return _transform

def try_convert_date(value: str):

    DATE_PATTERN = re.compile(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})')
    try:
        if try_convert_float(value) is not None:
            return None
        if try_convert_percent(value) is not None:
            return None
        if not any(c.isdigit() for c in value):
            return None
        matches = DATE_PATTERN.findall(value)
        for m in matches:
            dt = dateparser.parse(m, settings={'RETURN_AS_TIMEZONE_AWARE': False})
            if dt:
                return dt.strftime('%Y-%m-%d')

        date = dateparser.parse(value, settings={
            'RETURN_AS_TIMEZONE_AWARE': False,
            'RELATIVE_BASE': datetime(1, 1, 1)
        })
        # Extract only the month and day
        if date:
            if date.year == 1:
                # Year was not in the original string
                month_day = f"{date.month:02d}-{date.day:02d}"  # Format as MM-DD
                return month_day
            else:
                # Year was in the original string
                return date
        else:
            return None
    except:
        return None


def find_paragraph_blocks(line_index, paragraph_lines, x0_tol=2.0, font_size_tol=0.2, y_gap_min=0.1, y_gap_max=7.0):
    """
    Finds paragraph blocks even if lines are in different blocks, using y-coord and font info.
    """
    all_text = [row for row in line_index.query(key="text")]
    by_page = defaultdict(list)
    for row in all_text:
        by_page[row['page']].append(row)
    for page_rows in by_page.values():
        page_rows.sort(key=lambda r: r['y0'])

    paragraphs = []
    seen = set()

    for para in paragraph_lines:
        page, idx, y0 = para['page'], para['index'], para['y0']
        font_size = para['font_meta']['size'] if para['font_meta'] else None
        font_name = para['font_meta']['font'] if para['font_meta'] else None
        x0 = para['x0']

        if (page, idx) in seen:
            continue

        current_para = [para]
        seen.add((page, idx))
        prev_line = para
        prev_y1 = prev_line['y1']

        while True:
            # Find all unused lines below current line, on same page,
            # with similar x0, font, size, and y0 just below current y1.
            candidates = [
                row for row in by_page[page]
                if (page, row['index']) not in seen
                and abs(row['x0'] - x0) <= x0_tol
                and row['font_meta'] and row['font_meta']['font'] == font_name
                and (font_size is None or abs(row['font_meta']['size'] - font_size) < font_size_tol)
                and row['value'].strip()
                and -2.5 < (row['y0'] - prev_y1) < y_gap_max
            ]
            if not candidates:
                break
            # Always choose the closest y0 (lowest gap) as the true continuation
            next_row = min(candidates, key=lambda r: r['y0'])
            current_para.append(next_row)
            seen.add((page, next_row['index']))
            prev_line = next_row
            prev_y1 = prev_line['y1']

        # Merge logic as before
        full_text = " ".join(line["value"] for line in current_para if line.get("value"))
        all_bboxes = [line["bbox"] for line in current_para if line.get("bbox")]
        merged_bbox = Map.merge_all_boxes(all_bboxes) if all_bboxes else None
        first = current_para[0]
        merged = dict(first)
        merged["value"] = full_text
        merged["bbox"] = merged_bbox
        if merged_bbox and len(merged_bbox) == 4:
            merged["x0"], merged["y0"], merged["x1"], merged["y1"] = merged_bbox
            merged["x_span"] = merged["x1"] - merged["x0"]
            merged["y_span"] = merged["y1"] - merged["y0"]
        merged["lines_merged"] = [line["line"] for line in current_para]
        merged["blocks_merged"] = [line["block"] for line in current_para]
        merged["num_lines"] = len(current_para)
        paragraphs.append(merged)

    return paragraphs

class PDFTools:

    @staticmethod
    def combine_pages_into_one(src_doc):
        """
        Combines all pages of input Fitz Doc into one tall PDF,
        """
        combined_doc = fitz.open()
        original_page_count = len(src_doc)
        total_height = 0
        max_width = 0
        page_heights = []  # each page's height in the original doc
        for page_index in range(original_page_count):
            page = src_doc[page_index]
            width, height = page.rect.width, page.rect.height
            page_heights.append(height)
            total_height += height
            max_width = max(max_width, width)

        combined_page = combined_doc.new_page(width=max_width, height=total_height)

        page_breaks = []
        current_y = 0.0

        # Place each original page
        for page_index in range(original_page_count):
            page_breaks.append(current_y)
            page = src_doc[page_index]
            width, height = page.rect.width, page.rect.height

            target_rect = fitz.Rect(0, current_y, width, current_y + height)
            combined_page.show_pdf_page(target_rect, src_doc, page_index)
            current_y += height
        
        return combined_doc, page_breaks