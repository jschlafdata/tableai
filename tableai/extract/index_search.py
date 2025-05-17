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

class FitzTextIndex:
    def __init__(self, flattened_data: List[Tuple[str, Any]]):
        """
        Initialize the index with a list of (path, value) tuples from `flatten_fitz_dict`.
        """
        self.flattened_data = flattened_data
        self.page_metadata = {}
        self.unique_pages = set()
        self._build_indices()

    @staticmethod
    def flatten_fitz_dict(data, page_num: int, parent_key='', sep='.', result=None):
        if result is None:
            result = []
        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                FitzTextIndex.flatten_fitz_dict(v, page_num, new_key, sep, result)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_key = f"{parent_key}[{i}]"
                FitzTextIndex.flatten_fitz_dict(item, page_num, new_key, sep, result)
        else:
            result.append((page_num, parent_key, data))
        return result


    @classmethod
    def from_document(cls, doc: fitz.Document):
        """
        Build a FitzTextIndex from all pages in a fitz.Document.
        Also stores page-level metadata (e.g. width, height).
        """
        instance = cls(flattened_data=[])  # defer full init
        all_flattened = []
    
        for page_num in range(len(doc)):
            page = doc[page_num]
            text_dict = page.get_text("dict")
            flattened = cls.flatten_fitz_dict(text_dict, page_num)
            all_flattened.extend(flattened)
    
            # Store page-level metadata
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
    
        for local_idx, (page_num, path, value) in enumerate(self.flattened_data):
            global_idx = page_num * PAGE_STRIDE + local_idx
    
            wildcard_path = re.sub(r'\[\d+\]', '[*]', path)
            self.indices[path][value].add(global_idx)
            self.indices[wildcard_path][value].add(global_idx)
    
            self.page_map[global_idx] = page_num
            self.index_map[global_idx] = (page_num, path, value)
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
        print(page)
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
        """
        Query with support for wildcard paths, group-level intersection, field restricts, and page filtering.
    
        Returns:
            List of (page_num, global_idx, path, value)
        """
        if restrict:
            restrict = [restrict] if isinstance(restrict, str) else restrict
    
        page_filter = None
        if page is not None:
            page_filter = [page] if isinstance(page, int) else page
    
        # Empty query returns all values, optionally filtered
        if not kwargs:
            return [
                (self.page_map[idx], idx, path, value)
                for idx, (pg, path, value) in self.index_map.items()
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
    
        return [(self.page_map[i], i, self.index_map[i][1], self.index_map[i][2]) for i in sorted(final_indices)]


class LineTextIndex:
    def __init__(
        self,
        data: List[Tuple[int, int, str, Any]],
        page_metadata: Optional[Dict[int, Dict[str, Any]]] = None
    ):
        """
        Accepts a list of (page_num, index, path, value) and builds a searchable index.
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
        self._build_index()

    def _build_index(self):
        # First, ensure we're grouping by the right elements
        text_bbox_map = {}

        
        # First pass - collect all bboxes
        for page_num, idx, path, value in self.raw:
            parsed = self._parse_path(path)
            if parsed is None:
                continue
                
            block, line, span, key = parsed
            if key == "bbox" and isinstance(value, (list, tuple)) and len(value) == 4:
                # Store bbox with a key that includes all identifying info
                group_key = (page_num, block, line, span)
                text_bbox_map[group_key] = value
        
        # Second pass - associate text with bboxes
        for page_num, idx, path, value in self.raw:
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
                "path": path
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
                    # No bbox found, set default values
                    row["bbox"] = None
                    row["x0"] = row["y0"] = row["x1"] = row["y1"] = None
                
                # Normalize text value
                if isinstance(value, str):
                    row["normalized_value"] = normalize_recurring_text(value)
                else:
                    row["normalized_value"] = None
            
            self.index.append(row)
            self.by_page[page_num].append(row)

        fw_gaps = compute_full_width_v_whitespace(self.by_page, self.page_metadata, min_gap=5.0)
        for gap in fw_gaps:
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


def groupby(*keys: str, filterby: Callable[[List[Dict[str, Any]]], bool] = lambda g: True):
    """
    Return a transform function that groups rows by the given keys,
    then filters those groups with `filterby`.

    Args:
        *keys: field names to group by (e.g., 'value', 'page')
        filterby: a function that receives each group and returns True to keep it
    """
    def _transform(rows: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        grouped = defaultdict(list)
        for row in rows:
            key = tuple(row[k] for k in keys)
            grouped[key].append(row)

        return [group for group in grouped.values() if filterby(group)]

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
            meta = row['meta']
            page = row['page']
            meta = row.get('meta')
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
    
