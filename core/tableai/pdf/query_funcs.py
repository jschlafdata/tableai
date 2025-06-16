from typing import Any, Dict, List, Tuple, Optional, Union, Literal
import unicodedata
import dateparser
from datetime import datetime
import re
from tableai.pdf.coordinates import (
    Geometry,
    CoordinateMapping
)
import functools
from collections import defaultdict
from tableai.pdf.query import FitzSearchIndex, QueryParams
from tableai.pdf.generic_models import (
    HorizontalWhitespaceParams,
    GroupTouchingBoxesParams, 
    ParagraphsParams
)

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

def horizontal_whitespace(
    query_index: 'FitzSearchIndex',
    params: Optional[HorizontalWhitespaceParams] = None,
    **kwargs
) -> 'ResultSet':
    """Finds full-width whitespace blocks using a self-initializing parameter model."""
    p = params or HorizontalWhitespaceParams()
    p = p.model_copy(update=kwargs)

    transform_func = lambda rows: [r for r in rows if r["gap"] >= p.y_tolerance]

    query_params = QueryParams(
        page=p.page_number,
        key="full_width_v_whitespace",
        transform=transform_func,
        query_label=p.query_label
    )
    return query_index.query(params=query_params)


def group_vertically_touching_bboxes(
    whitespace_blocks: List[Tuple[float, ...]], 
    header_footer_blocks: List[Tuple[float, ...]],
    params: Optional[GroupTouchingBoxesParams] = None,
    **kwargs
) -> List[List[Tuple[float, ...]]]:
    """Groups vertically touching bboxes using a self-initializing parameter model."""
    p = params or GroupTouchingBoxesParams()
    p = p.model_copy(update=kwargs)

    bboxes = whitespace_blocks + header_footer_blocks
    
    # Handle edge case of an empty list
    if not bboxes:
        return []

    # --- OPTIMIZATION: Convert to a set for fast O(1) lookups later ---
    # This is the key to making the final filtering step efficient.
    header_footer_set: Set[Tuple[float, ...]] = set(header_footer_blocks)

    # 1. Sort bboxes by their top coordinate (y0)
    sorted_bboxes = sorted(bboxes, key=lambda b: b[1])

    # 2. Initialize the first group
    all_groups: List[List[Tuple[float, ...]]] = []
    current_group: List[Tuple[float, ...]] = [sorted_bboxes[0]]

    # 3. Iterate and group based on vertical proximity
    for i in range(1, len(sorted_bboxes)):
        current_bbox = sorted_bboxes[i]
        previous_bbox_in_chain = current_group[-1]
        
        gap = current_bbox[1] - previous_bbox_in_chain[3]  # current.y0 - previous.y1

        if gap <= p.y_tolerance:
            current_group.append(current_bbox)
        else:
            all_groups.append(current_group)
            current_group = [current_bbox]

    # 4. Add the last group
    if current_group:
        all_groups.append(current_group)
        
    # --- CORRECTED FILTERING LOGIC ---
    # Use a list comprehension that checks for membership in the efficient set.
    # For each 'group', keep it if 'any' 'bbox' in that group exists in the 'header_footer_set'.
    return [
        group for group in all_groups 
        if any(bbox in header_footer_set for bbox in group)
    ]


def find_paragraph_blocks(
    query_index: 'FitzSearchIndex', 
    paragraph_seed_lines: List[Dict[str, Any]],
    params: ParagraphsParams
) -> List[Dict[str, Any]]:
    """
    Finds and groups paragraph blocks, outputting a list of dictionaries that
    conform to the GroupbyQueryResult schema.
    """
    # Query all text once to have a pool of lines to search for continuations.
    # .to_dict() is efficient for repeated lookups.
    all_text_rows = query_index.query(QueryParams(key="text")).to_dict()
    
    by_page = defaultdict(list)
    for row in all_text_rows:
        by_page[row['page']].append(row)
    
    # Sort each page's text by vertical position
    for page_rows in by_page.values():
        page_rows.sort(key=lambda r: r.get('y0') or 0.0)

    # A set to track which lines have already been assigned to a paragraph
    seen = set()
    output_summaries = []

    for seed_line in paragraph_seed_lines:
        page, idx = seed_line['page'], seed_line['index']

        if (page, idx) in seen:
            continue

        # Start a new paragraph group with the seed line
        current_para_members = [seed_line]
        seen.add((page, idx))
        
        # Get font info from the seed line to ensure consistency
        font_size = seed_line['font_meta']['size'] if seed_line.get('font_meta') else None
        font_name = seed_line['font_meta']['font'] if seed_line.get('font_meta') else None
        
        prev_line = seed_line
        
        # ---- The Core Chaining Logic (largely the same) ----
        while True:
            # Find all unused lines below the current line that could be a continuation
            candidates = [
                row for row in by_page[page]
                if (page, row['index']) not in seen
                and row.get('x0') is not None
                and abs(row['x0'] - prev_line['x0']) <= params.x0_tol
                and row.get('font_meta') and row['font_meta']['font'] == font_name
                and (font_size is None or abs(row['font_meta']['size'] - font_size) < font_size_tol)
                and row.get('value', '').strip()
                and row.get('y0') is not None
                and 0 < (row['y0'] - prev_line['y1']) < params.y_gap_max
            ]
            if not candidates:
                break
            
            # The true next line is the one with the smallest vertical gap
            next_line = min(candidates, key=lambda r: r['y0'])
            
            current_para_members.append(next_line)
            seen.add((page, next_line['index']))
            prev_line = next_line

        if not current_para_members:
            continue

        first_member = current_para_members[0]
        
        summary = {
            # --- Key fields for GroupbyQueryResult ---
            "group_id": (first_member['page'], first_member['index']), # A unique ID for the paragraph
            "groupby_keys": ("paragraph_group",), # A conceptual key
            "member_count": len(current_para_members),
            "query_label": "paragraph",

            # --- Consistent fields (copied from first member) ---
            "page": first_member.get('page'),
            "region": first_member.get('region'),
            "physical_page": first_member.get('physical_page'),
            "physical_page_bounds": first_member.get('physical_page_bounds'),
            "meta": first_member.get('meta', {}),
            
            # --- Aggregated fields (collected from all members) ---
            "group_bboxes": [m['bbox'] for m in current_para_members if m.get('bbox')],
            "group_paths": [m['path'] for m in current_para_members if m.get('path')],
            "group_text": [m['value'] for m in current_para_members if m.get('value')],
            "group_indices": [m['index'] for m in current_para_members if m.get('index') is not None]
        }
        output_summaries.append(summary)

    return output_summaries

def paragraphs(
    query_index: 'FitzSearchIndex', 
    params: Optional[ParagraphsParams] = None,
    **kwargs
) -> 'ResultSet':
    """Finds and groups lines into paragraphs using a self-initializing parameter model."""
    p = params or ParagraphsParams()
    p = p.model_copy(update=kwargs)

    # Use functools.partial to pass the finalized params object to the helper
    transform_func = functools.partial(find_paragraph_blocks, query_index, params=p)

    bounds_filter_func = lambda r: (
        r.get("x_span") is not None and
        r.get("page_width_rel") is not None and r["page_width_rel"] > 0 and
        (r["x_span"] / r["page_width_rel"]) > p.width_threshold
    )

    query_params = QueryParams(
        key="text",
        bounds_filter=bounds_filter_func,
        transform=transform_func,
        query_label=p.query_label or "paragraph"
    )
    return query_index.query(params=query_params)


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