from typing import Any, Dict, List, Tuple, Optional, Union, Literal
import unicodedata
import dateparser
from datetime import datetime
import re
from tableai.pdf.coordinates import Map

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