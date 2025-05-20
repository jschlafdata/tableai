import fitz
import re
import hashlib
import itertools
from collections import defaultdict
from tableai.extract.index_search import (
    FitzTextIndex, 
    LineTextIndex,
    search_normalized_text
)

from tableai.extract.helpers import Map
from tableai.extract.query_engine import QueryEngine
import copy

def remove_special_characters(text):
    # This keeps only letters, numbers, and spaces
    return re.sub(r'[^A-Za-z0-9 ]+', '', text)

class TableHeaderExtractor:
    """
    Extracts and flattens table header metadata from a PDF file.
    Usage:
        extractor = TableHeaderExtractor(file_id="...")
        flat_list, y_meta = extractor.process(table_structures)
    """

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.index = None
        self.line_index = None
        self.page_metadata = None
        self.query_engine = None
        self.query_label = "table_headers"
        self.description = "Inference results translated to pdf table bounds"

    def build_index(self):
        doc = fitz.open(self.pdf_path)
        self.index = FitzTextIndex.from_document(doc)
        all_pages_text_index = self.index.query(
            **{"blocks[*].lines[*].spans[*].text": "*"}, restrict=["text", "font", "bbox"]
        )
        self.page_metadata = self.index.page_metadata
        self.line_index = LineTextIndex(all_pages_text_index, page_metadata=self.page_metadata)
        self.query_engine = QueryEngine(self.line_index)
        doc.close()

    def hash_columns(self, columns_dict):
        """Create a stable hash from all column names across hierarchy levels."""
        flat_columns = []
        for col_list in columns_dict.values():
            flat_columns.extend(col_list)
        col_string = '|'.join(flat_columns)
        return hashlib.md5(col_string.encode()).hexdigest()

    def extract_table_headers(self, table_structures):
        """
        Extracts table headers and groups them by table title, hierarchy, and columns.
        Returns merged_results in the expected structure.
        """
        if self.line_index is None:
            self.build_index()
        
        column_matches_meta = []
        for structure in table_structures:
            if not isinstance(structure, dict):
                continue

            for table in structure.get('tables', []):
                if not isinstance(table, dict):
                    continue

                table_index = table.get('table_index')
                table_title = table.get('title', f'Table {table_index}')
                table_hierarchy = defaultdict(list)
                for vals in table['column_metadata'].values():
                    for level_key, col_name in vals.items():
                        match = re.search(r'\d+', level_key)
                        if match:
                            level = int(match.group())
                            table_hierarchy[level].append(col_name)

                for hierarchy_level, cols in table_hierarchy.items():
                    col_search = ' '.join(cols)
                    column_matches = self.line_index.query(
                        key="text",
                        func=lambda rows: search_normalized_text(
                            rows, target=col_search, mode="word", best_match=False
                        ),
                        description=f"MATCHES FOR '{table_title}' COLUMNS LEVEL {hierarchy_level}"
                    )
                    if not column_matches:
                        col_search = remove_special_characters(col_search)
                        column_matches = self.line_index.query(
                            key="text",
                            func=lambda rows: search_normalized_text(
                                rows, target=col_search, mode="word", best_match=False
                            ),
                            description=f"MATCHES FOR '{table_title}' COLUMNS LEVEL {hierarchy_level}"
                        )

                    if column_matches:
                        for match in column_matches:
                            bounds = match['bboxes']
                            pages = list(set([x['page'] for x in match['spans']]))
                            if len(pages) == 1:
                                page = pages[0]
                                column_matches_meta.append({
                                    'table_index': table_index,
                                    'table_title': table_title,
                                    'page': page,
                                    'bounds': Map.merge_all_boxes(bounds),
                                    'columns': {hierarchy_level: cols},
                                    'hierarchy': hierarchy_level
                                })

        grouped_matches = defaultdict(list)
        for entry in column_matches_meta:
            col_hash = self.hash_columns(entry['columns'])
            key = (entry['table_title'], entry['hierarchy'], col_hash)
            grouped_matches[key].append(entry)

        merged_results = []
        for (table_title, hierarchy, col_hash), group in grouped_matches.items():
            pages = {}
            columns = group[0]['columns']
            table_index = group[0]['table_index']
            for entry in group:
                page_num = entry['page']
                if page_num not in pages:
                    pages[page_num] = []
                pages[page_num].append({
                    'bbox': entry['bounds'],
                    'hierarchy': hierarchy
                })
            merged_results.append({
                'table_title': table_title,
                'columns': columns,
                'pages': pages,
                'table_index': table_index
            })
        return merged_results

    def flatten_table_results(self, merged_results):
        """
        Returns:
            results (dict): { "pages": { page_num: [detailed dict, ...], ... } }
        """
        results = {"pages": defaultdict(list)}
        table_counter = 1
    
        for table in merged_results:
            table_title = table['table_title']
            columns = table['columns']
            pages = table['pages']
            for page_num, bboxes in pages.items():
                for bounds_index, bbox_item in enumerate(bboxes):
                    bbox = bbox_item['bbox']
                    # You may have to look up these details from your index
                    # For now, only bbox fields are shown
                    entry = {
                        "page": page_num,
                        "table_index": table_counter,
                        "bounds_index": bounds_index,
                        "bbox": bbox,
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3],
                        "x_span": bbox[2] - bbox[0],
                        "y_span": bbox[3] - bbox[1],
                        "table_title": table_title,
                        "columns": columns,
                        "hierarchy": bbox_item.get("hierarchy", None),
                        "meta": {
                            "width": self.page_metadata.get(page_num, {}).get("width"),
                            "height": self.page_metadata.get(page_num, {}).get("height"),
                        },
                        # Optional: Add path/index/etc if available
                    }
                    results["pages"][str(page_num)].append(entry)
            table_counter += 1
    
        # Convert defaultdict to dict for serialization
        results["pages"] = dict(results["pages"])
        return results

    def get_table_y_metadata(self, flat_dict):
        """
        Accepts:
            flat_dict: output of flatten_table_results (a dict with "pages" -> { page: [rows...] })
        Returns:
            List of dicts with min_y/max_y per table_index per page.
        """
        meta_list = []
        for page, items in flat_dict["pages"].items():
            # Group by table_index within each page, since multiple tables can exist per page
            tables = defaultdict(list)
            for row in items:
                tables[row["table_index"]].append(row)
            for table_index, table_rows in tables.items():
                min_y = min(r["y0"] for r in table_rows)
                max_y = max(r["y1"] for r in table_rows)
                table_title = table_rows[0]["table_title"]
                meta_list.append({
                    "table_index": table_index,
                    "table_title": table_title,
                    "page": int(page),
                    "min_y": min_y,
                    "max_y": max_y,
                })
        return meta_list

    def process(self, table_structures):
        """
        Run the full pipeline and return (flat_list, table_y_metadata)
        """
        merged_results = self.extract_table_headers(table_structures)
        flat_list = self.flatten_table_results(merged_results)
        table_y_metadata = self.get_table_y_metadata(flat_list)
        horizontal_whitespace = self.query_engine.get("Horizontal.Whitespace")
        hz_whitespace = list(itertools.chain.from_iterable(horizontal_whitespace['results']['pages'].values()))
        flat_list_values = list(itertools.chain.from_iterable(flat_list['pages'].values()))
        preceding_whitepace = {
            'pages': {'0': attach_preceding_whitespace(flat_list_values, hz_whitespace, whitespace_key="table_top_bounds")}
        }
        return self.render_api(flat_list), table_y_metadata

    def render_api(self, flat_list):

        page_metadata = self.line_index.page_metadata

        return {
            "query_label": self.query_label,
            "description": self.description,
            "pdf_metadata": page_metadata,
            "results": dict(flat_list)
        }


def attach_preceding_whitespace(table_headers, whitespace_blocks, whitespace_key="table_top_bounds"):
    """
    1) For each table header, find the single closest whitespace block whose bottom (y1)
       is <= the header's top (y0).

    2) If a table has multiple headers (same table_index), pick the largest block among those
       individually closest blocks and attach that same block to all headers in that table.

    Args:
        table_headers (List[Dict]): Each dict has at least:
            {
              'table_index': ...,
              'bbox': [x0, y0, x1, y1],
              ...
            }
        whitespace_blocks (List[Dict]): Each dict with at least:
            {
              'bbox': [x0, y0, x1, y1],
              ...
            }
        whitespace_key (str): The key to attach in the 'table_metadata' for the matched block.

    Returns:
        List[Dict] - A copy of the original table_headers, each with an added `table_metadata[whitespace_key]`.
    """

    # 1) Enumerate headers to keep their original order index.
    #    This also lets us store results in a parallel structure.
    indexed_headers = list(enumerate(table_headers))

    # For each header, find the single "closest" whitespace above that header's top.
    header_to_block = {}

    for i, header in indexed_headers:
        y0 = header['bbox'][1]  # header top
        closest_block = None
        closest_diff = float('inf')  # track minimal vertical distance

        for block in whitespace_blocks:
            block_bottom = block['bbox'][3]
            if block_bottom <= y0:
                diff = y0 - block_bottom
                if diff < closest_diff:
                    closest_diff = diff
                    closest_block = block

        # Store whichever "closest block" we found (None if none are above).
        header_to_block[i] = closest_block

    # 2) Group headers by table_index so we know which ones belong to the same table.
    table_groups = defaultdict(list)
    for i, header in indexed_headers:
        table_idx = header.get('table_index')
        table_groups[table_idx].append(i)

    # 3) For each table that has multiple headers, pick the "highest" block from among
    #    their individually closest blocks, then assign it to all headers in that table.
    for table_idx, header_indices in table_groups.items():
        if len(header_indices) > 1:
            # Multiple headers for this table => pick the largest block among them
            # by vertical height = (y1 - y0).
            blocks = [header_to_block[i] for i in header_indices if header_to_block[i] is not None]
            if blocks:
                # return the block that is HIGHEST on the page. Representing the start of that shared tables Y0 bounds.
                largest_block = min(blocks, key=lambda b: (b['bbox'][3] - b['bbox'][1]))
                # Reassign that block to all headers in this group
                for i in header_indices:
                    header_to_block[i] = largest_block

    # 4) Reconstruct output in original order
    print(f"HEADER TO BLOCK KEYS: {max(list(header_to_block.keys()))}")
    out = []
    for i, header in indexed_headers:
        new_header = header.copy()
        new_header.setdefault('table_metadata', {})

        this_block = header_to_block[i]
        if this_block:
            this_block_copy = copy.deepcopy(this_block)

            # 1) cut it for the current header
            this_block_copy['bbox'] = cut_bbox_vertically(this_block_copy['bbox'], fraction=0.5, cut_from='top')
            new_header['table_metadata'][whitespace_key] = this_block_copy

            # 2) If i+1 is valid and references a block, store it as "bottom_bounds"
            if (i + 1) in header_to_block and header_to_block[i + 1]:
                next_block_copy = copy.deepcopy(header_to_block[i + 1])
                if next_block_copy['bbox'][1] <= this_block_copy['bbox'][3]:
                    pass 
                else:
                    next_block_copy['bbox'] = cut_bbox_vertically(next_block_copy['bbox'], fraction=0.5, cut_from='bottom')
                    new_header['table_metadata']["table_bottom_bounds"] = next_block_copy

        out.append(new_header)

    return out


def cut_bbox_vertically(bbox, fraction=0.5, cut_from='top'):
    """
    Cut a portion of the vertical span from a bounding box by either raising y0
    (cutting from the top) or lowering y1 (cutting from the bottom).

    Args:
        bbox (tuple or list): (x0, y0, x1, y1)
        fraction (float): The fraction of the bbox's height to cut. 
            For example, 0.5 means 'cut half the height.'
        cut_from (str): Either 'top' or 'bottom'. 
            If 'top', we increase y0 by fraction*height.
            If 'bottom', we decrease y1 by fraction*height.

    Returns:
        tuple: A new bounding box (x0, new_y0, x1, new_y1) after the cut.
    """
    x0, y0, x1, y1 = bbox
    height = y1 - y0
    cut_amount = fraction * height

    if cut_from.lower() == 'top':
        # Increase y0 by cut_amount
        y0_new = y0 + cut_amount
        return (x0, y0_new, x1, y1)
    elif cut_from.lower() == 'bottom':
        # Decrease y1 by cut_amount
        y1_new = y1 - cut_amount
        return (x0, y0, x1, y1_new)
    else:
        # Fallback or raise an error
        raise ValueError("cut_from must be either 'top' or 'bottom'.")