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

    
    def hash_columns(self, columns_list):
        """
        Create a stable hash from columns, which is now a list of strings.
        For example: ['Process', 'Number Sales', 'Net Sales', ...]
        """
        col_string = '|'.join(columns_list)  # Join them into one string
        return hashlib.md5(col_string.encode('utf-8')).hexdigest()

    def extract_table_headers(self, table_structures):
        """
        Extracts table headers from the new schema where each 'tables' object contains:
          - 'headers_hierarchy': bool (just metadata; no special search behavior here)
          - 'columns': list of header labels
    
        There's no explicit 'table_index' or 'table_title' in this new schema.
        We use the dictionary key in 'tables' as our table identifier (table_id).
        
        Returns a list of merged_results in the structure:
        [
          {
            'table_id': <string or int key from 'tables'>,
            'columns': <list of strings>,
            'headers_hierarchy': <bool>,
            'pages': {
              <page_number>: [
                {
                  'bbox': [...],
                  # Additional fields if needed
                },
                ...
              ]
            }
          },
          ...
        ]
        """
        # Ensure we have a built index for searching
        if self.line_index is None:
            self.build_index()
    
        column_matches_meta = []
    
        for structure in table_structures:
            # Safeguard against malformed entries
            if not isinstance(structure, dict):
                continue
            
            # 'tables' is a dict whose keys are table IDs (strings), values are table data
            tables = structure.get('tables', {})
            for table_id_str, table_data in tables.items():
    
                # Columns is a list of strings
                columns = table_data.get('columns', [])
                if not columns:
                    continue
    
                # Optional: We still capture the bool for later usage
                headers_hierarchy = table_data.get('headers_hierarchy', False)
    
                # Combine column names into a single string for searching
                col_search = ' '.join(columns)
    
                # Search in our self.line_index for that combined header text
                column_matches = self.line_index.query(
                    key="text",
                    func=lambda rows: search_normalized_text(
                        rows, target=col_search, mode="word", best_match=False
                    ),
                    description=f"MATCHES FOR table_id='{table_id_str}' columns"
                )
    
                # If no matches are found, try removing special characters and search again
                if not column_matches:
                    col_search_cleaned = remove_special_characters(col_search)
                    column_matches = self.line_index.query(
                        key="text",
                        func=lambda rows: search_normalized_text(
                            rows, target=col_search_cleaned, mode="word", best_match=False
                        ),
                        description=f"MATCHES FOR table_id='{table_id_str}' columns (cleaned)"
                    )
    
                # If we found matches, record them
                if column_matches:
                    for match in column_matches:
                        bounds = match['bboxes']
                        # Each match can have multiple spans, possibly on multiple pages
                        pages = list({span['page'] for span in match['spans']})
                        # If we only want single-page matches, we can require len(pages) == 1
                        # Otherwise, if multi-page is valid, you can skip this check.
                        if len(pages) == 1:
                            page = pages[0]
                            column_matches_meta.append({
                                'table_id': table_id_str,
                                'columns': columns,
                                'headers_hierarchy': headers_hierarchy,
                                'page': page,
                                'bounds': Map.merge_all_boxes(bounds)
                            })
    
        # Group matches by (table_id, columns, headers_hierarchy)
        grouped_matches = defaultdict(list)
        for entry in column_matches_meta:
            col_hash = self.hash_columns(entry['columns'])  # or create your own hash/tuple
            key = (entry['table_id'], col_hash, entry['headers_hierarchy'])
            grouped_matches[key].append(entry)
    
        # Merge bounding boxes by page for each group
        merged_results = []
        for (table_id, col_hash, hierarchy_flag), group in grouped_matches.items():
            pages_dict = {}
            # columns are stored directly as a list in the first entry
            columns = group[0]['columns']
            headers_hierarchy = group[0]['headers_hierarchy']
    
            for entry in group:
                page_num = entry['page']
                if page_num not in pages_dict:
                    pages_dict[page_num] = []
                pages_dict[page_num].append({
                    'bbox': entry['bounds'],
                    # Add other fields if needed
                })
    
            merged_results.append({
                'table_id': table_id,
                'columns': columns,
                'headers_hierarchy': headers_hierarchy,
                'pages': pages_dict
            })

        for merged_item in merged_results:
            # merged_item['pages'] is a dict: {page_num: [ { 'bbox': (...)}, ... ]}
            for page_num, box_dicts in merged_item['pages'].items():
                unique_box_dicts = []
                seen_boxes = set()
                for bd in box_dicts:
                    # Convert the bbox tuple to something hashable (it already is)
                    bbox_tuple = bd['bbox']
                    # If not already seen, add it and mark as seen
                    if bbox_tuple not in seen_boxes:
                        unique_box_dicts.append(bd)
                        seen_boxes.add(bbox_tuple)
                # Overwrite the list with only the unique set
                merged_item['pages'][page_num] = unique_box_dicts
        
        return merged_results
    
    def flatten_table_results(self, merged_results):
        """
        Returns:
            results (dict): { "pages": { page_num: [detailed dict, ...], ... } }
        """
        results = {"pages": defaultdict(list)}
        column_hash_to_index = {}
        table_counter = 0

        results = {"pages": defaultdict(list)}
        column_hash_to_index = {}
        table_counter = 0  # Start from 0 or 1 as you prefer
    
        # Pass 1: Assign a unique table_index per unique columns
        for table in merged_results:
            col_hash = self.hash_columns(table['columns'])
            if col_hash not in column_hash_to_index:
                column_hash_to_index[col_hash] = table_counter
                table_counter += 1
        print(column_hash_to_index)
        for table in merged_results:
            columns = table['columns']
            col_hash = self.hash_columns(columns)
            table_index = column_hash_to_index[col_hash]
            pages = table['pages']
            for page_num, bboxes in pages.items():
                for bounds_index, bbox_item in enumerate(bboxes):
                    bbox = bbox_item['bbox']
                    entry = {
                        "page": page_num,
                        "table_index": table_index,
                        "bounds_index": bounds_index,
                        "bbox": bbox,
                        "x0": bbox[0],
                        "y0": bbox[1],
                        "x1": bbox[2],
                        "y1": bbox[3],
                        "x_span": bbox[2] - bbox[0],
                        "y_span": bbox[3] - bbox[1],
                        "table_title": '',
                        'col_hash': col_hash,
                        "columns": columns,
                        "hierarchy": bbox_item.get("hierarchy", None),
                        "meta": {
                            "width": self.page_metadata.get(page_num, {}).get("width"),
                            "height": self.page_metadata.get(page_num, {}).get("height"),
                        },
                    }
                    results["pages"][str(page_num)].append(entry)
    
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
        table_structures = detect_and_clean_duplicated_columns(table_structures)
        merged_results = self.extract_table_headers(table_structures)
        flat_list = self.flatten_table_results(merged_results)
        table_y_metadata = self.get_table_y_metadata(flat_list)
        horizontal_whitespace = self.query_engine.get("Horizontal.Whitespace")
        hz_whitespace = list(itertools.chain.from_iterable(horizontal_whitespace['results']['pages'].values()))
        flat_list_values = list(itertools.chain.from_iterable(flat_list['pages'].values()))

        large_hz_whitespace = self.query_engine.get("Horizontal.Whitespace", y_tolerance=10)

        bounds_builder = TableBoundsBuilder(
            page_metadata=self.line_index.page_metadata,
            large_whitespace_blocks=large_hz_whitespace
        )

        relative_bounds, refined_relative_bounds = bounds_builder.build_relative_bounds(
            table_headers=deduplicate_table_headers(flat_list_values),
            whitespace_blocks=hz_whitespace
        )
        table_bounds_metadata = {
            'pages': {'0': relative_bounds }
        }
        # return {'flat_list_values': flat_list_values, 'hz_whitespace': hz_whitespace, 'page_metadata': self.line_index.page_metadata}
        return self.render_api(table_bounds_metadata), {
            'refined_relative_bounds': refined_relative_bounds, 
            'relative_bounds': relative_bounds, 
            'flat_list_values': flat_list_values, 
            'page_metadata': self.line_index.page_metadata,
            'large_hz_whitespace': large_hz_whitespace, 
            'hz_whitespace': hz_whitespace
        }

    def render_api(self, flat_list):

        page_metadata = self.line_index.page_metadata

        return {
            "query_label": self.query_label,
            "description": self.description,
            "pdf_metadata": page_metadata,
            "results": dict(flat_list)
        }


def detect_and_clean_duplicated_columns(vision_output):
    """
    Detects and cleans duplicated column patterns in vision model output.
    Works across all pages to find duplicated vs clean table patterns.
    
    This handles cases where side-by-side tables are detected as:
    1. One table with duplicated columns: ["Date", "Batch", "Amount", "Date", "Batch", "Amount"]
    2. Separate tables with the non-duplicated columns: ["Date", "Batch", "Amount"]
    
    If both patterns exist anywhere in the document, removes the duplicated version.
    
    Args:
        vision_output (List[Dict]): List of page results from vision model
        
    Returns:
        List[Dict]: Cleaned vision output with duplicated tables removed
    """
    # Step 1: Collect all tables across all pages
    all_tables = []  # List of (page_idx, table_id, table_info, columns)
    
    for page_idx, page_result in enumerate(vision_output):
        if 'tables' not in page_result:
            continue
            
        for table_id, table_info in page_result['tables'].items():
            columns = table_info.get('columns', [])
            all_tables.append((page_idx, table_id, table_info, columns))
    
    # Step 2: Identify duplicated patterns across all pages
    duplicated_patterns = {}  # maps deduplicated_pattern -> {'duplicated': [...], 'clean': [...]}
    
    for page_idx, table_id, table_info, columns in all_tables:
        # Check if this table has duplicated columns
        dedup_result = detect_column_duplication(columns)
        
        if dedup_result['has_duplication']:
            # This table has duplicated columns
            dedup_pattern = tuple(dedup_result['deduplicated'])
            if dedup_pattern not in duplicated_patterns:
                duplicated_patterns[dedup_pattern] = {'duplicated': [], 'clean': []}
            
            duplicated_patterns[dedup_pattern]['duplicated'].append({
                'page_idx': page_idx,
                'table_id': table_id,
                'table_info': table_info,
                'original_columns': columns,
                'duplication_factor': dedup_result['duplication_factor']
            })
        else:
            # This table has clean columns - check if it matches any duplicated patterns
            clean_pattern = tuple(columns)
            if clean_pattern not in duplicated_patterns:
                duplicated_patterns[clean_pattern] = {'duplicated': [], 'clean': []}
            
            duplicated_patterns[clean_pattern]['clean'].append({
                'page_idx': page_idx,
                'table_id': table_id,
                'table_info': table_info,
                'columns': columns
            })
    
    # Step 3: Determine which tables to remove (across all pages)
    tables_to_remove = set()  # Set of (page_idx, table_id) tuples
    
    for pattern, tables_info in duplicated_patterns.items():
        duplicated = tables_info['duplicated']
        clean = tables_info['clean']
        
        if len(duplicated) > 0 and len(clean) > 0:
            # Both duplicated and clean versions exist - remove duplicated ones
            for dup_table in duplicated:
                tables_to_remove.add((dup_table['page_idx'], dup_table['table_id']))
                print(f"Removing duplicated table: page {dup_table['page_idx']}, table {dup_table['table_id']}")
                print(f"  Columns: {dup_table['original_columns']}")
                print(f"  Clean versions found: {len(clean)} tables")
                for clean_table in clean:
                    print(f"    -> page {clean_table['page_idx']}, table {clean_table['table_id']}")
    
    # Step 4: Build cleaned output
    cleaned_output = []
    total_removed = 0
    
    for page_idx, page_result in enumerate(vision_output):
        if 'tables' not in page_result:
            cleaned_output.append(page_result)
            continue
            
        # Filter out tables marked for removal
        cleaned_tables = {}
        original_count = len(page_result['tables'])
        
        for table_id, table_info in page_result['tables'].items():
            if (page_idx, table_id) not in tables_to_remove:
                cleaned_tables[table_id] = table_info
            else:
                total_removed += 1
        
        # Update the page result
        cleaned_page = page_result.copy()
        cleaned_page['tables'] = cleaned_tables
        cleaned_page['number_of_tables'] = len(cleaned_tables)
        
        if len(cleaned_tables) != original_count:
            print(f"Page {page_idx}: {original_count} -> {len(cleaned_tables)} tables")
        
        cleaned_output.append(cleaned_page)
    
    if total_removed > 0:
        print(f"\nTotal tables removed: {total_removed}")
    
    return cleaned_output


def detect_column_duplication(columns):
    """
    Detects if a list of columns contains exact duplications.
    
    Args:
        columns (List[str]): List of column names
        
    Returns:
        Dict: {
            'has_duplication': bool,
            'deduplicated': List[str],
            'duplication_factor': int,
            'pattern_analysis': Dict
        }
    """
    if not columns or len(columns) == 0:
        return {
            'has_duplication': False,
            'deduplicated': [],
            'duplication_factor': 1,
            'pattern_analysis': {}
        }
    
    # Try different duplication factors (2, 3, 4, etc.)
    for factor in range(2, len(columns) + 1):
        if len(columns) % factor == 0:
            chunk_size = len(columns) // factor
            chunks = [columns[i:i + chunk_size] for i in range(0, len(columns), chunk_size)]
            
            # Check if all chunks are identical
            if all(chunk == chunks[0] for chunk in chunks):
                return {
                    'has_duplication': True,
                    'deduplicated': chunks[0],
                    'duplication_factor': factor,
                    'pattern_analysis': {
                        'original_length': len(columns),
                        'chunk_size': chunk_size,
                        'chunks': chunks
                    }
                }
    
    return {
        'has_duplication': False,
        'deduplicated': columns,
        'duplication_factor': 1,
        'pattern_analysis': {}
    }

def deduplicate_table_headers(table_headers):
    """
    Deduplicates table headers based on table_index and bounds_index combination.
    
    Args:
        table_headers (List[Dict]): List of table header dictionaries
        
    Returns:
        List[Dict]: Deduplicated list of table headers
    """
    seen_combinations = set()
    deduplicated = []
    
    for header in table_headers:
        # Create a unique key from table_index and bounds_index
        table_index = header.get('table_index')
        bounds_index = header.get('bounds_index')
        key = (table_index, bounds_index)
        
        # Only add if we haven't seen this combination before
        if key not in seen_combinations:
            seen_combinations.add(key)
            deduplicated.append(header)
        else:
            print(f"Skipping duplicate: table_index={table_index}, bounds_index={bounds_index}")
    
    print(f"Original count: {len(table_headers)}, Deduplicated count: {len(deduplicated)}")
    return deduplicated

class TableBoundsBuilder:
    """
    A class to attach and process whitespace blocks that precede table headers
    in order to determine full table boundaries, including support for side-by-side tables.
    """
    def __init__(self, page_metadata, large_whitespace_blocks, top_bounds_key="table_top_bounds", bottom_bounds_key="table_bottom_bounds", table_bounds_key="full_table_bounds", stage=0):
        """
        Initializes the TableWhitespaceAttacher.

        Args:
            whitespace_key (str, optional): The key to store the whitespace block in the
                                            header's 'table_metadata'. Defaults to "table_top_bounds".
            page_meta (Dict, optional): Page metadata containing width and height information.
                                        Defaults to None.
        """
        self.top_bounds_key = top_bounds_key
        self.bottom_bounds_key = bottom_bounds_key
        self.table_bounds_key = table_bounds_key
        self.stage = stage
        self.large_whitespace_blocks = large_whitespace_blocks
        self.page_meta = page_metadata.get(stage) if page_metadata is not None else {}
        self.page_dimensions = self._extract_page_dimensions(self.page_meta)

    def build_relative_bounds(self, table_headers, whitespace_blocks):
        """
        Attaches whitespace blocks above table headers to help determine full table bounds.
        Enhanced to handle side-by-side tables with duplicated columns.
        """
        # 2. Find the closest whitespace block above each header
        headers_with_index = list(enumerate(table_headers))
        header_to_block_map = self._find_closest_whitespace_blocks(headers_with_index, whitespace_blocks)

        # 3. Group headers by table index
        table_groups = self._group_headers_by_table(headers_with_index)

        # 4. Assign shared whitespace blocks for multi-header tables
        self._assign_shared_blocks_for_tables(table_groups, header_to_block_map)

        # 5. Create the final output with attached whitespace metadata
        relative_bounds = self._create_output_with_metadata(
            headers_with_index,
            header_to_block_map
        )

        base_results = []
        for table in relative_bounds:
            base_results.append({
                'table_index': table['table_index'], 
                'headers_bbox': table['bbox'],
                'columns': table['columns'],
                'hierarchy': table['hierarchy'],
                'bounds_index': table['bounds_index'],
                'table_top_bounds': table['table_metadata'].get('table_top_bounds', {}).get('bbox', None),
                'table_bottom_bounds':  table['table_metadata'].get('table_bottom_bounds', {}).get('bbox', None),
                'table_full_bounds':  table['table_metadata'].get('full_table_bounds', {}).get('bbox', None)
            })
        
        base_results = self.correct_table_bounds(base_results)
        refined_relative_bounds = self.adjust_tables_for_whitespace(base_results, self.large_whitespace_blocks)
        
        # NEW: Enhanced logic to handle side-by-side tables
        refined_relative_bounds = self._handle_side_by_side_tables(refined_relative_bounds)

        refinement_map={}
        for refinement in refined_relative_bounds:
            table_index = refinement['table_index']
            bounds_index = refinement['bounds_index']
            tbl_key = f"{table_index}.{bounds_index}"
            bounds = refinement['table_full_bounds']
            refinement_map[tbl_key] = bounds

        for result in relative_bounds:
            try:
                table_index = result['table_index']
                bounds_index = result['bounds_index']
                tbl_key = f"{table_index}.{bounds_index}"
                top_bounds_block = result['table_metadata']['table_top_bounds']
                this_block_copy = copy.deepcopy(top_bounds_block)
                tpy0=top_bounds_block['bbox'][1]
                refined_bounds = refinement_map.get(tbl_key)
                
                if refined_bounds is None:
                    print(f"Warning: No refined bounds found for {tbl_key}")
                    continue
                    
                x0,y0,x1,y1 = refined_bounds
                
                # Special handling for bounds_index == 0 to preserve top boundary
                if bounds_index == 0:
                    refined_bounds = x0,tpy0,x1,y1
                    
                this_block_copy['bbox'] = refined_bounds
                x0,y0,x1,y1 = refined_bounds
                this_block_copy['x0'] = x0
                this_block_copy['y0'] = y0
                this_block_copy['x1'] = x1
                this_block_copy['y1'] = y1
                result['table_metadata']['refined_table_bounds'] = this_block_copy
                
                print(f"Final refined bounds for {tbl_key}: {refined_bounds}")
                
            except Exception as e:
                print(f"Relative bound refinement failed for {result.get('table_index', 'unknown')}.{result.get('bounds_index', 'unknown')}: {e}")
                pass
        return relative_bounds, refined_relative_bounds

    def _handle_side_by_side_tables(self, base_results, overlap_threshold=5):
        """
        Enhanced logic to handle side-by-side tables with duplicated columns.
        This method detects when tables with the same table_index are positioned
        side-by-side and splits their bounds vertically.
        
        Args:
            base_results (List[Dict]): List of table results
            overlap_threshold (float): Minimum overlap in y-coordinates to consider tables side-by-side
            
        Returns:
            List[Dict]: Modified results with properly split bounds for side-by-side tables
        """
        # Group tables by table_index
        table_groups = defaultdict(list)
        for i, table in enumerate(base_results):
            table_groups[table['table_index']].append((i, table))
        
        # Process each group to detect side-by-side arrangements
        for table_index, tables in table_groups.items():
            if len(tables) <= 1:
                continue
            
            # Check if these tables have the same column structure (indicating duplicates)
            first_table = tables[0][1]
            same_columns = all(
                table.get('columns', []) == first_table.get('columns', []) or
                table.get('col_hash') == first_table.get('col_hash')
                for _, table in tables
            )
            
            if not same_columns:
                continue
                
            # Sort tables by bounds_index to maintain order
            tables.sort(key=lambda x: x[1]['bounds_index'])
            
            # Detect side-by-side arrangements
            side_by_side_groups = self._detect_side_by_side_groups(tables, overlap_threshold)
            
            # Split bounds for side-by-side groups
            for group in side_by_side_groups:
                if len(group) > 1:
                    print(f"Found side-by-side group with {len(group)} tables for table_index {table_index}")
                    self._split_bounds_vertically(group, base_results)
        
        return base_results
    
    def _detect_side_by_side_groups(self, tables, overlap_threshold):
        """
        Detects which tables are positioned side-by-side based on y-coordinate overlap.
        
        Args:
            tables (List[Tuple]): List of (index, table) tuples
            overlap_threshold (float): Minimum overlap to consider tables side-by-side
            
        Returns:
            List[List[Tuple]]: Groups of side-by-side tables
        """
        groups = []
        processed = set()
        
        for i, (idx1, table1) in enumerate(tables):
            if idx1 in processed:
                continue
                
            current_group = [(idx1, table1)]
            processed.add(idx1)
            
            # Check for other tables that overlap vertically with table1
            for j, (idx2, table2) in enumerate(tables[i+1:], i+1):
                if idx2 in processed:
                    continue
                    
                if self._tables_overlap_vertically(table1, table2, overlap_threshold):
                    current_group.append((idx2, table2))
                    processed.add(idx2)
            
            if len(current_group) > 1:
                groups.append(current_group)
        
        return groups
    
    def _tables_overlap_vertically(self, table1, table2, threshold):
        """
        Checks if two tables overlap vertically (share y-coordinate space).
        
        Args:
            table1, table2 (Dict): Table dictionaries with bbox (header bounds)
            threshold (float): Minimum overlap required
            
        Returns:
            bool: True if tables overlap vertically
        """
        # Use 'bbox' for header bounds, fallback to 'headers_bbox' if it exists
        bbox1 = table1.get('bbox') or table1.get('headers_bbox')
        bbox2 = table2.get('bbox') or table2.get('headers_bbox')
        
        if not bbox1 or not bbox2:
            return False
            
        y1_top, y1_bottom = bbox1[1], bbox1[3]
        y2_top, y2_bottom = bbox2[1], bbox2[3]
        
        # Calculate overlap
        overlap_start = max(y1_top, y2_top)
        overlap_end = min(y1_bottom, y2_bottom)
        overlap = max(0, overlap_end - overlap_start)
        
        return overlap >= threshold
    
    def _split_bounds_vertically(self, side_by_side_group, base_results):
        """
        Splits the bounds vertically for side-by-side tables using simple page-wide bounds.
        Also harmonizes the vertical bounds with special handling for first occurrence.
        
        For side-by-side tables:
        - Left table: x0 = 0 (page min), x1 = midpoint
        - Right table: x0 = midpoint, x1 = page width (page max)
        - First occurrence (bounds_index=0): Keep original y0 (headers), extend y1 to group max
        - Subsequent occurrences: Use unified y0 and y1 (data regions only)
        
        Args:
            side_by_side_group (List[Tuple]): Group of (index, table) tuples that are side-by-side
            base_results (List[Dict]): The full results list to modify
        """
        # Sort by x-coordinate (left to right)
        side_by_side_group.sort(key=lambda x: x[1].get('bbox', x[1].get('headers_bbox', [0, 0, 0, 0]))[0])
        
        # Get page width from the first table's metadata or use a reasonable default
        page_width = self.page_dimensions.get('width', 652.0)  # Default based on your example
        if page_width == 0:
            # Try to get it from table metadata
            first_table = side_by_side_group[0][1]
            page_width = first_table.get('meta', {}).get('width', 652.0)
        
        # Calculate single midpoint between the two tables
        if len(side_by_side_group) == 2:
            _, left_table = side_by_side_group[0]
            _, right_table = side_by_side_group[1]
            
            # Get header bboxes
            left_bbox = left_table.get('bbox') or left_table.get('headers_bbox', [0, 0, 0, 0])
            right_bbox = right_table.get('bbox') or right_table.get('headers_bbox', [0, 0, 0, 0])
            
            # Calculate midpoint between right edge of left table and left edge of right table
            midpoint = (left_bbox[2] + right_bbox[0]) / 2
        else:
            # For more than 2 tables, use equal divisions
            midpoint = page_width / 2
        
        # Find the unified vertical bounds for the group
        min_y0 = min(table['table_full_bounds'][1] for _, table in side_by_side_group if table.get('table_full_bounds'))
        max_y1 = max(table['table_full_bounds'][3] for _, table in side_by_side_group if table.get('table_full_bounds'))
        
        # Find if there's a bounds_index=0 table (first occurrence with headers)
        first_occurrence_y0 = None
        for _, table in side_by_side_group:
            if table.get('bounds_index') == 0 and table.get('headers_bbox'):
                first_occurrence_y0 = table['headers_bbox'][3]
                break
        
        print(f"Page width: {page_width}, Midpoint: {midpoint}")
        print(f"Group vertical bounds: min_y0={min_y0}, max_y1={max_y1}, first_occurrence_y0={first_occurrence_y0}")
        print(f"Processing {len(side_by_side_group)} side-by-side tables:")
        
        # Apply simple bounds to each table
        for i, (table_idx, table) in enumerate(side_by_side_group):
            print(f"  Table {i}: idx={table_idx}, bounds_index={table.get('bounds_index')}")
            
            # Check if this table has bounds to modify
            if table.get('table_full_bounds') is None:
                print(f"    Skipping - no table_full_bounds")
                continue
                
            # Convert to list for modification
            bounds = list(table['table_full_bounds'])
            original_bounds = bounds.copy()
            
            # Set horizontal bounds using margin-aware logic
            if i == 0:  # Left table
                # Calculate right margin: midpoint - right edge of headers
                header_bbox = left_table.get('bbox') or left_table.get('headers_bbox', [0, 0, 0, 0])
                right_margin = midpoint - header_bbox[2]
                
                # Set bounds: x0 = headers_x0 - right_margin, x1 = midpoint
                bounds[0] = max(0, header_bbox[0] - right_margin)  # Don't go negative
                bounds[2] = midpoint
                print(f"    Left table margins: header_x0={header_bbox[0]}, header_x1={header_bbox[2]}, right_margin={right_margin}")
            else:  # Right table (or subsequent tables)
                # Calculate left margin: left edge of headers - midpoint  
                header_bbox = right_table.get('bbox') or right_table.get('headers_bbox', [0, 0, 0, 0])
                left_margin = header_bbox[0] - midpoint
                
                # Set bounds: x0 = midpoint, x1 = headers_x1 + left_margin
                bounds[0] = midpoint
                bounds[2] = min(page_width, header_bbox[2] + left_margin)  # Don't exceed page width
                print(f"    Right table margins: header_x0={header_bbox[0]}, header_x1={header_bbox[2]}, left_margin={left_margin}")
            
            # Set vertical bounds based on bounds_index
            if table.get('bounds_index') == 0:
                # First occurrence: keep original y0 (headers), extend to max y1
                bounds[3] = max_y1  # Extend bottom to group maximum
                print(f"    First occurrence: keeping y0={bounds[1]}, extending y1 to {max_y1}")
            else:
                # Subsequent occurrences: use unified bounds (data region only)
                if first_occurrence_y0 is not None:
                    bounds[1] = first_occurrence_y0  # Start after headers
                else:
                    bounds[1] = min_y0  # Fallback to group minimum
                bounds[3] = max_y1
                print(f"    Subsequent occurrence: unified bounds y0={bounds[1]}, y1={bounds[3]}")
            
            print(f"    Updated bounds: {original_bounds} -> [{bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]}]")
            
            # Update the table's bounds in base_results
            base_results[table_idx]['table_full_bounds'] = tuple(bounds)
            
            # Also update other bound types if they exist
            if table.get('table_top_bounds') is not None:
                top_bounds = list(table['table_top_bounds'])
                top_bounds[0] = bounds[0]
                top_bounds[2] = bounds[2]
                # For top bounds, keep original vertical bounds or adjust minimally
                base_results[table_idx]['table_top_bounds'] = tuple(top_bounds)
                print(f"    Updated table_top_bounds: x0={top_bounds[0]}, x1={top_bounds[2]}")
            
            if table.get('table_bottom_bounds') is not None:
                bottom_bounds = list(table['table_bottom_bounds'])
                bottom_bounds[0] = bounds[0]
                bottom_bounds[2] = bounds[2]
                # For bottom bounds, use the unified bottom
                bottom_bounds[3] = max_y1
                base_results[table_idx]['table_bottom_bounds'] = tuple(bottom_bounds)
                print(f"    Updated table_bottom_bounds: x0={bottom_bounds[0]}, x1={bottom_bounds[2]}")
        
        print("Side-by-side bounds splitting completed.")

    def _extract_page_dimensions(self, page_meta):
        """
        Extract page width and height from page metadata.

        Args:
            page_meta (Dict): Metadata containing at least 'width' and 'height' keys.

        Returns:
            Dict: A dictionary with 'width' and 'height' keys, defaulting to 0 if not found.
        """
        width = page_meta.get('width', 0)
        height = page_meta.get('height', 0)
        return {'width': width, 'height': height}

    def _find_closest_whitespace_blocks(self, indexed_headers, whitespace_blocks):
        """
        For each header, finds the closest whitespace block positioned above it.

        Assumptions:
        - A whitespace block is "above" a header if its bottom (y1) <= header's top (y0)
        - The "closest" block has the smallest vertical distance to the header

        Args:
            indexed_headers (List[Tuple]): List of (index, header) tuples
            whitespace_blocks (List[Dict]): List of whitespace blocks

        Returns:
            Dict: Mapping of header index to its closest whitespace block
        """
        header_to_block = {}

        for i, header in indexed_headers:
            y0 = header['bbox'][1]  # Header top position
            closest_block = None
            closest_diff = float('inf')

            for block in whitespace_blocks:
                block_bottom = block['bbox'][3]
                if block_bottom <= y0:
                    diff = y0 - block_bottom
                    if diff < closest_diff:
                        closest_diff = diff
                        closest_block = block

            header_to_block[i] = closest_block

        return header_to_block

    def _group_headers_by_table(self, indexed_headers):
        """
        Groups header indices by their table_index.

        Assumption:
        - Headers with the same table_index belong to the same table

        Args:
            indexed_headers (List[Tuple]): List of (index, header) tuples

        Returns:
            Dict: Mapping of table_index to list of header indices
        """
        table_groups = defaultdict(list)
        for i, header in indexed_headers:
            table_idx = header.get('table_index')
            table_groups[table_idx].append(i)
        return table_groups

    def _assign_shared_blocks_for_tables(self, table_groups, header_to_block_map):
        """
        For tables with multiple headers, assigns the highest block to all headers in that table.

        Assumptions:
        - Multiple headers in the same table should share the same whitespace block
        - The "highest" block is the one with the smallest height (y1 - y0)

        Args:
            table_groups (Dict): Mapping of table_index to list of header indices
            header_to_block_map (Dict): Mapping of header index to whitespace block

        Returns:
            None: Modifies header_to_block_map in place
        """
        for table_idx, header_indices in table_groups.items():
            if len(header_indices) > 1:
                # Get all non-None blocks from headers in this table
                blocks = [
                    header_to_block_map[i] for i in header_indices
                    if header_to_block_map[i] is not None
                ]

                if blocks:
                    # Find the highest block on the page (smallest vertical span)
                    highest_block = min(blocks, key=lambda b: b['bbox'][3])

                    # Assign this block to all headers in the table
                    for i in header_indices:
                        header_to_block_map[i] = highest_block

    def _create_output_with_metadata(self, indexed_headers, header_to_block_map):
        """
        Creates the final output with whitespace metadata attached to each header.

        This function:
        1. Adds the whitespace block to each header as table_metadata
        2. Processes adjacent headers to determine table bottom bounds
        3. Creates full table bounds by merging top and bottom bounds
        4. Handles special case for the last table extending to page bottom

        Assumptions:
        - The last table may extend to the bottom of the page
        - Headers should be processed in their original order
        - Each header needs a copy of its whitespace block

        Args:
            indexed_headers (List[Tuple]): List of (index, header) tuples
            header_to_block_map (Dict): Mapping of header index to whitespace block
            whitespace_key (str): Key to store the whitespace block in table_metadata
            page_dimensions (Dict): Dictionary containing page width and height

        Returns:
            List[Dict]: Headers with attached metadata
        """
        max_table_index = max(list(header_to_block_map.keys())) if header_to_block_map else -1
        out = []

        for i, header in indexed_headers:
            new_header = header.copy()
            new_header.setdefault('table_metadata', {})

            this_block = header_to_block_map.get(i)
            if this_block:
                # Create copies of the block for different purposes
                this_block_copy = copy.deepcopy(this_block)
                this_block_copy2 = copy.deepcopy(this_block)

                # Cut the whitespace block vertically for the current header
                this_block_copy['bbox'] = self._cut_bbox_vertically(
                    this_block_copy['bbox'], fraction=0.5, cut_from='top'
                )
                new_header['table_metadata'][self.top_bounds_key] = this_block_copy

                # Process bottom bounds, including special case for the last table
                self._process_bottom_bounds(
                    new_header,
                    header_to_block_map,
                    i,
                    this_block_copy,
                    this_block_copy2,
                    max_table_index
                )

            out.append(new_header)

        return out

    def _process_bottom_bounds(self, new_header, header_to_block_map,
                               current_index, top_block, full_block,
                               max_table_index):
        """
        Processes the bottom bounds of a table using the next header's whitespace block
        or the page bottom for the last table.

        Assumptions:
        - Bottom bounds are determined by the whitespace block of the next header
        - If the next block overlaps with the current one, no change is made
        - The last table (at max_table_index) extends to the bottom of the page

        Args:
            new_header (Dict): The header being processed
            header_to_block_map (Dict): Mapping of header index to whitespace block
            current_index (int): Index of the current header
            top_block (Dict): The whitespace block for the table's top bound
            full_block (Dict): Block to be used for the full table bounds
            max_table_index (int): The maximum header index (for identifying the last table)
            page_dimensions (Dict): Dictionary containing page width and height

        Returns:
            None: Modifies new_header in place
        """
        # 1. Special handling for the last table - it extends to the bottom of the page
        if current_index == max_table_index and self.page_dimensions['width'] and self.page_dimensions['height']:
            yy0 = top_block['bbox'][1]
            page_end_box = [0, yy0, self.page_dimensions['width'], self.page_dimensions['height']]
            full_block['bbox'] = page_end_box
            print(f"I AM HERE: full_block: {full_block}")
            new_header['table_metadata']["full_table_bounds"] = full_block
            return

        # 2. Regular case - determine bottom bounds from the next header's whitespace block
        next_index = current_index + 1
        if next_index in header_to_block_map and header_to_block_map[next_index]:
            next_block_copy = copy.deepcopy(header_to_block_map[next_index])

            # Skip if next block is above or overlaps with current block
            if next_block_copy['bbox'][1] <= top_block['bbox'][3]:
                return

            # Cut the next block vertically for bottom bounds
            next_block_copy['bbox'] = self._cut_bbox_vertically(
                next_block_copy['bbox'], fraction=0.5, cut_from='bottom'
            )
            new_header['table_metadata'][self.bottom_bounds_key] = next_block_copy

            # Create full table bounds by merging top and bottom bounds
            merged_table_bounds = Map.merge_all_boxes([
                next_block_copy['bbox'], top_block['bbox']
            ])
            full_block['bbox'] = merged_table_bounds
            new_header['table_metadata']["full_table_bounds"] = full_block

    def _cut_bbox_vertically(self, bbox, fraction=0.5, cut_from='top'):
        """
        Cuts a bounding box vertically at a specified fraction.

        Args:
            bbox (List[float]): Bounding box coordinates [x0, y0, x1, y1]
            fraction (float, optional): Fraction at which to cut. Defaults to 0.5.
            cut_from (str, optional): Where to apply the cut from ('top' or 'bottom'). Defaults to 'top'.

        Returns:
            List[float]: Modified bounding box
        """
        x0, y0, x1, y1 = bbox
        height = y1 - y0
        cut_height = height * fraction

        if cut_from == 'top':
            return [x0, y0 + cut_height, x1, y1]
        else:  # cut_from == 'bottom'
            return [x0, y0, x1, y1 - cut_height]
        
    def correct_table_bounds(self, base_results):
        # Step 1: Group tables by table_index
        table_indices = {}
        for i, table in enumerate(base_results):
            table_index = table['table_index']
            if table_index not in table_indices:
                table_indices[table_index] = []
            table_indices[table_index].append(i)
        
        # Step 2: Process tables with repeated indices
        for table_index, positions in table_indices.items():
            if len(positions) > 1:  # If this table_index appears more than once
                # Sort the tables by bounds_index
                sorted_positions = sorted(positions, 
                                        key=lambda pos: base_results[pos]['bounds_index'])
                
                for i in range(len(sorted_positions) - 1):
                    curr_pos = sorted_positions[i]
                    next_pos = sorted_positions[i + 1]
                    
                    curr_table = base_results[curr_pos]
                    next_table = base_results[next_pos]
                    
                    # Set the current table's bottom bounds to the y0 of the next table's headers
                    next_headers_y0 = next_table['headers_bbox'][1]
                    
                    # Update the current table's bottom bounds
                    if curr_table['table_bottom_bounds'] is None:
                        # Create table_bottom_bounds using x values from table_top_bounds
                        if curr_table['table_top_bounds'] is not None:
                            top_bounds = curr_table['table_top_bounds']
                            curr_table['table_bottom_bounds'] = [top_bounds[0], next_headers_y0 - 6, top_bounds[2], next_headers_y0]
                    
                    # Update the current table's full bounds
                    if curr_table['table_full_bounds'] is None:
                        # Create table_full_bounds using existing bounds
                        if curr_table['table_top_bounds'] is not None:
                            top_bounds = curr_table['table_top_bounds']
                            curr_table['table_full_bounds'] = [top_bounds[0], top_bounds[1], top_bounds[2], next_headers_y0]
                
                # For tables with bounds_index > 0, adjust their table_full_bounds to start after their headers
                for i in range(1, len(sorted_positions)):
                    curr_pos = sorted_positions[i]
                    curr_table = base_results[curr_pos]
                    
                    # Adjust the y0 value for table_full_bounds to start after this table's headers
                    headers_y1 = curr_table['headers_bbox'][3]
                    
                    if curr_table['table_full_bounds'] is not None:
                        curr_full_bounds = list(curr_table['table_full_bounds'])
                        curr_full_bounds[1] = headers_y1
                        curr_table['table_full_bounds'] = tuple(curr_full_bounds)
        
        return base_results

    def adjust_tables_for_whitespace(self, base_results, whitespace_data, min_header_distance=30):
        # Extract whitespace gaps from the provided data
        whitespace_gaps = []
        if 'results' in whitespace_data and 'pages' in whitespace_data['results']:
            for page_num, gaps in whitespace_data['results']['pages'].items():
                whitespace_gaps.extend(gaps)
        
        # Sort whitespace gaps by y0 (top position)
        whitespace_gaps.sort(key=lambda gap: gap['y0'])
        
        # For each table, check for overlap with whitespace gaps
        for table in base_results:
            # Skip if table doesn't have full bounds
            if table['table_full_bounds'] is None:
                continue
            
            # Convert tuple to list for modification if needed
            if isinstance(table['table_full_bounds'], tuple):
                table_bounds = list(table['table_full_bounds'])
            else:
                table_bounds = table['table_full_bounds']
            
            table_y0 = table_bounds[1]
            table_y1 = table_bounds[3]
            
            # Get the table header's bottom y-coordinate (if available)
            header_y1 = table_y0  # Default to table top if header info not available
            if 'header_bounds' in table and table['header_bounds'] is not None:
                if isinstance(table['header_bounds'], tuple):
                    header_y1 = table['header_bounds'][3]  # Bottom of header
                else:
                    header_y1 = table['header_bounds'][3]
            
            # Check each whitespace gap for overlap
            for gap in whitespace_gaps:
                gap_y0 = gap['y0']
                gap_y1 = gap['y1']
                
                # If the gap is fully within the table bounds
                if table_y0 < gap_y0 and table_y1 > gap_y1:
                    # Only adjust if the gap is at least min_header_distance away from the header
                    if gap_y0 - header_y1 >= min_header_distance:
                        # Adjust the table's y1 to end at the start of the whitespace gap
                        table_bounds[3] = gap_y0
                        
                        # Update the table's full bounds
                        if isinstance(table['table_full_bounds'], tuple):
                            table['table_full_bounds'] = tuple(table_bounds)
                        else:
                            table['table_full_bounds'] = table_bounds
                        
                        # Once we've adjusted based on a gap, we don't need to check other gaps
                        break
        
        return base_results

# class TableBoundsBuilder:
#     """
#     A class to attach and process whitespace blocks that precede table headers
#     in order to determine full table boundaries.
#     """
#     def __init__(self, page_metadata, large_whitespace_blocks, top_bounds_key="table_top_bounds", bottom_bounds_key="table_bottom_bounds", table_bounds_key="full_table_bounds", stage=0):
#         """
#         Initializes the TableWhitespaceAttacher.

#         Args:
#             whitespace_key (str, optional): The key to store the whitespace block in the
#                                             header's 'table_metadata'. Defaults to "table_top_bounds".
#             page_meta (Dict, optional): Page metadata containing width and height information.
#                                         Defaults to None.
#         """
#         self.top_bounds_key = top_bounds_key
#         self.bottom_bounds_key = bottom_bounds_key
#         self.table_bounds_key = table_bounds_key
#         self.stage = stage
#         self.large_whitespace_blocks = large_whitespace_blocks
#         self.page_meta = page_metadata.get(stage) if page_metadata is not None else {}
#         self.page_dimensions = self._extract_page_dimensions(self.page_meta)

#     def build_relative_bounds(self, table_headers, whitespace_blocks):
#         """
#         Attaches whitespace blocks above table headers to help determine full table bounds.

#         This method processes table headers and whitespace blocks to determine the vertical
#         boundaries of tables in a document. It finds whitespace areas that precede each table
#         header and attaches them as metadata to create a representation of the table's bounds.

#         Key assumptions:
#         - Table headers have a 'table_index' that groups headers belonging to the same table
#         - Tables are separated by whitespace blocks in the document
#         - Headers and whitespace blocks have 'bbox' coordinates in [x0, y0, x1, y1] format
#         - Lower y-values are higher on the page (y0 = top, y1 = bottom)
#         - For multi-header tables, all headers share the same whitespace block

#         Args:
#             table_headers (List[Dict]): Each dict must contain:
#                 {
#                     'table_index': int,            # Identifier for grouping headers of the same table
#                     'bbox': [x0, y0, x1, y1],      # Bounding box coordinates
#                     ...
#                 }
#             whitespace_blocks (List[Dict]): Each dict must contain:
#                 {
#                     'bbox': [x0, y0, x1, y1],      # Bounding box coordinates
#                     ...
#                 }

#         Returns:
#             List[Dict]: A copy of the original table_headers, each with added table_metadata containing
#                         the matched whitespace blocks and table bounds information.
#         """
#         # 2. Find the closest whitespace block above each header
#         headers_with_index = list(enumerate(table_headers))
#         header_to_block_map = self._find_closest_whitespace_blocks(headers_with_index, whitespace_blocks)

#         # 3. Group headers by table index
#         table_groups = self._group_headers_by_table(headers_with_index)

#         # 4. Assign shared whitespace blocks for multi-header tables
#         self._assign_shared_blocks_for_tables(table_groups, header_to_block_map)

#         # 5. Create the final output with attached whitespace metadata
#         relative_bounds = self._create_output_with_metadata(
#             headers_with_index,
#             header_to_block_map
#         )

#         base_results = []
#         for table in relative_bounds:
#             base_results.append({
#                 'table_index': table['table_index'], 
#                 'headers_bbox': table['bbox'],
#                 'columns': table['columns'],
#                 'hierarchy': table['hierarchy'],
#                 'bounds_index': table['bounds_index'],
#                 'table_top_bounds': table['table_metadata'].get('table_top_bounds', {}).get('bbox', None),
#                 'table_bottom_bounds':  table['table_metadata'].get('table_bottom_bounds', {}).get('bbox', None),
#                 'table_full_bounds':  table['table_metadata'].get('full_table_bounds', {}).get('bbox', None)
#             })
#         base_results = self.correct_table_bounds(base_results)
#         refined_relative_bounds = self.adjust_tables_for_whitespace(base_results, self.large_whitespace_blocks)

#         refinement_map={}
#         for refinement in refined_relative_bounds:
#             table_index = refinement['table_index']
#             bounds_index = refinement['bounds_index']
#             tbl_key = f"{table_index}.{bounds_index}"
#             bounds = refinement['table_full_bounds']
#             refinement_map[tbl_key] = bounds

#         for result in relative_bounds:
#             try:
#                 table_index = result['table_index']
#                 bounds_index = result['bounds_index']
#                 tbl_key = f"{table_index}.{bounds_index}"
#                 top_bounds_block = result['table_metadata']['table_top_bounds']
#                 this_block_copy = copy.deepcopy(top_bounds_block)
#                 tpy0=top_bounds_block['bbox'][1]
#                 refined_bounds = refinement_map[tbl_key]
#                 x0,y0,x1,y1 = refined_bounds
#                 if bounds_index == 0:
#                     refined_bounds = x0,tpy0,x1,y1
#                 this_block_copy['bbox'] = refined_bounds
#                 x0,y0,x1,y1 = refined_bounds
#                 this_block_copy['x0'] = x0
#                 this_block_copy['y0'] = y0
#                 this_block_copy['x1'] = x1
#                 this_block_copy['y1'] = y1
#                 result['table_metadata']['refined_table_bounds'] = this_block_copy
#             except:
#                 print(f"Relative bound refinement failed: {result}")
#                 pass
#         return relative_bounds, refined_relative_bounds

#     def _extract_page_dimensions(self, page_meta):
#         """
#         Extract page width and height from page metadata.

#         Args:
#             page_meta (Dict): Metadata containing at least 'width' and 'height' keys.

#         Returns:
#             Dict: A dictionary with 'width' and 'height' keys, defaulting to 0 if not found.
#         """
#         width = page_meta.get('width', 0)
#         height = page_meta.get('height', 0)
#         return {'width': width, 'height': height}

#     def _find_closest_whitespace_blocks(self, indexed_headers, whitespace_blocks):
#         """
#         For each header, finds the closest whitespace block positioned above it.

#         Assumptions:
#         - A whitespace block is "above" a header if its bottom (y1) <= header's top (y0)
#         - The "closest" block has the smallest vertical distance to the header

#         Args:
#             indexed_headers (List[Tuple]): List of (index, header) tuples
#             whitespace_blocks (List[Dict]): List of whitespace blocks

#         Returns:
#             Dict: Mapping of header index to its closest whitespace block
#         """
#         header_to_block = {}

#         for i, header in indexed_headers:
#             y0 = header['bbox'][1]  # Header top position
#             closest_block = None
#             closest_diff = float('inf')

#             for block in whitespace_blocks:
#                 block_bottom = block['bbox'][3]
#                 if block_bottom <= y0:
#                     diff = y0 - block_bottom
#                     if diff < closest_diff:
#                         closest_diff = diff
#                         closest_block = block

#             header_to_block[i] = closest_block

#         return header_to_block

#     def _group_headers_by_table(self, indexed_headers):
#         """
#         Groups header indices by their table_index.

#         Assumption:
#         - Headers with the same table_index belong to the same table

#         Args:
#             indexed_headers (List[Tuple]): List of (index, header) tuples

#         Returns:
#             Dict: Mapping of table_index to list of header indices
#         """
#         table_groups = defaultdict(list)
#         for i, header in indexed_headers:
#             table_idx = header.get('table_index')
#             table_groups[table_idx].append(i)
#         return table_groups

#     def _assign_shared_blocks_for_tables(self, table_groups, header_to_block_map):
#         """
#         For tables with multiple headers, assigns the highest block to all headers in that table.

#         Assumptions:
#         - Multiple headers in the same table should share the same whitespace block
#         - The "highest" block is the one with the smallest height (y1 - y0)

#         Args:
#             table_groups (Dict): Mapping of table_index to list of header indices
#             header_to_block_map (Dict): Mapping of header index to whitespace block

#         Returns:
#             None: Modifies header_to_block_map in place
#         """
#         for table_idx, header_indices in table_groups.items():
#             if len(header_indices) > 1:
#                 # Get all non-None blocks from headers in this table
#                 blocks = [
#                     header_to_block_map[i] for i in header_indices
#                     if header_to_block_map[i] is not None
#                 ]

#                 if blocks:
#                     # Find the highest block on the page (smallest vertical span)
#                     highest_block = min(blocks, key=lambda b: b['bbox'][3])

#                     # Assign this block to all headers in the table
#                     for i in header_indices:
#                         header_to_block_map[i] = highest_block

#     def _create_output_with_metadata(self, indexed_headers, header_to_block_map):
#         """
#         Creates the final output with whitespace metadata attached to each header.

#         This function:
#         1. Adds the whitespace block to each header as table_metadata
#         2. Processes adjacent headers to determine table bottom bounds
#         3. Creates full table bounds by merging top and bottom bounds
#         4. Handles special case for the last table extending to page bottom

#         Assumptions:
#         - The last table may extend to the bottom of the page
#         - Headers should be processed in their original order
#         - Each header needs a copy of its whitespace block

#         Args:
#             indexed_headers (List[Tuple]): List of (index, header) tuples
#             header_to_block_map (Dict): Mapping of header index to whitespace block
#             whitespace_key (str): Key to store the whitespace block in table_metadata
#             page_dimensions (Dict): Dictionary containing page width and height

#         Returns:
#             List[Dict]: Headers with attached metadata
#         """
#         max_table_index = max(list(header_to_block_map.keys())) if header_to_block_map else -1
#         out = []

#         for i, header in indexed_headers:
#             new_header = header.copy()
#             new_header.setdefault('table_metadata', {})

#             this_block = header_to_block_map.get(i)
#             if this_block:
#                 # Create copies of the block for different purposes
#                 this_block_copy = copy.deepcopy(this_block)
#                 this_block_copy2 = copy.deepcopy(this_block)

#                 # Cut the whitespace block vertically for the current header
#                 this_block_copy['bbox'] = self._cut_bbox_vertically(
#                     this_block_copy['bbox'], fraction=0.5, cut_from='top'
#                 )
#                 new_header['table_metadata'][self.top_bounds_key] = this_block_copy

#                 # Process bottom bounds, including special case for the last table
#                 self._process_bottom_bounds(
#                     new_header,
#                     header_to_block_map,
#                     i,
#                     this_block_copy,
#                     this_block_copy2,
#                     max_table_index
#                 )

#             out.append(new_header)

#         return out

#     def _process_bottom_bounds(self, new_header, header_to_block_map,
#                                current_index, top_block, full_block,
#                                max_table_index):
#         """
#         Processes the bottom bounds of a table using the next header's whitespace block
#         or the page bottom for the last table.

#         Assumptions:
#         - Bottom bounds are determined by the whitespace block of the next header
#         - If the next block overlaps with the current one, no change is made
#         - The last table (at max_table_index) extends to the bottom of the page

#         Args:
#             new_header (Dict): The header being processed
#             header_to_block_map (Dict): Mapping of header index to whitespace block
#             current_index (int): Index of the current header
#             top_block (Dict): The whitespace block for the table's top bound
#             full_block (Dict): Block to be used for the full table bounds
#             max_table_index (int): The maximum header index (for identifying the last table)
#             page_dimensions (Dict): Dictionary containing page width and height

#         Returns:
#             None: Modifies new_header in place
#         """
#         # 1. Special handling for the last table - it extends to the bottom of the page
#         if current_index == max_table_index and self.page_dimensions['width'] and self.page_dimensions['height']:
#             yy0 = top_block['bbox'][1]
#             page_end_box = [0, yy0, self.page_dimensions['width'], self.page_dimensions['height']]
#             full_block['bbox'] = page_end_box
#             print(f"I AM HERE: full_block: {full_block}")
#             new_header['table_metadata']["full_table_bounds"] = full_block
#             return

#         # 2. Regular case - determine bottom bounds from the next header's whitespace block
#         next_index = current_index + 1
#         if next_index in header_to_block_map and header_to_block_map[next_index]:
#             next_block_copy = copy.deepcopy(header_to_block_map[next_index])

#             # tolerance = 5.0  # or whatever threshold you want
#             # if next_block_copy['bbox'][1] <= top_block['bbox'][3] + tolerance:
#             #     # Possibly skip or partially adjust logic
#             #     pass
#             # # Skip if next block is above or overlaps with current block
#             if next_block_copy['bbox'][1] <= top_block['bbox'][3]:
#                 return

#             # Cut the next block vertically for bottom bounds
#             next_block_copy['bbox'] = self._cut_bbox_vertically(
#                 next_block_copy['bbox'], fraction=0.5, cut_from='bottom'
#             )
#             new_header['table_metadata'][self.bottom_bounds_key] = next_block_copy

#             # Create full table bounds by merging top and bottom bounds
#             merged_table_bounds = Map.merge_all_boxes([
#                 next_block_copy['bbox'], top_block['bbox']
#             ])
#             full_block['bbox'] = merged_table_bounds
#             new_header['table_metadata']["full_table_bounds"] = full_block

#     def _cut_bbox_vertically(self, bbox, fraction=0.5, cut_from='top'):
#         """
#         Cuts a bounding box vertically at a specified fraction.

#         Args:
#             bbox (List[float]): Bounding box coordinates [x0, y0, x1, y1]
#             fraction (float, optional): Fraction at which to cut. Defaults to 0.5.
#             cut_from (str, optional): Where to apply the cut from ('top' or 'bottom'). Defaults to 'top'.

#         Returns:
#             List[float]: Modified bounding box
#         """
#         x0, y0, x1, y1 = bbox
#         height = y1 - y0
#         cut_height = height * fraction

#         if cut_from == 'top':
#             return [x0, y0 + cut_height, x1, y1]
#         else:  # cut_from == 'bottom'
#             return [x0, y0, x1, y1 - cut_height]
        
#     def correct_table_bounds(self, base_results):
#         # Step 1: Group tables by table_index
#         table_indices = {}
#         for i, table in enumerate(base_results):
#             table_index = table['table_index']
#             if table_index not in table_indices:
#                 table_indices[table_index] = []
#             table_indices[table_index].append(i)
        
#         # Step 2: Process tables with repeated indices
#         for table_index, positions in table_indices.items():
#             if len(positions) > 1:  # If this table_index appears more than once
#                 # Sort the tables by bounds_index
#                 sorted_positions = sorted(positions, 
#                                         key=lambda pos: base_results[pos]['bounds_index'])
                
#                 for i in range(len(sorted_positions) - 1):
#                     curr_pos = sorted_positions[i]
#                     next_pos = sorted_positions[i + 1]
                    
#                     curr_table = base_results[curr_pos]
#                     next_table = base_results[next_pos]
                    
#                     # Set the current table's bottom bounds to the y0 of the next table's headers
#                     next_headers_y0 = next_table['headers_bbox'][1]
                    
#                     # Update the current table's bottom bounds
#                     if curr_table['table_bottom_bounds'] is None:
#                         # Create table_bottom_bounds using x values from table_top_bounds
#                         if curr_table['table_top_bounds'] is not None:
#                             top_bounds = curr_table['table_top_bounds']
#                             curr_table['table_bottom_bounds'] = [top_bounds[0], next_headers_y0 - 6, top_bounds[2], next_headers_y0]
                    
#                     # Update the current table's full bounds
#                     if curr_table['table_full_bounds'] is None:
#                         # Create table_full_bounds using existing bounds
#                         if curr_table['table_top_bounds'] is not None:
#                             top_bounds = curr_table['table_top_bounds']
#                             curr_table['table_full_bounds'] = [top_bounds[0], top_bounds[1], top_bounds[2], next_headers_y0]
                
#                 # For tables with bounds_index > 0, adjust their table_full_bounds to start after their headers
#                 for i in range(1, len(sorted_positions)):
#                     curr_pos = sorted_positions[i]
#                     curr_table = base_results[curr_pos]
                    
#                     # Adjust the y0 value for table_full_bounds to start after this table's headers
#                     headers_y1 = curr_table['headers_bbox'][3]
                    
#                     if curr_table['table_full_bounds'] is not None:
#                         curr_full_bounds = list(curr_table['table_full_bounds'])
#                         curr_full_bounds[1] = headers_y1
#                         curr_table['table_full_bounds'] = tuple(curr_full_bounds)
        
#         return base_results

#     def adjust_tables_for_whitespace(self, base_results, whitespace_data, min_header_distance=30):
#         # Extract whitespace gaps from the provided data
#         whitespace_gaps = []
#         if 'results' in whitespace_data and 'pages' in whitespace_data['results']:
#             for page_num, gaps in whitespace_data['results']['pages'].items():
#                 whitespace_gaps.extend(gaps)
        
#         # Sort whitespace gaps by y0 (top position)
#         whitespace_gaps.sort(key=lambda gap: gap['y0'])
        
#         # For each table, check for overlap with whitespace gaps
#         for table in base_results:
#             # Skip if table doesn't have full bounds
#             if table['table_full_bounds'] is None:
#                 continue
            
#             # Convert tuple to list for modification if needed
#             if isinstance(table['table_full_bounds'], tuple):
#                 table_bounds = list(table['table_full_bounds'])
#             else:
#                 table_bounds = table['table_full_bounds']
            
#             table_y0 = table_bounds[1]
#             table_y1 = table_bounds[3]
            
#             # Get the table header's bottom y-coordinate (if available)
#             header_y1 = table_y0  # Default to table top if header info not available
#             if 'header_bounds' in table and table['header_bounds'] is not None:
#                 if isinstance(table['header_bounds'], tuple):
#                     header_y1 = table['header_bounds'][3]  # Bottom of header
#                 else:
#                     header_y1 = table['header_bounds'][3]
            
#             # Check each whitespace gap for overlap
#             for gap in whitespace_gaps:
#                 gap_y0 = gap['y0']
#                 gap_y1 = gap['y1']
                
#                 # If the gap is fully within the table bounds
#                 if table_y0 < gap_y0 and table_y1 > gap_y1:
#                     # Only adjust if the gap is at least min_header_distance away from the header
#                     if gap_y0 - header_y1 >= min_header_distance:
#                         # Adjust the table's y1 to end at the start of the whitespace gap
#                         table_bounds[3] = gap_y0
                        
#                         # Update the table's full bounds
#                         if isinstance(table['table_full_bounds'], tuple):
#                             table['table_full_bounds'] = tuple(table_bounds)
#                         else:
#                             table['table_full_bounds'] = table_bounds
                        
#                         # Once we've adjusted based on a gap, we don't need to check other gaps
#                         break
        
#         return base_results