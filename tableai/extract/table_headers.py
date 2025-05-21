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
            table_headers=flat_list_values,
            whitespace_blocks=hz_whitespace
        )
        table_bounds_metadata = {
            'pages': {'0': relative_bounds }
        }
        # return {'flat_list_values': flat_list_values, 'hz_whitespace': hz_whitespace, 'page_metadata': self.line_index.page_metadata}
        return self.render_api(table_bounds_metadata), {'refined_relative_bounds': refined_relative_bounds, 'relative_bounds': relative_bounds}

    def render_api(self, flat_list):

        page_metadata = self.line_index.page_metadata

        return {
            "query_label": self.query_label,
            "description": self.description,
            "pdf_metadata": page_metadata,
            "results": dict(flat_list)
        }



class TableBoundsBuilder:
    """
    A class to attach and process whitespace blocks that precede table headers
    in order to determine full table boundaries.
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

        This method processes table headers and whitespace blocks to determine the vertical
        boundaries of tables in a document. It finds whitespace areas that precede each table
        header and attaches them as metadata to create a representation of the table's bounds.

        Key assumptions:
        - Table headers have a 'table_index' that groups headers belonging to the same table
        - Tables are separated by whitespace blocks in the document
        - Headers and whitespace blocks have 'bbox' coordinates in [x0, y0, x1, y1] format
        - Lower y-values are higher on the page (y0 = top, y1 = bottom)
        - For multi-header tables, all headers share the same whitespace block

        Args:
            table_headers (List[Dict]): Each dict must contain:
                {
                    'table_index': int,            # Identifier for grouping headers of the same table
                    'bbox': [x0, y0, x1, y1],      # Bounding box coordinates
                    ...
                }
            whitespace_blocks (List[Dict]): Each dict must contain:
                {
                    'bbox': [x0, y0, x1, y1],      # Bounding box coordinates
                    ...
                }

        Returns:
            List[Dict]: A copy of the original table_headers, each with added table_metadata containing
                        the matched whitespace blocks and table bounds information.
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
                refined_bounds = refinement_map[tbl_key]
                x0,y0,x1,y1 = refined_bounds
                if bounds_index == 0:
                    refined_bounds = x0,tpy0,x1,y1
                this_block_copy['bbox'] = refined_bounds
                x0,y0,x1,y1 = refined_bounds
                this_block_copy['x0'] = x0
                this_block_copy['y0'] = y0
                this_block_copy['x1'] = x1
                this_block_copy['y1'] = y1
                result['table_metadata']['refined_table_bounds'] = this_block_copy
            except:
                print(f"Relative bound refinement failed: {result}")
                pass
        return relative_bounds, refined_relative_bounds

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

            # tolerance = 5.0  # or whatever threshold you want
            # if next_block_copy['bbox'][1] <= top_block['bbox'][3] + tolerance:
            #     # Possibly skip or partially adjust logic
            #     pass
            # # Skip if next block is above or overlaps with current block
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