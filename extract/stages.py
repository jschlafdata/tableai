import argparse
import json
from pdf.file import DirectoryFileNode
# from core.extract.tables import Mathematics
# from core.objects.pdf import Pdf
# from data_writers.export import DictWriter
import fitz
from pathlib import Path
import os
from collections import defaultdict
import re
from extract.helpers import Map, Search, Stitch, Detect, Bounds
from data_loaders.ingest_files import FileReader
import hashlib
import numpy as np
from typing import Tuple, Dict, List, Any
from scipy.signal import find_peaks
from sklearn.cluster import KMeans


def identify_column_alignments(
    pdf_path: str, 
    table_bounds: Tuple[float, float, float, float], 
    num_columns: int,
    max_pages: int = None,
    use_kmeans_fallback: bool = True,
    smoothing_window: int = 3,
    alignment_threshold_ratio: float = 0.1
) -> Dict[str, Any]:
    """
    Identifies columns and their alignments within a table in a PDF document.
    
    Args:
        pdf_path (str): Path to the PDF file.
        table_bounds (Tuple[float, float, float, float]): (x0, y0, x1, y1) coordinates 
            defining the table area.
        num_columns (int): Expected number of columns in the table.
        max_pages (int, optional): Maximum number of pages to process. Defaults to None (all pages).
        use_kmeans_fallback (bool, optional): Whether to use a K-means fallback approach if 
            histogram-based detection fails. Defaults to True.
        smoothing_window (int, optional): Window size for simple smoothing of the histogram 
            before peak detection. Defaults to 3.
        alignment_threshold_ratio (float, optional): Proportion of column width used as threshold 
            for determining alignment (left/right/center). Defaults to 0.1 (10%).

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'column_boundaries': List of x-coordinates for column boundaries
            - 'column_alignments': List of alignment types ('left', 'center', 'right') for each column
            - 'alignment_lines': List of x-coordinates where alignment lines could be drawn
            - 'alignment_line_data': Detailed line info (dict) for each alignment line
            - 'column_elements': The grouped text elements for each column (useful for debugging).
    """
    
    # -----------------------------------------------------------------------
    # 1. Setup and text extraction
    # -----------------------------------------------------------------------
    doc = fitz.open(pdf_path)
    x0, y0, x1, y1 = table_bounds
    
    # Ensure bounding box is valid
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid table_bounds, x1 should be > x0 and y1 > y0.")
    
    results = {
        'column_boundaries': [],
        'column_alignments': [],
        'alignment_lines': [],
        'alignment_line_data': [],
        'column_elements': []
    }
    
    # Collect all text elements with their positions
    all_text_elements = []
    
    # Either go through all pages or the specified max_pages
    pages_to_process = range(len(doc)) if max_pages is None else range(min(max_pages, len(doc)))
    
    for page_num in pages_to_process:
        page = doc[page_num]
        
        # Extract text blocks within the table bounds on this page
        # NOTE: get_text("dict", clip=...) extracts text within clip region
        text_dict = page.get_text("dict", clip=(x0, y0, x1, y1))
        
        for block in text_dict.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span["text"].strip()
                    if not span_text:
                        continue
                    
                    bbox = span["bbox"]  # (x0, y0, x1, y1)
                    
                    # Filter out bounding boxes that might be outliers
                    # (e.g., negative width/height or completely outside table)
                    if bbox[0] < x0 - 20 or bbox[2] > x1 + 20:
                        # Hard cutoff to skip text far outside the intended region
                        continue

                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    if width <= 0 or height <= 0:
                        continue
                    
                    center_x = (bbox[0] + bbox[2]) / 2.0
                    all_text_elements.append({
                        'text': span_text,
                        'x0': bbox[0],
                        'y0': bbox[1],
                        'x1': bbox[2],
                        'y1': bbox[3],
                        'width': width,
                        'height': height,
                        'center_x': center_x,
                        'page': page_num
                    })
    
    doc.close()
    
    # If no text was extracted, return empty results
    if not all_text_elements:
        return results
    
    # -----------------------------------------------------------------------
    # 2. Identify column boundaries
    # -----------------------------------------------------------------------
    
    # We'll try two approaches:
    # A) Use a histogram of x positions (with optional smoothing + find_peaks).
    # B) Fallback to KMeans on center_x if histogram is insufficient (or if user requests).
    
    # 2.1. Create array of x-positions (center_x or edges)
    # You can experiment with center_x only, or x0/x1. 
    # For now let's combine x0 and x1 but focus on center_x for better column detection.
    x_positions = [elem['center_x'] for elem in all_text_elements]
    
    # Basic outlier filtering: keep only positions within the table
    x_positions = [x for x in x_positions if x0 <= x <= x1]
    if not x_positions:
        # If everything is out of bounds, return.
        return results
    
    # 2.2. Use histogram approach
    # Decide number of bins. Heuristic: at most 100, or maybe ~2*len(x_positions)**0.5
    num_bins = min(100, max(10, int(len(x_positions) ** 0.5 * 2)))
    hist_vals, bin_edges = np.histogram(x_positions, bins=num_bins, range=(x0, x1))
    
    # Optional smoothing of histogram to reduce noise
    if smoothing_window > 1:
        # Simple convolution smoothing
        kernel = np.ones(smoothing_window) / smoothing_window
        hist_vals = np.convolve(hist_vals, kernel, mode='same')
    
    # 2.3. Find peaks in histogram
    # Fine-tune `distance` or `prominence` as needed
    peaks_idx, props = find_peaks(hist_vals, prominence=1, distance=1)
    peak_xs = bin_edges[peaks_idx]  # approximate x-values for the peaks
    
    # Sort peaks by descending amplitude
    peak_amplitudes = hist_vals[peaks_idx]
    sorted_indices = np.argsort(peak_amplitudes)[::-1]
    peak_xs_sorted = peak_xs[sorted_indices]
    
    # 2.4. Select top (num_columns + 1) boundaries
    # We want #columns + 1 boundaries. But sometimes the table's left & right edges
    # won't be recognized as peaks. So let's ensure we have them in place.
    column_boundaries = []
    
    # Always include the extreme left and right of the table region:
    boundary_candidates = [x0, x1]
    
    # Then add the top peaks
    for px in peak_xs_sorted:
        # Keep them within table region
        if x0 < px < x1:
            boundary_candidates.append(px)
    
    # Sort boundary candidates
    boundary_candidates = sorted(list(set(boundary_candidates)))
    
    # If we have at least num_columns+1 boundaries from the histogram approach, we slice them.
    if len(boundary_candidates) >= num_columns + 1:
        # We want to choose the best subset that divides the region into roughly num_columns segments.
        # A simple approach: pick left & right extremes, then choose the (num_columns - 1) largest peaks in-between
        left_most = boundary_candidates[0]
        right_most = boundary_candidates[-1]
        in_between = boundary_candidates[1:-1]
        
        # If in_between is large, slice the top (num_columns - 1) from in_between
        # but we want them to be from the largest peaks or best distributed. 
        # For simplicity, let's just pick the largest peaks if they came from sorted list, 
        # or pick evenly if that approach is too involved.
        
        # Let’s pick evenly spaced from in_between. 
        # Or if you'd like, pick them in ascending order up to num_columns-1 if we trust the sorting by amplitude:
        #   chosen_in_between = in_between[:(num_columns - 1)]
        
        # For more robust selection, we could do additional logic here. We'll do a simple approach:
        step = max(1, len(in_between) // (num_columns))
        chosen_in_between = in_between[::step]
        chosen_in_between = chosen_in_between[:(num_columns - 1)]
        
        selected_boundaries = [left_most] + chosen_in_between + [right_most]
        
        # If we still don't have enough (rare edge case), we do a fallback to an even split:
        if len(selected_boundaries) < num_columns + 1:
            selected_boundaries = list(np.linspace(x0, x1, num_columns + 1))
        
        column_boundaries = sorted(selected_boundaries)
        
    else:
        # Not enough boundaries from peaks, fallback
        if use_kmeans_fallback:
            # KMeans fallback: cluster center_x into num_columns, then build boundaries around those clusters
            X = np.array(x_positions).reshape(-1, 1)
            kmeans = KMeans(n_clusters=num_columns, random_state=42)
            kmeans.fit(X)
            centers = sorted(kmeans.cluster_centers_.flatten())
            
            # Build boundaries by taking midpoints between adjacent cluster centers
            # plus the outer extremes
            boundaries = [x0]
            for i in range(len(centers) - 1):
                boundaries.append((centers[i] + centers[i+1]) / 2)
            boundaries.append(x1)
            column_boundaries = sorted(boundaries)
            
        else:
            # Even split fallback
            column_boundaries = list(np.linspace(x0, x1, num_columns + 1))

    # Store final boundaries
    results['column_boundaries'] = column_boundaries
    
    # -----------------------------------------------------------------------
    # 3. Group text elements by column
    # -----------------------------------------------------------------------
    column_elements = [[] for _ in range(num_columns)]
    
    for elem in all_text_elements:
        center_x = elem['center_x']
        # Find the correct boundary interval
        for i in range(num_columns):
            b_left = column_boundaries[i]
            b_right = column_boundaries[i+1]
            if b_left <= center_x <= b_right:
                column_elements[i].append(elem)
                break
    
    # Store for debugging
    results['column_elements'] = column_elements
    
    # -----------------------------------------------------------------------
    # 4. Determine alignment for each column
    # -----------------------------------------------------------------------
    for i in range(num_columns):
        elements = column_elements[i]
        
        if not elements:
            # If no elements, default to left alignment
            results['column_alignments'].append('left')
            results['alignment_lines'].append(column_boundaries[i])
            continue
        
        col_left = column_boundaries[i]
        col_right = column_boundaries[i+1]
        col_width = col_right - col_left
        
        # Collect counters
        left_aligned = 0
        right_aligned = 0
        center_aligned = 0
        
        # We’ll use this threshold to decide alignment
        threshold = col_width * alignment_threshold_ratio
        
        for elem in elements:
            dist_from_left = elem['x0'] - col_left
            dist_from_right = col_right - elem['x1']
            
            # Decide alignment
            if dist_from_left < threshold:
                left_aligned += 1
            elif dist_from_right < threshold:
                right_aligned += 1
            else:
                # Check center alignment
                elem_center = (elem['x0'] + elem['x1']) / 2
                col_center = (col_left + col_right) / 2
                if abs(elem_center - col_center) < threshold:
                    center_aligned += 1
                else:
                    # If it doesn't strongly fit left/right/center, you could
                    # increment a "mixed" category or skip. We'll skip here for clarity.
                    pass
        
        # Determine the dominant alignment
        total_count = left_aligned + right_aligned + center_aligned
        if total_count == 0:
            alignment = 'left'  # fallback
        else:
            # Compute proportions
            p_left = left_aligned / total_count
            p_right = right_aligned / total_count
            p_center = center_aligned / total_count
            
            # We can set a threshold of 0.6 for "dominant"
            if p_left > 0.6:
                alignment = 'left'
            elif p_right > 0.6:
                alignment = 'right'
            elif p_center > 0.6:
                alignment = 'center'
            else:
                # Mixed alignment, pick whichever is largest
                max_val = max(p_left, p_right, p_center)
                if max_val == p_left:
                    alignment = 'left'
                elif max_val == p_right:
                    alignment = 'right'
                else:
                    alignment = 'center'
        
        # Append alignment
        results['column_alignments'].append(alignment)
        
        # -------------------------------------------------------------------
        # 5. Calculate the best x-coordinate for an alignment guide
        # -------------------------------------------------------------------
        if alignment == 'left':
            left_edges = [e['x0'] for e in elements]
            alignment_line = np.median(left_edges) if left_edges else col_left
            
        elif alignment == 'right':
            right_edges = [e['x1'] for e in elements]
            alignment_line = np.median(right_edges) if right_edges else col_right
            
        else:  # 'center'
            centers = [(e['x0'] + e['x1']) / 2 for e in elements]
            alignment_line = np.median(centers) if centers else (col_left + col_right) / 2
        
        results['alignment_lines'].append(alignment_line)
    
    # -----------------------------------------------------------------------
    # 6. Build alignment line data
    # -----------------------------------------------------------------------
    table_y0, table_y1 = y0, y1
    alignment_line_data = []
    for i, line_x in enumerate(results['alignment_lines']):
        alignment_type = results['column_alignments'][i] if i < len(results['column_alignments']) else 'unknown'
        alignment_line_data.append({
            "x0": line_x,
            "y0": table_y0,
            "x1": line_x,
            "y1": table_y1,
            "alignment": alignment_type
        })
    
    results['alignment_line_data'] = alignment_line_data

    return results

class Stage1:
    def __init__(
        self,
        directory_file_node,
        header_bound=100,
        footer_bound=100,
        min_occurrences=2
    ):
        """
        :param input_dir: Directory to read PDFs from (recursive).
        :param output_dir: Base directory for saving the combined PDFs.
        :param header_bound: Number of points from top to keep (or skip) depending on logic.
        :param footer_bound: Number of points from bottom to keep (or skip).
        :param min_occurrences: Minimum times a piece of text must appear to be considered 'recurring'.
        """
        self.node = directory_file_node
        self.header_bound = header_bound
        self.footer_bound = footer_bound
        self.min_occurrences = min_occurrences

    def group_blocks_by_page(
            self, 
            page_width, 
            page_breaks, 
            page_height, 
            recurring_blocks, 
            header_bound=100, 
            footer_bound=100
        ):
        """
        1) We have 'page_breaks' and a list of bounding boxes in 'recurring_blocks',
        which are in the COMBINED PDF's global coordinate space.
        2) We'll group them by page, merge headers, merge footers, then produce
        final bounding boxes that:
            - span the full page width,
            - top bounding box = (0,0, page_width, yHeader) in local coords,
            then mapped back to global by adding page offset,
            - bottom bounding box = (0, yFooter, page_width, page_height).
        """
        total_height = page_height      # sum of all pages' heights
        # If multiple pages, average page height:
        if len(page_breaks) > 1:
            page_height = total_height / len(page_breaks)
        else:
            page_height = total_height

        # Dictionary for each page index: { "header": [...], "footer": [...], "body": [...] }
        blocks_by_page = {}
        # We'll create one entry PER PAGE, including the last page
        for i in range(len(page_breaks)):
            blocks_by_page[i] = {"header": [], "footer": [], "body": []}

        # ------------------------------------------------------
        # STEP A: Assign each block to its page, store in header/footer/body
        # ------------------------------------------------------
        for block in recurring_blocks:
            x0, y0, x1, y1 = block[:4]

            # figure out which page
            page_index = -1
            for i in range(len(page_breaks)):
                start_y = page_breaks[i]
                if i < len(page_breaks) - 1:
                    end_y = page_breaks[i + 1]
                else:
                    end_y = total_height
                
                # If block top >= start_y AND block bottom < end_y => belongs to page i
                if y0 >= start_y and y1 <= end_y:
                    page_index = i
                    break
            
            if page_index < 0:
                # crosses boundary or out of range
                continue

            # Local coords
            start_y = page_breaks[page_index]
            y0_local = y0 - start_y
            y1_local = y1 - start_y

            # Decide if it's in top region, bottom region, or body
            # e.g. if y1_local < header_bound => "header"
            # if y0_local > page_height - footer_bound => "footer"
            # else => "body"
            if y1_local <= header_bound:
                region = "header"
            elif y0_local >= (page_height - footer_bound):
                region = "footer"
            else:
                region = "body"

            blocks_by_page[page_index][region].append((x0, y0, x1, y1))

        # ------------------------------------------------------
        # STEP B: Merge header blocks & footer blocks, produce
        #         final "full-width" bounding boxes in global coords
        # ------------------------------------------------------
        final_blocks = []

        for i in range(len(page_breaks)):
            start_y = page_breaks[i]
            if i < len(page_breaks) - 1:
                end_y = page_breaks[i+1]
            else:
                end_y = total_height

            # Merge all header blocks
            header_merged = Map.merge_all_boxes(blocks_by_page[i]["header"])
            if header_merged:
                hx0, hy0, hx1, hy1 = header_merged
                # Convert to local
                hy0_local = hy0 - start_y
                hy1_local = hy1 - start_y

                # We want (0,0, page_width, yBoundLocal) in local coords,
                # where yBoundLocal = hy1_local
                yBoundLocal = hy1_local
                # Now map that back to global
                header_global = (
                    0,                     # x0
                    start_y + 0,           # y0
                    page_width,            # x1
                    start_y + yBoundLocal  # y1
                )
                final_blocks.append(header_global)

            # Merge all footer blocks
            footer_merged = Map.merge_all_boxes(blocks_by_page[i]["footer"])
            if footer_merged:
                fx0, fy0, fx1, fy1 = footer_merged
                # Convert to local
                fy0_local = fy0 - start_y
                fy1_local = fy1 - start_y

                # We want (0, yBoundLocal, page_width, page_height) in local coords
                # yBoundLocal = fy0_local
                yBoundLocal = fy0_local
                # Map back to global
                footer_global = (
                    0,
                    start_y + yBoundLocal,
                    page_width,
                    start_y + page_height
                )
                final_blocks.append(footer_global)
            
            # If you also want to do something with "body" region, you could do it here
            # But you said you only need a final header box & footer box

        return final_blocks

    def combine_pages_into_one(self):
        """
        Combines all pages of input_pdf_path into one tall PDF,
        then extracts recurring text blocks (keeping only top/bottom strips).
        Returns a metadata dict.
        """
        src_doc = fitz.open(self.node.stage_paths[0]["abs_path"])
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

        self.page_breaks = []
        current_y = 0.0

        # Place each original page
        for page_index in range(original_page_count):
            self.page_breaks.append(current_y)
            page = src_doc[page_index]
            width, height = page.rect.width, page.rect.height

            target_rect = fitz.Rect(0, current_y, width, current_y + height)
            combined_page.show_pdf_page(target_rect, src_doc, page_index)
            current_y += height

        
        combined_doc.save(self.node.stage_paths[1]["abs_path"])
        combined_doc.close()
        src_doc.close()

        # Extract text blocks from the single tall PDF
        self.blocks = Search.get_text_blocks_from_single_page(self.node.stage_paths[1]["abs_path"])

        # Identify recurring text, skipping middle region
        self.recurring = Search.find_recurring_text_blocks(
            self.blocks,
            self.page_breaks,
            total_height,
            page_heights,
            self.header_bound,
            self.footer_bound,
            self.min_occurrences
        )

        # Flatten to recurring_blocks array
        self.recurring_blocks = []
        for coords_list in self.recurring.values():
            self.recurring_blocks.extend(coords_list)


        grouped_blocks = self.group_blocks_by_page(
            page_width=max_width, 
            page_breaks=self.page_breaks, 
            page_height=total_height, 
            recurring_blocks=self.recurring_blocks, 
            header_bound=self.header_bound, 
            footer_bound=self.footer_bound
        )

        data_blocks =  Map.create_inverse_recurring_blocks(
                page_height=total_height, 
                page_width=max_width, 
                page_breaks=self.page_breaks, 
                recurring_blocks=grouped_blocks
        )

        data_blocks = Map.merge_overlapping_boxes(data_blocks)
        meta = {
                "page_width": max_width,
                "page_height": total_height,
                "page_breaks": self.page_breaks,
                "recurring_blocks": grouped_blocks,
                "data_blocks": data_blocks, 
        }
        return meta

    def run(self):
        """
        Main entry: scan all PDFs in input_dir, process them,
        and return a list of metadata dictionaries.
        """
        return self.combine_pages_into_one()
    

class Stage2:

    def __init__(
        self,
        directory_file_node,
        parameters: dict, 
        merge: bool=False,
        conf_path: str='llm/outputs/pdf_schema.yml',
    ):
        self.node = directory_file_node
        self.stage1_metadata = self.node.metadata["stage1"]
        self.conf_key = 'worldpay' if 'controlled' in self.node.sub_dir.lower() else 'wellsfargo' 
        self.pdf_config = FileReader.yaml(conf_path).get(self.conf_key, {})
        self.parameters = parameters
        self.merge = merge

    
    def run(self):
        
        self.node.load_pdf(2)
        pdf_coords = self.node.pdfs[2].to_dict()
        page1_coords = pdf_coords['coords'][0][0]
        self.height = page1_coords['height']
        self.width = page1_coords['width']
        
        conf_mappings={}
        table_key_dict={}
        for tbl_meta in self.pdf_config:
            table_key = hashlib.sha256(
                '|'.join(tbl_meta['columns'] + [self.conf_key]).encode("utf-8")
            ).hexdigest()
            col_key = hashlib.sha256(
                '|'.join(tbl_meta['columns']).encode("utf-8")
            ).hexdigest()
            conf_mappings[col_key] = table_key
            table_key_dict[table_key] = tbl_meta
            tbl_meta['table_key'] = table_key

        data_blocks=[]
        for block in self.stage1_metadata["data_blocks"]:
            block_meta = {
                'type': 'data_blocks',
                'rect': block
            }
            data_blocks.append(block_meta)


        placement_metadata, new_doc = Stitch.clip_and_place_pdf_regions(
            self.node.stage_paths[1]["abs_path"],
            data_blocks,
            source_page_number=0,
            layout='vertical',
            page_width=None,
            page_height=None,
            margin=20,
            gap=0,
            center_horizontally=False
        )

        new_doc.save(self.node.stage_paths[2]["abs_path"])

        white_space_blocks = Map.find_full_width_whitespace(
            self.node.stage_paths[2]["abs_path"],
            page_number=0, 
            min_height=10,
            include_images=True, 
            include_drawings=False
        )

        sequence_matches, matched_col_refs, header_bound, footer_bound, page_text_data = self.process_pdf(
            pdf_path=self.node.stage_paths[2]["abs_path"], 
            settings=self.pdf_config
        )

        grouped_tables, occupied_bounds = self.group_into_tables(
            sequence_matches, 
            matched_col_refs, 
            header_bound=header_bound, 
            footer_bound=footer_bound ,
            page_text_data=page_text_data,
            conf_mappings=conf_mappings, 
            table_key_dict=table_key_dict
        )

        grouped_tables = self.merge_duplicate_tables(grouped_tables)
        grouped_tables = grouped_tables['1']

        for table in grouped_tables:
            valid_x_max = Bounds.find_valid_x2_coordinates(table['table_bounds'], occupied_bounds, table["x_max"], min_width=20)
            table['expansions'] = valid_x_max
            if valid_x_max:
                table['table_bounds'][2] = valid_x_max
                curr_min_x = table['table_bounds'][0]
                if table['columns']:
                    x0,y0,x1,y1 = table['columns']
                    final_min_cols = min(curr_min_x, x0)
                    table['columns'] = (final_min_cols,y0,valid_x_max,y1)

        occupied_bounds = self.get_occupied_bounds(grouped_tables, x_margin=0, y_margin=0)
        inverse_tbl_bounds = Bounds.get_unoccupied_regions(
            self.width,
            self.height,
            occupied_bounds
        )

        for table in grouped_tables:
            table_bounds = table['table_bounds']
            table_columns = table['table_columns']
            # if table_columns:
            #     col_len = len(table_columns)
            #     table_row_meta = identify_column_alignments(self.node.stage_paths[2]["abs_path"], table_bounds, num_columns=col_len)
            #     table['table_row_meta'] = table_row_meta

        multi_tables = self.find_tables_with_multiple_internal_headers_and_whitespace(grouped_tables, white_space_blocks)
        slices = self.slice_tables_by_internal_headers(multi_tables)
        merged_tables = self.group_and_merge_tables_by_columns_hashed(slices)

        return grouped_tables, inverse_tbl_bounds, occupied_bounds, table_key_dict, self.width, self.height, merged_tables

    def process_pdf(self, pdf_path, settings):
        """
        Processes the PDF to detect table regions, columns, and totals.
        
        Args:
            pdf:      Path to PDF file
            table_ai: Some config or manager that has .config for thresholds
            settings: A list of dicts, each describing a 'table_name', 'totals', and 'columns'
            
        Returns:
            - sequence_matches: Dict of page -> { 'table_name': [...], 'column_matches': [...], 'totals_matches': [...] }
            - matched_col_refs: Dict of page -> [ (col_boxes, original_columns), ... ]
            - header_bound, footer_bound
            - page_text_data: Dict of page -> [ {'bbox': [x0, y0, x1, y1], 'text': '...'} ]
        """
        doc = fitz.open(pdf_path)
        header_bound, footer_bound = Detect.detect_header_footer_bounds(doc, **self.parameters)
        print(f"header_bound: {header_bound}")
        print(f"footer_bound: {footer_bound}")

        sequence_matches = defaultdict(lambda: defaultdict(list))
        matched_col_refs = defaultdict(list)
        
        # Collect all text spans with their bounding boxes
        page_text_data = {}

        # Helper: find, consolidate, merge, and store
        def _find_merge_and_store(pdf_words, target_str, original_cols, storage_dict, storage_key=None, target_type='table_name', table_key=None):
            """Returns merged bounding boxes. Optionally stores them in storage_dict[storage_key]."""
            raw_matches = Search.find_sequential_mathces(pdf_words, target_str, original_cols, target_type=target_type)
            consolidated = Map.consolidate_matches(raw_matches)
            merged = Map.merge_overlapping_boxes(consolidated)
            if storage_key and merged:
                storage_dict[storage_key].extend(merged)
            return merged

        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            
            # Extract words (pdf_words)
            text_dict = page.get_text("dict")
            blocks = text_dict["blocks"]
            
            # Collect all text spans for this page
            page_text_data[page_number] = []
            
            pdf_words = []
            for block in blocks:
                if block["type"] == 0:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            pdf_words.append(span)
                            
                            # Add to page_text_data if it has content
                            if span['text'].strip():
                                page_text_data[page_number].append({
                                    'bbox': span['bbox'],
                                    'text': span['text']
                                })

            for table in settings:
                # Normalize strings (remove whitespace, make lower)
                tbl_name = ''.join(table['table_name'].split()).lower()
                totals_name = ''.join(table['totals'].split()).lower()
                columns_str = ''.join(table['columns']).lower()
                original_columns = table['columns']
                table_key = table['table_key']

                # 1) Table name
                tbl_merged = _find_merge_and_store(
                    pdf_words, tbl_name, original_columns,
                    sequence_matches[page_number], 'table_name', 'table_name', table_key
                )
                
                # 2) Columns
                col_merged = _find_merge_and_store(
                    pdf_words, columns_str, original_columns,
                    sequence_matches[page_number], 'column_matches', 'column_matches', table_key
                )
                # Track references (keeps same behavior: appended once for each bounding box)
                for _ in col_merged:
                    matched_col_refs[page_number].append((col_merged, original_columns))

                # 3) Totals
                #    - We find & merge but don't store directly. We first filter out overlaps.
                tot_merged = _find_merge_and_store(
                    pdf_words, totals_name, original_columns,
                    {}, None, 'totals_matches', table_key  # don't store in sequence_matches just yet
                )
                all_existing = tbl_merged + col_merged
                final_totals = []
                for tot_box in tot_merged:
                    # Keep tot_box only if it doesn't overlap with known table or column boxes
                    if not any(Map.is_overlapping(tot_box, eb) for eb in all_existing):
                        final_totals.append(tot_box)
                
                if final_totals:
                    # print(len(final_totals))
                    sequence_matches[page_number]['totals_matches'].extend(final_totals)

        # -----------------------------------------
        # Optionally re-merge totals on each page
        # -----------------------------------------
        for page, matches in sequence_matches.items():
            if 'totals_matches' in matches:
                matches['totals_matches'] = Map.merge_overlapping_boxes(matches['totals_matches'])
            if 'table_name' in matches:
                matches['table_name'] = Map.merge_overlapping_boxes(matches['table_name'])

        return sequence_matches, dict(matched_col_refs), header_bound, footer_bound, page_text_data

    # ----------------------------------------------------------------------
    # group_into_tables (Refactored)
    # ----------------------------------------------------------------------
    def group_into_tables(
        self,
        pdf_data,
        column_metadata=None,
        page_height=792,
        header_bound=None,
        footer_bound=None,
        page_width=612,
        page_text_data=None,
        conf_mappings={}, 
        table_key_dict={}
    ):
        """
        Group PDF coordinates into logical tables based on spatial relationships.
        Creates flattened output of unique groups of [table_name, columns, totals, table_bounds].
        """

        # If footer_bound is set, use it for the bottom of the page
        if footer_bound is not None:
            page_height = footer_bound

        result = {}

        # Use Map.is_overlapping for the simple case
        def _boxes_overlap(box1, box2, min_overlap_ratio=0.0):
            """
            Extended overlap check. If min_overlap_ratio=0.0, just call Map.is_overlapping.
            Otherwise do a partial area check.
            """
            # Basic check:
            if not Map.is_overlapping(box1, box2):
                return False

            if min_overlap_ratio <= 0:
                return True

            # If a ratio is needed, calculate intersection area vs box1 area
            x0_1, y0_1, x1_1, y1_1 = box1
            x0_2, y0_2, x1_2, y1_2 = box2

            inter_x0 = max(x0_1, x0_2)
            inter_y0 = max(y0_1, y0_2)
            inter_x1 = min(x1_1, x1_2)
            inter_y1 = min(y1_1, y1_2)
            inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)

            box1_area = (x1_1 - x0_1) * (y1_1 - y0_1)
            if box1_area <= 0:
                return False

            overlap_ratio = inter_area / float(box1_area)
            return overlap_ratio >= min_overlap_ratio

        def _find_closest_box(sorted_boxes, ref_edge, above=True):
            """
            Finds the single closest bounding box in sorted_boxes that is strictly
            above or below the reference edge (y).
            """
            closest = None
            min_distance = float('inf')

            for box in sorted_boxes:
                # box = [x0, y0, x1, y1]
                y_top, y_bottom = box[1], box[3]

                if above:
                    # Only consider boxes whose bottom < ref_edge
                    if y_bottom < ref_edge:
                        dist = ref_edge - y_bottom
                        if dist < min_distance:
                            min_distance = dist
                            closest = box
                else:
                    # Only consider boxes whose top > ref_edge
                    if y_top > ref_edge:
                        dist = y_top - ref_edge
                        if dist < min_distance:
                            min_distance = dist
                            closest = box
            return closest

        def _find_max_text_width(page_num, y_min, y_max, margin=20):
            """
            Look at all text spans in page_text_data for vertical range [y_min,y_max],
            find the rightmost x1. This helps define table's right boundary.
            """
            if not page_text_data or page_num not in page_text_data:
                return None

            max_x = 0
            for span in page_text_data[page_num]:
                x0, sy0, x1, sy1 = span['bbox']
                if (sy0 <= y_max and sy1 >= y_min):
                    max_x = max(max_x, x1)

            return min(max_x + margin, page_width) if max_x > 0 else None

        # ------------------------------------------------
        # Build an initial list of tables
        # ------------------------------------------------
        for page_num, page_content in pdf_data.items():
            page_key = str(page_num)
            result[page_key] = []

            table_names  = page_content.get('table_name', [])
            column_boxes = page_content.get('column_matches', [])
            totals_boxes = page_content.get('totals_matches', [])

            # Sort each by top-y
            sorted_names  = sorted(table_names,  key=lambda b: b[1])
            sorted_cols   = sorted(column_boxes, key=lambda b: b[1])
            sorted_totals = sorted(totals_boxes, key=lambda b: b[1])

            # For each column match, find the closest table name above & the closest totals below
            for col_box in sorted_cols:
                col_x0, col_y0, col_x1, col_y1 = col_box

                closest_name   = _find_closest_box(sorted_names,  ref_edge=col_y0, above=True)
                closest_totals = _find_closest_box(sorted_totals, ref_edge=col_y1, above=False)

                y_max = closest_totals[3] if closest_totals else page_height
                x_min = min(col_x0, closest_totals[0]) if closest_totals else col_x0
                x_max = col_x1  # default

                # Possibly expand based on actual text
                max_text_x = _find_max_text_width(page_num, col_y0, y_max)
                if max_text_x and max_text_x > x_max:
                    x_max = max_text_x
                else:
                    # If we have totals, consider totals' right edge
                    if closest_totals and closest_totals[2] > x_max:
                        x_max = closest_totals[2]

                table_bounds = [x_min, col_y0, col_x1, y_max]

                table_info = {
                    'table_name':     closest_name,
                    'columns':        col_box,
                    'totals':         closest_totals,
                    'table_bounds':   table_bounds,
                    'x_max':          x_max, 
                    'internal_headers_y': [col_y1]
                }
                result[page_key].append(table_info)

        # ------------------------------------------------
        # Find all other column headers within each table
        # ------------------------------------------------
        for page_num, page_content in pdf_data.items():
            page_key = str(page_num)
            if page_key not in result:
                continue

            col_boxes = page_content.get('column_matches', [])

            for table in result[page_key]:
                tbl_x0, tbl_y0, tbl_x1, tbl_y1 = table['table_bounds']
                original_col_y = table['columns'][3]  # bottom Y of main col

                for cbox in col_boxes:
                    cx0, cy0, cx1, cy1 = cbox
                    # skip the original column
                    if cy1 == original_col_y:
                        continue
                    # if the column is inside our table, add it
                    if (cx0 >= tbl_x0 and cx1 <= tbl_x1 and cy0 >= tbl_y0 and cy1 <= tbl_y1):
                        table['internal_headers_y'].append(cy1)

                # Sort Y for consistency
                table['internal_headers_y'].sort()

        # ------------------------------------------------
        # If we have external column metadata, associate it
        # ------------------------------------------------
        if column_metadata:
            for page_key, tables in result.items():
                page_int = int(page_key)
                if page_int not in column_metadata:
                    continue

                meta_entries = column_metadata[page_int]  # list of (coords_list, col_names)
                for table in tables:
                    y_min = table['table_bounds'][1]
                    y_max = table['table_bounds'][3]

                    best_match = None
                    best_overlap = 0.0

                    for coords_list, col_names in meta_entries:
                        total_coords = len(coords_list)
                        if not total_coords:
                            continue

                        overlap_count = 0
                        for coord in coords_list:
                            _, ctop, _, cbot = coord
                            # Check any vertical overlap
                            if (ctop >= y_min and ctop <= y_max) \
                            or (cbot >= y_min and cbot <= y_max) \
                            or (ctop <= y_min and cbot >= y_max):
                                overlap_count += 1

                        ratio = overlap_count / total_coords
                        if ratio > best_overlap:
                            best_overlap = ratio
                            best_match = col_names

                    if best_match and best_overlap > 0.5:
                        col_key = hashlib.sha256(
                            '|'.join(best_match).encode("utf-8")
                        ).hexdigest()
                        # conf_mappings, table_key_dict
                        table['table_columns'] = best_match
                        table['table_key'] = conf_mappings[col_key]

        occupied_bounds = self.get_occupied_bounds(result)
        return result, occupied_bounds

    def get_occupied_bounds(self, result, x_margin=-1, y_margin=-5):
        occupied_bounds = []
        _tables=[]
        if isinstance(result, dict):
            for page_key, tables in result.items():
                for table in tables:
                    _tables.append(table)
        elif isinstance(result, list):
            _tables = result
        for table in _tables:
            tbl_bounds = table['table_bounds']
            tbl_name   = table['table_name']
            # Occupied region is the table, plus the name box
            if tbl_bounds:
                # Optionally add negative margin if you want them to combine more easily
                expanded_tb = Bounds.apply_margins_to_bounds(tbl_bounds, x_margin=x_margin, y_margin=y_margin)
                occupied_bounds.append(expanded_tb)
            if tbl_name:
                expanded_name = Bounds.apply_margins_to_bounds(tbl_name, x_margin=x_margin, y_margin=y_margin)
                occupied_bounds.append(expanded_name)

            # Drop totals if no overlap
            tbox = table['totals']
            if tbox and not Map.is_overlapping(tbl_bounds, tbox):
                table['totals'] = None
        return occupied_bounds
    # ----------------------------------------------------------------------
    # merge_duplicate_tables (unchanged except using Map.is_overlapping)
    # ----------------------------------------------------------------------
    def merge_duplicate_tables(self, result):
        """
        Merges tables that have overlapping 'table_bounds', combining columns, headers, etc.
        Repeats until no further merges occur.
        """

        def _merge_two_tables(tbl_a, tbl_b):
            # Union of bounding boxes
            ax0, ay0, ax1, ay1 = tbl_a['table_bounds']
            bx0, by0, bx1, by1 = tbl_b['table_bounds']
            tbl_a['table_bounds'] = [
                min(ax0, bx0),
                min(ay0, by0),
                max(ax1, bx1),
                max(ay1, by1)
            ]

            # Merge internal_headers_y
            headers_a = tbl_a.get('internal_headers_y', [])
            headers_b = tbl_b.get('internal_headers_y', [])
            tbl_a['internal_headers_y'] = sorted(set(headers_a + headers_b))

            # Merge table_columns
            cols_a = tbl_a.get('table_columns')
            cols_b = tbl_b.get('table_columns')
            if cols_a and cols_b:
                # deduplicate while preserving order
                combined = list(dict.fromkeys(cols_a + cols_b))
                tbl_a['table_columns'] = combined
            elif not cols_a and cols_b:
                tbl_a['table_columns'] = cols_b

            # Merge totals
            tot_a = tbl_a.get('totals')
            tot_b = tbl_b.get('totals')
            if tot_b:
                if not tot_a:
                    tbl_a['totals'] = tot_b
                else:
                    # pick whichever extends further down
                    if tot_b[3] > tot_a[3]:
                        tbl_a['totals'] = tot_b

        for page_str, tables in result.items():
            if not tables:
                continue

            changed = True
            while changed:
                changed = False
                merged_tables = []

                for tbl in tables:
                    found_merge = False
                    for m_tbl in merged_tables:
                        if Map.is_overlapping(m_tbl['table_bounds'], tbl['table_bounds']):
                            _merge_two_tables(m_tbl, tbl)
                            found_merge = True
                            changed = True
                            break
                    if not found_merge:
                        merged_tables.append(tbl)

                tables = merged_tables

            result[page_str] = tables

        return result
    
    def hash_columns(self, columns):
        """
        Given a list of column names, generate a hash (SHA-256) to use as a dictionary key.
        """
        # Create a single string from the column names
        combined_columns = "|".join(columns)
        # Generate a SHA-256 hash
        return hashlib.sha256(combined_columns.encode("utf-8")).hexdigest()

    def group_and_merge_tables_by_columns_hashed(self, tables):
        """
        Given a list of dictionaries (each having keys 'original_table' and 'sub_tables'),
        group the tables by a SHA-256 hash of their 'table_columns' in 'original_table',
        then merge the 'sub_tables' for those that share the same hash (i.e., same columns).
        
        Args:
            tables (list): A list of dictionaries. Each dictionary has:
                {
                    "original_table": {
                        "table_columns": [ ... ],
                        ...other keys...
                    },
                    "sub_tables": [
                        ... bounding-box info ...
                    ]
                }
        
        Returns:
            dict: A dictionary whose keys are hashes of column names,
                and whose values are merged sub-table data for all tables with those columns.
        """
        grouped = defaultdict(list)
        
        # 1. Group tables by the hash of their table_columns
        for table_item in tables:
            column_list = table_item["original_table"]["table_columns"]
            # Generate a stable hash key
            columns_key = self.hash_columns(column_list)
            grouped[columns_key].append(table_item)
        
        # 2. Merge sub_tables for each group
        merged_results = {}

        for columns_hash_key, group_items in grouped.items():
            merged_sub_tables = []
            original_tables = []
            tots=[]
            cols=[]
            names=[]
            
            for item in group_items:
                merged_sub_tables.extend(item["sub_tables"])
                
                table_name = item["original_table"].get('table_name', None)
                if table_name:
                    names.append(table_name)
                columns = item["original_table"].get('columns', None)
                if columns:
                    cols.append(columns)
                
                totals = item["original_table"].get('totals', None)
                if totals:
                    tots.append(totals)

                table_columns = item["original_table"].get('table_columns', None)
                
                original_tables.append(item["original_table"])

            
            # Build a result dict with the list of original_table metadata 
            # and a single list of all merged sub_tables
            merged_results[columns_hash_key] = {
                "merged_sub_tables": merged_sub_tables,
                "table_name": names[0] if names else [],
                "columns": cols[0] if cols else [],
                "totals": tots[0] if tots else [],
                "col_names": table_columns
            }
        
        return {idx: v for idx, (k,v) in enumerate(merged_results.items())}


    def find_tables_with_multiple_internal_headers_and_whitespace(self, tables, white_space_blocks):
        """
        stage2_data: The 'stage2' dictionary containing:
            {
            'tables': [...],
            'whitespace_blocks': [...],
            ...
            }
        Returns a list of dicts; each dict has:
        {
            'table': <the table object>,
            'whitespace_blocks': [ list of whitespace blocks overlapping in Y-dimension ]
        }
        """

        results = []

        # 1. Loop over all tables
        for table in tables:
            internal_headers = table.get("internal_headers_y", [])
            # 2. Check if there are multiple internal header lines
            if len(internal_headers) > 1:
                tb_x0, tb_y0, tb_x1, tb_y1 = table.get("table_bounds", [0, 0, 0, 0])

                overlapping_whitespace_blocks = []

                # 3. Collect whitespace blocks that overlap in Y dimension
                for wb in white_space_blocks:
                    wb_x0, wb_y0, wb_x1, wb_y1 = wb

                    # Overlap in Y occurs if:
                    # - The whitespace block's top is below the table's bottom
                    # - The whitespace block's bottom is above the table's top
                    # In other words, they are NOT "completely above or completely below" each other.
                    if not (wb_y1 <= tb_y0 or wb_y0 >= tb_y1):
                        overlapping_whitespace_blocks.append(wb)

                if overlapping_whitespace_blocks:
                    results.append({
                        "table": table,
                        "whitespace_blocks": overlapping_whitespace_blocks
                    })
            elif len(internal_headers) == 1:
                results.append({
                        "table": table,
                        "whitespace_blocks": []
                    })

        return results


    def slice_tables_by_internal_headers(self, multi_header_tables):
        """
        Given the output from find_tables_with_multiple_internal_headers_and_whitespace(...),
        this function handles all scenarios:
        - 0 internal headers: sub_tables = []
        - 1 internal header: exactly 1 slice
        - N internal headers: N slices

        Slicing logic (if N > 0):
        If 'internal_headers_y' = [H1, H2, ..., HN],
        we create slices:
            Slice 1: from H1 -> (earliest whitespace or H2)
            Slice 2: from H2 -> (earliest whitespace or H3)
            ...
            Slice N-1: from H(N-1) -> (earliest whitespace or HN)
            Slice N: from HN -> (earliest whitespace or the table bottom)

        Whitespace rule:
        - We look for any whitespace whose top (wb_y0) is
            strictly greater than the slice's start_y and
            less than or equal to its default_end_y.
        - If such whitespace exists, the slice ends at
            the earliest (lowest) wb_y0.

        :param multi_header_tables: List of dicts, each with:
        {
            "table": {
                "table_bounds": [tb_x0, tb_y0, tb_x1, tb_y1],
                "internal_headers_y": [...],
                ...
            },
            "whitespace_blocks": [
                [wb_x0, wb_y0, wb_x1, wb_y1],
                ...
            ]
        }

        :return: A list of dicts, each with:
        {
            "original_table": <the original table object>,
            "sub_tables": [
                [x0, y0, x1, y1],
                [x0, y0, x1, y1],
                ...
            ]
        }
        """

        results = []

        for entry in multi_header_tables:
            table_obj = entry["table"]
            tots = table_obj["totals"]
            if tots:
                totX0, totY0, totX1, totY1 = tots
            
            tb_x0, tb_y0, tb_x1, tb_y1 = table_obj["table_bounds"]
            whitespace_blocks = entry.get("whitespace_blocks", [])

            # Sort headers in ascending order
            internal_headers = sorted(table_obj.get("internal_headers_y", []))
            n = len(internal_headers)

            if n == 0:
                # No internal headers -> no slices
                results.append({
                    "original_table": table_obj,
                    "sub_tables": []
                })
                continue

            sub_tables = []

            # We produce exactly N slices for N headers
            # For each i in [0..N-1]:
            #   start = internal_headers[i]
            #   end   = internal_headers[i+1] if i+1 < N else table_bottom
            for i in range(n):
                start_y = internal_headers[i]
                if i < n - 1:
                    default_end_y = internal_headers[i + 1]
                else:
                    # Last header: slice goes down to table bottom
                    default_end_y = tb_y1

                # Find any whitespace top (wb_y0) strictly greater than start_y
                # and within [start_y..default_end_y].
                wb_candidates = []
                for wb in whitespace_blocks:
                    wb_x0, wb_y0, wb_x1, wb_y1 = wb
                    if (wb_y0 > start_y) and (wb_y0 <= default_end_y):
                        wb_candidates.append(wb_y0)

                # If there's whitespace, pick the earliest (lowest) top
                if wb_candidates:
                    slice_end_y = min(wb_candidates)
                else:
                    slice_end_y = default_end_y

                # Clamp slice to table bounds
                slice_y0 = max(tb_y0, start_y)
                slice_y1 = min(tb_y1, slice_end_y)

                # Add the slice if it has positive height
                if slice_y1 > slice_y0:
                    sub_tables.append([tb_x0, slice_y0, tb_x1, slice_y1])

            len_sub_tables = len(sub_tables)
            if len_sub_tables>0:
                last_tbl = sub_tables[len_sub_tables-1]
                if totY1 >= last_tbl[1]:
                    last_tbl[3] = totY1
                    sub_tables[len_sub_tables-1] = last_tbl

            results.append({
                "original_table": table_obj,
                "sub_tables": sub_tables
            })

        return results