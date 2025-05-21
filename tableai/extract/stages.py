import argparse
import json
# from core.extract.tables import Mathematics
# from core.objects.pdf import Pdf
# from data_writers.export import DictWriter
import fitz
from pathlib import Path
import os
from collections import defaultdict
import re
from tableai.extract.helpers import Map, Search, Stitch, Detect, Bounds
import hashlib
import numpy as np
from typing import Tuple, Dict, List, Any
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import tempfile 
import shutil


class Stage1:
    def __init__(
        self,
        directory_file_node,
        tables_search_config={},
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
        table_key_dict: dict,
        parameters: dict, 
        merge: bool=False,
        classification: str = None, 
    ):
        self.node = directory_file_node
        self.stage1_metadata = self.node.extraction_metadata["stage1"]
        self.classification = classification
        self.table_key_dict = table_key_dict
        self.parameters = parameters
        self.merge = merge

    def run(self):
        
        conf_mappings={}
        table_dict = {entry["table_key"]: entry for entry in self.table_key_dict}
        
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

        self.node.current_stage = 2
        page_meta = self.node.fetch_pdf_metadata()
        page1_coords = page_meta['coords'][0][0]
        self.height = page_meta['height']
        self.width = page_meta['width']

        white_space_blocks = Map.find_full_width_whitespace(
            self.node.stage_paths[2]["abs_path"],
            page_number=0, 
            min_height=10,
            include_images=True, 
            include_drawings=False
        )

        sequence_matches, matched_col_refs, header_bound, footer_bound, page_text_data, box_metadata_map = self.process_pdf(
            pdf_path=self.node.stage_paths[2]["abs_path"], 
            settings=self.table_key_dict
        )

        grouped_tables, occupied_bounds = self.group_into_tables(
            sequence_matches, 
            matched_col_refs, 
            header_bound=header_bound, 
            footer_bound=footer_bound ,
            page_text_data=page_text_data,
            conf_mappings=conf_mappings, 
            table_key_dict=table_dict,
            box_metadata_map=box_metadata_map
        )

        grouped_tables = self.merge_duplicate_tables(grouped_tables)
        if grouped_tables:
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
                table_columns = table.get('table_columns', [])

            multi_tables = self.find_tables_with_multiple_internal_headers_and_whitespace(grouped_tables, white_space_blocks)
            slices = self.slice_tables_by_internal_headers(multi_tables)
            merged_tables = self.group_and_merge_tables_by_columns_hashed(slices)

            return grouped_tables, inverse_tbl_bounds, occupied_bounds, self.table_key_dict, self.width, self.height, merged_tables, white_space_blocks, table_dict
        else:
            return [], [], [],  {}, self.width, self.height, [], []

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

        sequence_matches = defaultdict(lambda: defaultdict(list))
        matched_col_refs = defaultdict(list)
        
        # Collect all text spans with their bounding boxes
        page_text_data = {}
        storage_counter = 0
        box_metadata_map = defaultdict(dict)

        # Helper: find, consolidate, merge, and store
        def _find_merge_and_store(pdf_words, target_str, original_cols, storage_dict, storage_key=None, target_type='table_name', table_key=None):
            """Returns merged bounding boxes. Optionally stores them in storage_dict[storage_key]."""
            raw_matches = Search.find_sequential_mathces(pdf_words, target_str, original_cols, target_type=target_type)
            consolidated = Map.consolidate_matches(raw_matches)
            merged = Map.merge_overlapping_boxes(consolidated)
            if storage_key and merged:
                for m in merged:
                    box_metadata_map[tuple(m)]["table_key"] = table_key
                storage_dict[storage_key].extend(merged)
            return merged

        for page_index, page in enumerate(doc):
            storage_counter += 1
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
                storage_counter += 1
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
                        box_metadata_map[tuple(tot_box)]["table_key"] = table_key
                        final_totals.append(tot_box)
                
                if final_totals:
                    # print(len(final_totals))
                    sequence_matches[page_number]['totals_matches'].extend(final_totals)

        # -----------------------------------------
        # Optionally re-merge totals on each page
        # -----------------------------------------
        # for page, matches in sequence_matches.items():
        #     if 'totals_matches' in matches:
        #         final_tots = Map.merge_overlapping_boxes(matches['totals_matches'])
        #         matches['totals_matches'] = Map.merge_overlapping_boxes(matches['totals_matches'])
        #     if 'table_name' in matches:
        #         final_merge = Map.merge_overlapping_boxes(matches['table_name'])
        #         table_key = box_metadata_map.get(tuple(final_merge), {}).get("table_key")
        #         matches['table_name'] = final_merge

        return sequence_matches, dict(matched_col_refs), header_bound, footer_bound, page_text_data, dict(box_metadata_map)

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
        table_key_dict={},
        box_metadata_map={}
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
                table_key = box_metadata_map.get(tuple(col_box), {}).get("table_key")

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
                    'internal_headers_y': [col_y1],
                    'table_key': table_key,
                    'col_names': table_key_dict.get(table_key, {}).get('columns')
                }
                print(f"FOUND TABLE META!! {table_key}")
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
            column_list = table_item.get("original_table", {}).get("table_columns")
            if column_list:
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
                table_key = item["original_table"].get('table_key', None)
                # print(f"FOUND ANOYTHET KEY {table_key}")
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
                "col_names": table_columns,
                "tabke_key": table_key
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
    
import pandas as pd

class TableBuilder:
    @staticmethod
    def nested_dict_to_dataframe(data: dict) -> pd.DataFrame:
        # Get the number of rows by checking any column
        num_rows = max(len(col) for col in data.values())
        
        # Create a list of rows
        rows = []
        for row_idx in range(num_rows):
            row = []
            for col_idx in sorted(data.keys()):
                value = data[col_idx].get(row_idx, [''])[0]  # Default to empty string if missing
                row.append(value)
            rows.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def extract_table_from_pdf(pdf_path: str, page_num: int, value_cells: List[Dict[str, Any]], spanning_text) -> pd.DataFrame:
        """
        Extract text from a PDF to build a table based on bbox coordinates.
        
        Args:
            pdf_path: Path to the PDF file
            page_num: Page number to extract from (0-based in PyMuPDF)
            value_cells: Array of cell metadata with bbox coordinates
            
        Returns:
            DataFrame representing the extracted table
        """
        # try:
        # Open the PDF document
        pdf_document = fitz.open(pdf_path)
        
        # Get the specified page (PyMuPDF uses 0-based indexing)
        page = pdf_document[page_num]
        text_dict = page.get_text("dict")
        
        spans = []
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        for span in line["spans"]:
                            spans.append({
                                'text': span['text'],
                                'bbox': (span['bbox'][0], span['bbox'][1], span['bbox'][2], span['bbox'][3])
                            })
        
        # Extract text for each cell based on bbox
        # Extract text for each cell based on bbox
        spanning_bboxes=[]
        if spanning_text:
            spanning_bboxes = [x['bbox'] for x in spanning_text]
        mapped_values = defaultdict(lambda: defaultdict(list))
        test_mapped_values = defaultdict(lambda: defaultdict(list))
        for cell in value_cells:
            row_idx = cell.get('row_index', '-1')
            col_idx = cell['col_index']
            bbox = cell['bbox']
    
            for span in spans:
                text = span['text']
                span_bbox = span['bbox']

    
                if Map.is_overlapping(bbox, span_bbox):
                    # and not any(Map.is_overlapping(span_bbox, box) for box in spanning_text)
                    overlaps = []
                    if spanning_bboxes:
                        for box in spanning_bboxes:
                            overlaps.append(Map.is_overlapping(span_bbox, box))
                    if any(overlaps):
                        print(f"ignoring text: {text}")
                        pass
                    else:
                        pct_contained = Map.percent_contained(span_bbox, bbox)
                        mapped_values[col_idx][row_idx].append((text, pct_contained))
            
        filtered_mapped_values = defaultdict(dict)
        for col_idx, row_map in mapped_values.items():
            for row_idx, matches in row_map.items():
                if matches:
                    # Each match is (text, pct_contained)
                    best_match = max(matches, key=lambda x: x[1])  # max by pct_contained
                    filtered_mapped_values[col_idx][row_idx] = [best_match[0]]
        
        return dict({k:dict(v) for k,v in filtered_mapped_values.items()})
    
    @staticmethod
    def get_totals(total_values):
        col_totals = {}
        for col_idx, row_data in total_values.items():
            for _, vals in row_data.items():
                tot = ''.join(vals)
                tot_clean = tot.replace(',', '').replace('$', '')
                try:
                    col_totals[col_idx]=float(tot_clean)
                except:
                    for t in vals:
                        try:
                            col_totals[col_idx]=float(t.replace(',', '').replace('$', ''))
                            break
                        except:
                            pass
                    pass
        return col_totals


    @staticmethod    
    def convert_to_float(df, columns=None):
        """
        Convert specified DataFrame columns to float type, handling various formats.
        If conversion fails or the entire string doesn't represent a number, returns the original string.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame containing columns to convert.
        columns : list or None, default None
            List of column names to convert. If None, attempts to convert all columns
            that contain numeric-like values.
            
        Returns:
        --------
        tuple
            (pandas.DataFrame, dict): 
                - DataFrame with converted columns
                - Dictionary mapping column names to their data types
        """
        # Create a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Dictionary to store column data types
        column_dtypes = {}
        
        # If columns not specified, try all columns
        if columns is None:
            # Try to identify numeric-like columns by checking if they contain digits
            columns = []
            for col in df.columns:
                # Skip if already a numeric type
                if pd.api.types.is_numeric_dtype(df[col]):
                    columns.append(col)
                    column_dtypes[col] = str(df[col].dtype)
                    continue
                    
                # Check if column contains any digits
                if df[col].astype(str).str.contains(r'\d').any():
                    columns.append(col)
                else:
                    # Store original dtype for non-processed columns
                    column_dtypes[col] = str(df[col].dtype)
        else:
            # Store original dtype for non-processed columns
            for col in df.columns:
                if col not in columns:
                    column_dtypes[col] = str(df[col].dtype)
        
        # Common currency symbols and their adjacent formatting
        currency_pattern = r'[$€£¥₹₽₩₺₴₸R\฿\u20A0-\u20CF]'
        
        # Process each column
        for col in columns:
            # Skip if already a numeric type and store dtype
            if pd.api.types.is_numeric_dtype(result_df[col]):
                column_dtypes[col] = str(result_df[col].dtype)
                continue
                
            # Make a copy of the original column for fallback
            original_values = result_df[col].copy()
            original_dtype = str(original_values.dtype)
            
            # Convert column to string type first to handle non-string values
            result_df[col] = result_df[col].astype(str)
            
            # Apply the conversion function to each value
            result_df[col] = pd.Series(
                [TableBuilder._try_convert_to_float(val, currency_pattern) 
                 for val in result_df[col]], 
                index=result_df.index
            )
            
            # Check the data type after conversion
            if pd.api.types.is_numeric_dtype(result_df[col]):
                column_dtypes[col] = str(result_df[col].dtype)
            else:
                # If mixed types, force to object dtype
                result_df[col] = result_df[col].astype('object')
                column_dtypes[col] = 'object'
        
        return result_df, column_dtypes

    @staticmethod
    def _try_convert_to_float(value, currency_pattern):
        """
        Try to convert a value to float if it represents a pure number.
        Returns the original value if the string contains non-numeric content
        after cleaning.
        
        Parameters:
        -----------
        value : str
            String value to try to convert
        currency_pattern : str
            Regex pattern for currency symbols
            
        Returns:
        --------
        float or original
            Float if conversion succeeds, otherwise original value
        """
        if pd.isna(value) or value.strip() == '':
            return value
        
        original = value
        value = str(value).strip()

        if '/' in value:
            return value
        
        # Extract potential numeric parts
        # First, check if we have a simple percentage
        if value.endswith('%') and re.match(r'^[\d.,]+%$', value):
            try:
                return float(value.rstrip('%')) / 100
            except:
                return original
        
        # Save a copy of the original cleaned string to check later
        original_cleaned = value
        
        # Remove currency symbols
        value = re.sub(currency_pattern, '', value)
        
        # Handle simple European number format
        if re.search(r'^\d{1,3}(?:\.\d{3})+,\d+$', value):
            value = value.replace('.', '')
            value = value.replace(',', '.')
        # Handle simple US/UK number format
        elif re.search(r'^\d{1,3}(?:,\d{3})+\.\d+$', value) or re.search(r'^\d{1,3}(?:,\d{3})+$', value):
            value = value.replace(',', '')
        # Handle simple number with comma as decimal
        elif value.count(',') == 1 and value.count('.') == 0:
            value = value.replace(',', '.')
        
        # Clean the string of any non-numeric chars except decimal and minus
        cleaned = re.sub(r'[^\d.-]', '', value)
        
        # Fix multiple decimal points (just keep the first one)
        parts = cleaned.split('.')
        if len(parts) > 2:
            cleaned = parts[0] + '.' + ''.join(parts[1:])
        
        # Handle parentheses for negative numbers
        if original.startswith('(') and original.endswith(')'):
            cleaned = '-' + cleaned.replace('-', '')
        
        # The critical check: if the original string has non-numeric characters
        # that aren't just formatting (like currency, commas, etc.), return original
        
        # Check if there's any alphabetic content (excluding % for percentages)
        if re.search(r'[a-zA-Z]', original_cleaned.replace('%', '')):
            return original
        
        # Check for other non-numeric content that would indicate this isn't meant to be a number
        if '+' in original_cleaned and not re.match(r'^[+\-]?[\d.,]+$', original_cleaned.replace('%', '')):
            # Special check for strings like "$0.10 + 2.10%" which might be transaction fees
            if re.search(r'^\s*' + currency_pattern + r'?\s*[\d.,]+\s*\+\s*' + 
                       currency_pattern + r'?\s*[\d.,]+%?\s*$', original_cleaned):
                # Let's try to convert only the numeric part before the + symbol
                # This is a common pattern for transaction fees
                match = re.search(r'^\s*' + currency_pattern + r'?\s*([\d.,]+)\s*\+', original_cleaned)
                if match:
                    try:
                        return float(match.group(1).replace(',', ''))
                    except:
                        pass
            # If it's not a simple fee formula or can't be converted, return the original
            return original
        
        # Now try to convert to float
        try:
            # Final check to make sure this seems like a proper number
            if re.match(r'^[-+]?\d*\.?\d+$', cleaned):
                return float(cleaned)
            else:
                return original
        except (ValueError, TypeError):
            return original


    @staticmethod
    def get_the_table(pdf_path, page_num, totals_cells, value_cells, col_names, spanning_text):

        table_frame = pd.DataFrame([])
        total_values={}
        col_totals={}
        table_values={}
        table_frame_dict={}
        data_types={}
        sum_tests={}

        try:
            total_values = TableBuilder.extract_table_from_pdf(
                pdf_path=pdf_path, 
                page_num=page_num, 
                value_cells=totals_cells,
                spanning_text=spanning_text
            )
            
            col_totals = TableBuilder.get_totals(total_values)
        except:
            pass

        try:
            table_values = TableBuilder.extract_table_from_pdf(
                pdf_path=pdf_path, 
                page_num=page_num, 
                value_cells=value_cells,
                spanning_text=spanning_text
            )
        except:
            pass

        if table_values:
            try: 
                table_frame = TableBuilder.nested_dict_to_dataframe(table_values)
                table_frame.columns = col_names
                table_frame = table_frame.applymap(lambda x: ''.join(x))
                table_frame, data_types = TableBuilder.convert_to_float(table_frame)
                table_frame_dict = table_frame.to_dict()

                try:
                    if col_totals:
                        for idx, (col_name, table_data) in enumerate(table_frame_dict.items()):
                            if col_totals.get(idx) and table_data:
                                tot = round(col_totals.get(idx), 1)
                                col_sum = round(sum(table_data.values()), 1)
                                sum_tests[idx] = {'extracted': tot, 'calculated': col_sum, 'result': tot == col_sum}
                except:
                    pass
            except:
                pass
        
        return col_totals, table_values, table_frame_dict, data_types, sum_tests

    @staticmethod
    def main(pdf_path, page_num, totals_cells, value_cells, col_names, spanning_text):
        
        col_totals, table_values, table_frame, data_types, sum_tests = TableBuilder.get_the_table(
            pdf_path=pdf_path, 
            page_num=int(page_num), 
            totals_cells=totals_cells, 
            value_cells=value_cells, 
            col_names=col_names, 
            spanning_text=spanning_text
        )
        return col_totals, table_values, table_frame, data_types, sum_tests

class Stage3: 

    def __init__(
        self,
        directory_file_node,
        table_key_dict: dict,
        parameters: dict, 
        merge: bool=False,
        classification: str = None, 
    ):
        self.node = directory_file_node
        self.table_key_dict = table_key_dict
        self.stage2_metadata = self.node.extraction_metadata["stage2"]
        self.classification = classification
        self.parameters = parameters
        self.merge = merge
    
    def process(self):
        headers_slice = self.find_tables_with_multiple_internal_headers_and_whitespace(self.stage2_metadata)
        multi_tbls = self.slice_tables_by_internal_headers(headers_slice)
        merged_tables = self.group_and_merge_tables_by_columns_hashed(multi_tbls)

        stage3_meta = self.create_pdf_from_tables_autosize(
            self.node.stage_paths[2]["abs_path"],
            merged_tables,
            output_pdf_path=self.node.stage_paths[3]["abs_path"],
            page_width=self.stage2_metadata["page_width"],
            margin=10,
            gap=1
        )

        clean_stage3 = {}
        for pg_index, meta in stage3_meta.items():
            page_width = meta['page_width']
            page_height = meta['page_height']
            items = meta['placed_items']
            table_key = meta['table_key']
            cols = meta['column_names']
            new_tables=[]
            tots=[]
            max_y=[]
            columns_box=[]
            for it in items:
                if it['type'] == 'sub_table':
                    tbl_box = it['new_bbox']
                    scaled_totY0 = it.get('totY0', None)
                    scaled_totY1 = it.get('totY1', None)
                    if scaled_totY0:
                        tots.append(scaled_totY0)
                    if scaled_totY1:
                        max_y.append(scaled_totY1)
                    new_tables.append(tbl_box)
                if it['type'] == 'columns':
                    columns_box = it['new_bbox']

            new_tables = Map.merge_all_boxes(new_tables)
            x0,y0,x1,y1 = new_tables
            clean_tots = [x for x in tots if x<=y1]
            _y1=None
            y1=None
            if clean_tots:
                tot_bound = max(clean_tots)
                y1 = tot_bound - 1
                new_tables = x0,y0,x1,y1
            if max_y:
                _y1 = max(max_y)                
            
            spanning_text = TableExtractor.find_spanning_text(
                pdf_path=self.node.stage_paths[3]["abs_path"], 
                page_number=int(pg_index), 
                table_bbox=new_tables, 
                min_coverage=0.8
            )
            
            column_bounds = TableExtractor.get_column_bboxes(
                pdf_path=self.node.stage_paths[3]["abs_path"], 
                page_number=int(pg_index), 
                header_bbox=new_tables, 
                column_names=cols,
                spanning_text=spanning_text
            )

            column_texts, tot_map, blocks_df = TableExtractor.extract_column_texts(
                pdf_path=self.node.stage_paths[3]["abs_path"], 
                page_num=int(pg_index), 
                col_bbox=column_bounds, 
                tolerance=0.0,
                row_tolerance=5
            )


            # row_bounds = TableExtractor.get_table_row_bounds(cells, y_tolerance=2.0)
            try:
                row_cells_df = TableExtractor.get_table_row_bounds_as_dataframe(
                    column_texts, 
                    tot_map, 
                    y_tolerance=2.0
                )
                row_cells = TableExtractor.clean_rows(row_cells_df, columns_box)
                
                row_cells['ROW_BBOX'] = row_cells.sort_values('y0').apply(lambda x : [x.ROW_BBOX[0],x.y0, x.ROW_BBOX[2], x.ROW_BBOX[3]], axis=1)
                row_cells[['x0', 'x1', 'y1']] = row_cells.sort_values('y0').apply(lambda x : [x.ROW_BBOX[0],x.ROW_BBOX[2], x.ROW_BBOX[3]], axis=1).apply(pd.Series)
                row_bounds, min_x0, max_x1 = TableExtractor.row_bounds_df_to_list(row_cells)

                if _y1 and y1:
                    clean_tots_bounds = [min_x0, y1, max_x1, _y1]
                    totals_cells = TableExtractor.split_totals_box_into_cells(
                        totals_bbox=clean_tots_bounds, 
                        col_bboxes=column_bounds
                    )

                table_cells = TableExtractor.get_all_cell_bboxes(
                    row_bounds=row_bounds,
                    col_bboxes=column_bounds
                )
            except:
                row_bounds=[]
                table_cells=[]
                clean_tots_bounds=[]
                totals_cells=[]
                print(f'FAILED DATAFRAME: {self.node.stage_paths[3]["abs_path"]}')
            
            try:
                col_totals, table_values, table_frame_dict, data_types, sum_tests = TableBuilder.main(
                    pdf_path=self.node.stage_paths[3]["abs_path"], 
                    page_num=pg_index, 
                    totals_cells=totals_cells, 
                    value_cells=table_cells, 
                    col_names=['_'.join(x['columnName'].split()) for x in column_bounds],
                    spanning_text=spanning_text
                )
                print('hooray!! tables!!')
            except:
                print('booo! failed!!')
                col_totals={}
                table_values={}
                table_frame_dict={}
                data_types={}
                sum_tests={}

            clean_stage3[pg_index] = {
                'page_width': page_width, 
                'page_height': page_height, 
                'table_key': table_key, 
                'new_tables': new_tables,
                'scaled_totY0': tot_bound,
                'col_bbox': column_bounds,
                'spanning_text': spanning_text,
                'table_cells': column_texts,
                'row_bounds': row_bounds,
                'value_cells': table_cells,
                'totals_bbox': clean_tots_bounds,
                'totals_cells': totals_cells,
                'col_totals': col_totals, 
                'table_values': table_values, 
                'table_frame_dict': table_frame_dict, 
                'data_types': data_types,
                'sum_tests': sum_tests
            }

        return {'pages': stage3_meta, 'clean_stage3': clean_stage3}


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
            original_table = table_item.get('original_table', {})
            column_list = original_table.get('table_columns', [])
            if column_list:
                table_key = original_table['table_key']
                grouped[table_key].append(table_item)
        
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
                "col_names": table_columns,
                "table_key": columns_hash_key
            }
        
        return {idx: v for idx, (k,v) in enumerate(merged_results.items())}


    def find_tables_with_multiple_internal_headers_and_whitespace(self, stage2_data):
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
        for table in stage2_data.get("tables", []):
            internal_headers = table.get("internal_headers_y", [])
            # 2. Check if there are multiple internal header lines
            if len(internal_headers) > 1:
                tb_x0, tb_y0, tb_x1, tb_y1 = table.get("table_bounds", [0, 0, 0, 0])

                overlapping_whitespace_blocks = []

                # 3. Collect whitespace blocks that overlap in Y dimension
                for wb in stage2_data.get("whitespace_blocks", []):
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

    def create_pdf_from_tables_autosize(
        self, 
        input_pdf_path,
        merged_tables,
        output_pdf_path="merged_tables_output.pdf",
        page_width=612,
        margin=50,
        gap=10
    ):
        new_doc = fitz.open()
        placement_metadata = {}

        def compute_scale_fit_width(clip_rect_width, clip_rect_height, max_width):
            if clip_rect_width <= 0:
                return (0, 0, 1.0)
            scale = min(1.0, max_width / clip_rect_width)
            return (clip_rect_width * scale, clip_rect_height * scale, scale)

        for idx, table_info in merged_tables.items():
            table_key = table_info["table_key"]
            boxes_to_place = []
            col_names= table_info.get('col_names', [])
            tots = table_info.get('totals')
            # print(f"TOTS: {tots}")
            if tots:
                totX0, totY0, totX1, totY1 = tots

            if "columns" in table_info:
                boxes_to_place.append(("columns", table_info["columns"]))

            for stb in table_info.get("merged_sub_tables", []):
                boxes_to_place.append(("sub_table", stb))

            total_height = 0
            max_content_width = page_width - 2 * margin
            scaled_dims = []

            for type_label, bbox in boxes_to_place:
                clip_rect = fitz.Rect(*bbox)
                scaled_w, scaled_h, scale_factor = compute_scale_fit_width(
                    clip_rect.width, clip_rect.height, max_content_width
                )
                scaled_dims.append((type_label, scaled_w, scaled_h, scale_factor))
                total_height += scaled_h + gap

            if boxes_to_place:
                total_height -= gap

            page_height = max(total_height + 2 * margin, margin * 2 + 50)
            new_page = new_doc.new_page(width=page_width, height=page_height)
            y_cursor = margin
            placed_items = []

            for (type_label, bbox), (_, scaled_w, scaled_h, scale_factor) in zip(boxes_to_place, scaled_dims):
                clip_rect = fitz.Rect(*bbox)
                source_page_number = 0  # Adjust if needed

                # -- TEMP COPY and REDACT EVERYTHING OUTSIDE clip_rect --
                tmp_pdf_path = tempfile.mktemp(suffix=".pdf")
                shutil.copy(input_pdf_path, tmp_pdf_path)
                tmp_doc = fitz.open(tmp_pdf_path)
                tmp_page = tmp_doc.load_page(source_page_number)
                page_rect = tmp_page.rect
                x0, y0, x1, y1 = clip_rect

                # Redact around the clip rectangle
                if y0 > page_rect.y0:
                    tmp_page.add_redact_annot(fitz.Rect(page_rect.x0, page_rect.y0, page_rect.x1, y0), fill=(1, 1, 1))
                if y1 < page_rect.y1:
                    tmp_page.add_redact_annot(fitz.Rect(page_rect.x0, y1, page_rect.x1, page_rect.y1), fill=(1, 1, 1))
                if x0 > page_rect.x0:
                    tmp_page.add_redact_annot(fitz.Rect(page_rect.x0, y0, x0, y1), fill=(1, 1, 1))
                if x1 < page_rect.x1:
                    tmp_page.add_redact_annot(fitz.Rect(x1, y0, page_rect.x1, y1), fill=(1, 1, 1))

                tmp_page.apply_redactions(images=True, graphics=True, text=True)

                # Compute new placement bounds
                dest_x0 = (page_width - scaled_w) / 2
                dest_y0 = y_cursor
                dest_x1 = dest_x0 + scaled_w
                dest_y1 = dest_y0 + scaled_h
                dest_rect = fitz.Rect(dest_x0, dest_y0, dest_x1, dest_y1)

                scaled_totY0 = None
                scaled_totY1 = None
                if totY0 >= y0 and totY1 <= y1 and type_label == "sub_table":
                    scaled_totY0 = (totY0 - y0) * scale_factor + dest_y0
                    scaled_totY1 = (totY1 - y0) * scale_factor + dest_y0
                # if type_label == "sub_table" and tots:
                #     scaled_totY0 = (totY0 - y0) * scale_factor + dest_y0
                #     scaled_totY1 = (totY1 - y1) * scale_factor + dest_y1

                # Copy the redacted clip from the cleaned temp doc
                new_page.show_pdf_page(
                    dest_rect,
                    tmp_doc,
                    source_page_number,
                    clip=clip_rect
                )

                placed_items.append({
                    "type": type_label,
                    "original_bbox": bbox,
                    "new_bbox": [dest_x0, dest_y0, dest_x1, dest_y1],
                    "scale_factor": scale_factor,
                    "source_page_number": source_page_number,
                    "totY0": scaled_totY0, 
                    "totY1": scaled_totY1
                })

                y_cursor += scaled_h + gap
                tmp_doc.close()
                os.remove(tmp_pdf_path)

            placement_metadata[idx] = {
                "new_pdf_page_index": new_page.number,
                "page_width": page_width,
                "page_height": page_height,
                "placed_items": placed_items,
                "column_names": col_names,
                "table_key": table_key
            }

        if len(new_doc) > 0:
            new_doc.save(output_pdf_path, garbage=4, deflate=True, clean=True)
            print(f"Created {output_pdf_path} with {len(merged_tables)} pages.")
            return placement_metadata
        else:
            return


class TableExtractor:

    @staticmethod
    def extract_column_texts(pdf_path, page_num, col_bbox, tolerance=0.0, row_tolerance=0):
        """
        Extract text and bounding boxes within each column using span-level data,
        and include the column index.

        Parameters
        ----------
        doc : fitz.Document
            The opened PDF document.
        page_num : int
            Page number (0-based).
        col_bbox : list of dict
            List of columns with their bounding boxes and names.
        tolerance : float
            Margin of error for edge overlap when matching word bounds.

        Returns
        -------
        dict
            Mapping of columnName to a list of (text, bbox, col_index) tuples found within its bounding box.
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]

        column_texts = {col['columnName']: [] for col in col_bbox}
        current_line=0
        all_blocks=[]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    x0, y0, x1, y1 = span["bbox"]
                    text = span["text"].strip()
                    if not text:
                        continue

                    for idx, col in enumerate(col_bbox):
                        current_line+=5
                        col_name = col["columnName"]
                        cx0, cy0, cx1, cy1 = col["coords"]

                        # Apply tolerance
                        adj_x0 = cx0 - tolerance
                        adj_y0 = cy0 - tolerance
                        adj_x1 = cx1 + tolerance
                        adj_y1 = cy1 + tolerance

                        # Check for span overlap in X and Y
                        if (adj_x0 <= x0 <= adj_x1 or adj_x0 <= x1 <= adj_x1 or (x0 <= adj_x0 and x1 >= adj_x1)) and \
                        (adj_y0 <= y0 <= adj_y1 or adj_y0 <= y1 <= adj_y1 or (y0 <= adj_y0 and y1 >= adj_y1)):
                            column_texts[col_name].append((text, span["bbox"], idx, span["font"], current_line))
                            all_blocks.append([x0, y0, x1, y1, current_line])
                            break  # Avoid assigning the same span to multiple columns

        doc.close()

        tot_map={}
        blocks_df = pd.DataFrame(all_blocks)
        if len(blocks_df)>0:
            blocks_df.columns = ['x0', 'y0', 'x1', 'y1', 'row_num']
            
            blocks_df['y0_lag'] = blocks_df['y0'].shift(1)
            blocks_df['row_diff'] = blocks_df.apply(lambda x : x['y0']-x['y0_lag'], axis=1)        
            group_ids = []
            group = 0
            
            for idx, current in enumerate(blocks_df['row_diff'].values.tolist()):
                if current > row_tolerance:
                    group += 1
                group_ids.append(group)
            
            blocks_df['group_ids'] = group_ids
            blocks_df['min_y0'] = blocks_df.groupby('group_ids')['y0'].transform('min')
            blocks_df['max_y1'] = blocks_df.groupby('group_ids')['y1'].transform('max')

            blocks_df['y0_lag'] = blocks_df.apply(lambda x: x.y0_lag if pd.notna(x.y0_lag) else x.y0, axis=1)
            
            for _, grp in blocks_df[['row_num', 'x0', 'min_y0', 'x1', 'max_y1']].iterrows():
                tot_map[grp['row_num']] = [grp['x0'], grp['min_y0'], grp['x1'], grp['max_y1']]
            
        return column_texts, tot_map, blocks_df

    @staticmethod
    def get_table_row_bounds_as_dataframe(table_cells, tot_map, y_tolerance=5.0):
        """
        Group table cells into rows using Y-overlap and return as a flat DataFrame.
        
        Parameters
        ----------
        table_cells : dict
            Output from extract_column_texts, mapping column name to (text, bbox, index) tuples.
        y_tolerance : float
            Tolerance for grouping bbox y-ranges into the same row.
            
        Returns
        -------
        pd.DataFrame
            Columns:
            - TEXT: cell text
            - COLNAME: column name
            - COORDS: cell bounding box [x0, y0, x1, y1]
            - COL_IDX: column index (position in column layout)
            - ROW_BBOX: bounding box of the full row [x0, y0, x1, y1]
            - x0, y0, x1, y1: unpacked span box values (for sorting/debug)
        """
        # Collect all spans
        all_spans = []
        for col_name, entries in table_cells.items():
            for text, bbox, col_index, font, col_line in entries:
                x0, y0, x1, y1 = bbox
                # Calculate the center y-coordinate for better grouping
                y_center = (y0 + y1) / 2
                all_spans.append((text, bbox, col_name, x0, y0, x1, y1, col_index, font, y_center, col_line))
        
        # Sort spans by y-center coordinate for better grouping
        all_spans.sort(key=lambda x: x[9])  # y_center
        
        # Group into rows with improved overlap detection
        rows = []
        for text, bbox, col_name, x0, y0, x1, y1, col_index, font, y_center, col_line in all_spans:
            placed = False
            
            # Try to place in existing row
            for row in rows:
                # Use the y_center to determine if in same row
                # Check if within tolerance of the row's average y_center
                if abs(row['y_center'] - y_center) <= y_tolerance:
                    # Update row boundaries
                    row['y0'] = min(row['y0'], y0)
                    row['y1'] = max(row['y1'], y1)
                    row['x0'] = min(row['x0'], x0)
                    row['x1'] = max(row['x1'], x1)
                    row['cells'].append((text, bbox, col_name, col_index, font, col_line))
                    row['bbox'] = [row['x0'], row['y0'], row['x1'], row['y1']]
                    
                    # Update the average y_center of this row
                    # This is to adjust the row's center as new cells are added
                    n_cells = len(row['cells'])
                    row['y_center'] = ((n_cells - 1) * row['y_center'] + y_center) / n_cells
                    
                    placed = True
                    break
                    
            # Create new row if not placed
            if not placed:
                rows.append({
                    'y0': y0,
                    'y1': y1,
                    'x0': x0,
                    'x1': x1,
                    'bbox': [x0, y0, x1, y1],
                    'cells': [(text, bbox, col_name, col_index, font, col_line)],
                    'y_center': y_center  # Track average center of the row
                })
        
        # Flatten to DataFrame
        records = []
        for i, row in enumerate(rows):
            for text, bbox, col_name, col_index, font, col_line in row['cells']:
                records.append({
                    "TEXT": text,
                    "COLNAME": col_name,
                    "COORDS": bbox,
                    "COL_IDX": col_index,
                    "ROW_BBOX": row["bbox"],
                    "ROW_IDX": i,  # Add row index for easier grouping
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                    "font": font,
                    "col_line": col_line
                })
        
        # Create DataFrame, sort by row index and column index
        df = pd.DataFrame(records)
        df['CLEAN_BBOX'] = df['col_line'].map(tot_map)
        df[['xx0','yy0','xx1','yy1']] = df['CLEAN_BBOX'].apply(pd.Series)
        df.drop(columns={'x0','y0','x1','y1'}, inplace=True)
        df.rename(columns={'xx0':'x0','yy0':'y0','xx1':'x1','yy1':'y1'}, inplace=True)
        return df

    @staticmethod
    def merge_row_group(group):
        texts = group['TEXT'].tolist()
        bboxes = group['COORDS'].tolist()  # or 'bbox'
        merged_bbox = Map.merge_all_boxes(bboxes)
        
        return pd.Series({
            'TEXT': ' '.join(texts),  # or use '\n'.join(...) if multiline
            'ROW_BBOX': merged_bbox
        })

    @staticmethod
    def clean_rows(row_bounds_df, columns_box):
        row_bounds_df.drop(columns={'ROW_BBOX'}, inplace = True)
        row_bounds_df['ROW_BBOX'] = row_bounds_df.apply(lambda x: [x.x0, x.y0, x.x1, x.y1], axis=1)
        mask = row_bounds_df.apply(lambda x: not Map.is_overlapping((x.x0, x.y0, x.x1, x.y1), columns_box), axis=1)
        filtered_df = row_bounds_df[mask]
        merged_rows = (
            filtered_df
            .sort_values('y0')
            .groupby(['COLNAME', 'y0', 'font', 'COL_IDX'])
            .apply(TableExtractor.merge_row_group)
            .reset_index()
        )
        return merged_rows


    @staticmethod
    def clip_row_y1_to_next_y0(df):
        df = df.copy().sort_values('y0').reset_index(drop=True)
        df['next_y0'] = df['y0'].shift(-1)
        df['y1'] = df[['y1', 'next_y0']].min(axis=1)
        return df.drop(columns='next_y0')

    @staticmethod
    def row_bounds_df_to_list(row_bounds_df):
        # print(row_bounds_df.columns)
        min_x0 = row_bounds_df['x0'].min()
        max_x1 = row_bounds_df['x1'].max()
        row_bounds_df_sorted = row_bounds_df.sort_values(['y0', 'COL_IDX']).drop_duplicates('y0')
        row_bounds_df_sorted['y_gap'] = row_bounds_df_sorted['y0'].diff().fillna(0)

        row_bounds_df_sorted = row_bounds_df_sorted.reset_index().reset_index().drop(columns={'index'}).rename(columns={'level_0': 'index'})
        true_row_bounds = TableExtractor.clip_row_y1_to_next_y0(row_bounds_df_sorted).apply(lambda x : [min_x0, x.y0, max_x1, x.y1], axis=1).tolist()
        return true_row_bounds, min_x0, max_x1

    @staticmethod
    def get_all_cell_bboxes(row_bounds, col_bboxes):
        """
        Combine row and column bounding boxes to compute bounding boxes for all cells.

        Parameters
        ----------
        row_bounds : list of list
            Each row is [x0, y0, x1, y1] — the vertical bounds of a table row.
        col_bboxes : list of dict
            Each dict must have 'columnName' and 'coords' (a 4-tuple [x0, y0, x1, y1]).

        Returns
        -------
        list of dict
            Each dict represents one cell:
            {
                'row_index': int,
                'col_index': int,
                'columnName': str,
                'bbox': [x0, y0, x1, y1]
            }
        """
        cells = []

        for row_idx, row in enumerate(row_bounds):
            row_x0, row_y0, row_x1, row_y1 = row

            for col_idx, col in enumerate(col_bboxes):
                col_x0, _, col_x1, _ = col['coords']

                # The intersection box: use col x-range + row y-range
                cell_bbox = [col_x0, row_y0, col_x1, row_y1]

                cells.append({
                    'row_index': row_idx,
                    'col_index': col_idx,
                    'columnName': col['columnName'],
                    'bbox': cell_bbox
                })

        return cells

    @staticmethod
    def split_totals_box_into_cells(totals_bbox, col_bboxes):
        """
        Split a totals row bbox into individual cells using the X boundaries from col_bboxes.

        Parameters
        ----------
        totals_bbox : list
            A 4-tuple [x0, y0, x1, y1] representing the full totals row.
        col_bboxes : list of dict
            Each dict must contain 'coords': [x0, y0, x1, y1] and 'columnName'.

        Returns
        -------
        list of dict
            Each dict contains:
            - 'columnName': name of the column
            - 'bbox': [x0, y0, x1, y1] for the cell in the totals row
            - 'col_index': index of the column
        """
        x0, y0, x1, y1 = totals_bbox

        cell_boxes = []
        for i, col in enumerate(col_bboxes):
            col_x0, _, col_x1, _ = col['coords']
            cell_bbox = [col_x0, y0, col_x1, y1]
            cell_boxes.append({
                "columnName": col['columnName'],
                "bbox": cell_bbox,
                "col_index": i
            })

        return cell_boxes

    @staticmethod
    def find_column_breaks_using_newlines(pdf_path, page_number, table_bbox):
        """
        Find potential column breaks by analyzing newline patterns.
        
        This finds x-positions where newlines commonly occur, which likely indicate
        column boundaries in a table.
        
        Parameters
        ----------
        pdf_path: str
            File path to the PDF.
        page_number: int
            The page index (0-based) to work with.
        table_bbox: list or tuple of float
            The bounding box [x0, y0, x1, y1] representing the table bounds.
            
        Returns
        -------
        list
            List of x-positions that likely represent column breaks.
        """
        # Open the PDF and get the specified page
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        
        # Get raw text with newlines preserved
        raw_text = page.get_text("text")
        
        # Get text with position information
        text_with_pos = page.get_text("dict")
        blocks = text_with_pos["blocks"]
        
        # Get table dimensions
        table_x0, table_y0, table_x1, table_y1 = table_bbox
        
        # Store potential column break positions
        newline_positions = []
        line_end_positions = []
        
        # Extract positions of newlines
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    line_bbox = line["bbox"]
                    
                    # Check if line is within or overlaps with table
                    if (line_bbox[1] >= table_y0 and line_bbox[3] <= table_y1) or \
                       (line_bbox[1] < table_y1 and line_bbox[3] > table_y0):
                        
                        # Record right edge of line as potential column break
                        line_end_positions.append(line_bbox[2])
                        
                        # Check for newlines in spans
                        for span in line["spans"]:
                            text = span.get("text", "")
                            if "\n" in text:
                                # Record position of span right edge
                                newline_positions.append(span["bbox"][2])
        
        # Combine positions and find clusters
        all_positions = newline_positions + line_end_positions
        
        # Skip if no positions found
        if not all_positions:
            doc.close()
            return []
        
        # Sort positions
        all_positions.sort()
        
        # Cluster nearby positions (within threshold)
        threshold = 5.0  # PDF units
        clusters = []
        current_cluster = [all_positions[0]]
        
        for pos in all_positions[1:]:
            if pos - current_cluster[-1] <= threshold:
                # Add to current cluster
                current_cluster.append(pos)
            else:
                # Start new cluster
                if current_cluster:
                    # Use average of cluster
                    clusters.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [pos]
        
        # Add final cluster
        if current_cluster:
            clusters.append(sum(current_cluster) / len(current_cluster))
        
        # Filter clusters that are too close to table edges
        edge_threshold = 20.0  # PDF units
        filtered_clusters = []
        
        for pos in clusters:
            if pos > table_x0 + edge_threshold and pos < table_x1 - edge_threshold:
                filtered_clusters.append(pos)
        
        doc.close()
        return filtered_clusters
    
    @staticmethod
    def get_column_bboxes(pdf_path, page_number, header_bbox, column_names, spanning_text=None):
        """
        Given a PDF path, a page number, a bounding box (in PDF coordinates), and a list of column names,
        split the bounding box horizontally into sub-boxes for each column, ignoring spanning text.
        
        Parameters
        ----------
        pdf_path: str
            File path to the PDF.
        page_number: int
            The page index (0-based) to work with.
        header_bbox: list or tuple of float
            The bounding box [x0, y0, x1, y1] representing the header bounds in PDF coordinates.
        column_names: list of str
            Names of the columns in the table header.
        spanning_text: list of dict, optional
            List of spanning text items to ignore. Each item should have a 'bbox' key
            with coordinates [x0, y0, x1, y1].
            
        Returns
        -------
        list
            A list of column objects with format:
            [
                {
                    'columnName': str,
                    'coords': [x0, y0, x1, y1]
                },
                ...
            ]
        """
        # Initialize spanning_text if None
        if spanning_text is None:
            spanning_text = []

        # Open the PDF and get the specified page
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        
        # Find potential column breaks using newline analysis
        newline_breaks = TableExtractor.find_column_breaks_using_newlines(
            pdf_path, page_number, header_bbox
        )
        
        # Extract text spans
        text_dict = page.get_text("dict")
        blocks = text_dict["blocks"]
        pdf_words = []
        
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    for span in line["spans"]:
                        # Store newline information with span
                        span['has_newline'] = '\n' in span.get('text', '')
                        
                        # Skip spans that overlap with spanning text
                        if not Map.is_in_spanning_text(span['bbox'], spanning_text):
                            pdf_words.append(span)
        
        # Find matches for each column name
        matches = {}
        
        # Group column names to handle duplicates
        grouped = defaultdict(list)
        for idx, name in enumerate(column_names):
            grouped[name].append(idx)

        # Create unique column identifiers
        all_unique_cols = []
        for col_name, indexes in grouped.items():
            unique_cols = [(f"{col_name}_{idx}", col_name, idx) for idx, _ in enumerate(indexes)]
            for c in unique_cols:
                all_unique_cols.append(c)

        # Find matches for unique columns
        for unique_col, target_string, target_index in all_unique_cols:
            raw_matches = Search.find_sequential_mathces(
                pdf_words, target_string, column_names, target_type='column_matches'
            )            
            if raw_matches:
                for index, match in enumerate(raw_matches):
                    if match and index == target_index:
                        matches[unique_col] = match
        
        # Group spans by column
        final_cols = defaultdict(list)
        for col, col_matches in matches.items():
            final_cols[col].extend(col_matches)
            
            # Add spans that overlap with column headers horizontally
            for span in pdf_words:
                bbox = span['bbox']
                # Skip if it's in spanning text
                if Map.is_in_spanning_text(bbox, spanning_text):
                    continue
                    
                for col_match in col_matches:
                    if Map.is_x_overlapping(bbox, col_match):
                        if Map.is_overlapping(bbox, header_bbox):
                            final_cols[col].append(bbox)
                            break  # Stop checking once overlap is found for this span
        
        # Merge column bounds
        column_bounds = []
        for col_name, boxes in final_cols.items():
            merged_box = Map.merge_all_boxes(boxes)
            if merged_box:
                # Add a small buffer (padding) to each column to avoid exact boundary issues
                padding = 1.0  # Small padding in PDF units
                column_bounds.append({
                    'name': col_name,
                    'bbox': merged_box,
                    'x0': merged_box[0] - padding,  # Add padding to left
                    'x1': merged_box[2] + padding   # Add padding to right
                })
        
        # Sort columns by x-position
        column_bounds.sort(key=lambda x: x['x0'])
        
        # Create non-overlapping column boundaries
        final_bounds = []
        for i, column in enumerate(column_bounds):
            # First column starts at table boundary
            if i == 0:
                left_bound = header_bbox[0]
            else:
                prev_column = column_bounds[i-1]
                
                # Check if there's a newline break between these columns
                newline_break_x = None
                for break_x in newline_breaks:
                    # Check if break is between the two columns
                    if prev_column['x1'] < break_x and break_x < column['x0']:
                        newline_break_x = break_x
                        break
                
                if newline_break_x:
                    # Use the newline break as the boundary
                    left_bound = newline_break_x
                else:
                    # Fall back to improved midpoint calculation
                    
                    # Calculate the midpoint, but ensure minimum gap
                    mid = (prev_column['x1'] + column['x0']) / 2
                    
                    # Ensure there's at least a minimum gap between columns
                    min_gap = 2.0  # Minimum gap in PDF units
                    
                    # If columns are too close, ensure proper separation
                    if column['x0'] - prev_column['x1'] < min_gap:
                        # Find a point that's proportionally between the columns
                        # based on the relative widths of the columns
                        prev_width = prev_column['x1'] - prev_column['x0']
                        curr_width = column['x1'] - column['x0']
                        total_width = prev_width + curr_width
                        
                        # Calculate weighted midpoint based on column widths
                        weight_prev = prev_width / total_width
                        weight_curr = curr_width / total_width
                        
                        # Place boundary proportionally between columns
                        mid = prev_column['x1'] * weight_prev + column['x0'] * weight_curr
                    
                    left_bound = mid
            
            # Last column extends to table boundary
            if i == len(column_bounds) - 1:
                right_bound = header_bbox[2]
            else:
                next_column = column_bounds[i+1]
                
                # Check if there's a newline break between these columns
                newline_break_x = None
                for break_x in newline_breaks:
                    # Check if break is between the two columns
                    if column['x1'] < break_x and break_x < next_column['x0']:
                        newline_break_x = break_x
                        break
                
                if newline_break_x:
                    # Use the newline break as the boundary
                    right_bound = newline_break_x
                else:
                    # Fall back to improved midpoint calculation
                    
                    # Apply the same improved boundary calculation for the right bound
                    mid = (column['x1'] + next_column['x0']) / 2
                    
                    min_gap = 2.0
                    if next_column['x0'] - column['x1'] < min_gap:
                        curr_width = column['x1'] - column['x0']
                        next_width = next_column['x1'] - next_column['x0']
                        total_width = curr_width + next_width
                        
                        weight_curr = curr_width / total_width
                        weight_next = next_width / total_width
                        
                        mid = column['x1'] * weight_curr + next_column['x0'] * weight_next
                    
                    right_bound = mid
            
            # Skip creating a column if boundaries overlap with spanning text
            column_box = [left_bound, header_bbox[1], right_bound, header_bbox[3]]
            skip_column = False
            
            for span_item in spanning_text:
                span_bbox = span_item.get('bbox', None)
                if span_bbox and Map.is_overlapping(column_box, span_bbox):
                    # Calculate the percentage of overlap
                    col_width = column_box[2] - column_box[0]
                    col_height = column_box[3] - column_box[1]
                    col_area = col_width * col_height
                    
                    overlap_x0 = max(column_box[0], span_bbox[0])
                    overlap_y0 = max(column_box[1], span_bbox[1])
                    overlap_x1 = min(column_box[2], span_bbox[2])
                    overlap_y1 = min(column_box[3], span_bbox[3])
                    
                    if overlap_x1 > overlap_x0 and overlap_y1 > overlap_y0:
                        overlap_width = overlap_x1 - overlap_x0
                        overlap_height = overlap_y1 - overlap_y0
                        overlap_area = overlap_width * overlap_height
                        
                        # If the overlap is significant (> 50% of column area), skip this column
                        if overlap_area / col_area > 0.5:
                            skip_column = True
                            break
            
            if not skip_column:
                final_bounds.append({
                    'columnName': column['name'],
                    'coords': column_box
                })
        
        doc.close()
        return final_bounds

    @staticmethod
    def find_spanning_text(pdf_path, page_number, table_bbox, min_coverage=0.8):
        """
        Find text that spans horizontally across a significant portion of the table.
        
        Parameters
        ----------
        pdf_path : str
            Path to the PDF file
        page_number : int
            Page number (0-based index)
        table_bbox : list or tuple
            Table bounding box [x0, y0, x1, y1]
        min_coverage : float, optional
            Minimum horizontal coverage ratio required (0.0-1.0), default is 0.8 (80%)
            
        Returns
        -------
        list
            List of dictionaries with spanning text information:
            [
                {
                    'text': str,           # Text content
                    'bbox': [x0,y0,x1,y1], # Bounding box
                    'coverage': float      # Horizontal coverage ratio
                },
                ...
            ]
        """
        # Extract table dimensions
        table_x0, table_y0, table_x1, table_y1 = table_bbox
        table_width = table_x1 - table_x0
        
        # Open the PDF and get the specified page
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        
        # Extract text spans within the table area
        text_dict = page.get_text("dict")
        blocks = text_dict["blocks"]
        
        # Store spans that meet the coverage criteria
        spanning_text = []
        
        # Process each text span
        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    # Check if line is within table bounds vertically
                    line_y0 = line["bbox"][1]
                    line_y1 = line["bbox"][3]
                    
                    if (line_y0 >= table_y0 and line_y1 <= table_y1) or \
                       (line_y0 <= table_y0 and line_y1 >= table_y0) or \
                       (line_y0 <= table_y1 and line_y1 >= table_y1):
                        
                        # Calculate horizontal coverage
                        line_x0 = line["bbox"][0]
                        line_x1 = line["bbox"][2]
                        
                        # Clip to table bounds
                        clipped_x0 = max(line_x0, table_x0)
                        clipped_x1 = min(line_x1, table_x1)
                        
                        # Calculate horizontal overlap
                        horizontal_overlap = clipped_x1 - clipped_x0
                        
                        # Calculate coverage ratio
                        if horizontal_overlap > 0:
                            coverage_ratio = horizontal_overlap / table_width
                            
                            # Check if it meets the minimum coverage
                            if coverage_ratio >= min_coverage:
                                # Combine text from all spans in the line
                                line_text = ""
                                for span in line["spans"]:
                                    line_text += span["text"]
                                
                                # Store the result
                                spanning_text.append({
                                    'text': line_text,
                                    'bbox': [clipped_x0, line_y0, clipped_x1, line_y1],
                                    'coverage': coverage_ratio
                                })
        
        doc.close()
        return spanning_text
    