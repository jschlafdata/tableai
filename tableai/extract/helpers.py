import os
import re
import json
import yaml
import fitz
from collections import defaultdict
from typing import Tuple, Optional
from tableai.data_loaders.files import FileReader
import os
import shutil
import tempfile

class Detect:
    @staticmethod
    def detect_header_footer_bounds(
        doc: fitz.Document, 
        **kwargs
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Analyzes a PDF to detect consistent header and footer regions and returns their boundary positions.
        
        Args:
            doc:            fitz.Document object
            header_threshold: 
            footer_threshold:
            min_occurrence_ratio: Minimum ratio of pages that must contain the header/footer
            
        Returns:
            Tuple containing:
            - header_bound: Y coordinate for bottom of header region (None if no header detected)
            - footer_bound: Y coordinate for top of footer region (None if no footer detected)
        """
        num_pages = len(doc)
        # Extract thresholds from kwargs
        header_threshold, footer_threshold, min_occurrence_ratio = [
            v for k, v in kwargs['extract']['coordinates']['bounds']['detect_header_footer_bounds'].items()
            if k != 'description'
        ]
        
        # Collect data about possible headers and footers
        header_blocks_data = defaultdict(list)  # Maps text -> list of (y0, y1)
        footer_blocks_data = defaultdict(list)  # Maps text -> list of (y0, y1)

        for page in doc:
            blocks = page.get_text("blocks")
            page_height = page.rect.height
            
            for block in blocks:
                x0, y0, x1, y1, text, block_no, _ = block
                block_text = text.strip()
                
                # Check for potential header
                if y0 <= header_threshold and block_text:
                    header_blocks_data[block_text].append((y0, y1))
                
                # Check for potential footer
                if (page_height - y1) <= footer_threshold and block_text:
                    footer_blocks_data[block_text].append((y0, y1))
        
        # Filter blocks that appear in at least min_occurrence_ratio of pages
        header_candidates = {
            text: coords for text, coords in header_blocks_data.items()
            if len(coords) >= min_occurrence_ratio * num_pages
        }
        footer_candidates = {
            text: coords for text, coords in footer_blocks_data.items()
            if len(coords) >= min_occurrence_ratio * num_pages
        }
        
        # Find the max y1 for headers (bottom boundary) and min y0 for footers (top boundary)
        header_bound = (
            max(max(y1 for _, y1 in coords) for coords in header_candidates.values())
            if header_candidates else None
        )
        footer_bound = (
            min(min(y0 for y0, _ in coords) for coords in footer_candidates.values())
            if footer_candidates else None
        )
            
        return header_bound, footer_bound

    @staticmethod
    def collect_page_text_data(pdf_path):
        """
        Collect all text spans with their bounding boxes from each page of a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary mapping page numbers to lists of text spans
            Format: {page_number: [{'bbox': [x0, y0, x1, y1], 'text': '...'}]}
        """
        page_text_data = {}
        
        doc = fitz.open(pdf_path)
        
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            page_text_data[page_number] = []
            
            # Get all text from the page
            text_dict = page.get_text("dict")
            blocks = text_dict["blocks"]
            
            # Extract all spans
            for block in blocks:
                if block["type"] == 0:  # Text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # Get coordinates and text
                            x0, y0, x1, y1 = span['bbox']
                            text = span['text']
                            
                            # Skip empty spans
                            if text.strip():
                                page_text_data[page_number].append({
                                    'bbox': [x0, y0, x1, y1],
                                    'text': text
                                })
        
        doc.close()
        return page_text_data

class Search:
    @staticmethod
    def find_sequential_mathces(
        pdf_words, 
        target_string, 
        source_cols,
        return_original=False,
        target_type='table_name'
    ):
        """
        Searches for a sequence of words in pdf_words that, when concatenated (and lowercased),
        match the target_string.

        Args:
            pdf_words: List[Dict], each dict with:
                {
                    'text': 'Paid By',
                    'bbox': (x0, y0, x1, y1)
                }
            target_string:  The string we want to find, e.g. 'datebatchpaidbyamount'
            source_cols:    Original columns (kept for possible reference)
            return_original:If True, returns both bounding boxes and references
            
        Returns:
            If return_original=False, returns just a list of matches, where each match
            is a list of bounding boxes (one box per matching word).
            
            If return_original=True, returns a tuple:
              (all_matches, [(boxes_for_this_match, source_cols), ... ])
        """
        normalized_target = re.sub(r"\s+", "", target_string).lower()
        target_len = len(normalized_target)

        all_matches = []
        def normalize_word(txt):
            return re.sub(r"\s+", "", txt).lower()
        
        if target_type in ('column_matches', 'totals_matches'):
            # Check each pdf_word index as a possible start of the match
            for start_index in range(len(pdf_words)):
                match_bboxes = []
                matched_length = 0

                for w in range(start_index, len(pdf_words)):
                    current_word_norm = normalize_word(pdf_words[w]['text'])
                    needed_sub = normalized_target[matched_length : matched_length + len(current_word_norm)]
                    
                    if current_word_norm == needed_sub:
                        match_bboxes.append(pdf_words[w]['bbox'])
                        matched_length += len(current_word_norm)

                        if matched_length == target_len:
                            all_matches.append(match_bboxes)
                            break
                    else:
                        # Mismatch, stop this attempt
                        break
            
            if return_original:
                match_references = [(match, source_cols) for match in all_matches]
                return all_matches, match_references
            else:
                return all_matches
        else:
            all_matches = []

            # For each possible start index
            for start_index in range(len(pdf_words)):
                matched_length = 0
                match_bboxes = []

                w = start_index
                # Keep going until we consume the entire target or fail
                while w < len(pdf_words) and matched_length < target_len:
                    current_norm = normalize_word(pdf_words[w]['text'])
                    bbox = pdf_words[w]['bbox']

                    local_index = 0  # pointer in current_norm
                    while local_index < len(current_norm) and matched_length < target_len:
                        if current_norm[local_index] == normalized_target[matched_length]:
                            # If not already in our matched list, add the bounding box
                            if not match_bboxes or match_bboxes[-1] != bbox:
                                match_bboxes.append(bbox)

                            matched_length += 1
                            local_index += 1
                        else:
                            # mismatch in mid-word => end this attempt
                            break

                    # If we are in mid-word and mismatch occurs, we stop this start_index attempt
                    if local_index < len(current_norm) and matched_length < target_len:
                        break

                    w += 1  # next pdf_word

                # If we matched the entire target
                if matched_length == target_len:
                    all_matches.append(match_bboxes)

            if return_original:
                match_references = [(match, source_cols) for match in all_matches]
                return all_matches, match_references
            else:
                return all_matches

    @staticmethod
    def find_recurring_text_blocks(
        blocks,
        page_breaks,
        total_height,
        page_heights,
        header_bound,
        footer_bound,
        min_occurrences
    ):
        """
        Identifies text that appears >= self.min_occurrences times,
        skipping blocks fully in the middle region [header_bound..(page_height - footer_bound)]
        for each original page. Also normalizes "Page X of Y" text so that
        "Page 1 of 5", "Page 2 of 5", etc. are treated as the same string.
        Page breaks represent "virtual" breaks of pages, after combining a multi page pdf into 1 page.
        """

        def normalize_page_number_text(txt):
            t = txt.lower().strip()
            # "page X of Y" => "page xx of xx"
            t = re.sub(r'page\s*\d+\s*of\s*\d+', 'page xx of xx', t)
            # "page X" => "page xx"
            t = re.sub(r'page\s*\d+', 'page xx', t)
            return t

        text_map = defaultdict(list)

        for (x0, y0, x1, y1, txt) in blocks:
            # 1) Determine which original page this block belongs to
            page_index = None
            for i in range(len(page_breaks)):
                start_y = page_breaks[i]
                end_y = (
                    page_breaks[i+1]
                    if i < len(page_breaks) - 1
                    else total_height
                )
                if y0 >= start_y and y1 <= end_y:
                    page_index = i
                    break

            # If the block doesn't fully fit on one page, skip it
            if page_index is None:
                continue

            # 2) Local coords for that page
            start_y = page_breaks[page_index]
            page_h = page_heights[page_index]
            y0_local = y0 - start_y
            y1_local = y1 - start_y

            # 3) Skip blocks that are fully in the middle region
            #    [header_bound..(page_height - footer_bound)]
            if (header_bound is not None) and (footer_bound is not None):
                mid_start = header_bound
                mid_end = page_h - footer_bound
                if (y0_local >= mid_start) and (y1_local <= mid_end):
                    continue

            # 4) Normalize the text for page-number patterns
            normalized_txt = normalize_page_number_text(txt)

            # 5) Keep the block under the normalized text key
            text_map[normalized_txt].append((x0, y0, x1, y1))

        # 6) Filter by min_occurrences
        recurring_dict = {}
        for norm_txt, coords_list in text_map.items():
            if len(coords_list) >= min_occurrences:
                recurring_dict[norm_txt] = coords_list

        return recurring_dict

    @staticmethod
    def get_text_blocks_from_single_page(pdf_path):
        """
        Extracts text blocks from the 1st page of a single-page PDF (the combined PDF).
        Returns list of tuples: (x0, y0, x1, y1, text).
        """
        doc = fitz.open(pdf_path)
        if len(doc) == 0:
            doc.close()
            return []

        page = doc[0]
        blocks = page.get_text("blocks")
        results = []
        for b in blocks:
            x0, y0, x1, y1, text, block_no = b[:6]
            txt = text.strip()
            if txt:
                results.append((x0, y0, x1, y1, txt))
        doc.close()
        return results


class Map:
    @staticmethod
    def is_overlapping(boxA, boxB):
        """Return True if boxA and boxB overlap at all."""
        x0A, y0A, x1A, y1A = boxA
        x0B, y0B, x1B, y1B = boxB
        
        if x1A < x0B or x1B < x0A:  # One rectangle is left of the other
            return False
        if y1A < y0B or y1B < y0A:  # One rectangle is above the other
            return False
        return True

    @staticmethod
    def is_x_overlapping(boxA, boxB):
        """
        Return True if the horizontal (x-axis) bounds of boxA and boxB overlap.
        Ignores vertical positioning.
        
        Parameters:
            boxA, boxB: Lists or tuples in the form [x0, y0, x1, y1]
        
        Returns:
            bool: True if x-ranges intersect, False otherwise.
        """
        x0A, _, x1A, _ = boxA
        x0B, _, x1B, _ = boxB

        return not (x1A < x0B or x1B < x0A)

    @staticmethod
    def is_in_spanning_text(bbox, spanning_text_list):
        """Check if a bounding box overlaps with any spanning text."""
        if not spanning_text_list:
            return False
        
        for spanning_item in spanning_text_list:
            span_bbox = spanning_item.get('bbox', None)
            if span_bbox and Map.is_overlapping(bbox, span_bbox):
                return True
                
        return False

    @staticmethod
    def merge_all_boxes(bboxes):
        """
        Takes a list of bounding boxes [ (x0,y0,x1,y1), ... ]
        and merges all of them into a single bounding box that encompasses all boxes.
        Returns a single bounding box (x0,y0,x1,y1).
        
        If the input list is empty, returns None.
        """
        if not bboxes:
            return None
        
        # Extract all coordinates
        xs0 = [box[0] for box in bboxes]
        ys0 = [box[1] for box in bboxes]
        xs1 = [box[2] for box in bboxes]
        ys1 = [box[3] for box in bboxes]
        
        # Create a single bounding box that encompasses all
        merged_box = (
            min(xs0),
            min(ys0),
            max(xs1),
            max(ys1)
        )
        
        return merged_box

    @staticmethod
    def percent_contained(inner, outer) -> float:
        """
        Returns the percentage (0.0 to 1.0) of `inner`'s area that is contained within `outer`.

        If there's no intersection, returns 0.0.
        If `inner` is fully contained in `outer`, returns 1.0.
        """
        x0_i, y0_i, x1_i, y1_i = inner
        x0_o, y0_o, x1_o, y1_o = outer

        # Calculate intersection box
        x0 = max(x0_i, x0_o)
        y0 = max(y0_i, y0_o)
        x1 = min(x1_i, x1_o)
        y1 = min(y1_i, y1_o)

        inter_width = max(0.0, x1 - x0)
        inter_height = max(0.0, y1 - y0)
        intersection_area = inter_width * inter_height

        inner_area = max(0.0, (x1_i - x0_i) * (y1_i - y0_i))
        if inner_area == 0:
            return 0.0

        return intersection_area / inner_area
    
    @staticmethod
    def merge_overlapping_boxes(bboxes):
        """
        Takes a list of bounding boxes [ (x0,y0,x1,y1), ... ]
        and merges any that overlap, returning a new list.
        """
        merged = []
        for box in bboxes:
            has_merged = False
            for i, existing in enumerate(merged):
                if Map.is_overlapping(box, existing):
                    merged_box = (
                        min(existing[0], box[0]),
                        min(existing[1], box[1]),
                        max(existing[2], box[2]),
                        max(existing[3], box[3])
                    )
                    merged[i] = merged_box
                    has_merged = True
                    break
            if not has_merged:
                merged.append(box)
        return merged

    @staticmethod
    def consolidate_matches(matches):
        """
        Takes a list of lists of bounding boxes, e.g.:
            [ [ (x0,y0,x1,y1), ... ], [ ... ], ... ]
        and returns a list of single bounding boxes,
        where each bounding box encloses all of the boxes in that match.
        """
        consolidated = []
        for match in matches:
            xs0 = [b[0] for b in match]
            ys0 = [b[1] for b in match]
            xs1 = [b[2] for b in match]
            ys1 = [b[3] for b in match]

            consolidated.append((
                min(xs0), 
                min(ys0),
                max(xs1), 
                max(ys1)
            ))
        return consolidated

    @staticmethod
    def find_full_width_whitespace(pdf_path, page_number=0, min_height=20, include_images=True, include_drawings=True):
        """
        Identify all vertical 'full-width' whitespace bands on a single PDF page,
        each of which is at least `min_height` tall.

        Returns:
            A list of lists [x0, y0, x1, y1], each spanning the entire page width
            (from x=0 to x=page_width) and representing a whitespace region 
            with height >= min_height.

        Args:
            pdf_path (str) : Path to the PDF file.
            page_number (int): 0-based index of the page to analyze.
            min_height (float): Minimum height (in PDF points) to qualify 
                                as an empty 'band' of whitespace.
            include_images (bool): Whether to consider images as occupied space.
            include_drawings (bool): Whether to consider drawings as occupied space.
        """
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        page_rect = page.rect  # full page rectangle
        page_width = page_rect.width
        page_height = page_rect.height

        # -------------------------------------------------------------------
        # 1) Collect bounding boxes from text, images, and drawn objects
        # -------------------------------------------------------------------
        occupied_intervals = []

        # (a) Text blocks
        text_blocks = page.get_text("blocks")
        # Each block is of the form: (x0, y0, x1, y1, text, block_type, ...)
        # We only need the coordinates:
        for blk in text_blocks:
            x0, y0, x1, y1 = blk[0], blk[1], blk[2], blk[3]
            # If the block has some visible height
            if y1 > y0:
                # Ensure all y values are within page boundaries
                y0 = max(0, min(y0, page_height))
                y1 = max(0, min(y1, page_height))
                occupied_intervals.append((y0, y1))

        # (b) Images (if enabled)
        if include_images:
            for img_info in page.get_images(full=True):
                if "bbox" in img_info:
                    bbox = img_info["bbox"]
                    # Ensure all y values are within page boundaries
                    y0 = max(0, min(bbox.y0, page_height))
                    y1 = max(0, min(bbox.y1, page_height))
                    if y1 > y0:
                        occupied_intervals.append((y0, y1))
                else:
                    # Skip images with no bounding box information
                    pass

        # (c) Drawing objects (lines, shapes, etc.) - if enabled
        if include_drawings:
            for drawing_cmd in page.get_drawings():
                r = drawing_cmd["rect"]  # bounding box as fitz.Rect
                # Ensure all y values are within page boundaries
                y0 = max(0, min(r.y0, page_height))
                y1 = max(0, min(r.y1, page_height))
                # only add if it has nonzero height
                if y1 > y0:
                    occupied_intervals.append((y0, y1))

        # Close the source doc if we're done
        doc.close()

        # -------------------------------------------------------------------
        # 2) Merge the vertical intervals to get total occupied coverage
        # -------------------------------------------------------------------
        if not occupied_intervals:
            # If there's no content, the entire page is whitespace
            return [[0, 0, page_width, page_height]]

        # Sort intervals by their starting y
        occupied_intervals.sort(key=lambda iv: iv[0])
        merged = []
        current_start, current_end = occupied_intervals[0]

        for i in range(1, len(occupied_intervals)):
            iv_start, iv_end = occupied_intervals[i]
            if iv_start <= current_end:
                # Overlaps or touches - merge them
                current_end = max(current_end, iv_end)
            else:
                # No overlap, push the previous interval, start a new one
                merged.append((current_start, current_end))
                current_start, current_end = iv_start, iv_end

        # Add the last interval
        merged.append((current_start, current_end))

        # -------------------------------------------------------------------
        # 3) Compute the complement: the "empty" intervals in [0, page_height]
        # -------------------------------------------------------------------
        whitespace_coords = []

        # Gap before the first occupied interval
        if merged[0][0] > 0:
            gap_start = 0
            gap_end = merged[0][0]
            gap_height = gap_end - gap_start
            if gap_height >= min_height:
                whitespace_coords.append([0, gap_start, page_width, gap_end])

        # Gaps between merged intervals
        for i in range(len(merged) - 1):
            current_end_y = merged[i][1]
            next_start_y = merged[i+1][0]
            gap_height = next_start_y - current_end_y
            if gap_height >= min_height:
                whitespace_coords.append([0, current_end_y, page_width, next_start_y])

        # Gap after the last interval
        if merged[-1][1] < page_height:
            gap_start = merged[-1][1]
            gap_end = page_height
            gap_height = gap_end - gap_start
            if gap_height >= min_height:
                whitespace_coords.append([0, gap_start, page_width, gap_end])

        # Debug: check for any negative values
        for i, coords in enumerate(whitespace_coords):
            for j, value in enumerate(coords):
                if value < 0:
                    print(f"Warning: Negative value {value} at whitespace_coords[{i}][{j}]")
                    # Force to be non-negative
                    whitespace_coords[i][j] = 0

        return whitespace_coords

    @staticmethod
    def create_inverse_recurring_blocks(page_height, page_width, page_breaks, recurring_blocks):
        """
        Creates inverse recurring blocks (areas outside of headers and footers)
        
        Args:
            page_height (float): Total height of the combined PDF pages
            page_width (float): Width of the PDF page
            page_breaks (list): Array of y-coordinates where pages break
            recurring_blocks (list): Array of recurring blocks [page_idx, y1, x2, y2]
            
        Returns:
            list: Array of inverse blocks [page_idx, y1, x2, y2]
        """
        # Sort recurring blocks by y1 coordinate
        recurring_blocks.sort(key=lambda block: block[1])
        
        # Initialize inverse blocks array
        inverse_blocks = []
        
        # For each page break, find the blocks that belong to that page
        for i in range(len(page_breaks)):
            page_start = page_breaks[i]
            page_end = page_breaks[i + 1] if i < len(page_breaks) - 1 else page_height
            
            # Find all recurring blocks on this page
            blocks_on_page = [block for block in recurring_blocks 
                            if block[1] >= page_start and block[2] <= page_end]
            
            # If no blocks on this page, the entire page is an inverse block
            if not blocks_on_page:
                inverse_blocks.append([0, page_start, page_width, page_end])
                continue
            
            # Sort blocks on this page by y1 coordinate
            blocks_on_page.sort(key=lambda block: block[1])
            
            # Add inverse block from page start to first recurring block
            if blocks_on_page[0][1] > page_start:
                inverse_blocks.append([0, page_start, page_width, blocks_on_page[0][1]])
            
            # Add inverse blocks between recurring blocks
            for j in range(len(blocks_on_page) - 1):
                current_block = blocks_on_page[j]
                next_block = blocks_on_page[j + 1]
                
                if current_block[3] < next_block[1]:
                    inverse_blocks.append([0, current_block[3], page_width, next_block[1]])
            
            # Add inverse block from last recurring block to page end
            last_block = blocks_on_page[-1]
            if last_block[3] < page_end:
                inverse_blocks.append([0, last_block[3], page_width, page_end])
        
        return inverse_blocks

class Stitch:

    @staticmethod
    def clip_and_place_pdf_regions(
        input_pdf_path,
        regions,
        source_page_number=0,
        layout='vertical',
        page_width=None,
        page_height=None,
        margin=20,
        gap=10,
        center_horizontally=True
    ):
        """
        Clips specified regions from a PDF page and places them in a new PDF according to the specified layout.
        
        Args:
            input_pdf_path (str): Path to the source PDF
            regions (list): List of region dictionaries with format:
                        [{'rect': [x0, y0, x1, y1], 'type': 'optional_label'}, ...]
            output_pdf_path (str, optional): Path where the output PDF will be saved. If None, doesn't save.
            source_page_number (int): Page number in the source PDF to extract from (default: 0)
            layout (str): 'vertical' (stack regions vertically) or 'custom' (use provided page dimensions)
            page_width (int): Width of output page (required for 'custom' layout, auto-calculated for 'vertical')
            page_height (int): Height of output page (required for 'custom' layout, auto-calculated for 'vertical')
            margin (int): Margin around the content (default: 20 points)
            gap (int): Gap between regions when stacking vertically (default: 10 points)
            center_horizontally (bool): Whether to center regions horizontally (default: True)
        
        Returns:
            tuple: (placement_metadata, document_object) - Metadata about placements and the created document
        """
        if not regions:
            print("No regions provided.")
            return None, None
        
        # Convert regions to fitz.Rect objects
        region_rects = []
        for r in regions:
            if isinstance(r, dict) and 'rect' in r:
                rect_data = r['rect']
                if isinstance(rect_data, (list, tuple)):
                    region_rects.append(fitz.Rect(*rect_data))
                elif isinstance(rect_data, fitz.Rect):
                    region_rects.append(rect_data)
            elif isinstance(r, (list, tuple)) and len(r) == 4:
                region_rects.append(fitz.Rect(*r))
            elif isinstance(r, fitz.Rect):
                region_rects.append(r)
            else:
                raise ValueError(f"Unsupported region format: {r}")
        
        # Create new PDF document
        new_doc = fitz.open()
        
        # Calculate page dimensions based on layout
        if layout == 'vertical':
            max_width = max(r.width for r in region_rects)
            total_height = sum(r.height for r in region_rects)
            if len(region_rects) > 1:
                total_height += gap * (len(region_rects) - 1)
            
            # Add margins to page dimensions, not to content placement
            page_width = max_width + (2 * margin if margin > 0 else 0)
            page_height = total_height + (2 * margin if margin > 0 else 0)
        elif layout == 'custom':
            if not page_width or not page_height:
                raise ValueError("Custom layout requires specifying page_width and page_height")
        else:
            raise ValueError("Unsupported layout type. Use 'vertical' or 'custom'")
        
        # Create the output page
        new_page = new_doc.new_page(width=page_width, height=page_height)
        
        # Initial y position includes margin
        y_cursor = margin
        
        placement_metadata = {
            "page_width": page_width,
            "page_height": page_height,
            "placed_items": []
        }
        
        # Process each region
        for i, rect in enumerate(region_rects):
            region = regions[i] if i < len(regions) and isinstance(regions[i], dict) else {"type": f"region_{i}"}
            
            # Create temporary copy of input PDF
            tmp_pdf_path = tempfile.mktemp(suffix=".pdf")
            shutil.copy(input_pdf_path, tmp_pdf_path)
            tmp_doc = fitz.open(tmp_pdf_path)
            tmp_page = tmp_doc.load_page(source_page_number)
            page_rect = tmp_page.rect
            
            # Redact everything outside the target region
            x0, y0, x1, y1 = rect
            
            # Top area
            if y0 > page_rect.y0:
                tmp_page.add_redact_annot(fitz.Rect(page_rect.x0, page_rect.y0, page_rect.x1, y0), fill=(1, 1, 1))
            # Bottom area
            if y1 < page_rect.y1:
                tmp_page.add_redact_annot(fitz.Rect(page_rect.x0, y1, page_rect.x1, page_rect.y1), fill=(1, 1, 1))
            # Left area
            if x0 > page_rect.x0:
                tmp_page.add_redact_annot(fitz.Rect(page_rect.x0, y0, x0, y1), fill=(1, 1, 1))
            # Right area
            if x1 < page_rect.x1:
                tmp_page.add_redact_annot(fitz.Rect(x1, y0, page_rect.x1, y1), fill=(1, 1, 1))
            
            # Apply redactions
            tmp_page.apply_redactions(images=True, graphics=True, text=True)
            
            # Determine placement in the output
            block_width, block_height = rect.width, rect.height
            
            # Calculate x position (centered or not)
            if center_horizontally:
                dest_x0 = (page_width - block_width) / 2
            else:
                dest_x0 = margin
                
            dest_y0 = y_cursor
            
            # Create a precise destination rectangle with exact dimensions
            dest_rect = fitz.Rect(dest_x0, dest_y0, dest_x0 + block_width, dest_y0 + block_height)
            
            # Add to output page with exact clipping
            new_page.show_pdf_page(
                dest_rect,
                tmp_doc,
                source_page_number,
                clip=rect
            )
            
            # Store metadata about this placement
            placement_metadata["placed_items"].append({
                "type": region.get("type", f"region_{i}"),
                "original_rect": [x0, y0, x1, y1],
                "placed_rect": [dest_x0, dest_y0, dest_x0 + block_width, dest_y0 + block_height],
                "source_page": source_page_number
            })
            
            # Update cursor for next placement if using vertical layout
            if layout == 'vertical':
                y_cursor += block_height + gap
                
            # Clean up temporary file
            tmp_doc.close()
            os.remove(tmp_pdf_path)
        
        return placement_metadata, new_doc
    

class Bounds:
    @staticmethod
    def apply_margins_to_bounds(bounds_list, x_margin=5, y_margin=5):
        """
        Apply x and y margins to a list of bounding box coordinates.
        
        Args:
            bounds_list (list): List of [x0, y0, x1, y1] coordinates
            x_margin (float): Margin to apply horizontally (inward from each side)
            y_margin (float): Margin to apply vertically (inward from top and bottom)
            
        Returns:
            list: New list of coordinates with margins applied
        """
        
        x0, y0, x1, y1 = bounds_list
        
        # Apply margins (shrink the box from all sides)
        new_x0 = x0 + x_margin
        new_y0 = y0 + y_margin
        new_x1 = x1 - x_margin
        new_y1 = y1 - y_margin
        
        # Make sure we don't create invalid boxes
        if new_x1 > new_x0 and new_y1 > new_y0:
            return [new_x0, new_y0, new_x1, new_y1]
        else:
            return bounds_list

    @staticmethod
    def get_unoccupied_regions(page_width, page_height, table_bounds):
        """
        Return a list of rectangles [x0, y0, x1, y1] representing all unoccupied
        space on a page after subtracting the specified table_bounds (tables).

        Assumes a coordinate system where (0,0) is at the lower-left,
        and (page_width, page_height) is at the upper-right.
        If your PDF uses a different origin (e.g., top-left), you may need
        to flip y-coordinates or reorder them.

        Args:
            page_width  (float): width of the PDF page
            page_height (float): height of the PDF page
            table_bounds (list): list of bounding boxes [x0,y0,x1,y1] for 'occupied' areas
        Returns:
            free_rects (list): list of [x0,y0,x1,y1] covering all unoccupied regions
        """

        # ----------------------------------------------------------------------
        # 1) Normalize each bounding box so that x0 < x1 and y0 < y1.
        #    If y-coordinates are "top-down," you might invert them: y0' = page_height - y0
        # ----------------------------------------------------------------------
        normalized_tables = []
        for box in table_bounds:
            x0, y0, x1, y1 = box

            # Ensure x0 < x1
            if x1 < x0:
                x0, x1 = x1, x0

            # Ensure y0 < y1
            if y1 < y0:
                y0, y1 = y1, y0

            # Clip to page just in case
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(page_width, x1)
            y1 = min(page_height, y1)

            # If there's still a valid area, keep it
            if x1 > x0 and y1 > y0:
                normalized_tables.append([x0, y0, x1, y1])

        # ----------------------------------------------------------------------
        # 2) Start free space as the entire page: [0, 0, page_width, page_height]
        # ----------------------------------------------------------------------
        free_rects = [[0.0, 0.0, page_width, page_height]]

        # ----------------------------------------------------------------------
        # 3) Subtraction function to remove one 'occupied' box from free_rects
        # ----------------------------------------------------------------------
        def subtract_box_from_free(free_list, occ_box):
            """Subtract 'occ_box' from each rect in free_list. Return updated list."""
            new_free_list = []
            ox0, oy0, ox1, oy1 = occ_box

            for fr in free_list:
                fx0, fy0, fx1, fy1 = fr

                # Check if there's overlap
                # Overlap in x-dimension if [fx0, fx1] intersects [ox0, ox1]
                overlap_x0 = max(fx0, ox0)
                overlap_x1 = min(fx1, ox1)
                # Overlap in y-dimension if [fy0, fy1] intersects [oy0, oy1]
                overlap_y0 = max(fy0, oy0)
                overlap_y1 = min(fy1, oy1)

                no_overlap = (overlap_x1 <= overlap_x0) or (overlap_y1 <= overlap_y0)
                if no_overlap:
                    # No intersection => keep the entire free rect
                    new_free_list.append(fr)
                else:
                    # There's some overlap => carve out the overlap from 'fr'

                    # 1) Region *above* the overlap
                    if overlap_y1 < fy1:
                        new_free_list.append([fx0, overlap_y1, fx1, fy1])
                    # 2) Region *below* the overlap
                    if overlap_y0 > fy0:
                        new_free_list.append([fx0, fy0, fx1, overlap_y0])
                    # 3) Region *left* of the overlap
                    if overlap_x0 > fx0:
                        new_free_list.append([fx0, overlap_y0, overlap_x0, overlap_y1])
                    # 4) Region *right* of the overlap
                    if overlap_x1 < fx1:
                        new_free_list.append([overlap_x1, overlap_y0, fx1, overlap_y1])

            return new_free_list

        # ----------------------------------------------------------------------
        # 4) Subtract each table bounding box from 'free_rects'
        # ----------------------------------------------------------------------
        for occ_box in normalized_tables:
            free_rects = subtract_box_from_free(free_rects, occ_box)

            # Debug: Uncomment if you want to see step-by-step
            # print("After subtracting", occ_box, "free_rects =")
            # for r in free_rects:
            #    print("   ", r)

        # ----------------------------------------------------------------------
        # 5) Sort or filter out any degenerate boxes (w=0 or h=0)
        # ----------------------------------------------------------------------
        final_free_rects = []
        for r in free_rects:
            x0, y0, x1, y1 = r
            if x1 > x0 and y1 > y0:
                final_free_rects.append(r)

        # Optional: sort by y, then x
        final_free_rects.sort(key=lambda rect: (rect[1], rect[0]))

        return final_free_rects
    
    @staticmethod
    def find_valid_x2_coordinates(check_table, occupied_bounds, x_max, min_width=1, expansion_max=20):
        """
        Find the optimal x2 coordinate for check_table such that it doesn't overlap
        any tables in occupied_bounds.
        
        Args:
            check_table: [x0, y0, x1, y1] - The table we're checking
            occupied_bounds: List of [x0, y0, x1, y1] - Existing tables
            x_max: Maximum allowed x-coordinate (page width)
            min_width: Minimum required width for the table
            expansion_max: Maximum allowed expansion from current x1 value
            
        Returns:
            The optimal x2 coordinate, or None if no valid expansion is possible
        """
        x0, y0, x1, y1 = check_table
        
        # Calculate bounds for expansion
        min_x2 = x0 + min_width  # Minimum width required
        max_possible_x2 = min(x_max, x1 + expansion_max)  # Maximum allowed expansion
        
        # If current width is already at or beyond the maximum allowed, return None
        if x1 >= max_possible_x2:
            return None
        
        # Find potential overlaps (tables with vertical overlap)
        potentially_overlapping = []
        for bounds in occupied_bounds:
            o_x0, o_y0, o_x1, o_y1 = bounds
            
            # Skip if this is the same table (coordinates match)
            if x0 == o_x0 and y0 == o_y0 and x1 == o_x1 and y1 == o_y1:
                continue
                
            # Check if there's a vertical overlap
            has_vertical_overlap = (
                (o_y0 <= y0 and o_y1 >= y0) or  # Other table overlaps top of check_table
                (o_y0 <= y1 and o_y1 >= y1) or  # Other table overlaps bottom of check_table
                (o_y0 >= y0 and o_y1 <= y1) or  # Other table is within check_table vertically
                (o_y0 <= y0 and o_y1 >= y1)     # check_table is within other table vertically
            )
            
            if has_vertical_overlap:
                potentially_overlapping.append(bounds)
        
        # If no potential overlaps, we can expand to the maximum allowed
        if not potentially_overlapping:
            return max_possible_x2
        
        # Find the leftmost table to the right of our current table
        min_right_edge = max_possible_x2
        
        for bounds in potentially_overlapping:
            o_x0, o_y0, o_x1, o_y1 = bounds
            
            # If this table is to the right of our current table
            if o_x0 > x1:
                min_right_edge = min(min_right_edge, o_x0)
        
        # If we found a constraining table to the right, return the position right before it
        if min_right_edge < max_possible_x2:
            return min_right_edge
        
        # Otherwise return the maximum allowed expansion
        return max_possible_x2