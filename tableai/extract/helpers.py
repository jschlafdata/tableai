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
from typing import Dict, Any, List

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
    def bbox_overlaps(bbox1, bbox2):
        """
        Returns True if bbox1 overlaps bbox2.
        bbox = (x0, y0, x1, y1)
        """
        if not bbox1 or not bbox2:
            return False
        x0a, y0a, x1a, y1a = bbox1
        x0b, y0b, x1b, y1b = bbox2
        return not (x1a <= x0b or x1b <= x0a or y1a <= y0b or y1b <= y0a)

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
    def is_fully_contained(inner, outer) -> bool:
        """
        Returns True if the `inner` box is fully contained within the `outer` box.

        Boxes are in (x0, y0, x1, y1) format.
        """
        x0_i, y0_i, x1_i, y1_i = inner
        x0_o, y0_o, x1_o, y1_o = outer

        return (
            x0_i >= x0_o and
            y0_i >= y0_o and
            x1_i <= x1_o and
            y1_i <= y1_o
        )

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
    def translate_bbox_to_virtual_page(bbox, page_num, pdf_metadata):
        """
        Translate a bbox from its original page coords to a single virtual page.

        Args:
            bbox: (x0, y0, x1, y1) for the given page
            page_num: which page this bbox is from (int)
            pdf_metadata: dict like {page_num: {'width': ..., 'height': ...}, ...}
        
        Returns:
            (x0_new, y0_new, x1_new, y1_new) on the single virtual page
        """
        # Stack pages vertically; y offset is the sum of all previous page heights
        y_offset = sum(pdf_metadata[p]['height'] for p in range(page_num))
        x0, y0, x1, y1 = bbox
        return (x0, y0 + y_offset, x1, y1 + y_offset)

    @staticmethod
    def get_virtual_page_size(pdf_metadata):
        """
        Return (width, height) of the combined virtual page.
        Assumes all pages are the same width (common in PDFs).
        """
        total_height = sum(m['height'] for m in pdf_metadata.values())
        first_page = min(pdf_metadata.keys())
        width = pdf_metadata[first_page]['width']
        return (width, total_height)

    @staticmethod
    def create_inverse_blocks(page_width, page_height, recurring_blocks):
        """
        Returns the inverse of recurring blocks for a single page.
        The result is a list of blocks covering all vertical space *not* in any recurring block.
        
        Args:
            page_width (float): Width of the page
            page_height (float): Height of the page
            recurring_blocks (list): List of blocks [x0, y0, x1, y1]
        
        Returns:
            list: Inverse blocks as [x0, y0, x1, y1]
        """
        if not recurring_blocks:
            # No recurring blocks? The whole page is inverse!
            return [[0, 0, page_width, page_height]]
        
        # Sort by y0 (top of block)
        recurring_blocks = sorted(recurring_blocks, key=lambda block: block[1])
        inverse_blocks = []
        
        # 1. From top of page to first block
        first = recurring_blocks[0]
        if first[1] > 0:
            inverse_blocks.append([0, 0, page_width, first[1]])
        
        # 2. Between blocks
        for i in range(len(recurring_blocks) - 1):
            curr = recurring_blocks[i]
            nxt = recurring_blocks[i + 1]
            if curr[3] < nxt[1]:  # Only if there is a gap
                inverse_blocks.append([0, curr[3], page_width, nxt[1]])
        
        # 3. From last block to bottom of page
        last = recurring_blocks[-1]
        if last[3] < page_height:
            inverse_blocks.append([0, last[3], page_width, page_height])
        
        return inverse_blocks


class Stitch:

    @staticmethod
    def combine_pages_into_one(src_doc, node):
        """
        Combines all pages of input_pdf_path into one tall PDF,
        then extracts recurring text blocks (keeping only top/bottom strips).
        Returns a metadata dict.
        """
        print(f"COMBINGING PAGES INTO ONE!!")
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

        
        combined_doc.save(node.stage_paths[1]["abs_path"])
        combined_doc.close()
        src_doc.close()

    @staticmethod
    def clip_and_place_pdf_regions(
        input_pdf_path,
        regions,
        source_page_number=0,
        layout='vertical',
        page_width=None,
        page_height=None,
        margin=20,
        gap=25,
        center_horizontally=True
    ):
        """
        Clips specified regions from a PDF page and places them in a new PDF according to the specified layout.
        Expects regions as a list of (x0, y0, x1, y1) tuples/lists.

        Returns:
            tuple: (placement_metadata, document_object)
        """
        if not regions:
            print("No regions provided.")
            return None, None

        # Convert regions to fitz.Rect objects
        region_rects = [fitz.Rect(*r['bbox']) for r in regions if isinstance(r, dict)]
        if not region_rects:
            region_rects = [fitz.Rect(*r) for r in regions if isinstance(r, list|tuple)]

        print(f"{region_rects}")

        # Create new PDF document
        new_doc = fitz.open()

        # Calculate page dimensions based on layout
        if layout == 'vertical':
            max_width = max(r.width for r in region_rects)
            total_height = sum(r.height for r in region_rects)
            if len(region_rects) > 1:
                total_height += gap * (len(region_rects) - 1)
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
            # Create temporary copy of input PDF
            tmp_pdf_path = tempfile.mktemp(suffix=".pdf")
            shutil.copy(input_pdf_path, tmp_pdf_path)
            tmp_doc = fitz.open(tmp_pdf_path)
            tmp_page = tmp_doc.load_page(source_page_number)
            page_rect = tmp_page.rect

            x0, y0, x1, y1 = rect

            # Redact everything outside the target region
            if y0 > page_rect.y0:
                tmp_page.add_redact_annot(fitz.Rect(page_rect.x0, page_rect.y0, page_rect.x1, y0), fill=(1, 1, 1))
            if y1 < page_rect.y1:
                tmp_page.add_redact_annot(fitz.Rect(page_rect.x0, y1, page_rect.x1, page_rect.y1), fill=(1, 1, 1))
            if x0 > page_rect.x0:
                tmp_page.add_redact_annot(fitz.Rect(page_rect.x0, y0, x0, y1), fill=(1, 1, 1))
            if x1 < page_rect.x1:
                tmp_page.add_redact_annot(fitz.Rect(x1, y0, page_rect.x1, y1), fill=(1, 1, 1))

            tmp_page.apply_redactions(images=True, graphics=True, text=True)

            block_width, block_height = rect.width, rect.height

            # Calculate x position (centered or not)
            if center_horizontally:
                dest_x0 = (page_width - block_width) / 2
            else:
                dest_x0 = margin

            dest_y0 = y_cursor
            dest_rect = fitz.Rect(dest_x0, dest_y0, dest_x0 + block_width, dest_y0 + block_height)

            # Place region onto output page
            new_page.show_pdf_page(
                dest_rect,
                tmp_doc,
                source_page_number,
                clip=rect
            )

            placement_metadata["placed_items"].append({
                "original_rect": [x0, y0, x1, y1],
                "placed_rect": [dest_x0, dest_y0, dest_x0 + block_width, dest_y0 + block_height],
                "source_page": source_page_number
            })

            if layout == 'vertical':
                y_cursor += block_height + gap

            tmp_doc.close()
            os.remove(tmp_pdf_path)

        return new_doc

    @staticmethod
    def redact_pdf_regions(
        input_doc: fitz.Document,
        redaction_dict: Dict[str, List[Dict[str, Any]]]
    ) -> fitz.Document:
        """
        Redact specified regions in a PDF.

        Args:
            input_doc (fitz.Document): Source PDF document.
            redaction_dict (dict): {
                page_index (str or int): [
                    {"bbox": [x0, y0, x1, y1], ...}, ...
                ], ...
            }

        Returns:
            fitz.Document: In-memory redacted PDF document.
        """
        # 1. Save a temp copy of input PDF to disk
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_source:
            input_doc.save(temp_source.name)
            temp_source_path = temp_source.name

        # 2. Open the temp copy
        src_doc = fitz.open(temp_source_path)

        # 3. Save another temp path for the redacted output
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_output:
            temp_output_path = temp_output.name

        # 4. Copy the source doc to output temp path (so we can edit in-place)
        shutil.copy(temp_source_path, temp_output_path)

        # 5. Open as output doc for redaction
        out_doc = fitz.open(temp_output_path)

        # 6. Iterate and redact
        for page_idx_str, regions in redaction_dict.items():
            try:
                page_idx = int(page_idx_str)
            except Exception:
                continue  # skip bad keys

            page = out_doc[page_idx]

            for region in regions:
                bbox = region.get('bbox')
                if not bbox or len(bbox) != 4:
                    continue
                x0, y0, x1, y1 = bbox
                rect = fitz.Rect(x0, y0, x1, y1)
                # Add redaction annotation
                page.add_redact_annot(rect, fill=(1, 1, 1))  # White fill (can be black if desired)

            # Apply all redactions on this page
            page.apply_redactions(images=True, graphics=True, text=True)

        # 7. Save to in-memory buffer, reload as fitz.Document, cleanup temp files
        pdf_bytes = out_doc.tobytes()
        out_doc.close()
        src_doc.close()
        try:
            os.remove(temp_source_path)
            os.remove(temp_output_path)
        except Exception:
            pass

        return fitz.open("pdf", pdf_bytes)

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