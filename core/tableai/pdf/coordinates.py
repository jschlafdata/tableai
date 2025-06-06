from __future__ import annotations

__all__ = ["Map"]

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
