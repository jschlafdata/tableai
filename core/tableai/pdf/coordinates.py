from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any, Union, Protocol
from dataclasses import dataclass
import fitz
from tableai.pdf.generic_types import BBox, BBoxList, Point
from tableai.pdf.pdf_page import VirtualPageBounds, RegionsByPage
import fitz  # PyMuPDF
from pathlib import Path

__all__ = ["Geometry", "CoordinateMapping", "ManipulateDoc"]


class ManipulateDoc:
    @staticmethod
    def clip_and_place_pdf_regions(
        source_pdf: Union[str, Path, bytes, fitz.Document],
        regions: List[Union[Dict[str, List[float]], List[float], Tuple[float, ...]]],
        source_page_number: int = 0,
        layout: str = 'vertical',
        margin: int = 20,
        gap: int = 0,
        center_horizontally: bool = True
    ) -> Tuple[Dict[str, Any], fitz.Document]:
        """
        Clips specified regions from a PDF page and places them onto a single new page
        using a robust redaction method to ensure no extraneous data is carried over.

        Args:
            source_pdf: The source PDF, can be a file path, bytes, or fitz.Document.
            regions: A list of regions to clip, e.g., [{'bbox': [x0, y0, x1, y1]}].
            source_page_number: The 0-indexed page in the source PDF to clip from.
            layout: 'vertical' or 'preserve_positions'.
            margin: Space around the content on the new page.
            gap: Space between regions in 'vertical' layout.
            center_horizontally: If True, centers content in 'vertical' layout.

        Returns:
            A tuple containing:
            - placement_metadata (dict): Details about the new page and placed items.
            - new_doc (fitz.Document): The new PDF document object.
        """
        if not regions:
            raise ValueError("No regions provided to clip.")

        # --- 1. Normalize region inputs ---
        region_rects = []
        for r in regions:
            bbox = r.get('bbox') if isinstance(r, dict) else r
            if bbox and isinstance(bbox, (list, tuple)):
                region_rects.append(fitz.Rect(bbox))
        if not region_rects:
            raise ValueError("Regions list is empty or in an unrecognized format.")

        print(region_rects)

        # --- 2. Load source document and page efficiently ---
        source_doc_was_opened = False
        if isinstance(source_pdf, (str, Path, bytes)):
            source_doc = fitz.open(stream=source_pdf if isinstance(source_pdf, bytes) else None, filename=source_pdf if not isinstance(source_pdf, bytes) else None)
            source_doc_was_opened = True
        elif isinstance(source_pdf, fitz.Document):
            source_doc = source_pdf
        else:
            raise TypeError("source_pdf must be a path, bytes, or fitz.Document object.")
        
        source_page = source_doc.load_page(source_page_number)
        page_boundary = source_page.rect

        # --- 3. Calculate new page dimensions ---
        if layout == 'vertical':
            max_width = max(r.width for r in region_rects)
            total_height = sum(r.height for r in region_rects) + gap * (len(region_rects) - 1)
            page_width = max_width + 2 * margin
            page_height = total_height + 2 * margin
        elif layout == 'preserve_positions':
            overall_bbox = fitz.Rect()
            for rect in region_rects:
                overall_bbox.include_rect(rect)
            page_width = overall_bbox.width + 2 * margin
            page_height = overall_bbox.height + 2 * margin
            origin_x, origin_y = overall_bbox.x0, overall_bbox.y0
        else:
            raise ValueError(f"Unsupported layout type '{layout}'. Use 'vertical' or 'preserve_positions'.")
        
        # --- 4. Create new document and page ---
        new_doc = fitz.open()
        new_page = new_doc.new_page(width=page_width, height=page_height)
        print(page_width)
        print(f"page_height: {page_height}")
        placement_metadata = {"page_width": page_width, "page_height": page_height, "layout_used": layout, "placed_items": []}
        y_cursor = margin

        # --- 5. Process each region using IN-MEMORY redaction ---
        for i, clip_rect in enumerate(region_rects):
            # Create an in-memory, temporary document containing just the source page
            temp_doc = fitz.open()
            temp_doc.insert_pdf(source_doc, from_page=source_page_number, to_page=source_page_number)
            temp_page = temp_doc[0]

            # Redact everything *outside* the desired clip_rect
            # Top rectangle
            temp_page.add_redact_annot(fitz.Rect(page_boundary.x0, page_boundary.y0, page_boundary.x1, clip_rect.y0), fill=(1,1,1))
            # Bottom rectangle
            temp_page.add_redact_annot(fitz.Rect(page_boundary.x0, clip_rect.y1, page_boundary.x1, page_boundary.y1), fill=(1,1,1))
            # Left rectangle
            temp_page.add_redact_annot(fitz.Rect(page_boundary.x0, clip_rect.y0, clip_rect.x0, clip_rect.y1), fill=(1,1,1))
            # Right rectangle
            temp_page.add_redact_annot(fitz.Rect(clip_rect.x1, clip_rect.y0, page_boundary.x1, clip_rect.y1), fill=(1,1,1))

            # Apply the redactions to truly remove the content
            temp_page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE) # Keeps original quality

            # Calculate the destination rectangle on the new page
            if layout == 'vertical':
                dest_x0 = (page_width - clip_rect.width) / 2 if center_horizontally else margin
                dest_y0 = y_cursor
                y_cursor += clip_rect.height + gap
            elif layout == 'preserve_positions':
                dest_x0 = (clip_rect.x0 - origin_x) + margin
                dest_y0 = (clip_rect.y0 - origin_y) + margin
            
            dest_rect = fitz.Rect(dest_x0, dest_y0, dest_x0 + clip_rect.width, dest_y0 + clip_rect.height)
            
            # Place the content of the redacted temporary page onto the new page
            new_page.show_pdf_page(dest_rect, temp_doc, 0, clip=clip_rect)
            
            temp_doc.close() # Close the in-memory temp doc

            placement_metadata["placed_items"].append({
                "original_bbox": list(clip_rect), "placed_bbox": list(dest_rect), "source_page": source_page_number
            })
        
        return placement_metadata, new_doc

@dataclass
class CoordinateMapping:
    """Handles coordinate transformations between different coordinate systems."""
    
    @staticmethod
    def absolute_to_relative(bbox: BBox, page_bounds: VirtualPageBounds) -> BBox:
        """Convert absolute coordinates to page-relative coordinates."""
        x0, y0, x1, y1 = bbox
        return (
            x0 - page_bounds.x0,
            y0 - page_bounds.y0,
            x1 - page_bounds.x0,
            y1 - page_bounds.y0
        )
    
    @staticmethod
    def relative_to_absolute(bbox: BBox, page_bounds: VirtualPageBounds) -> BBox:
        """Convert page-relative coordinates to absolute coordinates."""
        x0, y0, x1, y1 = bbox
        return (
            x0 + page_bounds.x0,
            y0 + page_bounds.y0,
            x1 + page_bounds.x0,
            y1 + page_bounds.y0
        )
    
    @staticmethod
    def scale_for_display(
        bbox: BBox, 
        scale_x: float, 
        scale_y: float, 
        offset_x: float = 0, 
        offset_y: float = 0
    ) -> BBox:
        """Scale coordinates for display rendering."""
        x0, y0, x1, y1 = bbox
        return (
            (x0 - offset_x) * scale_x,
            (y0 - offset_y) * scale_y,
            (x1 - offset_x) * scale_x,
            (y1 - offset_y) * scale_y
        )

class Geometry:
    @staticmethod
    def is_overlapping(boxA: BBox, boxB: BBox) -> bool:
        """Return True if boxA and boxB overlap at all."""
        x0A, y0A, x1A, y1A = boxA
        x0B, y0B, x1B, y1B = boxB
        
        if x1A < x0B or x1B < x0A:  # One rectangle is left of the other
            return False
        if y1A < y0B or y1B < y0A:  # One rectangle is above the other
            return False
        return True

    @staticmethod
    def bbox_overlaps(bbox1: Optional[BBox], bbox2: Optional[BBox]) -> bool:
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
    def is_x_overlapping(boxA: BBox, boxB: BBox) -> bool:
        """
        Return True if the horizontal (x-axis) bounds of boxA and boxB overlap.
        Ignores vertical positioning.
        
        Parameters:
            boxA, boxB: Tuples in the form (x0, y0, x1, y1)
        
        Returns:
            bool: True if x-ranges intersect, False otherwise.
        """
        x0A, _, x1A, _ = boxA
        x0B, _, x1B, _ = boxB

        return not (x1A < x0B or x1B < x0A)

    @staticmethod
    def is_in_spanning_text(bbox: BBox, spanning_text_list: List[Any]) -> bool:
        """Check if a bounding box overlaps with any spanning text."""
        if not spanning_text_list:
            return False
        
        for spanning_item in spanning_text_list:
            span_bbox = spanning_item.get('bbox', None)
            if span_bbox and Geometry.is_overlapping(bbox, span_bbox):
                return True
        return False

    @staticmethod
    def merge_all_boxes(bboxes: BBoxList) -> Optional[BBox]:
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
        merged_box: BBox = (
            min(xs0),
            min(ys0),
            max(xs1),
            max(ys1)
        )
        
        return merged_box
    
    @staticmethod
    def is_fully_contained(inner: BBox, outer: BBox) -> bool:
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
    def percent_contained(inner: BBox, outer: BBox) -> float:
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
    def merge_overlapping_boxes(bboxes: BBoxList) -> BBoxList:
        """
        Takes a list of bounding boxes [ (x0,y0,x1,y1), ... ]
        and merges any that overlap, returning a new list.
        """
        merged: BBoxList = []
        for box in bboxes:
            has_merged = False
            for i, existing in enumerate(merged):
                if Geometry.is_overlapping(box, existing):
                    merged_box: BBox = (
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
    def scale_y(bbox: Optional[BBox] = None, y_offset: float = 0, *args: float) -> BBox:
        """
        Scale a bounding box by adding y_offset to y coordinates.
        
        Args:
            bbox: Optional bbox tuple (x0, y0, x1, y1)
            y_offset: Offset to add to y coordinates
            *args: Alternative way to pass x0, y0, x1, y1 as separate arguments
        
        Returns:
            Scaled bounding box
        """
        if bbox:
            return (bbox[0], bbox[1] + y_offset, bbox[2], bbox[3] + y_offset)
        else:
            if len(args) != 4:
                raise ValueError("Must provide either bbox tuple or 4 coordinate arguments")
            return (args[0], args[1] + y_offset, args[2], args[3] + y_offset)

    @staticmethod
    def inverse_page_blocks(regions_by_page: RegionsByPage) -> BBoxList:
        """
        Returns the inverse of recurring blocks treating all pages as continuous coordinate space.
        
        Args:
            regions_by_page: Result from process_noise_regions function
                Format: {page_num: {'bboxes': [...], 'page_width': float, 'page_height': float}}
        
        Returns:
            list: Inverse blocks as [(x0, y0, x1, y1), ...] covering gaps between noise regions
        """
        # Collect all noise regions from all pages into a single list
        all_regions: BBoxList = []
        page_width: Optional[float] = None
        
        for page_num, page_data in regions_by_page.items():
            page_width = float(page_data['page_width'])  # Should be same for all pages
            page_bboxes = page_data.get('bboxes', [])
            if isinstance(page_bboxes, list):
                all_regions.extend(page_bboxes)
        
        if not all_regions or page_width is None:
            # No noise regions? Return empty list since we don't know document bounds
            return []
        
        # Sort all noise regions by y0 (top coordinate)
        sorted_regions = sorted(all_regions, key=lambda region: region[1])
        
        inverse_blocks: BBoxList = []
        
        # 1. From top of document (y=0) to first noise region
        first = sorted_regions[0]
        if first[1] > 0:
            inverse_blocks.append((0, 0, page_width, first[1]))
        
        # 2. Between noise regions (find gaps)
        for i in range(len(sorted_regions) - 1):
            curr = sorted_regions[i]
            nxt = sorted_regions[i + 1]
            if curr[3] < nxt[1]:  # Only if there is a gap between regions
                inverse_blocks.append((0, curr[3], page_width, nxt[1]))
        
        return inverse_blocks

    # Additional utility methods for comprehensive bbox operations
    
    @staticmethod
    def bbox_area(bbox: BBox) -> float:
        """Calculate the area of a bounding box."""
        x0, y0, x1, y1 = bbox
        return max(0.0, (x1 - x0) * (y1 - y0))
    
    @staticmethod
    def bbox_width(bbox: BBox) -> float:
        """Calculate the width of a bounding box."""
        x0, _, x1, _ = bbox
        return max(0.0, x1 - x0)
    
    @staticmethod
    def bbox_height(bbox: BBox) -> float:
        """Calculate the height of a bounding box."""
        _, y0, _, y1 = bbox
        return max(0.0, y1 - y0)
    
    @staticmethod
    def bbox_center(bbox: BBox) -> Point:
        """Calculate the center point of a bounding box."""
        x0, y0, x1, y1 = bbox
        return ((x0 + x1) / 2, (y0 + y1) / 2)