from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any, Union, Protocol
from dataclasses import dataclass
import fitz

__all__ = ["Geometry", "CoordinateMapping", "BBox"]

# Type definitions
BBox = Tuple[float, float, float, float]
BBoxList = List[BBox]
Point = Tuple[float, float]

@dataclass
class VirtualPageBounds:
    """Represents bounds for a virtual page."""
    page_number: int
    x0: float
    y0: float
    x1: float
    y1: float
    
    @property
    def width(self) -> float:
        return self.x1 - self.x0
    
    @property
    def height(self) -> float:
        return self.y1 - self.y0
    
    @property
    def rect(self) -> fitz.Rect:
        return fitz.Rect(self.x0, self.y0, self.x1, self.y1)
    
    @property
    def tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

RegionsByPage = Dict[int, Dict[str, Union[BBoxList, float]]]

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
    def is_in_spanning_text(bbox: BBox, spanning_text_list: List[SpanningTextItem]) -> bool:
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

# @dataclass
# class CoordinateMapping:
#     """Handles coordinate transformations between different coordinate systems."""
    
#     @staticmethod
#     def absolute_to_relative(bbox: Tuple[float, float, float, float], 
#                            page_bounds: 'VirtualPageBounds') -> Tuple[float, float, float, float]:
#         """Convert absolute coordinates to page-relative coordinates."""
#         x0, y0, x1, y1 = bbox
#         return (
#             x0 - page_bounds.x0,
#             y0 - page_bounds.y0,
#             x1 - page_bounds.x0,
#             y1 - page_bounds.y0
#         )
    
#     @staticmethod
#     def relative_to_absolute(bbox: Tuple[float, float, float, float], 
#                            page_bounds: 'VirtualPageBounds') -> Tuple[float, float, float, float]:
#         """Convert page-relative coordinates to absolute coordinates."""
#         x0, y0, x1, y1 = bbox
#         return (
#             x0 + page_bounds.x0,
#             y0 + page_bounds.y0,
#             x1 + page_bounds.x0,
#             y1 + page_bounds.y0
#         )
    
#     @staticmethod
#     def scale_for_display(bbox: Tuple[float, float, float, float], 
#                          scale_x: float, scale_y: float, 
#                          offset_x: float = 0, offset_y: float = 0) -> Tuple[float, float, float, float]:
#         """Scale coordinates for display rendering."""
#         x0, y0, x1, y1 = bbox
#         return (
#             (x0 - offset_x) * scale_x,
#             (y0 - offset_y) * scale_y,
#             (x1 - offset_x) * scale_x,
#             (y1 - offset_y) * scale_y
#         )

# class Geometry:
#     @staticmethod
#     def is_overlapping(boxA, boxB):
#         """Return True if boxA and boxB overlap at all."""
#         x0A, y0A, x1A, y1A = boxA
#         x0B, y0B, x1B, y1B = boxB
        
#         if x1A < x0B or x1B < x0A:  # One rectangle is left of the other
#             return False
#         if y1A < y0B or y1B < y0A:  # One rectangle is above the other
#             return False
#         return True

#     @staticmethod
#     def bbox_overlaps(bbox1, bbox2):
#         """
#         Returns True if bbox1 overlaps bbox2.
#         bbox = (x0, y0, x1, y1)
#         """
#         if not bbox1 or not bbox2:
#             return False
#         x0a, y0a, x1a, y1a = bbox1
#         x0b, y0b, x1b, y1b = bbox2
#         return not (x1a <= x0b or x1b <= x0a or y1a <= y0b or y1b <= y0a)

#     @staticmethod
#     def is_x_overlapping(boxA, boxB):
#         """
#         Return True if the horizontal (x-axis) bounds of boxA and boxB overlap.
#         Ignores vertical positioning.
        
#         Parameters:
#             boxA, boxB: Lists or tuples in the form [x0, y0, x1, y1]
        
#         Returns:
#             bool: True if x-ranges intersect, False otherwise.
#         """
#         x0A, _, x1A, _ = boxA
#         x0B, _, x1B, _ = boxB

#         return not (x1A < x0B or x1B < x0A)

#     @staticmethod
#     def is_in_spanning_text(bbox, spanning_text_list):
#         """Check if a bounding box overlaps with any spanning text."""
#         if not spanning_text_list:
#             return False
        
#         for spanning_item in spanning_text_list:
#             span_bbox = spanning_item.get('bbox', None)
#             if span_bbox and Geometry.is_overlapping(bbox, span_bbox):
#                 return True
                
#         return False

#     @staticmethod
#     def merge_all_boxes(bboxes):
#         """
#         Takes a list of bounding boxes [ (x0,y0,x1,y1), ... ]
#         and merges all of them into a single bounding box that encompasses all boxes.
#         Returns a single bounding box (x0,y0,x1,y1).
        
#         If the input list is empty, returns None.
#         """
#         if not bboxes:
#             return None
        
#         # Extract all coordinates
#         xs0 = [box[0] for box in bboxes]
#         ys0 = [box[1] for box in bboxes]
#         xs1 = [box[2] for box in bboxes]
#         ys1 = [box[3] for box in bboxes]
        
#         # Create a single bounding box that encompasses all
#         merged_box = (
#             min(xs0),
#             min(ys0),
#             max(xs1),
#             max(ys1)
#         )
        
#         return merged_box
    
#     @staticmethod
#     def is_fully_contained(inner, outer) -> bool:
#         """
#         Returns True if the `inner` box is fully contained within the `outer` box.

#         Boxes are in (x0, y0, x1, y1) format.
#         """
#         x0_i, y0_i, x1_i, y1_i = inner
#         x0_o, y0_o, x1_o, y1_o = outer

#         return (
#             x0_i >= x0_o and
#             y0_i >= y0_o and
#             x1_i <= x1_o and
#             y1_i <= y1_o
#         )

#     @staticmethod
#     def percent_contained(inner, outer) -> float:
#         """
#         Returns the percentage (0.0 to 1.0) of `inner`'s area that is contained within `outer`.

#         If there's no intersection, returns 0.0.
#         If `inner` is fully contained in `outer`, returns 1.0.
#         """
#         x0_i, y0_i, x1_i, y1_i = inner
#         x0_o, y0_o, x1_o, y1_o = outer

#         # Calculate intersection box
#         x0 = max(x0_i, x0_o)
#         y0 = max(y0_i, y0_o)
#         x1 = min(x1_i, x1_o)
#         y1 = min(y1_i, y1_o)

#         inter_width = max(0.0, x1 - x0)
#         inter_height = max(0.0, y1 - y0)
#         intersection_area = inter_width * inter_height

#         inner_area = max(0.0, (x1_i - x0_i) * (y1_i - y0_i))
#         if inner_area == 0:
#             return 0.0

#         return intersection_area / inner_area
    
#     @staticmethod
#     def merge_overlapping_boxes(bboxes):
#         """
#         Takes a list of bounding boxes [ (x0,y0,x1,y1), ... ]
#         and merges any that overlap, returning a new list.
#         """
#         merged = []
#         for box in bboxes:
#             has_merged = False
#             for i, existing in enumerate(merged):
#                 if Geometry.is_overlapping(box, existing):
#                     merged_box = (
#                         min(existing[0], box[0]),
#                         min(existing[1], box[1]),
#                         max(existing[2], box[2]),
#                         max(existing[3], box[3])
#                     )
#                     merged[i] = merged_box
#                     has_merged = True
#                     break
#             if not has_merged:
#                 merged.append(box)
#         return merged

#     @staticmethod
#     def scale_y(bbox: Optional[list] = None, y_offset: Optional[int]=0, *args):
#         if bbox:
#             return (bbox[0], bbox[1] + y_offset, bbox[2], bbox[3] + y_offset)
#         else:
#             return (args[0], args[1] + y_offset, args[2], args[3] + y_offset)

#     @staticmethod
#     def inverse_page_blocks(regions_by_page):
#         """
#         Returns the inverse of recurring blocks treating all pages as continuous coordinate space.
        
#         Args:
#             noise_regions_by_page (dict): Result from process_noise_regions function
#                 Format: {page_num: {'bboxes': [...], 'page_width': float, 'page_height': float}}
        
#         Returns:
#             list: Inverse blocks as [x0, y0, x1, y1] covering gaps between noise regions
#         """
#         # Collect all noise regions from all pages into a single list
#         all_regions = []
#         page_width = None
        
#         for page_num, page_data in regions_by_page.items():
#             page_width = page_data['page_width']  # Should be same for all pages
#             all_regions.extend(page_data['bboxes'])
        
#         if not all_regions:
#             # No noise regions? Return empty list since we don't know document bounds
#             return []
        
#         # Sort all noise regions by y0 (top coordinate)
#         sorted_regions = sorted(all_regions, key=lambda region: region[1])
        
#         inverse_blocks = []
        
#         # 1. From top of document (y=0) to first noise region
#         first = sorted_regions[0]
#         if first[1] > 0:
#             inverse_blocks.append([0, 0, page_width, first[1]])
        
#         # 2. Between noise regions (find gaps)
#         for i in range(len(sorted_regions) - 1):
#             curr = sorted_regions[i]
#             nxt = sorted_regions[i + 1]
#             if curr[3] < nxt[1]:  # Only if there is a gap between regions
#                 inverse_blocks.append([0, curr[3], page_width, nxt[1]])
        
#         return inverse_blocks