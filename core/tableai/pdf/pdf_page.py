from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any, Union, Protocol
from tableai.pdf.generic_types import BBox, BBoxList, Point
from dataclasses import dataclass
import fitz

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


class VirtualPageManager:
    """Centralized manager for virtual page coordinate mapping and metadata."""
    
    def __init__(self, virtual_page_metadata: Optional[Dict] = None):
        """Initialize with virtual page metadata."""
        self.metadata = virtual_page_metadata or {}
        self._page_bounds_cache: Dict[int, VirtualPageBounds] = {}
        self._virtual_breaks: List[Tuple[float, int]] = []
        self._setup_lookup_structures()
    
    def _setup_lookup_structures(self):
        """Setup efficient lookup structures for virtual page mapping."""
        if "page_breaks" in self.metadata:
            self._virtual_breaks = sorted(self.metadata["page_breaks"])
        
        # Cache VirtualPageBounds objects
        if "page_bounds" in self.metadata:
            for page_num_str, bounds in self.metadata["page_bounds"].items():
                page_num = int(page_num_str)
                self._page_bounds_cache[page_num] = VirtualPageBounds(
                    page_number=page_num,
                    x0=bounds[0], y0=bounds[1], 
                    x1=bounds[2], y1=bounds[3]
                )
    
    def get_virtual_page_number(self, y_coordinate: float) -> int:
        """Get virtual page number for a given y-coordinate using binary search."""
        if not self._virtual_breaks:
            return 0
        
        # Using bisect_right would be slightly more efficient, but this is clear.
        left, right = 0, len(self._virtual_breaks) - 1
        result_page = 0
        
        while left <= right:
            mid = (left + right) // 2
            y_start, page_num = self._virtual_breaks[mid]
            
            if y_coordinate >= y_start:
                result_page = page_num
                left = mid + 1
            else:
                right = mid - 1
        
        return result_page
    
    def get_page_bounds(self, page_number: int) -> Optional[VirtualPageBounds]:
        """Get bounds for a virtual page."""
        return self._page_bounds_cache.get(page_number)
    
    def get_all_page_bounds(self) -> Dict[int, VirtualPageBounds]:
        """Get all virtual page bounds."""
        return self._page_bounds_cache.copy()
    
    def bbox_to_virtual_page_coords(self, bbox: Tuple[float, float, float, float]) -> Tuple[VirtualPageBounds, Tuple[float, float, float, float]]:
        """Convert a bbox to virtual page coordinates."""
        page_number = self.get_virtual_page_number(bbox[1])
        page_bounds = self.get_page_bounds(page_number)
        
        if not page_bounds:
            raise ValueError(f"No bounds found for virtual page {page_number}")
        
        relative_coords = self.absolute_to_relative(bbox, page_bounds)
        return page_bounds, relative_coords

    def absolute_to_relative(self, bbox: BBox, page_bounds: VirtualPageBounds) -> BBox:
        """Convert absolute coordinates to page-relative coordinates."""
        x0, y0, x1, y1 = bbox
        return (
            x0 - page_bounds.x0,
            y0 - page_bounds.y0,
            x1 - page_bounds.x0,
            y1 - page_bounds.y0
        )
    
    def get_region_in_page(self, bbox: Tuple[float, float, float, float]) -> str:
        """Determine if bbox is in header or footer region of its virtual page."""
        page_bounds, relative_coords = self.bbox_to_virtual_page_coords(bbox)
        
        # Calculate midpoint relative to virtual page
        mid_y_relative = (relative_coords[1] + relative_coords[3]) / 2.0
        
        return "header" if mid_y_relative < (page_bounds.height / 2) else "footer"
    
    def filter_bboxes_by_page_limit(self, bboxes: List[Tuple[float, float, float, float]], 
                                   page_limit: int) -> List[Tuple[float, float, float, float]]:
        """Filter bboxes to only include those within the page limit."""
        return [bbox for bbox in bboxes if self.get_virtual_page_number(bbox[1]) <= page_limit]
    
    def get_document_bounds_with_limit(self, page_limit: int) -> Optional[fitz.Rect]:
        """Get document bounds up to the specified page limit."""
        if page_limit < 0:
            return None
        
        max_x = max_y = 0
        
        for page_num, bounds in self._page_bounds_cache.items():
            if page_num <= page_limit:
                max_x = max(max_x, bounds.x1)
                max_y = max(max_y, bounds.y1)
        
        return fitz.Rect(0, 0, max_x, max_y) if max_y > 0 else None
    
    def get_view_dimensions_with_limit(self, page_limit: int) -> Tuple[float, float]:
        """Get view dimensions up to the specified page limit."""
        bounds_rect = self.get_document_bounds_with_limit(page_limit)
        return (bounds_rect.width, bounds_rect.height) if bounds_rect else (0, 0)
    
    @property
    def page_count(self) -> int:
        """Get total number of virtual pages."""
        return self.metadata.get("page_count", 0)
    
    @property
    def combined_doc_width(self) -> float:
        """Get combined document width."""
        return self.metadata.get("combined_doc_width", 0)
    
    @property
    def combined_doc_height(self) -> float:
        """Get combined document height."""
        return self.metadata.get("combined_doc_height", 0)