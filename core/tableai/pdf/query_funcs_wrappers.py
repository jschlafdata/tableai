
from tableai.pdf.generic_results import GroupbyQueryResult
from tableai.pdf.generic_types import (
    GroupFunction, 
    BBox, 
    BBoxList
)
from typing import Optional, List,  Dict, Any, Callable, Tuple
from tableai.pdf.coordinates import (
    Geometry, 
    CoordinateMapping
)

# Field accessor functions that return aggregation functions with type safety
def merge_all_bboxes(field_name: str) -> GroupFunction:
    """
    Returns a function that merges all bboxes from the specified field.
    
    Args:
        field_name: Name of the field containing the list of bboxes
                   (e.g., 'group_bboxes', 'group_bboxes_rel')
    
    Returns:
        Function that takes a group and returns merged bbox or None
    """
    __trace_ignore__ = True
    def _merge_bboxes(group: 'GroupbyQueryResult') -> Optional[BBox]:
        __trace_ignore__ = True
        bboxes_to_merge: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes_to_merge:
            return None
        return Geometry.merge_all_boxes(bboxes_to_merge)
    
    return _merge_bboxes

def merge_overlapping_bboxes(field_name: str) -> GroupFunction:
    """
    Returns a function that merges overlapping bboxes from the specified field.
    
    Args:
        field_name: Name of the field containing the list of bboxes
    
    Returns:
        Function that takes a group and returns list of merged bboxes
    """
    def _merge_overlapping(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return []
        return Geometry.merge_overlapping_boxes(bboxes)
    
    return _merge_overlapping

def concat_text(field_name: str, delimiter: str = '|') -> GroupFunction:
    """
    Returns a function that concatenates text from the specified field.
    
    Args:
        field_name: Name of the field containing the list of text strings
        delimiter: String to join the text with
    
    Returns:
        Function that takes a group and returns concatenated text
    """
    def _concat_text(group: 'GroupbyQueryResult') -> str:
        text_list: List[str] = getattr(group, field_name, [])
        return delimiter.join(text_list)
    
    return _concat_text

def get_field(field_name: str) -> GroupFunction:
    """
    Returns a function that extracts a specific field from the group.
    
    Args:
        field_name: Name of the field to extract
    
    Returns:
        Function that takes a group and returns the field value
    """
    def _get_field(group: 'GroupbyQueryResult') -> Any:
        return getattr(group, field_name, None)
    
    return _get_field

def count_items(field_name: str) -> GroupFunction:
    """
    Returns a function that counts items in a list field.
    
    Args:
        field_name: Name of the field containing a list
    
    Returns:
        Function that takes a group and returns the count
    """
    def _count_items(group: 'GroupbyQueryResult') -> int:
        items = getattr(group, field_name, [])
        return len(items) if items else 0
    
    return _count_items

def check_overlap_with_bbox(field_name: str, target_bbox: BBox) -> GroupFunction:
    """
    Returns a function that checks if any bbox in the field overlaps with the target bbox.
    
    Args:
        field_name: Name of the field containing bboxes
        target_bbox: The bbox to check overlap against
    
    Returns:
        Function that takes a group and returns True if any bbox overlaps
    """
    def _check_overlap(group: 'GroupbyQueryResult') -> bool:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return False
        
        for bbox in bboxes:
            if Geometry.bbox_overlaps(bbox, target_bbox):
                return True
        return False
    
    return _check_overlap

def check_x_overlap_with_bbox(field_name: str, target_bbox: BBox) -> GroupFunction:
    """
    Returns a function that checks if any bbox in the field has x-overlap with the target bbox.
    
    Args:
        field_name: Name of the field containing bboxes
        target_bbox: The bbox to check x-overlap against
    
    Returns:
        Function that takes a group and returns True if any bbox has x-overlap
    """
    def _check_x_overlap(group: 'GroupbyQueryResult') -> bool:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return False
        
        for bbox in bboxes:
            if Geometry.is_x_overlapping(bbox, target_bbox):
                return True
        return False
    
    return _check_x_overlap

def is_fully_contained_in(field_name: str, outer_bbox: BBox, index: int = 0) -> GroupFunction:
    """
    Returns a function that checks if a bbox is fully contained within the outer bbox.
    
    Args:
        field_name: Name of the field containing bboxes
        outer_bbox: The containing bbox
        index: Index of the bbox to check (default: 0)
    
    Returns:
        Function that takes a group and returns True if bbox is fully contained
    """
    def _is_contained(group: 'GroupbyQueryResult') -> bool:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes or len(bboxes) <= index:
            return False
        
        return Geometry.is_fully_contained(bboxes[index], outer_bbox)
    
    return _is_contained

def percent_contained_in(field_name: str, outer_bbox: BBox, index: int = 0) -> GroupFunction:
    """
    Returns a function that calculates what percentage of a bbox is contained in the outer bbox.
    
    Args:
        field_name: Name of the field containing bboxes
        outer_bbox: The containing bbox
        index: Index of the bbox to check (default: 0)
    
    Returns:
        Function that takes a group and returns percentage contained (0.0 to 1.0)
    """
    def _percent_contained(group: 'GroupbyQueryResult') -> float:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes or len(bboxes) <= index:
            return 0.0
        
        return Geometry.percent_contained(bboxes[index], outer_bbox)
    
    return _percent_contained

def scale_bboxes_y(field_name: str, y_offset: float) -> GroupFunction:
    """
    Returns a function that scales all bboxes in a field by a y-offset.
    
    Args:
        field_name: Name of the field containing bboxes
        y_offset: Offset to apply to y coordinates
    
    Returns:
        Function that takes a group and returns scaled bboxes
    """
    def _scale_y(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return []
        
        return [Geometry.scale_y(bbox, y_offset) for bbox in bboxes]
    
    return _scale_y

def transform_bboxes(field_name: str, transform_func: Callable[[BBox], BBox]) -> GroupFunction:
    """
    Returns a function that applies a custom transformation to all bboxes in a field.
    
    Args:
        field_name: Name of the field containing bboxes
        transform_func: Function that takes a bbox and returns a transformed bbox
    
    Returns:
        Function that takes a group and returns transformed bboxes
    """
    def _transform(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return []
        
        return [transform_func(bbox) for bbox in bboxes]
    
    return _transform


def expand_bboxes(field_name: str, margin: float) -> GroupFunction:
    """
    Returns a function that expands all bboxes in a field by a margin.
    
    Args:
        field_name: Name of the field containing bboxes
        margin: Margin to expand in all directions
    
    Returns:
        Function that takes a group and returns expanded bboxes
    """
    def _expand(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return []
        
        return [Geometry.expand_bbox(bbox, margin) for bbox in bboxes]
    
    return _expand

def contract_bboxes(field_name: str, margin: float) -> GroupFunction:
    """
    Returns a function that contracts all bboxes in a field by a margin.
    
    Args:
        field_name: Name of the field containing bboxes
        margin: Margin to contract in all directions
    
    Returns:
        Function that takes a group and returns contracted bboxes
    """
    def _contract(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return []
        
        return [Geometry.contract_bbox(bbox, margin) for bbox in bboxes]
    
    return _contract

def bbox_centers(field_name: str) -> GroupFunction:
    """
    Returns a function that calculates center points for all bboxes in a field.
    
    Args:
        field_name: Name of the field containing bboxes
    
    Returns:
        Function that takes a group and returns list of center points
    """
    def _centers(group: 'GroupbyQueryResult') -> List[Tuple[float, float]]:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return []
        
        return [Geometry.bbox_center(bbox) for bbox in bboxes]
    
    return _centers


def sort_bboxes(field_name: str, sort_by: str = 'top_left') -> GroupFunction:
    """
    Returns a function that sorts bboxes by position or size.
    
    Args:
        field_name: Name of the field containing bboxes
        sort_by: Sorting method - 'top_left', 'center', 'area', 'width', 'height'
    
    Returns:
        Function that takes a group and returns sorted bboxes
    """
    def _sort(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        if not bboxes:
            return []
        
        return Geometry.sort_bboxes_by_position(bboxes, sort_by)
    
    return _sort

def absolute_to_relative_bboxes(field_name: str, page_bounds_field: str) -> GroupFunction:
    """
    Returns a function that converts absolute bboxes to page-relative coordinates.
    
    Args:
        field_name: Name of the field containing bboxes
        page_bounds_field: Name of the field containing page bounds object
    
    Returns:
        Function that takes a group and returns relative bboxes
    """
    def _to_relative(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        page_bounds = getattr(group, page_bounds_field, None)
        
        if not bboxes or not page_bounds:
            return []
        
        return [CoordinateMapping.absolute_to_relative(bbox, page_bounds) for bbox in bboxes]
    
    return _to_relative

def relative_to_absolute_bboxes(field_name: str, page_bounds_field: str) -> GroupFunction:
    """
    Returns a function that converts relative bboxes to absolute coordinates.
    
    Args:
        field_name: Name of the field containing bboxes
        page_bounds_field: Name of the field containing page bounds object
    
    Returns:
        Function that takes a group and returns absolute bboxes
    """
    def _to_absolute(group: 'GroupbyQueryResult') -> BBoxList:
        bboxes: Optional[BBoxList] = getattr(group, field_name, None)
        page_bounds = getattr(group, page_bounds_field, None)
        
        if not bboxes or not page_bounds:
            return []
        
        return [CoordinateMapping.relative_to_absolute(bbox, page_bounds) for bbox in bboxes]
    
    return _to_absolute