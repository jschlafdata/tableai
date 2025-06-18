from typing import TypeAlias, Tuple, Optional, List, Dict, Any, Union, Protocol, Callable
from tableai.pdf.generic_results import GroupbyQueryResult

# Type definitions for better type safety
BBox = Tuple[float, float, float, float]
BBoxList = List[BBox]
BoundingBoxes: TypeAlias = List[Tuple[float, float, float, float]]
GroupFunction = Callable[[GroupbyQueryResult], Any]
Point = Tuple[float, float]
TextBlocks: TypeAlias = List[str]

class HasBBoxField(Protocol):
    """Protocol for objects that have bbox fields."""
    def __getattribute__(self, name: str) -> Any: ...