"""
Pure‑function utilities for bounding‑box (BBox) and page‑space geometry.

All public helpers build on a *single* set of primitives:
    ├─ bbox_width(), bbox_height(), bbox_area()
    ├─ translate(), scale()
    ├─ intersection(), overlaps()
    └─ union()
Everything else is a very thin wrapper around the above ‑– so there is zero
redundancy and every behavioural change propagates automatically.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Union, Any
from dataclasses import dataclass

x0         = float
x1         = float
y0         = float
y1         = float
BBox       = Tuple[x0, x1, y0, y1]
BBoxList   = List[BBox]

@dataclass
class Box:
    """A semantic region with bounds and metadata."""
    bbox: BBox
    box_type: str  # table, text, image, header, footer
    page_number: int
    metadata: Dict[str, Any] = None

    @property
    def rect(self):
        x0, y0, x1, y1 = self.bbox
        return float(x0), float(y0), float(x1), float(y1)

Boxes: List[Box]

# -----------------------------------------------------------------------------
# ░░  CORE PRIMITIVES  ░░
# -----------------------------------------------------------------------------

def bbox_width(b: BBox)  -> float: return max(0.0, b[2] - b[0])
def bbox_height(b: BBox) -> float: return max(0.0, b[3] - b[1])
def bbox_area(b: BBox)   -> float: return bbox_width(b) * bbox_height(b)

def midpoint(p1: Union[x0,y0], p2: Union[x1, y1]):
    return (p1 + p2)/2

def translate(b: BBox, dx: float = 0.0, dy: float = 0.0) -> BBox:
    """Move bbox by (dx, dy)."""
    x0, y0, x1, y1 = b
    return (x0 + dx, y0 + dy, x1 + dx, y1 + dy)

def scale(b: BBox, sx: float = 1.0, sy: float = 1.0) -> BBox:
    """Scale bbox about origin."""
    x0, y0, x1, y1 = b
    return (x0 * sx, y0 * sy, x1 * sx, y1 * sy)

def intersection(a: BBox, b: BBox) -> Optional[BBox]:
    """Return the intersecting bbox or *None* if the boxes do not overlap."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:                       # no overlap
        return None
    return (x0, y0, x1, y1)

def overlaps(a: BBox, b: BBox) -> bool:
    """True if *any* intersection exists (inclusive of touching edges)."""
    return intersection(a, b) is not None

def union(a: BBox, b: BBox) -> BBox:
    """The smallest bbox that encloses *both* boxes (does not require overlap)."""
    return (min(a[0], b[0]),
            min(a[1], b[1]),
            max(a[2], b[2]),
            max(a[3], b[3]))