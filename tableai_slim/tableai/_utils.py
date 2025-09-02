import dateparser
import re
from datetime import datetime
from typing import Dict, Any
import fitz
from typing import Optional, Any, List, Dict, Union, Tuple

from ._coordinates import BBox, Box

def ext_pdf_date(meta_date: str = None) -> str:
    date_l = list(filter(None, re.split(':|-|D', meta_date)))
    date = date_l[0][:8]
    return datetime.strptime(date, "%Y%m%d").date().strftime('%Y-%m-%d')

def ext_pdf_metadata(
    doc: fitz.Document, 
    metadata_accessors: Optional[Tuple] = ("creator", "title", "author", "subject", "keywords", "producer"),
    date_fields: Optional[Tuple] = ("creationDate", "modDate")
) -> Dict | None:
    trailer = doc.pdf_trailer()
    meta_version = trailer.get("Version") if isinstance(trailer, dict) else None
    md = doc.metadata
    metadata = {k: md[k] for k in metadata_accessors if md.get(k)}
    metadata['tag'] = "|".join(f"{k}|{''.join(md[k].split())}" for k in metadata_accessors if md.get(k))
    metadata['raw'] = md
    metadata['trailer'] = trailer
    metadata['format'] = md.get('format', None)
    metadata['dates'] = {k: ext_pdf_date(md[k]) for k in date_fields if md.get(k)}
    return metadata

def ext_page_content_frame(page: fitz.Page) -> BBox:
    content_frame = fitz.Rect()
    for b in page.get_text("blocks"):
        content_frame |= b[:4]
    return Box(
        bbox=tuple(content_frame), 
        box_type='PAGE_CONTENT_FRAME', 
        page_number=page.number
    )

def calc_inverse_page_rects(
    box: Union[Box, Tuple[float, float, float, float]],
    page: fitz.Page,
    clamp_nonnegative: bool = True,
) -> Dict[str, Any]:
    """
    Compute the inverse (outside) regions of `box` with respect to `page`.
    Coordinates assume PyMuPDF's y-down system.

    Returns:
      {
        "rects": {
          "top":    (x0, y0, x1, y1),
          "bottom": (x0, y0, x1, y1),
          "left":   (x0, y0, x1, y1),
          "right":  (x0, y0, x1, y1),
        },
        "inverse_box_sizes": {"left": float, "right": float, "top": float, "bottom": float},
        "content_rect": fitz.Rect,
        "page_rect": fitz.Rect,
        "size": (page_width, page_height),
      }
    """
    if hasattr(box, "bbox"):
        c_x0, c_y0, c_x1, c_y1 = box.bbox
    else:
        c_x0, c_y0, c_x1, c_y1 = map(float, box)

    pr = page.rect
    p_x0, p_y0, p_x1, p_y1 = pr.x0, pr.y0, pr.x1, pr.y1

    # Build rectangles always clipped to the page; works even if `box` overflows the page.
    # Vertical split lines
    top_y2 = min(max(c_y0, p_y0), p_y1)   # bottom of top strip
    bot_y1 = max(min(c_y1, p_y1), p_y0)   # top of bottom strip
    # Horizontal split lines
    left_x1 = min(max(c_x0, p_x0), p_x1)  # right edge of left strip
    right_x0 = max(min(c_x1, p_x1), p_x0) # left edge of right strip
    # Vertical span for side strips (clipped to page); collapse if no overlap
    v_y0 = top_y2
    v_y1 = bot_y1
    if v_y1 < v_y0:
        v_y1 = v_y0  # zero-height if the box is completely above or below the page

    rects = {
        "top":    (p_x0, p_y0, p_x1, top_y2),
        "bottom": (p_x0, bot_y1, p_x1, p_y1),
        "left":   (p_x0, v_y0, left_x1, v_y1),
        "right":  (right_x0, v_y0, p_x1, v_y1),
    }

    # Distances to page edges (optionally clamped to 0)
    if clamp_nonnegative:
        inverse_box_sizes = {
            "left":   left_x1 - p_x0,
            "right":  p_x1 - right_x0,
            "top":    top_y2 - p_y0,
            "bottom": p_y1 - bot_y1,
        }
    else:
        inverse_box_sizes = {
            "left":   c_x0 - p_x0,
            "right":  p_x1 - c_x1,
            "top":    c_y0 - p_y0,
            "bottom": p_y1 - c_y1,
        }

    return {
        "rects": rects,
        "inverse_box_sizes": inverse_box_sizes
    }

def ext_page_metadata(pg: fitz.Page) -> Dict[str, Any]:
    """
    Build lightweight, page-specific metadata for a single Page.
    Uses your content frame + inverse rects.
    """
    p = pg._fitz_page  # fitz.Page
    blocks = p.get_text("blocks") or []
    content_frame_box = ext_page_content_frame(p)  # -> Box
    inv = calc_inverse_page_rects(box=content_frame_box, page=p, clamp_nonnegative=True)

    return {
        "page_number": pg.number,
        "width": pg.width,
        "height": pg.height,
        "rotation": pg.rotation,
        "has_text": bool(blocks),
        "block_count": len(blocks),
        "content_frame": content_frame_box.bbox,     # (x0, y0, x1, y1)
        "inverse_rects": inv["rects"],               # dict of 4 tuples
        "margins": inv["inverse_box_sizes"]
    }
