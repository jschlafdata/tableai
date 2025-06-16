from __future__ import annotations

import base64
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4
from typing import Optional, List, Tuple, Union, Dict, Any, Callable, TYPE_CHECKING, TypeVar, Generic, Type
import fitz
from pydantic import BaseModel, field_validator, model_validator, ValidationError, Field, create_model, field_serializer
from io import BytesIO
import base64
from PIL import Image, ImageDraw, ImageFont
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import fitz
import numpy as np
from enum import Enum

from tableai.pdf.query import (
    FitzSearchIndex, 
    GroupbyTransform
)
from tableai.pdf.coordinates import (
    Geometry,
    CoordinateMapping
)

from tableai.pdf.generic_models import (
    TextNormalizer, 
    WhitespaceGenerator
)

from tableai.readers.files import FileReader  # Import your FileReader

__all__ = ["PDFModel"]

try:
    from IPython.display import display
    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False


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
        
        relative_coords = CoordinateMapping.absolute_to_relative(bbox, page_bounds)
        return page_bounds, relative_coords
    
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


@dataclass
class PDFMetadata:

    trailer: Optional[str] = None
    version: Optional[str] = None
    meta_tag: Optional[str] = None
    metadata_accessors: Optional[Tuple] = ("creator", "title", "author", "subject", "keywords")
    _doc: Optional[fitz.Document] = None
    _context: Optional[dict] = None

    def __post_init__(self):
        trailer = self._doc.pdf_trailer()
        self.meta_version = trailer.get("Version") if isinstance(trailer, dict) else None
        md = self._doc.metadata
        self.meta_tag = "|".join(f"{k}|{''.join(md[k].split())}" for k in self.metadata_accessors if md.get(k))
        self._doc.close()
    
    @classmethod
    def scan_header_ocr(cls, doc: fitz.Document, y1: Optional[int]=None):
        pass

class LoadType(str, Enum):
    FULL = "full"
    FIRST = "first"

class PDFModel(BaseModel):
    """
    High‑level container that:
    1. Loads an input PDF (local path / bytes / S3) via `FileReader`.
    2. Stitches all pages vertically into *one* tall page (“combined doc”).
    3. Builds a virtual‑page index (`FitzSearchIndex`) that understands page breaks.
    4. Offers rich helpers for rendering, cropping, and annotation.
    """

    # ----------------------- Core public attributes ------------------------ #
    path: Union[str, Path, bytes, bytearray]
    load_type: Optional[LoadType] = LoadType.FULL
    s3_client: Optional[Any] = None
    text_normalizer: Optional[TextNormalizer] = TextNormalizer(
        patterns={
            r'page\s*\d+\s*of\s*\d+': 'page xx of xx',
            r'page\s*\d+': 'page xx'
        }
    )
    whitespace_generator: Optional[WhitespaceGenerator] = WhitespaceGenerator(min_gap=5.0)

    # Populated automatically by the validator
    doc: Optional[fitz.Document] = None
    name: Optional[str] = None
    query_index: Optional["FitzSearchIndex"] = None
    pdf_metadata: Optional[PDFMetadata] = None
    virtual_page_metadata: Optional[dict] = None
    
    # NEW: Centralized virtual page manager
    vpm: Optional[VirtualPageManager] = None

    class Config:
        arbitrary_types_allowed = True

    # --------------------------------------------------------------------- #
    #                     INITIALISATION & METADATA                         #
    # --------------------------------------------------------------------- #
    @model_validator(mode="after")
    def _initialise_and_index(self) -> "PDFModel":
        """Load, stitch, index and harvest metadata in a *single* pass."""
        # 1. Load original multi‑page PDF via project reader
        try:
            original_doc = FileReader.pdf(self.path, s3_client=self.s3_client)
            if not isinstance(original_doc, fitz.Document):
                raise TypeError(
                    f"FileReader returned {type(original_doc)}, expected fitz.Document"
                )
            if isinstance(self.path, (str, Path)):
                self.name = Path(self.path).stem
        except Exception as exc:
            raise ValueError(f"Could not load PDF: {exc}") from exc

        # 2. Build combined document (+ virtual‑page metadata)
        self.doc, self.virtual_page_metadata = self._combine_pages_and_get_metadata(
            original_doc=original_doc, 
            load_type=self.load_type
        )

        self.pdf_metadata = PDFMetadata(_doc=original_doc)
        
        # NEW: Instantiate the VirtualPageManager with the generated metadata
        if self.virtual_page_metadata:
            self.vpm = VirtualPageManager(self.virtual_page_metadata)
        else:
            raise ValueError("Failed to generate virtual page metadata.")        

        self.query_index = FitzSearchIndex.from_pdf_model(self, text_normalizer=self.text_normalizer, whitespace_generator=self.whitespace_generator)

        return self

    # --------------------------------------------------------------------- #
    #                       INTERNAL BUILD HELPERS                          #
    # --------------------------------------------------------------------- #
    @staticmethod
    def _combine_pages_and_get_metadata(
        original_doc: fitz.Document,
        load_type: LoadType,
        LOAD_FIRST_PAGE_ONLY: bool = False
    ) -> Tuple[fitz.Document, dict]:
        """Return a vertically‑stitched document + rich per‑page metadata."""
        # This method's responsibility is to create the raw metadata.
        # It remains largely unchanged as it is the source of truth for the VPM.
        
        combined = fitz.open()
        total_h = 0.0
        max_w = 0.0

        if load_type == LoadType.FIRST:
            LOAD_FIRST_PAGE_ONLY = True

        # Determine the effective page range
        page_range = [original_doc[0]] if LOAD_FIRST_PAGE_ONLY else original_doc
        page_count = 1 if LOAD_FIRST_PAGE_ONLY else len(original_doc)

        page_dims: List[dict] = []
        for i, pg in enumerate(page_range):
            dims = {"width": pg.rect.width, "height": pg.rect.height}
            page_dims.append(dims)
            total_h += dims["height"]
            max_w = max(max_w, dims["width"])

        combined_page = combined.new_page(width=max_w, height=total_h)

        vpage_bounds: Dict[int, Tuple[float, float, float, float]] = {}
        vpage_breaks: List[Tuple[float, int]] = []
        content_areas = {
            "margin_bboxes": [],
            "content_bboxes": [],
            "pages": {},
        }

        y_offset = 0.0
        for i, pg in enumerate(page_range):
            # Use original page index (i) for consistency
            page_idx = i if not LOAD_FIRST_PAGE_ONLY else 0
            
            vpage_breaks.append((y_offset, page_idx))
            tgt_rect = fitz.Rect(0, y_offset, pg.rect.width, y_offset + pg.rect.height)
            vpage_bounds[page_idx] = tuple(tgt_rect)
            combined_page.show_pdf_page(tgt_rect, original_doc, page_idx)

            local_content = fitz.Rect()
            for b in pg.get_text("blocks"):
                local_content |= b[:4]

            if not local_content.is_empty:
                content_rel = tuple(local_content)
                margin_rel = Geometry.inverse_page_blocks(
                    {page_idx: {"bboxes": [content_rel], "page_width": pg.rect.width, "page_height": pg.rect.height}}
                )
                content_abs = Geometry.scale_y(content_rel, y_offset)
                margin_abs = [Geometry.scale_y(b, y_offset) for b in margin_rel]

                content_areas["content_bboxes"].append(content_abs)
                content_areas["margin_bboxes"].extend(margin_abs)
                content_areas["pages"][page_idx] = {"content_bbox(rel)": content_rel, "margin_bboxes(rel)": margin_rel}
            else:
                content_areas["pages"][page_idx] = {"content_bbox(rel)": None, "margin_bboxes(rel)": []}

            y_offset += pg.rect.height

        vpage_breaks.sort()
        metadata = {
            "page_count": page_count,
            "page_bounds": vpage_bounds,
            "page_breaks": vpage_breaks,
            "page_content_areas": content_areas,
            "combined_doc_width": max_w,
            "combined_doc_height": total_h,
            "original_page_dims": page_dims,
        }
        return combined, metadata

    # --------------------------------------------------------------------- #
    #                          PUBLIC PROPERTIES                            #
    # --------------------------------------------------------------------- #
    @property
    def pages(self) -> int:
        """Total *virtual* pages."""
        # UPDATED: Delegate to VirtualPageManager
        return self.vpm.page_count if self.vpm else 0

    # --------------------------------------------------------------------- #
    #                         INTERNAL UTILITIES                            #
    # --------------------------------------------------------------------- #
    def _filter_by_page_limit(
        self,
        page_limit: int,
        *,
        highlight_boxes: Optional[Union[List[tuple], Dict]] = None,
        crop_boxes: Optional[List] = None,
    ) -> Tuple[Optional[Union[List[tuple], Dict]], Optional[List]]:
        """Apply `page_limit` to crop‑ and highlight‑inputs in a single pass."""
        if not self.vpm:
            return highlight_boxes, crop_boxes

        # UPDATED: Delegate filtering logic to VPM where possible
        filt_crop = self.vpm.filter_bboxes_by_page_limit(crop_boxes, page_limit) if crop_boxes else None

        if isinstance(highlight_boxes, list):
            filt_high = self.vpm.filter_bboxes_by_page_limit(highlight_boxes, page_limit)
        elif isinstance(highlight_boxes, dict):
            # For dicts, we still need to iterate but can use VPM's checker
            filt_high = {
                k: {**d, "boxes": [b for b in d.get("boxes", []) if self.vpm.get_virtual_page_number(b[1]) <= page_limit]}
                for k, d in highlight_boxes.items()
            }
            filt_high = {k: v for k, v in filt_high.items() if v["boxes"]}
        else:
            filt_high = highlight_boxes

        return filt_high, filt_crop

    # --------------- Shared drawing / annotation helper ------------------ #
    def _annotate_base64_image(
        self, *, base64_img: str, page_number: Optional[int], page_limit: Optional[int],
        highlight_boxes: Optional[Union[List[tuple], Dict]], grid: bool, box_color: str,
        box_width: int, font_size: int,
    ) -> str:
        """Overlay highlight boxes (and optionally a y‑grid) on a base64 image."""
        if (not highlight_boxes and not grid) or not _IPYTHON_AVAILABLE:
            return base64_img

        img_raw = base64.b64decode(base64_img)
        img = Image.open(BytesIO(img_raw)).convert("RGB")
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        # UPDATED: All coordinate logic now uses the VirtualPageManager
        if page_number is not None:
            bounds = self.vpm.get_page_bounds(page_number)
            if not bounds: return base64_img # Should not happen
            view_w, view_h = bounds.width, bounds.height
            off_x, off_y = bounds.x0, bounds.y0
        else:
            off_x = off_y = 0
            if page_limit is not None:
                view_w, view_h = self.vpm.get_view_dimensions_with_limit(page_limit)
            else:
                view_w = self.vpm.combined_doc_width
                view_h = self.vpm.combined_doc_height

        if view_w == 0 or view_h == 0:
            return base64_img

        sx = img.width / view_w
        sy = img.height / view_h

        def draw_bbox(rect, clr):
            # UPDATED: Use CoordinateMapping for scaling
            px = CoordinateMapping.scale_for_display(rect, sx, sy, off_x, off_y)
            if px[2] > px[0] and px[3] > px[1]:
                draw.rectangle(px, outline=clr, width=box_width)
            return px

        if isinstance(highlight_boxes, dict):
            for lbl, data in (highlight_boxes or {}).items():
                col = data.get("color", box_color)
                for b in data.get("boxes", []):
                    px_rect = draw_bbox(b, col)
                    ax, ay = px_rect[0], px_rect[1]
                    txt_bbox = draw.textbbox((ax, ay - box_width), lbl, font=font, anchor="lb")
                    pad = 3
                    bg = (txt_bbox[0] - pad, txt_bbox[1] - pad, txt_bbox[2] + pad, txt_bbox[3] + pad)
                    draw.rectangle(bg, fill=col)
                    draw.text((ax, ay - box_width), lbl, font=font, fill="white", anchor="lb")
        elif isinstance(highlight_boxes, list):
            for b in highlight_boxes:
                draw_bbox(b, box_color)

        # --------------------------- Optional grid ------------------------ #
        if grid:
            try:
                gfont = ImageFont.truetype("Arial.ttf", max(10, int(font_size * 0.8)))
            except IOError:
                gfont = ImageFont.load_default()

            grid_color = (255, 20, 147, 150)  # semi‑transparent deep‑pink
            y_pdf = (int(off_y) // 100) * 100
            while y_pdf < off_y + view_h:
                if y_pdf >= off_y:
                    y_pix = (y_pdf - off_y) * sy
                    draw.line([(0, y_pix), (img.width, y_pix)], fill=grid_color, width=1)
                    label = str(int(y_pdf))
                    tb = draw.textbbox((5, y_pix + 2), label, font=gfont, anchor="lt")
                    draw.rectangle(tb, fill="white")
                    draw.text((5, y_pix + 2), label, font=gfont, fill="black", anchor="lt")
                y_pdf += 100

        out = BytesIO()
        img.save(out, format="PNG")
        return base64.b64encode(out.getvalue()).decode("utf-8")

    # Retain legacy name for backward compatibility
    def _add_highlight_boxes_to_base64(self, *args, **kwargs) -> str:
        kwargs.pop('zoom', None) # zoom not used in _annotate_base64_image
        return self._annotate_base64_image(*args, grid=False, **kwargs)

    # --------------------------------------------------------------------- #
    #                              RENDERING                                #
    # --------------------------------------------------------------------- #
    def get_page(self, page_number: Optional[int] = None) -> fitz.Page:
        """Always returns *combined* page (index 0) after bound checking."""
        if not self.doc or not self.vpm:
            raise RuntimeError("PDF document not loaded or VPM not initialized")
        if page_number is not None and (page_number < 0 or page_number >= self.vpm.page_count):
            raise IndexError(f"Virtual page {page_number} out of range (0-{self.vpm.page_count-1})")
        return self.doc[0]

    def _render_combined_crops(
        self,
        page: fitz.Page,
        crop_boxes: List[Tuple[float, float, float, float]],
        zoom: float = 1.0,
        spacing: int = 10,
    ) -> str:
        """Render *multiple* crops then stack vertically with white spacer."""
        mat = fitz.Matrix(zoom, zoom)

        rendered: List[Image.Image] = []
        full_h = 0
        max_w = 0

        for box in crop_boxes:
            pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(*box))
            img = Image.open(BytesIO(pix.tobytes("png")))
            rendered.append(img)
            full_h += img.height
            max_w = max(max_w, img.width)

        if len(rendered) > 1:
            full_h += spacing * (len(rendered) - 1)

        canvas = Image.new("RGB", (max_w, full_h), "white")
        y = 0
        for i, im in enumerate(rendered):
            canvas.paste(im, (0, y))
            y += im.height + (spacing if i < len(rendered) - 1 else 0)

        out = BytesIO()
        canvas.save(out, format="PNG")
        return base64.b64encode(out.getvalue()).decode("utf-8")
    
    # UPDATED: These two methods are now just simple delegations to the VPM.
    def _get_limited_view_dimensions(self, page_limit: int) -> Tuple[float, float]:
        """Dimensions (w,h) up to `page_limit`."""
        return self.vpm.get_view_dimensions_with_limit(page_limit) if self.vpm else (0, 0)

    def _get_document_bounds_with_limit(self, page_limit: int) -> Optional[fitz.Rect]:
        """Bounding rect (0,0,x_max,y_max) limited to given virtual page."""
        return self.vpm.get_document_bounds_with_limit(page_limit) if self.vpm else None

    # --------------------------- Public helpers --------------------------- #
    def get_page_base64(
        self, *, page_number: Optional[int] = None, zoom: float = 1.0,
        crop_boxes: Optional[List] = None, spacing: int = 10, page_limit: Optional[int] = None,
        highlight_boxes: Optional[Union[List[tuple], Dict]] = None, box_color: str = "red",
        box_width: int = 2, font_size: int = 12,
    ) -> str:
        """Return rendered PNG (base64) for a view, with optional annotations."""
        if not self.vpm: raise RuntimeError("VPM not initialized")
        
        page = self.get_page(page_number)

        if page_limit is not None:
            highlight_boxes, crop_boxes = self._filter_by_page_limit(
                page_limit, highlight_boxes=highlight_boxes, crop_boxes=crop_boxes
            )

        if crop_boxes:
            b64 = self._render_combined_crops(page, crop_boxes, zoom, spacing)
        else:
            clip_rect = None
            # UPDATED: Get clipping rectangle from the VPM.
            if page_number is not None:
                bounds = self.vpm.get_page_bounds(page_number)
                clip_rect = bounds.rect if bounds else None
            elif page_limit is not None:
                clip_rect = self.vpm.get_document_bounds_with_limit(page_limit)

            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip_rect)
            b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

        return self._annotate_base64_image(
            base64_img=b64, page_number=page_number, page_limit=page_limit,
            highlight_boxes=highlight_boxes, grid=False, box_color=box_color,
            box_width=box_width, font_size=font_size,
        )

    # --------------------------------------------------------------------- #
    #                              DISPLAY                                  #
    # --------------------------------------------------------------------- #
    def show(self, *args, **kwargs):
        """Render in‑notebook image with optional highlights/grid."""
        # This high-level method remains largely the same, but now orchestrates
        # the refactored, cleaner helper methods.
        if not _IPYTHON_AVAILABLE:
            print("IPython/Jupyter is required to display images.")
            return

        page_number = kwargs.get("page_number")
        page_limit = kwargs.get("page_limit")
        highlight_boxes = kwargs.get("highlight_boxes")
        crop_boxes = kwargs.get("crop_boxes")

        if page_limit is not None:
            highlight_boxes, crop_boxes = self._filter_by_page_limit(
                page_limit, highlight_boxes=highlight_boxes, crop_boxes=crop_boxes
            )
            kwargs["highlight_boxes"] = highlight_boxes
            kwargs["crop_boxes"] = crop_boxes

        b64 = self.get_page_base64(
            page_number=page_number,
            crop_boxes=crop_boxes,
            spacing=kwargs.get("spacing", 10),
            zoom=kwargs.get("zoom", 1.0),
            page_limit=page_limit,
            highlight_boxes=None, # handled by _annotate_base64_image next
        )

        # Annotate separately for grid support
        b64 = self._annotate_base64_image(
            base64_img=b64,
            page_number=page_number, page_limit=page_limit,
            highlight_boxes=highlight_boxes, grid=kwargs.get("grid", False),
            box_color=kwargs.get("box_color", "red"),
            box_width=kwargs.get("box_width", 2),
            font_size=kwargs.get("font_size", 12),
        )

        display(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))

    # --------------------------------------------------------------------- #
    #                        GEOMETRY CONVERSION                            #
    # --------------------------------------------------------------------- #
    def page_bbox_to_combined(self, page_number: int, page_relative_bbox: tuple) -> tuple:
        """Translate *page‑relative* bbox → *combined‑doc* coordinates."""
        # UPDATED: Delegate to CoordinateMapping
        if not self.vpm:
            raise RuntimeError("VPM not initialized")
        
        page_bounds = self.vpm.get_page_bounds(page_number)
        if not page_bounds:
            raise ValueError(f"Invalid page number: {page_number}")
        
        return CoordinateMapping.relative_to_absolute(page_relative_bbox, page_bounds)

    # --------------------------------------------------------------------- #
    #                               ITERATION                               #
    # --------------------------------------------------------------------- #
    def __iter__(self):
        """Yield virtual page indices (0‑based)."""
        yield from range(self.pages)



# class PDFModel(BaseModel):
#     """
#     High‑level container that:
#     1. Loads an input PDF (local path / bytes / S3) via `FileReader`.
#     2. Stitches all pages vertically into *one* tall page (“combined doc”).
#     3. Builds a virtual‑page index (`LineTextIndex`) that understands page breaks.
#     4. Offers rich helpers for rendering, cropping, and annotation.
#     """

#     # ----------------------- Core public attributes ------------------------ #
#     path: Union[str, Path, bytes, bytearray]
#     s3_client: Optional[object] = None
#     vpm: Optional[VirtualPageManager] = None

#     # Populated automatically by the validator
#     doc: Optional[fitz.Document] = None
#     name: Optional[str] = None
#     query_index: Optional["LineTextIndex"] = None
#     meta_version: Optional[str] = None
#     meta_tag: Optional[str] = None
#     virtual_page_metadata: Optional[dict] = None

#     class Config:
#         arbitrary_types_allowed = True

#     # --------------------------------------------------------------------- #
#     #                     INITIALISATION & METADATA                         #
#     # --------------------------------------------------------------------- #
#     @model_validator(mode="after")
#     def _initialise_and_index(self) -> "PDFModel":
#         """Load, stitch, index and harvest metadata in a *single* pass."""
#         # 1. Load original multi‑page PDF via project reader
#         try:
#             original_doc = FileReader.pdf(self.path, s3_client=self.s3_client)
#             if not isinstance(original_doc, fitz.Document):
#                 raise TypeError(
#                     f"FileReader returned {type(original_doc)}, expected fitz.Document"
#                 )
#             if isinstance(self.path, (str, Path)):
#                 self.name = Path(self.path).stem
#         except Exception as exc:
#             raise ValueError(f"Could not load PDF: {exc}") from exc

#         # 2. Build combined document (+ virtual‑page metadata)
#         self.doc, self.virtual_page_metadata = self._combine_pages_and_get_metadata(
#             original_doc
#         )

#         # 3. Construct virtual‑aware text index
#         self.query_index = LineTextIndex.from_document(
#             self.doc, virtual_page_metadata=self.virtual_page_metadata
#         )

#         # 4. Extract key PDF metadata
#         trailer = self.doc.pdf_trailer()
#         self.meta_version = trailer.get("Version") if isinstance(trailer, dict) else None
#         md = self.doc.metadata
#         wanted = ("creator", "title", "author", "subject", "keywords")
#         self.meta_tag = "|".join(f"{k}|{''.join(md[k].split())}" for k in wanted if md.get(k))

#         original_doc.close()
#         return self

#     # --------------------------------------------------------------------- #
#     #                       INTERNAL BUILD HELPERS                          #
#     # --------------------------------------------------------------------- #
#     @staticmethod
#     def _combine_pages_and_get_metadata(
#         original_doc: fitz.Document,
#     ) -> Tuple[fitz.Document, dict]:
#         """Return a vertically‑stitched document + rich per‑page metadata."""
#         combined = fitz.open()
#         total_h = 0.0
#         max_w = 0.0

#         page_dims: List[dict] = []
#         for pg in original_doc:
#             dims = {"width": pg.rect.width, "height": pg.rect.height}
#             page_dims.append(dims)
#             total_h += dims["height"]
#             max_w = max(max_w, dims["width"])

#         combined_page = combined.new_page(width=max_w, height=total_h)

#         vpage_bounds: Dict[int, Tuple[float, float, float, float]] = {}
#         vpage_breaks: List[Tuple[float, int]] = []
#         content_areas = {
#             "margin_bboxes": [],
#             "content_bboxes": [],
#             "pages": {},
#         }

#         y_offset = 0.0
#         for i, pg in enumerate(original_doc):
#             vpage_breaks.append((y_offset, i))
#             tgt_rect = fitz.Rect(0, y_offset, pg.rect.width, y_offset + pg.rect.height)
#             vpage_bounds[i] = tuple(tgt_rect)
#             combined_page.show_pdf_page(tgt_rect, original_doc, i)

#             # detect real content area on original page
#             local_content = fitz.Rect()
#             for b in pg.get_text("blocks"):
#                 local_content |= b[:4]

#             if not local_content.is_empty:
#                 # relative & absolute bboxes
#                 content_rel = tuple(local_content)
#                 margin_rel = Map.inverse_page_blocks(
#                     {
#                         i: {
#                             "bboxes": [content_rel],
#                             "page_width": pg.rect.width,
#                             "page_height": pg.rect.height,
#                         }
#                     }
#                 )
#                 content_abs = Map.scale_y(content_rel, y_offset)
#                 margin_abs = [Map.scale_y(b, y_offset) for b in margin_rel]

#                 content_areas["content_bboxes"].append(content_abs)
#                 content_areas["margin_bboxes"].extend(margin_abs)
#                 content_areas["pages"][i] = {
#                     "content_bbox(rel)": content_rel,
#                     "margin_bboxes(rel)": margin_rel,
#                 }
#             else:  # blank page
#                 content_areas["pages"][i] = {
#                     "content_bbox(rel)": None,
#                     "margin_bboxes(rel)": [],
#                 }

#             y_offset += pg.rect.height

#         vpage_breaks.sort()
#         metadata = {
#             "page_count": len(original_doc),
#             "page_bounds": vpage_bounds,
#             "page_breaks": vpage_breaks,
#             "page_content_areas": content_areas,
#             "combined_doc_width": max_w,
#             "combined_doc_height": total_h,
#             "original_page_dims": page_dims,
#         }
#         return combined, metadata

#     # --------------------------------------------------------------------- #
#     #                          PUBLIC PROPERTIES                            #
#     # --------------------------------------------------------------------- #
#     @property
#     def pages(self) -> int:
#         """Total *virtual* pages."""
#         return self.virtual_page_metadata.get("page_count", 0) if self.virtual_page_metadata else 0

#     # --------------------------------------------------------------------- #
#     #                         INTERNAL UTILITIES                            #
#     # --------------------------------------------------------------------- #
#     def _filter_by_page_limit(
#         self,
#         page_limit: int,
#         *,
#         highlight_boxes: Optional[Union[List[tuple], Dict]] = None,
#         crop_boxes: Optional[List] = None,
#     ) -> Tuple[Optional[Union[List[tuple], Dict]], Optional[List]]:
#         """
#         Apply `page_limit` to crop‑ and highlight‑inputs in a single pass.
#         """
#         if not hasattr(self, "query_index") or not hasattr(self.query_index, "_get_virtual_page_num"):
#             return highlight_boxes, crop_boxes

#         def within(bbox):
#             return self.query_index._get_virtual_page_num(bbox[1]) <= page_limit

#         filt_crop = [b for b in (crop_boxes or []) if within(b)] if crop_boxes else None

#         if isinstance(highlight_boxes, list):
#             filt_high = [b for b in highlight_boxes if within(b)]
#         elif isinstance(highlight_boxes, dict):
#             filt_high = {
#                 k: {**d, "boxes": [b for b in d.get("boxes", []) if within(b)]}
#                 for k, d in highlight_boxes.items()
#             }
#             filt_high = {k: v for k, v in filt_high.items() if v["boxes"]}
#         else:
#             filt_high = highlight_boxes  # None or unsupported type untouched

#         return filt_high, filt_crop

#     # --------------- Shared drawing / annotation helper ------------------ #
#     def _annotate_base64_image(
#         self,
#         *,
#         base64_img: str,
#         page_number: Optional[int],
#         page_limit: Optional[int],
#         highlight_boxes: Optional[Union[List[tuple], Dict]],
#         grid: bool,
#         box_color: str,
#         box_width: int,
#         font_size: int,
#     ) -> str:
#         """Overlay highlight boxes (and optionally a y‑grid) on a base64 image."""
#         if not highlight_boxes and not grid:
#             return base64_img

#         img_raw = base64.b64decode(base64_img)
#         img = Image.open(BytesIO(img_raw)).convert("RGB")
#         draw = ImageDraw.Draw(img)

#         # font
#         try:
#             font = ImageFont.truetype("Arial.ttf", font_size)
#         except IOError:
#             font = ImageFont.load_default()

#         # --------------------- coordinate transforms ---------------------- #
#         if page_number is not None:
#             x0, y0, x1, y1 = self.virtual_page_metadata["page_bounds"][page_number]
#             view_w, view_h = x1 - x0, y1 - y0
#             off_x, off_y = x0, y0
#         else:
#             off_x = off_y = 0
#             if page_limit is not None:
#                 view_w, view_h = self._get_limited_view_dimensions(page_limit)
#             else:
#                 view_w = self.virtual_page_metadata["combined_doc_width"]
#                 view_h = self.virtual_page_metadata["combined_doc_height"]

#         if view_w == 0 or view_h == 0:  # defensive
#             return base64_img

#         sx = img.width / view_w
#         sy = img.height / view_h

#         # ------------------------- Draw highlights ------------------------ #
#         def draw_bbox(rect, clr):
#             rx0, ry0, rx1, ry1 = rect
#             px = [(rx0 - off_x) * sx, (ry0 - off_y) * sy, (rx1 - off_x) * sx, (ry1 - off_y) * sy]
#             if px[2] > px[0] and px[3] > px[1]:
#                 draw.rectangle(px, outline=clr, width=box_width)
#             return px

#         if isinstance(highlight_boxes, dict):
#             for lbl, data in (highlight_boxes or {}).items():
#                 col = data.get("color", box_color)
#                 for b in data.get("boxes", []):
#                     px_rect = draw_bbox(b, col)
#                     # label background & text
#                     ax, ay = px_rect[0], px_rect[1]
#                     txt_bbox = draw.textbbox((ax, ay - box_width), lbl, font=font, anchor="lb")
#                     pad = 3
#                     bg = (
#                         txt_bbox[0] - pad,
#                         txt_bbox[1] - pad,
#                         txt_bbox[2] + pad,
#                         txt_bbox[3] + pad,
#                     )
#                     draw.rectangle(bg, fill=col)
#                     draw.text((ax, ay - box_width), lbl, font=font, fill="white", anchor="lb")
#         elif isinstance(highlight_boxes, list):
#             for b in highlight_boxes:
#                 draw_bbox(b, box_color)

#         # --------------------------- Optional grid ------------------------ #
#         if grid:
#             try:
#                 gfont = ImageFont.truetype("Arial.ttf", max(10, int(font_size * 0.8)))
#             except IOError:
#                 gfont = ImageFont.load_default()

#             grid_color = (255, 20, 147, 150)  # semi‑transparent deep‑pink
#             y_pdf = (int(off_y) // 100) * 100
#             while y_pdf < off_y + view_h:
#                 if y_pdf >= off_y:
#                     y_pix = (y_pdf - off_y) * sy
#                     draw.line([(0, y_pix), (img.width, y_pix)], fill=grid_color, width=1)
#                     label = str(int(y_pdf))
#                     tb = draw.textbbox((5, y_pix + 2), label, font=gfont, anchor="lt")
#                     draw.rectangle(tb, fill="white")
#                     draw.text((5, y_pix + 2), label, font=gfont, fill="black", anchor="lt")
#                 y_pdf += 100

#         # re‑encode
#         out = BytesIO()
#         img.save(out, format="PNG")
#         return base64.b64encode(out.getvalue()).decode("utf-8")

#     # Retain legacy name for backward compatibility  ------------------- #
#     def _add_highlight_boxes_to_base64(  # noqa: N802  (legacy API)
#         self,
#         base64_img: str,
#         highlight_boxes,
#         page_number,
#         page_limit,
#         box_color,
#         box_width,
#         font_size,
#         zoom,
#     ) -> str:  # pylint: disable=too-many-arguments
#         return self._annotate_base64_image(
#             base64_img=base64_img,
#             page_number=page_number,
#             page_limit=page_limit,
#             highlight_boxes=highlight_boxes,
#             grid=False,
#             box_color=box_color,
#             box_width=box_width,
#             font_size=font_size,
#         )

#     # --------------------------------------------------------------------- #
#     #                              RENDERING                                #
#     # --------------------------------------------------------------------- #
#     def get_page(self, page_number: Optional[int] = None) -> fitz.Page:
#         """Always returns *combined* page (index 0) after bound checking."""
#         if not self.doc:
#             raise RuntimeError("PDF document not loaded")
#         if page_number is not None and (page_number < 0 or page_number >= self.pages):
#             raise IndexError(f"Virtual page {page_number} out of range")
#         return self.doc[0]

#     # ------------ Core low‑level image generation (no annotations) -------- #
#     def _render_combined_crops(
#         self,
#         page: fitz.Page,
#         crop_boxes: List[Tuple[float, float, float, float]],
#         zoom: float = 1.0,
#         spacing: int = 10,
#     ) -> str:
#         """Render *multiple* crops then stack vertically with white spacer."""
#         mat = fitz.Matrix(zoom, zoom)

#         rendered: List[Image.Image] = []
#         full_h = 0
#         max_w = 0

#         for box in crop_boxes:
#             pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(*box))
#             img = Image.open(BytesIO(pix.tobytes("png")))
#             rendered.append(img)
#             full_h += img.height
#             max_w = max(max_w, img.width)

#         if len(rendered) > 1:
#             full_h += spacing * (len(rendered) - 1)

#         canvas = Image.new("RGB", (max_w, full_h), "white")
#         y = 0
#         for i, im in enumerate(rendered):
#             canvas.paste(im, (0, y))
#             y += im.height + (spacing if i < len(rendered) - 1 else 0)

#         out = BytesIO()
#         canvas.save(out, format="PNG")
#         return base64.b64encode(out.getvalue()).decode("utf-8")

#     def _get_limited_view_dimensions(self, page_limit: int) -> Tuple[float, float]:
#         """Dimensions (w,h) up to `page_limit`."""
#         w = h = 0.0
#         for pn, (x0, y0, x1, y1) in self.virtual_page_metadata.get("page_bounds", {}).items():
#             if pn <= page_limit:
#                 w = max(w, x1)
#                 h = max(h, y1)
#         return w, h

#     def _get_document_bounds_with_limit(self, page_limit: int) -> Optional[fitz.Rect]:
#         """Bounding rect (0,0,x_max,y_max) limited to given virtual page."""
#         w, h = self._get_limited_view_dimensions(page_limit)
#         return fitz.Rect(0, 0, w, h) if h > 0 else None

#     # --------------------------- Public helpers --------------------------- #
#     def get_page_base64(
#         self,
#         *,
#         page_number: Optional[int] = None,
#         zoom: float = 1.0,
#         crop_boxes: Optional[List] = None,
#         spacing: int = 10,
#         page_limit: Optional[int] = None,
#         highlight_boxes: Optional[Union[List[tuple], Dict]] = None,
#         box_color: str = "red",
#         box_width: int = 2,
#         font_size: int = 12,
#     ) -> str:
#         """
#         Return rendered PNG (base64) for:
#         * a single virtual page,
#         * full combined doc,
#         * or combined custom crops.

#         Optional highlight‑box and page‑limit support.
#         """
#         page = self.get_page(page_number)

#         # filter inputs by page_limit first
#         if page_limit is not None:
#             highlight_boxes, crop_boxes = self._filter_by_page_limit(
#                 page_limit, highlight_boxes=highlight_boxes, crop_boxes=crop_boxes
#             )

#         if crop_boxes:
#             b64 = self._render_combined_crops(page, crop_boxes, zoom, spacing)
#         else:
#             clip_rect = None
#             if page_number is not None:
#                 clip_rect = fitz.Rect(*self.virtual_page_metadata["page_bounds"][page_number])
#             elif page_limit is not None:
#                 clip_rect = self._get_document_bounds_with_limit(page_limit)

#             pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip_rect)
#             b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

#         # overlay highlights if requested
#         b64 = self._annotate_base64_image(
#             base64_img=b64,
#             page_number=page_number,
#             page_limit=page_limit,
#             highlight_boxes=highlight_boxes,
#             grid=False,
#             box_color=box_color,
#             box_width=box_width,
#             font_size=font_size,
#         )
#         return b64

#     # --------------------------------------------------------------------- #
#     #                              DISPLAY                                  #
#     # --------------------------------------------------------------------- #
#     def show(
#         self,
#         page_number: Optional[int] = None,
#         highlight_boxes: Optional[Union[List[tuple], Dict]] = None,
#         crop_boxes: Optional[List] = None,
#         box_color: str = "red",
#         box_width: int = 2,
#         font_size: int = 12,
#         grid: bool = False,
#         zoom: float = 1.0,
#         spacing: int = 10,
#         page_limit: Optional[int] = None,
#     ):
#         """Render in‑notebook image with optional highlights/grid."""
#         if not _IPYTHON_AVAILABLE:
#             print("IPython/Jupyter is required to display images.")
#             return

#         # page‑limit filtering
#         if page_limit is not None:
#             highlight_boxes, crop_boxes = self._filter_by_page_limit(
#                 page_limit, highlight_boxes=highlight_boxes, crop_boxes=crop_boxes
#             )

#         # raw render (no annotations)
#         b64 = self.get_page_base64(
#             page_number=page_number,
#             crop_boxes=crop_boxes,
#             spacing=spacing,
#             zoom=zoom,
#             page_limit=page_limit,
#             highlight_boxes=None,  # handled later
#         )

#         # annotate (highlights + grid)
#         b64 = self._annotate_base64_image(
#             base64_img=b64,
#             page_number=page_number,
#             page_limit=page_limit,
#             highlight_boxes=highlight_boxes,
#             grid=grid,
#             box_color=box_color,
#             box_width=box_width,
#             font_size=font_size,
#         )

#         display(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))

#     # --------------------------------------------------------------------- #
#     #                        GEOMETRY CONVERSION                            #
#     # --------------------------------------------------------------------- #
#     def page_bbox_to_combined(self, page_number: int, page_relative_bbox: tuple) -> tuple:
#         """Translate *page‑relative* bbox → *combined‑doc* coordinates."""
#         vp = self.virtual_page_metadata["page_bounds"][page_number]
#         off_x, off_y = vp[0], vp[1]
#         x0, y0, x1, y1 = page_relative_bbox
#         return (x0 + off_x, y0 + off_y, x1 + off_x, y1 + off_y)

#     # --------------------------------------------------------------------- #
#     #                               ITERATION                               #
#     # --------------------------------------------------------------------- #
#     def __iter__(self):
#         """Yield virtual page indices (0‑based)."""
#         yield from range(self.pages)


# class PDFModel(BaseModel):
#     path: Union[str, Path, bytes, bytearray]
#     s3_client: Optional[object] = None
#     doc: Optional[fitz.Document] = None
#     name: Optional[str] = None
#     query_index: Optional['LineTextIndex'] = None
#     meta_version: Optional[str] = None
#     meta_tag: Optional[str] = None
#     virtual_page_metadata: Optional[dict] = None
    
#     class Config:
#         arbitrary_types_allowed = True

#     @model_validator(mode="after")
#     def initialize_and_metadata(self) -> "PDFModel":
#         """Single Pydantic validator that loads, combines, indexes, and extracts metadata."""
#         # 1. Load the original multi-page PDF
#         try:
#             temp_doc = FileReader.pdf(self.path, s3_client=self.s3_client)
#             if not isinstance(temp_doc, fitz.Document):
#                 raise ValueError(f"FileReader returned {type(temp_doc)}, expected fitz.Document")
            
#             if isinstance(self.path, (str, Path)):
#                 self.name = Path(self.path).stem

#         except Exception as e:
#             raise ValueError(f"Could not load temporary PDF document: {e}")

#         # 2. Combine into a single tall document and gather virtual metadata
#         self.doc, self.virtual_page_metadata = self._combine_pages_and_get_metadata(temp_doc)
        
#         # 3. Build the "virtually aware" LineTextIndex
#         self.query_index = LineTextIndex.from_document(
#             self.doc, 
#             virtual_page_metadata=self.virtual_page_metadata
#         )
        
#         # 5. Extract Document Metadata
#         md = self.doc.metadata
#         meta_fields = {k: md.get(k) for k in ["creator", "title", "author", "subject", "keywords"]}
#         pdf_trailer = self.doc.pdf_trailer()
#         self.meta_version = pdf_trailer.get("Version") if isinstance(pdf_trailer, dict) else None
#         self.meta_tag = '|'.join(f"{k}|{''.join(v.split())}" for k, v in meta_fields.items() if v).strip()
#         temp_doc.close()
        
#         return self

#     @staticmethod
#     def _combine_pages_and_get_metadata(original_doc: fitz.Document) -> Tuple[fitz.Document, dict]:
#         """
#         Combines pages, creates virtual page metadata, and calculates the
#         translated content area for each virtual page.
#         """
#         combined_doc = fitz.open()
#         total_height, max_width = 0.0, 0.0
        
#         # Get dimensions for all original pages
#         original_page_dims = []
#         for page in original_doc:
#             dims = {"width": page.rect.width, "height": page.rect.height}
#             original_page_dims.append(dims)
#             total_height += dims["height"]
#             max_width = max(max_width, dims["width"])
    
#         # Create the combined page
#         combined_page = combined_doc.new_page(width=max_width, height=total_height)

#         virtual_page_bounds = {}
#         virtual_page_breaks = []
#         virtual_page_content_areas = {
#             "margin_bboxes": [],   # Flat list of all margin boxes in *absolute* coordinates
#             "content_bboxes": [],  # Flat list of all content boxes in *absolute* coordinates
#             "pages": {}            # Per-page dictionary with *relative* coordinates
#         }
#         current_y = 0.0
        
#         for i, page in enumerate(original_doc):
#             virtual_page_breaks.append((current_y, i))
            
#             target_rect = fitz.Rect(0, current_y, page.rect.width, current_y + page.rect.height)
#             virtual_page_bounds[i] = tuple(target_rect) 
            
#             combined_page.show_pdf_page(target_rect, original_doc, i)
    
#             # --- NEW LOGIC FOR POPULATING THE DATA STRUCTURE ---
            
#             # 1. Calculate the content area on the original page (relative coordinates)
#             local_content_area_rect = fitz.Rect()
#             for b in page.get_text("blocks"):
#                 local_content_area_rect |= b[:4]
    
#             # 2. Process only if the page has content
#             if not local_content_area_rect.is_empty:
#                 # A. Convert to a plain tuple for storing (relative coords)
#                 content_bbox_rel = tuple(local_content_area_rect)
#                 # B. Calculate margin boxes (relative coords)
#                 margin_bboxes_rel = Map.inverse_page_blocks({i: {'bboxes': [content_bbox_rel], 'page_width': page.rect.width, 'page_height': page.rect.height}})
    
#                 # C. Translate to absolute coordinates for the combined document
#                 content_bbox_abs = Map.scale_y(bbox=content_bbox_rel, y_offset=current_y)
#                 margin_bboxes_abs = [Map.scale_y(bbox=bbox, y_offset=current_y) for bbox in margin_bboxes_rel]
    
#                 # D. Populate the main dictionary
#                 virtual_page_content_areas["content_bboxes"].append(content_bbox_abs)
#                 virtual_page_content_areas["margin_bboxes"].extend(margin_bboxes_abs) # Use extend for lists
                
#                 # E. Populate the page-specific dictionary with relative data
#                 virtual_page_content_areas["pages"][i] = {
#                     'content_bbox(rel)': content_bbox_rel, 
#                     'margin_bboxes(rel)': margin_bboxes_rel
#                 }
#             else:
#                 # Handle blank pages explicitly for consistency
#                 virtual_page_content_areas["pages"][i] = {
#                     'content_bbox(rel)': None,
#                     'margin_bboxes(rel)': []
#                 }
#             # Increment the vertical offset for the next page
#             current_y += page.rect.height
        
#         # Sort breaks by y-coordinate for efficient lookup
#         virtual_page_breaks.sort()
        
#         metadata = {
#             "page_count": len(original_doc),
#             "page_bounds": virtual_page_bounds,
#             "page_breaks": virtual_page_breaks,
#             "page_content_areas": virtual_page_content_areas, 
#             "combined_doc_width": max_width,
#             "combined_doc_height": total_height,
#             "original_page_dims": original_page_dims,
#         }
        
#         return combined_doc, metadata
    
#     @property
#     def pages(self) -> int:
#         """Returns the number of virtual pages."""
#         return self.virtual_page_metadata.get("page_count", 0) if self.virtual_page_metadata else 0

#     def get_page(self, page_number: Optional[int] = None) -> fitz.Page:
#         """Always returns the combined document page."""
#         if not self.doc:
#             raise ValueError("PDF document not loaded")
#         if page_number is not None and (page_number < 0 or page_number >= self.pages):
#             raise IndexError(f"Virtual page number {page_number} out of range")
#         return self.doc[0]

#     def _render_combined_crops(self, page, crop_boxes: List, zoom: Optional[float]=1, spacing: Optional[int] = 10) -> str:
#         """
#         Version with spacing between cropped regions.
#         """
#         from PIL import Image
#         import io
        
#         mat = fitz.Matrix(zoom, zoom)
#         crop_images = []
#         total_height = 0
#         max_width = 0
        
#         # Render each crop box
#         for bbox in crop_boxes:
#             clip_rect = fitz.Rect(*bbox)
#             pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            
#             img_data = pix.tobytes("png")
#             img = Image.open(io.BytesIO(img_data))
#             crop_images.append(img)
            
#             total_height += img.height
#             max_width = max(max_width, img.width)
        
#         # Add spacing between images (except after last one)
#         if len(crop_images) > 1:
#             total_height += spacing * (len(crop_images) - 1)
        
#         # Create combined image with white background
#         combined = Image.new('RGB', (max_width, total_height), 'white')
        
#         # Paste each crop with spacing
#         y_offset = 0
#         for i, img in enumerate(crop_images):
#             combined.paste(img, (0, y_offset))
#             y_offset += img.height
#             if i < len(crop_images) - 1:  # Add spacing except after last image
#                 y_offset += spacing
        
#         # Convert back to base64
#         buffer = io.BytesIO()
#         combined.save(buffer, format='PNG')
#         return base64.b64encode(buffer.getvalue()).decode("utf-8")

#     def _filter_by_page_limit(self, page_limit: int, highlight_boxes=None, crop_boxes=None):
#         """
#         Unified function to filter all bbox-based inputs by page limit.
        
#         Args:
#             page_limit: Maximum virtual page number to include
#             highlight_boxes: Optional highlight boxes (list or dict format)
#             crop_boxes: Optional crop boxes (list format)
        
#         Returns:
#             Tuple of (filtered_highlight_boxes, filtered_crop_boxes)
#         """
#         if not hasattr(self, 'query_index') or not hasattr(self.query_index, '_get_virtual_page_num'):
#             # Fallback: return original inputs if we can't determine page numbers
#             return highlight_boxes, crop_boxes
        
#         def is_bbox_within_limit(bbox):
#             """Helper to check if a bbox is within the page limit."""
#             page_num = self.query_index._get_virtual_page_num(bbox[1])  # y0 coordinate
#             return page_num <= page_limit
        
#         # Filter crop_boxes
#         filtered_crop_boxes = None
#         if crop_boxes:
#             filtered_crop_boxes = [bbox for bbox in crop_boxes if is_bbox_within_limit(bbox)]
        
#         # Filter highlight_boxes
#         filtered_highlight_boxes = None
#         if highlight_boxes:
#             if isinstance(highlight_boxes, list):
#                 filtered_highlight_boxes = [bbox for bbox in highlight_boxes if is_bbox_within_limit(bbox)]
            
#             elif isinstance(highlight_boxes, dict):
#                 filtered_highlight_boxes = {}
#                 for label, data in highlight_boxes.items():
#                     boxes = data.get("boxes", [])
#                     filtered_boxes = [bbox for bbox in boxes if is_bbox_within_limit(bbox)]
                    
#                     if filtered_boxes:  # Only include groups that have boxes within the limit
#                         filtered_highlight_boxes[label] = {**data, "boxes": filtered_boxes}
#             else:
#                 filtered_highlight_boxes = highlight_boxes
        
#         return filtered_highlight_boxes, filtered_crop_boxes
        

#     def get_page_base64(self, page_number: Optional[int] = None, zoom: Optional[float] = 1, 
#                        crop_boxes: Optional[List] = None, spacing: Optional[int] = 10, 
#                        page_limit: Optional[int] = None,
#                        highlight_boxes: Optional[Union[List[tuple], Dict]] = None,
#                        box_color: str = "red", box_width: int = 2, font_size: int = 12) -> str:
#         """
#         Renders a virtual page or entire document as base64 with optional annotations.
        
#         Args:
#             page_number: Virtual page number to render (None for entire document)
#             zoom: Zoom factor for rendering
#             crop_boxes: Optional list of bboxes to crop and combine into one image
#                        Format: [(x0, y0, x1, y1), ...]
#             spacing: Pixels of white space between cropped regions
#             page_limit: Maximum virtual page number to include (stops at this page)
#             highlight_boxes: Optional highlight boxes to draw on the image (same format as show())
#             box_color: Default color for highlight boxes
#             box_width: Width of highlight box outlines
#             font_size: Font size for highlight box labels
        
#         Returns:
#             Base64 encoded PNG image with annotations
#         """
#         page = self.get_page(page_number)
        
#         # Apply page_limit filtering at the start
#         if page_limit is not None:
#             highlight_boxes, crop_boxes = self._filter_by_page_limit(
#                 page_limit, 
#                 highlight_boxes=highlight_boxes, 
#                 crop_boxes=crop_boxes
#             )
        
#         # If crop_boxes provided, render each crop and combine them
#         if crop_boxes:
#             base64_img = self._render_combined_crops(page, crop_boxes, zoom, spacing)
#         else:
#             # Original logic for single page/document rendering
#             clip_rect = None
#             if page_number is not None:
#                 # Clip to the specific virtual page bounds
#                 page_bounds = self.virtual_page_metadata["page_bounds"].get(page_number)
#                 if not page_bounds:
#                     raise KeyError(f"Bounding box for virtual page {page_number} not found.")
#                 clip_rect = fitz.Rect(*page_bounds)
#             else:
#                 # Handle page_limit for entire document rendering
#                 if page_limit is not None:
#                     clip_rect = self._get_document_bounds_with_limit(page_limit)
            
#             mat = fitz.Matrix(zoom, zoom)
#             pix = page.get_pixmap(matrix=mat, clip=clip_rect)
#             img_bytes = pix.tobytes("png")
#             base64_img = base64.b64encode(img_bytes).decode("utf-8")
        
#         # If highlight_boxes provided, draw them on the image
#         if highlight_boxes:
#             base64_img = self._add_highlight_boxes_to_base64(
#                 base64_img, highlight_boxes, page_number, page_limit, 
#                 box_color, box_width, font_size, zoom
#             )
        
#         return base64_img
    
    
#     def _add_highlight_boxes_to_base64(self, base64_img: str, highlight_boxes, page_number, 
#                                       page_limit, box_color, box_width, font_size, zoom) -> str:
#         """
#         Draws highlight boxes on a base64 image and returns the annotated base64 image.
#         """
#         from PIL import Image, ImageDraw, ImageFont
#         import io
#         import base64
        
#         # Convert base64 to PIL Image
#         img_bytes = base64.b64decode(base64_img)
#         img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
#         draw = ImageDraw.Draw(img)
        
#         # Load font
#         try:
#             font = ImageFont.truetype("Arial.ttf", font_size)
#         except IOError:
#             font = ImageFont.load_default()
        
#         # Calculate coordinate transformation (same logic as show())
#         if page_number is not None:
#             vp_bounds = self.virtual_page_metadata["page_bounds"][page_number]
#             offset_x, offset_y = vp_bounds[0], vp_bounds[1]
#             view_width, view_height = vp_bounds[2] - offset_x, vp_bounds[3] - offset_y
#         else:
#             offset_x, offset_y = 0, 0
#             if page_limit is not None:
#                 view_width, view_height = self._get_limited_view_dimensions(page_limit)
#             else:
#                 view_width = self.virtual_page_metadata["combined_doc_width"]
#                 view_height = self.virtual_page_metadata["combined_doc_height"]
        
#         if view_width == 0 or view_height == 0:
#             return base64_img  # Return original if no valid dimensions
        
#         scale_x = img.width / view_width
#         scale_y = img.height / view_height
        
#         # Draw highlight boxes (same logic as show())
#         if isinstance(highlight_boxes, dict):
#             for label, data in highlight_boxes.items():
#                 color = data.get("color", box_color)
#                 boxes = data.get("boxes", [])
                
#                 for individual_box in boxes:
#                     x0, y0, x1, y1 = individual_box
                    
#                     # Draw the bounding box
#                     box_in_pixels = [
#                         (x0 - offset_x) * scale_x,
#                         (y0 - offset_y) * scale_y,
#                         (x1 - offset_x) * scale_x,
#                         (y1 - offset_y) * scale_y,
#                     ]
#                     if not (box_in_pixels[2] > box_in_pixels[0] and box_in_pixels[3] > box_in_pixels[1]):
#                         continue
                    
#                     draw.rectangle(box_in_pixels, outline=color, width=box_width)
    
#                     # Draw the label
#                     px0, py0 = box_in_pixels[0], box_in_pixels[1]
#                     text_anchor = (px0, py0 - box_width)
#                     text_bbox = draw.textbbox(text_anchor, label, font=font, anchor="lb")
                    
#                     bg_padding = 3
#                     bg_rect = (
#                         text_bbox[0] - bg_padding,
#                         text_bbox[1] - bg_padding,
#                         text_bbox[2] + bg_padding,
#                         text_bbox[3] + bg_padding
#                     )
                    
#                     draw.rectangle(bg_rect, fill=color)
#                     draw.text(text_anchor, label, font=font, fill="white", anchor="lb")
    
#         elif isinstance(highlight_boxes, list):
#             for x0, y0, x1, y1 in highlight_boxes:
#                 box_in_pixels = [
#                     (x0 - offset_x) * scale_x,
#                     (y0 - offset_y) * scale_y,
#                     (x1 - offset_x) * scale_x,
#                     (y1 - offset_y) * scale_y,
#                 ]
#                 if (box_in_pixels[2] > box_in_pixels[0] and box_in_pixels[3] > box_in_pixels[1]):
#                     draw.rectangle(box_in_pixels, outline=box_color, width=box_width)
        
#         # Convert back to base64
#         buffer = io.BytesIO()
#         img.save(buffer, format='PNG')
#         return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    
#     def _get_limited_view_dimensions(self, page_limit: int) -> tuple:
#         """Gets view dimensions up to the specified page limit."""
#         page_bounds = self.virtual_page_metadata.get("page_bounds", {})
        
#         max_x = 0
#         max_y = 0
        
#         for page_num, bounds in page_bounds.items():
#             if page_num <= page_limit:
#                 x0, y0, x1, y1 = bounds
#                 max_x = max(max_x, x1)
#                 max_y = max(max_y, y1)
        
#         return max_x, max_y
    
    
#     def _get_document_bounds_with_limit(self, page_limit: int) -> Optional[fitz.Rect]:
#         """Gets document bounds up to the specified page limit."""
#         page_bounds = self.virtual_page_metadata.get("page_bounds", {})
        
#         if not page_bounds or page_limit < 0:
#             return None
        
#         max_y = 0
#         max_x = 0
        
#         for page_num, bounds in page_bounds.items():
#             if page_num <= page_limit:
#                 x0, y0, x1, y1 = bounds
#                 max_x = max(max_x, x1)
#                 max_y = max(max_y, y1)
        
#         if max_y > 0:
#             return fitz.Rect(0, 0, max_x, max_y)
        
#         return None
    
    
#     def show(
#         self, 
#         page_number: Optional[int] = None, 
#         highlight_boxes: Optional[Union[List[tuple], Dict]] = None, 
#         crop_boxes: Optional[List] = None,
#         box_color: str = "red", 
#         box_width: int = 2,
#         font_size: int = 12,
#         grid=False,
#         zoom: Optional[float]=1, 
#         spacing: Optional[int] = 10,
#         page_limit: Optional[int] = None
#     ):
#         """
#         Displays a rendered PDF page with optional highlight boxes.
    
#         Args:
#             page_number (Optional[int]): The virtual page to display. If None, displays the entire document.
#             highlight_boxes (Optional[Union[List[tuple], Dict]]): Boxes to draw.
#             crop_boxes (Optional[List]): Optional list of bboxes to crop and combine
#             box_color (str): The default color for boxes if not specified in the dict schema.
#             box_width (int): The width of the box outlines.
#             font_size (int): The font size for labels when using the dict schema.
#             grid (bool): Whether to show y-coordinate grid lines
#             zoom (Optional[float]): Zoom factor for rendering
#             spacing (Optional[int]): Pixels of white space between cropped regions
#             page_limit (Optional[int]): Maximum virtual page number to include (stops at this page)
#         """
#         if not _IPYTHON_AVAILABLE:
#             print("IPython/Jupyter is required to display images.")
#             return
    
#         # UNIFIED FILTERING: Apply page_limit to all inputs at the start
#         if page_limit is not None:
#             highlight_boxes, crop_boxes = self._filter_by_page_limit(
#                 page_limit, 
#                 highlight_boxes=highlight_boxes, 
#                 crop_boxes=crop_boxes
#             )
    
#         img_base64 = self.get_page_base64(
#             page_number=page_number, 
#             crop_boxes=crop_boxes, 
#             spacing=spacing, 
#             zoom=zoom,
#             page_limit=page_limit  # Still pass it for document bounds calculation
#         )
#         img_bytes = base64.b64decode(img_base64)
#         img = Image.open(BytesIO(img_bytes)).convert("RGB")
#         draw = ImageDraw.Draw(img)
    
#         try:
#             font = ImageFont.truetype("Arial.ttf", font_size)
#         except IOError:
#             font = ImageFont.load_default()
    
#         if page_number is not None:
#             vp_bounds = self.virtual_page_metadata["page_bounds"][page_number]
#             offset_x, offset_y = vp_bounds[0], vp_bounds[1]
#             view_width, view_height = vp_bounds[2] - offset_x, vp_bounds[3] - offset_y
#         else:
#             offset_x, offset_y = 0, 0
#             if page_limit is not None:
#                 # Calculate view dimensions based on page_limit
#                 view_width, view_height = self._get_limited_view_dimensions(page_limit)
#             else:
#                 view_width = self.virtual_page_metadata["combined_doc_width"]
#                 view_height = self.virtual_page_metadata["combined_doc_height"]
    
#         if view_width == 0 or view_height == 0:
#             display(img)
#             return
    
#         scale_x = img.width / view_width
#         scale_y = img.height / view_height
    
#         # Handle highlight boxes (already filtered by this point)
#         if isinstance(highlight_boxes, dict):
#             for label, data in highlight_boxes.items():
#                 color = data.get("color", box_color)
#                 boxes = data.get("boxes", [])
                
#                 for individual_box in boxes:
#                     x0, y0, x1, y1 = individual_box
                    
#                     # Draw the bounding box
#                     box_in_pixels = [
#                         (x0 - offset_x) * scale_x,
#                         (y0 - offset_y) * scale_y,
#                         (x1 - offset_x) * scale_x,
#                         (y1 - offset_y) * scale_y,
#                     ]
#                     if not (box_in_pixels[2] > box_in_pixels[0] and box_in_pixels[3] > box_in_pixels[1]):
#                         continue
                    
#                     draw.rectangle(box_in_pixels, outline=color, width=box_width)
    
#                     # Draw the label
#                     px0, py0 = box_in_pixels[0], box_in_pixels[1]
#                     text_anchor = (px0, py0 - box_width)
#                     text_bbox = draw.textbbox(text_anchor, label, font=font, anchor="lb")
                    
#                     bg_padding = 3
#                     bg_rect = (
#                         text_bbox[0] - bg_padding,
#                         text_bbox[1] - bg_padding,
#                         text_bbox[2] + bg_padding,
#                         text_bbox[3] + bg_padding
#                     )
                    
#                     draw.rectangle(bg_rect, fill=color)
#                     draw.text(text_anchor, label, font=font, fill="white", anchor="lb")
    
#         elif isinstance(highlight_boxes, list):
#             for x0, y0, x1, y1 in highlight_boxes:
#                 box_in_pixels = [
#                     (x0 - offset_x) * scale_x,
#                     (y0 - offset_y) * scale_y,
#                     (x1 - offset_x) * scale_x,
#                     (y1 - offset_y) * scale_y,
#                 ]
#                 if (box_in_pixels[2] > box_in_pixels[0] and box_in_pixels[3] > box_in_pixels[1]):
#                     draw.rectangle(box_in_pixels, outline=box_color, width=box_width)
    
#         # Grid logic (unchanged)
#         if grid:
#             try:
#                 grid_font_size = max(10, int(font_size * 0.8))
#                 grid_font = ImageFont.truetype("Arial.ttf", grid_font_size)
#             except IOError:
#                 grid_font = ImageFont.load_default()
            
#             grid_color = (255, 20, 147, 150)
#             start_y_pdf = offset_y
#             end_y_pdf = offset_y + view_height
#             tick_y_pdf = (int(start_y_pdf) // 100) * 100
            
#             while tick_y_pdf < end_y_pdf:
#                 if tick_y_pdf >= start_y_pdf:
#                     y_pixel = (tick_y_pdf - offset_y) * scale_y
#                     draw.line([(0, y_pixel), (img.width, y_pixel)], fill=grid_color, width=1)
                    
#                     label_text = str(tick_y_pdf)
#                     text_anchor = (5, y_pixel + 2)
#                     text_bbox = draw.textbbox(text_anchor, label_text, font=grid_font, anchor="lt")
#                     draw.rectangle(text_bbox, fill="white")
#                     draw.text(text_anchor, label_text, font=grid_font, fill="black", anchor="lt")
#                 tick_y_pdf += 100
        
#         display(img)
            
#     def __iter__(self):
#         """Iterates through virtual pages."""
#         for i in range(self.pages):
#             yield i
            
#     def page_bbox_to_combined(self, page_number: int, page_relative_bbox: tuple) -> tuple:
#         """Converts a page-relative bbox to combined document coordinates."""
#         if not self.virtual_page_metadata:
#             raise ValueError("Document not initialized")

#         vp_bounds = self.virtual_page_metadata["page_bounds"][page_number]
#         offset_x, offset_y = vp_bounds[0], vp_bounds[1]
        
#         pr_x0, pr_y0, pr_x1, pr_y1 = page_relative_bbox
        
#         return (pr_x0 + offset_x, pr_y0 + offset_y, pr_x1 + offset_x, pr_y1 + offset_y)


# please update this function to eliminate all duplicated or unused logic. DO not impact existing functionality. 

# ALWAYS return FULL updated code and script, do not abbreviate.

# class PDFModel(BaseModel):
#     """
#     High‑level container that:
#     1. Loads an input PDF (local path / bytes / S3) via `FileReader`.
#     2. Stitches all pages vertically into *one* tall page (“combined doc”).
#     3. Builds a virtual‑page index (`LineTextIndex`) that understands page breaks.
#     4. Offers rich helpers for rendering, cropping, and annotation.
#     """

#     # ----------------------- Core public attributes ------------------------ #
#     path: Union[str, Path, bytes, bytearray]
#     s3_client: Optional[object] = None

#     # Populated automatically by the validator
#     doc: Optional[fitz.Document] = None
#     name: Optional[str] = None
#     query_index: Optional["LineTextIndex"] = None
#     meta_version: Optional[str] = None
#     meta_tag: Optional[str] = None
#     virtual_page_metadata: Optional[dict] = None

#     class Config:
#         arbitrary_types_allowed = True

#     # --------------------------------------------------------------------- #
#     #                     INITIALISATION & METADATA                         #
#     # --------------------------------------------------------------------- #
#     @model_validator(mode="after")
#     def _initialise_and_index(self) -> "PDFModel":
#         """Load, stitch, index and harvest metadata in a *single* pass."""
#         # 1. Load original multi‑page PDF via project reader
#         try:
#             original_doc = FileReader.pdf(self.path, s3_client=self.s3_client)
#             if not isinstance(original_doc, fitz.Document):
#                 raise TypeError(
#                     f"FileReader returned {type(original_doc)}, expected fitz.Document"
#                 )
#             if isinstance(self.path, (str, Path)):
#                 self.name = Path(self.path).stem
#         except Exception as exc:
#             raise ValueError(f"Could not load PDF: {exc}") from exc

#         # 2. Build combined document (+ virtual‑page metadata)
#         self.doc, self.virtual_page_metadata = self._combine_pages_and_get_metadata(
#             original_doc
#         )

#         # 3. Construct virtual‑aware text index
#         self.query_index = LineTextIndex.from_document(
#             self.doc, virtual_page_metadata=self.virtual_page_metadata
#         )

#         # 4. Extract key PDF metadata
#         trailer = self.doc.pdf_trailer()
#         self.meta_version = trailer.get("Version") if isinstance(trailer, dict) else None
#         md = self.doc.metadata
#         wanted = ("creator", "title", "author", "subject", "keywords")
#         self.meta_tag = "|".join(f"{k}|{''.join(md[k].split())}" for k in wanted if md.get(k))

#         original_doc.close()
#         return self

#     # --------------------------------------------------------------------- #
#     #                       INTERNAL BUILD HELPERS                          #
#     # --------------------------------------------------------------------- #
#     @staticmethod
#     def _combine_pages_and_get_metadata(
#         original_doc: fitz.Document,
#     ) -> Tuple[fitz.Document, dict]:
#         """Return a vertically‑stitched document + rich per‑page metadata."""
#         combined = fitz.open()
#         total_h = 0.0
#         max_w = 0.0

#         page_dims: List[dict] = []
#         for pg in original_doc:
#             dims = {"width": pg.rect.width, "height": pg.rect.height}
#             page_dims.append(dims)
#             total_h += dims["height"]
#             max_w = max(max_w, dims["width"])

#         combined_page = combined.new_page(width=max_w, height=total_h)

#         vpage_bounds: Dict[int, Tuple[float, float, float, float]] = {}
#         vpage_breaks: List[Tuple[float, int]] = []
#         content_areas = {
#             "margin_bboxes": [],
#             "content_bboxes": [],
#             "pages": {},
#         }

#         y_offset = 0.0
#         for i, pg in enumerate(original_doc):
#             vpage_breaks.append((y_offset, i))
#             tgt_rect = fitz.Rect(0, y_offset, pg.rect.width, y_offset + pg.rect.height)
#             vpage_bounds[i] = tuple(tgt_rect)
#             combined_page.show_pdf_page(tgt_rect, original_doc, i)

#             # detect real content area on original page
#             local_content = fitz.Rect()
#             for b in pg.get_text("blocks"):
#                 local_content |= b[:4]

#             if not local_content.is_empty:
#                 # relative & absolute bboxes
#                 content_rel = tuple(local_content)
#                 margin_rel = Map.inverse_page_blocks(
#                     {
#                         i: {
#                             "bboxes": [content_rel],
#                             "page_width": pg.rect.width,
#                             "page_height": pg.rect.height,
#                         }
#                     }
#                 )
#                 content_abs = Map.scale_y(content_rel, y_offset)
#                 margin_abs = [Map.scale_y(b, y_offset) for b in margin_rel]

#                 content_areas["content_bboxes"].append(content_abs)
#                 content_areas["margin_bboxes"].extend(margin_abs)
#                 content_areas["pages"][i] = {
#                     "content_bbox(rel)": content_rel,
#                     "margin_bboxes(rel)": margin_rel,
#                 }
#             else:  # blank page
#                 content_areas["pages"][i] = {
#                     "content_bbox(rel)": None,
#                     "margin_bboxes(rel)": [],
#                 }

#             y_offset += pg.rect.height

#         vpage_breaks.sort()
#         metadata = {
#             "page_count": len(original_doc),
#             "page_bounds": vpage_bounds,
#             "page_breaks": vpage_breaks,
#             "page_content_areas": content_areas,
#             "combined_doc_width": max_w,
#             "combined_doc_height": total_h,
#             "original_page_dims": page_dims,
#         }
#         return combined, metadata

#     # --------------------------------------------------------------------- #
#     #                          PUBLIC PROPERTIES                            #
#     # --------------------------------------------------------------------- #
#     @property
#     def pages(self) -> int:
#         """Total *virtual* pages."""
#         return self.virtual_page_metadata.get("page_count", 0) if self.virtual_page_metadata else 0

#     # --------------------------------------------------------------------- #
#     #                         INTERNAL UTILITIES                            #
#     # --------------------------------------------------------------------- #
#     def _filter_by_page_limit(
#         self,
#         page_limit: int,
#         *,
#         highlight_boxes: Optional[Union[List[tuple], Dict]] = None,
#         crop_boxes: Optional[List] = None,
#     ) -> Tuple[Optional[Union[List[tuple], Dict]], Optional[List]]:
#         """
#         Apply `page_limit` to crop‑ and highlight‑inputs in a single pass.
#         """
#         if not hasattr(self, "query_index") or not hasattr(self.query_index, "_get_virtual_page_num"):
#             return highlight_boxes, crop_boxes

#         def within(bbox):
#             return self.query_index._get_virtual_page_num(bbox[1]) <= page_limit

#         filt_crop = [b for b in (crop_boxes or []) if within(b)] if crop_boxes else None

#         if isinstance(highlight_boxes, list):
#             filt_high = [b for b in highlight_boxes if within(b)]
#         elif isinstance(highlight_boxes, dict):
#             filt_high = {
#                 k: {**d, "boxes": [b for b in d.get("boxes", []) if within(b)]}
#                 for k, d in highlight_boxes.items()
#             }
#             filt_high = {k: v for k, v in filt_high.items() if v["boxes"]}
#         else:
#             filt_high = highlight_boxes  # None or unsupported type untouched

#         return filt_high, filt_crop

#     # --------------- Shared drawing / annotation helper ------------------ #
#     def _annotate_base64_image(
#         self,
#         *,
#         base64_img: str,
#         page_number: Optional[int],
#         page_limit: Optional[int],
#         highlight_boxes: Optional[Union[List[tuple], Dict]],
#         grid: bool,
#         box_color: str,
#         box_width: int,
#         font_size: int,
#     ) -> str:
#         """Overlay highlight boxes (and optionally a y‑grid) on a base64 image."""
#         if not highlight_boxes and not grid:
#             return base64_img

#         img_raw = base64.b64decode(base64_img)
#         img = Image.open(BytesIO(img_raw)).convert("RGB")
#         draw = ImageDraw.Draw(img)

#         # font
#         try:
#             font = ImageFont.truetype("Arial.ttf", font_size)
#         except IOError:
#             font = ImageFont.load_default()

#         # --------------------- coordinate transforms ---------------------- #
#         if page_number is not None:
#             x0, y0, x1, y1 = self.virtual_page_metadata["page_bounds"][page_number]
#             view_w, view_h = x1 - x0, y1 - y0
#             off_x, off_y = x0, y0
#         else:
#             off_x = off_y = 0
#             if page_limit is not None:
#                 view_w, view_h = self._get_limited_view_dimensions(page_limit)
#             else:
#                 view_w = self.virtual_page_metadata["combined_doc_width"]
#                 view_h = self.virtual_page_metadata["combined_doc_height"]

#         if view_w == 0 or view_h == 0:  # defensive
#             return base64_img

#         sx = img.width / view_w
#         sy = img.height / view_h

#         # ------------------------- Draw highlights ------------------------ #
#         def draw_bbox(rect, clr):
#             rx0, ry0, rx1, ry1 = rect
#             px = [(rx0 - off_x) * sx, (ry0 - off_y) * sy, (rx1 - off_x) * sx, (ry1 - off_y) * sy]
#             if px[2] > px[0] and px[3] > px[1]:
#                 draw.rectangle(px, outline=clr, width=box_width)
#             return px

#         if isinstance(highlight_boxes, dict):
#             for lbl, data in (highlight_boxes or {}).items():
#                 col = data.get("color", box_color)
#                 for b in data.get("boxes", []):
#                     px_rect = draw_bbox(b, col)
#                     # label background & text
#                     ax, ay = px_rect[0], px_rect[1]
#                     txt_bbox = draw.textbbox((ax, ay - box_width), lbl, font=font, anchor="lb")
#                     pad = 3
#                     bg = (
#                         txt_bbox[0] - pad,
#                         txt_bbox[1] - pad,
#                         txt_bbox[2] + pad,
#                         txt_bbox[3] + pad,
#                     )
#                     draw.rectangle(bg, fill=col)
#                     draw.text((ax, ay - box_width), lbl, font=font, fill="white", anchor="lb")
#         elif isinstance(highlight_boxes, list):
#             for b in highlight_boxes:
#                 draw_bbox(b, box_color)

#         # --------------------------- Optional grid ------------------------ #
#         if grid:
#             try:
#                 gfont = ImageFont.truetype("Arial.ttf", max(10, int(font_size * 0.8)))
#             except IOError:
#                 gfont = ImageFont.load_default()

#             grid_color = (255, 20, 147, 150)  # semi‑transparent deep‑pink
#             y_pdf = (int(off_y) // 100) * 100
#             while y_pdf < off_y + view_h:
#                 if y_pdf >= off_y:
#                     y_pix = (y_pdf - off_y) * sy
#                     draw.line([(0, y_pix), (img.width, y_pix)], fill=grid_color, width=1)
#                     label = str(int(y_pdf))
#                     tb = draw.textbbox((5, y_pix + 2), label, font=gfont, anchor="lt")
#                     draw.rectangle(tb, fill="white")
#                     draw.text((5, y_pix + 2), label, font=gfont, fill="black", anchor="lt")
#                 y_pdf += 100

#         # re‑encode
#         out = BytesIO()
#         img.save(out, format="PNG")
#         return base64.b64encode(out.getvalue()).decode("utf-8")

#     # Retain legacy name for backward compatibility  ------------------- #
#     def _add_highlight_boxes_to_base64(  # noqa: N802  (legacy API)
#         self,
#         base64_img: str,
#         highlight_boxes,
#         page_number,
#         page_limit,
#         box_color,
#         box_width,
#         font_size,
#         zoom,
#     ) -> str:  # pylint: disable=too-many-arguments
#         return self._annotate_base64_image(
#             base64_img=base64_img,
#             page_number=page_number,
#             page_limit=page_limit,
#             highlight_boxes=highlight_boxes,
#             grid=False,
#             box_color=box_color,
#             box_width=box_width,
#             font_size=font_size,
#         )

#     # --------------------------------------------------------------------- #
#     #                              RENDERING                                #
#     # --------------------------------------------------------------------- #
#     def get_page(self, page_number: Optional[int] = None) -> fitz.Page:
#         """Always returns *combined* page (index 0) after bound checking."""
#         if not self.doc:
#             raise RuntimeError("PDF document not loaded")
#         if page_number is not None and (page_number < 0 or page_number >= self.pages):
#             raise IndexError(f"Virtual page {page_number} out of range")
#         return self.doc[0]

#     # ------------ Core low‑level image generation (no annotations) -------- #
#     def _render_combined_crops(
#         self,
#         page: fitz.Page,
#         crop_boxes: List[Tuple[float, float, float, float]],
#         zoom: float = 1.0,
#         spacing: int = 10,
#     ) -> str:
#         """Render *multiple* crops then stack vertically with white spacer."""
#         mat = fitz.Matrix(zoom, zoom)

#         rendered: List[Image.Image] = []
#         full_h = 0
#         max_w = 0

#         for box in crop_boxes:
#             pix = page.get_pixmap(matrix=mat, clip=fitz.Rect(*box))
#             img = Image.open(BytesIO(pix.tobytes("png")))
#             rendered.append(img)
#             full_h += img.height
#             max_w = max(max_w, img.width)

#         if len(rendered) > 1:
#             full_h += spacing * (len(rendered) - 1)

#         canvas = Image.new("RGB", (max_w, full_h), "white")
#         y = 0
#         for i, im in enumerate(rendered):
#             canvas.paste(im, (0, y))
#             y += im.height + (spacing if i < len(rendered) - 1 else 0)

#         out = BytesIO()
#         canvas.save(out, format="PNG")
#         return base64.b64encode(out.getvalue()).decode("utf-8")

#     def _get_limited_view_dimensions(self, page_limit: int) -> Tuple[float, float]:
#         """Dimensions (w,h) up to `page_limit`."""
#         w = h = 0.0
#         for pn, (x0, y0, x1, y1) in self.virtual_page_metadata.get("page_bounds", {}).items():
#             if pn <= page_limit:
#                 w = max(w, x1)
#                 h = max(h, y1)
#         return w, h

#     def _get_document_bounds_with_limit(self, page_limit: int) -> Optional[fitz.Rect]:
#         """Bounding rect (0,0,x_max,y_max) limited to given virtual page."""
#         w, h = self._get_limited_view_dimensions(page_limit)
#         return fitz.Rect(0, 0, w, h) if h > 0 else None

#     # --------------------------- Public helpers --------------------------- #
#     def get_page_base64(
#         self,
#         *,
#         page_number: Optional[int] = None,
#         zoom: float = 1.0,
#         crop_boxes: Optional[List] = None,
#         spacing: int = 10,
#         page_limit: Optional[int] = None,
#         highlight_boxes: Optional[Union[List[tuple], Dict]] = None,
#         box_color: str = "red",
#         box_width: int = 2,
#         font_size: int = 12,
#     ) -> str:
#         """
#         Return rendered PNG (base64) for:
#         * a single virtual page,
#         * full combined doc,
#         * or combined custom crops.

#         Optional highlight‑box and page‑limit support.
#         """
#         page = self.get_page(page_number)

#         # filter inputs by page_limit first
#         if page_limit is not None:
#             highlight_boxes, crop_boxes = self._filter_by_page_limit(
#                 page_limit, highlight_boxes=highlight_boxes, crop_boxes=crop_boxes
#             )

#         if crop_boxes:
#             b64 = self._render_combined_crops(page, crop_boxes, zoom, spacing)
#         else:
#             clip_rect = None
#             if page_number is not None:
#                 clip_rect = fitz.Rect(*self.virtual_page_metadata["page_bounds"][page_number])
#             elif page_limit is not None:
#                 clip_rect = self._get_document_bounds_with_limit(page_limit)

#             pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip_rect)
#             b64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")

#         # overlay highlights if requested
#         b64 = self._annotate_base64_image(
#             base64_img=b64,
#             page_number=page_number,
#             page_limit=page_limit,
#             highlight_boxes=highlight_boxes,
#             grid=False,
#             box_color=box_color,
#             box_width=box_width,
#             font_size=font_size,
#         )
#         return b64

#     # --------------------------------------------------------------------- #
#     #                              DISPLAY                                  #
#     # --------------------------------------------------------------------- #
#     def show(
#         self,
#         page_number: Optional[int] = None,
#         highlight_boxes: Optional[Union[List[tuple], Dict]] = None,
#         crop_boxes: Optional[List] = None,
#         box_color: str = "red",
#         box_width: int = 2,
#         font_size: int = 12,
#         grid: bool = False,
#         zoom: float = 1.0,
#         spacing: int = 10,
#         page_limit: Optional[int] = None,
#     ):
#         """Render in‑notebook image with optional highlights/grid."""
#         if not _IPYTHON_AVAILABLE:
#             print("IPython/Jupyter is required to display images.")
#             return

#         # page‑limit filtering
#         if page_limit is not None:
#             highlight_boxes, crop_boxes = self._filter_by_page_limit(
#                 page_limit, highlight_boxes=highlight_boxes, crop_boxes=crop_boxes
#             )

#         # raw render (no annotations)
#         b64 = self.get_page_base64(
#             page_number=page_number,
#             crop_boxes=crop_boxes,
#             spacing=spacing,
#             zoom=zoom,
#             page_limit=page_limit,
#             highlight_boxes=None,  # handled later
#         )

#         # annotate (highlights + grid)
#         b64 = self._annotate_base64_image(
#             base64_img=b64,
#             page_number=page_number,
#             page_limit=page_limit,
#             highlight_boxes=highlight_boxes,
#             grid=grid,
#             box_color=box_color,
#             box_width=box_width,
#             font_size=font_size,
#         )

#         display(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"))

#     # --------------------------------------------------------------------- #
#     #                        GEOMETRY CONVERSION                            #
#     # --------------------------------------------------------------------- #
#     def page_bbox_to_combined(self, page_number: int, page_relative_bbox: tuple) -> tuple:
#         """Translate *page‑relative* bbox → *combined‑doc* coordinates."""
#         vp = self.virtual_page_metadata["page_bounds"][page_number]
#         off_x, off_y = vp[0], vp[1]
#         x0, y0, x1, y1 = page_relative_bbox
#         return (x0 + off_x, y0 + off_y, x1 + off_x, y1 + off_y)

#     # --------------------------------------------------------------------- #
#     #                               ITERATION                               #
#     # --------------------------------------------------------------------- #
#     def __iter__(self):
#         """Yield virtual page indices (0‑based)."""
#         yield from range(self.pages)