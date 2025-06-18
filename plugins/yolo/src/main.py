from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Any, Tuple
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException, Form, File, Depends
from pydantic import BaseModel, Field

# from tableai.pdf.pdf_model import PDFModel
# Local inference - models have been moved inside this module
from .inference import YOLOTableDetector, YOLODetectionResult, PageDimensions

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

from tableai.pdf.tools import (
    FitzSearchIndex
)
from tableai.pdf.coordinates import (
    Geometry,
    CoordinateMapping
)
from tableai.pdf.pdf_page import VirtualPageManager
from tableai.pdf.generic_params import (
    TextNormalizer, 
    WhitespaceGenerator
)
from tableai.readers.files import FileReader
__all__ = ["PDFModel"]

try:
    from IPython.display import display
    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False


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

    def img_bbox_to_page_bbox(
        self,
        bbox_px: tuple[int, int, int, int],
        image_dims: tuple[int, int],
        page_number: int,
        zoom: float = 2.0
    ) -> tuple[float, float, float, float]:
        """
        Convert a YOLO box expressed in *render-image* pixels back to PDF
        *page* coordinates.

        `page_to_numpy()` renders with the same `zoom`, so scaling is:
            pdf_x = px_x * (page_width / img_width)
            pdf_y = px_y * (page_height / img_height)
        """
        x1_px, y1_px, x2_px, y2_px = bbox_px
        img_w, img_h = image_dims
        page_w=self.vpm.combined_doc_width
        page_h=self.vpm.combined_doc_height
        sx, sy = page_w / img_w, page_h / img_h

        return (x1_px * sx, y1_px * sy, x2_px * sx, y2_px * sy)

    # --------------------------------------------------------------------- #
    #                               ITERATION                               #
    # --------------------------------------------------------------------- #
    def __iter__(self):
        """Yield virtual page indices (0‑based)."""
        yield from range(self.pages)

# ────────────────────────────────────────────────────────────────
# FastAPI setup
# ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="TableAI YOLO Detection Service",
    description="Standalone YOLO-based table detection service",
    version="0.3.2"  # Version bump for the fix
)

_detector_cache: dict[str, YOLOTableDetector] = {}

# ────────────────────────────────────────────────────────────────
# Pydantic Models for API Layer
# ────────────────────────────────────────────────────────────────

#
# >>> PngDetectionParams class is REMOVED. It was the source of the problem. <<<
#

class DetectionResponse(BaseModel):
    # This model is now for error responses
    success: bool
    detections: List = []
    page_dimensions: Optional[Dict[int, PageDimensions]] = None
    page_count: Optional[int] = None
    coordinate_system: str = "pdf_points"
    model_used: str = ""
    message: str = ""

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    cached_models: List[str]

# ... (Utility helpers like _get_detector and the /health route remain the same) ...

def _get_detector(model_type: str) -> YOLOTableDetector:
    available_models = YOLOTableDetector.list_available_models()
    if model_type not in available_models:
        raise HTTPException(status_code=400, detail=f"Invalid model_type '{model_type}'. Valid options: {list(available_models.keys())}")
    if model_type not in _detector_cache:
        _detector_cache[model_type] = YOLOTableDetector()
    return _detector_cache[model_type]

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", service="tableai-yolo-service", version="0.3.2", cached_models=list(_detector_cache.keys()))

# ─────────────────── THE FIX IS IN THIS FUNCTION SIGNATURE ───────────────────────
@app.post(
    "/detect/png",
    response_model=Union[YOLODetectionResult, DetectionResponse],
    tags=["Detection"]
)
async def detect_png(
    # Each form field is now a separate argument, which is the correct pattern.
    image_file: UploadFile = File(..., description="PNG image of a single document page."),
    pdf_file: Optional[UploadFile] = File(None, description="The original PDF file for coordinate mapping."),
    page_num: int = Form(..., description="The 0-indexed page number the image corresponds to."),
    zoom: float = Form(2.0, description="The zoom factor used to generate the image."),
    dataset: str = Form("keremberke", description="Which YOLO model to load."),
    coordinate_system: str = Form("pdf_points", description="Coordinate system (for compatibility, not used in logic)."),
    model_overrides: Optional[str] = Form(None, description="JSON string for model overrides."),
    selector: Optional[str] = Form(None, description="Selector (currently unused)."),
):
    """
    Detect tables in a single page image (PNG). Requires the original PDF
    and page number for accurate coordinate translation.
    """
    try:
        if pdf_file is None:
            raise HTTPException(status_code=400, detail="The 'pdf_file' is required for coordinate mapping.")
        if pdf_file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Uploaded 'pdf_file' must be a PDF.")

        pdf_bytes = await pdf_file.read()
        pdf_model = PDFModel(path=pdf_bytes)

        # Use the arguments directly, not from a 'params' object
        if not (0 <= page_num < pdf_model.pages):
            raise HTTPException(status_code=400, detail=f"Invalid page_num {page_num}. PDF has {pdf_model.pages} pages (0 to {pdf_model.pages - 1}).")

        image_bytes = await image_file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if np_img is None:
            raise HTTPException(status_code=400, detail="Could not decode the uploaded image file.")

        detector = _get_detector(dataset)
        overrides = json.loads(model_overrides) if model_overrides else None

        result = detector.run_on_image(
            np_img=np_img, pdf_model=pdf_model, page_num=page_num,
            zoom=zoom, model_overrides=overrides, model_name=dataset
        )
        print(f"result: {result}")
        return result

    except HTTPException as http_exc:
        raise http_exc
    except Exception as exc:
        return DetectionResponse(
            success=False,
            message=f"An unexpected error occurred: {str(exc)}",
            model_used=dataset # Use dataset directly
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# from __future__ import annotations

# import json
# from pathlib import Path
# from typing import List, Dict, Optional, Union, Any

# import cv2  # <-- ADD
# import numpy as np # <-- ADD

# from fastapi import FastAPI, UploadFile, HTTPException, Form, File
# from pydantic import BaseModel
# # from tableai.readers.files import FileReader # No longer used
# from tableai.pdf.pdf_model import PDFModel

# # Local inference
# from .inference import YOLOTableDetector

# # ────────────────────────────────────────────────────────────────
# # FastAPI setup
# # ────────────────────────────────────────────────────────────────
# app = FastAPI(
#     title="TableAI YOLO Detection Service",
#     description="Standalone YOLO-based table detection service (param-compatible with Detectron endpoint)",
#     version="0.2.1" #<-- Version bump
# )

# # Single detector cache (keeps weights on disk / in memory)
# _detector_cache: dict[str, YOLOTableDetector] = {}   # keyed by model_type

# # ────────────────────────────────────────────────────────────────
# # Pydantic response models (UNCHANGED)
# # ────────────────────────────────────────────────────────────────
# class DetectionResponse(BaseModel):
#     success: bool
#     detections: Any = []
#     page_dimensions: Optional[Dict[int, Dict[str, int]]] = None
#     page_count: Optional[int] = None
#     coordinate_system: str = "pdf_points"
#     model_used: str = ""
#     message: str = ""


# class HealthResponse(BaseModel):
#     status: str
#     service: str
#     version: str
#     cached_models: List[str]


# # ────────────────────────────────────────────────────────────────
# # Utility helpers (UNCHANGED)
# # ────────────────────────────────────────────────────────────────
# def _get_detector(model_type: str) -> YOLOTableDetector:
#     """
#     Return a (cached) YOLOTableDetector for the requested `model_type`.
#     """
#     if model_type not in YOLOTableDetector.AVAILABLE_MODELS:
#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 f"Invalid model_type '{model_type}'. "
#                 f"Valid options: {list(YOLOTableDetector.AVAILABLE_MODELS.keys())}"
#             )
#         )
#     if model_type not in _detector_cache:
#         _detector_cache[model_type] = YOLOTableDetector()
#     return _detector_cache[model_type]


# # ────────────────────────────────────────────────────────────────
# # Routes
# # ────────────────────────────────────────────────────────────────
# @app.get("/health", response_model=HealthResponse)
# def health() -> HealthResponse:
#     return HealthResponse(
#         status="healthy",
#         service="tableai-yolo-service",
#         version="0.2.1", # <-- Version bump
#         cached_models=list(_detector_cache.keys()),
#     )

# # ────────────────── NEW ENDPOINT FOR PNG ─────────────────────────
# @app.post("/detect/png", response_model=DetectionResponse)
# async def detect_png(
#     image_file: UploadFile = File(..., description="PNG image of a single document page."),
#     pdf_file: Optional[UploadFile] = File(None),
#     page_num: int = Form(..., description="The 0-indexed page number the image corresponds to."),
#     zoom: str = Form("2.0", description="The zoom factor used to generate the image."),
#     coordinate_system: str = Form("pdf_points"),
#     model_overrides: Optional[str] = Form(None),
#     selector: Optional[str] = Form(None),
#     dataset: str = Form("keremberke", description="Which YOLO model to load."),
# ):
#     """
#     Detect tables in a single page image (PNG). Requires the original PDF path
#     and page number for accurate coordinate translation.
#     """
#     debug_info = {}
#     try:
#         zoom_val = float(zoom)
#         model_type = dataset
#         debug_info.update(
#             zoom_raw=zoom, zoom_parsed=zoom_val,
#             coordinate_system_raw=coordinate_system, coordinate_system_parsed=coordinate_system,
#             dataset_raw=dataset, model_type=model_type,
#             pdf_file=pdf_file, page_num=page_num
#         )

#         if pdf_file is not None:
#             if pdf_file.content_type != "application/pdf":
#                 raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
#             pdf_bytes = await pdf_file.read()
#             pdf_source = pdf_bytes
#         else:
#             raise HTTPException(status_code=400, detail="Provide either 'pdf_file' or 'pdf_path'")

#         overrides = json.loads(model_overrides) if model_overrides else None
#         pdf_model = PDFModel(path=pdf_source)
#         # 2. Load image into numpy array
#         image_bytes = await image_file.read()
#         np_arr = np.frombuffer(image_bytes, np.uint8)
#         np_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

#         if not (0 <= page_num < pdf_model.pages):
#             raise HTTPException(
#                 status_code=400,
#                 detail=f"Invalid page_num {page_num}. PDF has {pdf_model.pages} pages (0 to {pdf_model.pages - 1})."
#             )

#         # 4. Get detector and overrides
#         detector = _get_detector(model_type)
#         overrides = json.loads(model_overrides) if model_overrides else None

#         # 5. Run detection on the single image
#         result = detector.run_on_image(
#             np_img=np_img,
#             pdf_model=pdf_model,
#             page_num=page_num,
#             zoom=zoom_val,
#             model_overrides=overrides,
#             model_name=model_type
#         )
        
#         # 6. Format response (same as before)
#         detections: List[Dict[str, Union[int, float, tuple]]] = []
#         for p_num, bbox_list in result["tbl_coordinates"].items():
#             for bbox in bbox_list:
#                 detections.append({"page": p_num, "bbox": bbox, "confidence": 1.0})
        
#         return DetectionResponse(
#             success=True,
#             detections=result,
#             page_dimensions=result.get("page_dimensions"),
#             page_count=pdf_model.pages,
#             coordinate_system=coordinate_system,
#             model_used=model_type,
#             message=f"Detected {len(detections)} tables on page {page_num}. "
#                     f"Debug: {debug_info}"
#         )

#     except Exception as exc:
#         return DetectionResponse(
#             success=False, message=f"Error: {str(exc)}. Debug: {debug_info}"
#         )


# # ────────────────── ORIGINAL PDF ENDPOINT (UNCHANGED) ────────────
# @app.post("/detect/pdf", response_model=DetectionResponse)
# async def detect_pdf(
#     pdf_file: Optional[UploadFile] = File(None),
#     pdf_path: Optional[str] = Form(None),
#     zoom: Optional[str] = Form(None),
#     coordinate_system: Optional[str] = Form(None),
#     model_overrides: Optional[str] = Form(None),
#     selector: Optional[str] = Form(None),
#     dataset: Optional[str] = Form(None),
# ):
#     """
#     Detect tables in a PDF. *Interface identical to Detectron* so front-end /
#     client code can call either service without changes.
#     """

#     debug_info = {}
#     try:
#         zoom_val = float(zoom) if zoom else 2.0
#         coord_system = coordinate_system or "pdf"
#         model_type = dataset or "keremberke"
#         debug_info.update(
#             zoom_raw=zoom, zoom_parsed=zoom_val,
#             coordinate_system_raw=coordinate_system, coordinate_system_parsed=coord_system,
#             dataset_raw=dataset, model_type=model_type,
#         )
#         if pdf_file is not None:
#             if pdf_file.content_type != "application/pdf":
#                 raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
#             pdf_bytes = await pdf_file.read()
#             pdf_source = pdf_bytes
#         elif pdf_path:
#             pdf_source = pdf_path.strip()
#         else:
#             raise HTTPException(status_code=400, detail="Provide either 'pdf_file' or 'pdf_path'")

#         overrides = json.loads(model_overrides) if model_overrides else None

#         pdf_model = PDFModel(path=pdf_source)
#         detector = _get_detector(model_type)
#         # This now calls the original .run() method
#         result = detector.run(
#             pdf_model,
#             zoom=zoom_val,
#             model_overrides=overrides,
#             model_name=model_type
#         )

#         detections: List[Dict[str, Union[int, float, tuple]]] = []
#         for page_num, bbox_list in result["tbl_coordinates"].items():
#             for bbox in bbox_list:
#                 detections.append({"page": page_num, "bbox": bbox, "confidence": 1.0})

#         return DetectionResponse(
#             success=True,
#             detections=result,
#             page_dimensions=result.get("page_dimensions"),
#             page_count=pdf_model.pages,
#             coordinate_system=coord_system,
#             model_used=model_type,
#             message=f"Detected {len(detections)} tables across {pdf_model.pages} pages. "
#                     f"Debug: {debug_info}"
#         )

#     except Exception as exc:
#         return DetectionResponse(
#             success=False, message=f"Error: {str(exc)}. Debug: {debug_info}"
#         )


# @app.get("/models/info")
# def models_info():
#     """Mirror Detectron `/models/info` shape."""
#     return {
#         "available_models": list(YOLOTableDetector.AVAILABLE_MODELS.keys()),
#         "cached_models": list(_detector_cache.keys()),
#         "coordinate_systems": ["image_pixels", "pdf_points"],
#         "endpoints": {
#             "/detect/pdf": "PDF detection (processes full PDF).",
#             "/detect/png": "Single page PNG detection (requires original PDF path).", # <-- ADDED
#             "/health": "Health check",
#             "/models/info": "Model information",
#         },
#     }


# ────────────────────────────────────────────────────────────────
# Local dev entry point (UNCHANGED)
# ────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# from __future__ import annotations

# import json
# from typing import List, Dict, Optional, Union, Any

# from fastapi import FastAPI, UploadFile, HTTPException, Form, File
# from pydantic import BaseModel
# from tableai.readers.files import FileReader
# from tableai.pdf.models import PDFModel

# # Local inference
# from .inference import YOLOTableDetector

# # ────────────────────────────────────────────────────────────────
# # FastAPI setup
# # ────────────────────────────────────────────────────────────────
# app = FastAPI(
#     title="TableAI YOLO Detection Service",
#     description="Standalone YOLO-based table detection service (param-compatible with Detectron endpoint)",
#     version="0.2.0"
# )

# # Single detector cache (keeps weights on disk / in memory)
# _detector_cache: dict[str, YOLOTableDetector] = {}   # keyed by model_type


# # ────────────────────────────────────────────────────────────────
# # Pydantic response models
# # ────────────────────────────────────────────────────────────────
# class DetectionResponse(BaseModel):
#     success: bool
#     detections: Any = []
#     page_dimensions: Optional[Dict[int, Dict[str, int]]] = None
#     page_count: Optional[int] = None
#     coordinate_system: str = "pdf_points"
#     model_used: str = ""
#     message: str = ""


# class HealthResponse(BaseModel):
#     status: str
#     service: str
#     version: str
#     cached_models: List[str]


# # ────────────────────────────────────────────────────────────────
# # Utility helpers
# # ────────────────────────────────────────────────────────────────
# def _get_detector(model_type: str) -> YOLOTableDetector:
#     """
#     Return a (cached) YOLOTableDetector for the requested `model_type`.
#     """
#     if model_type not in YOLOTableDetector.AVAILABLE_MODELS:
#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 f"Invalid model_type '{model_type}'. "
#                 f"Valid options: {list(YOLOTableDetector.AVAILABLE_MODELS.keys())}"
#             )
#         )

#     if model_type not in _detector_cache:
#         _detector_cache[model_type] = YOLOTableDetector()
#     return _detector_cache[model_type]


# # ────────────────────────────────────────────────────────────────
# # Routes
# # ────────────────────────────────────────────────────────────────
# @app.get("/health", response_model=HealthResponse)
# def health() -> HealthResponse:
#     return HealthResponse(
#         status="healthy",
#         service="tableai-yolo-service",
#         version="0.2.0",
#         cached_models=list(_detector_cache.keys()),
#     )


# @app.post("/detect/pdf", response_model=DetectionResponse)
# async def detect_pdf(
#     # ── parameters copied 1-for-1 from Detectron endpoint ─────────
#     pdf_file: Optional[UploadFile] = File(None),
#     pdf_path: Optional[str] = Form(None),
#     zoom: Optional[str] = Form(None),
#     coordinate_system: Optional[str] = Form(None),
#     model_overrides: Optional[str] = Form(None),
#     selector: Optional[str] = Form(None),
#     dataset: Optional[str] = Form(None),          # keep same name but maps to YOLO model_type
# ):
#     """
#     Detect tables in a PDF. *Interface identical to Detectron* so front-end /
#     client code can call either service without changes.

#     • `dataset`   → which YOLO model to load (keremberke, foduucom, doclaynet).
#     • `selector`  is accepted for parity but **ignored** by YOLO (kept for future use).
#     """

#     debug_info = {}  # will be appended to `message` for easier troubleshooting
#     try:
#         zoom_val = float(zoom) if zoom else 2.0
#         coord_system = coordinate_system or "pdf"
#         model_type = dataset or "keremberke"       # default YOLO model
#         debug_info.update(
#             zoom_raw=zoom, zoom_parsed=zoom_val,
#             coordinate_system_raw=coordinate_system, coordinate_system_parsed=coord_system,
#             dataset_raw=dataset, model_type=model_type,
#         )
#         if pdf_file is not None:
#             if pdf_file.content_type != "application/pdf":
#                 raise HTTPException(status_code=400, detail="Uploaded file must be a PDF")
#             pdf_bytes = await pdf_file.read()
#             pdf_source = pdf_bytes
#         elif pdf_path:
#             pdf_source = pdf_path.strip()
#         else:
#             raise HTTPException(status_code=400, detail="Provide either 'pdf_file' or 'pdf_path'")

#         overrides = None
#         if model_overrides:
#             try:
#                 overrides = json.loads(model_overrides)
#             except json.JSONDecodeError:
#                 raise HTTPException(status_code=400, detail="Invalid JSON in model_overrides")

#         pdf_model = PDFModel(path=pdf_source)
#         detector = _get_detector(model_type)
#         result = detector.run(
#             pdf_model,
#             zoom=zoom_val,
#             model_overrides=overrides,
#             model_name=model_type
#         )

#         detections: List[Dict[str, Union[int, float, tuple]]] = []
#         for page_num, bbox_list in result["tbl_coordinates"].items():
#             for bbox in bbox_list:
#                 detections.append(
#                     {"page": page_num, "bbox": bbox, "confidence": 1.0}
#                 )

#         return DetectionResponse(
#             success=True,
#             detections=result,
#             page_dimensions=result.get("page_dimensions"),
#             page_count=pdf_model.pages,
#             coordinate_system=coord_system,
#             model_used=model_type,
#             message=f"Detected {len(detections)} tables across {pdf_model.pages} pages. "
#                     f"Debug: {debug_info}"
#         )

#     except Exception as exc:
#         return DetectionResponse(
#             success=False,
#             detections=[],
#             page_dimensions=None,
#             page_count=None,
#             coordinate_system=coordinate_system or "pdf",
#             model_used=dataset or "keremberke",
#             message=f"Error: {str(exc)}. Debug: {debug_info}"
#         )


# @app.get("/models/info")
# def models_info():
#     """Mirror Detectron `/models/info` shape."""
#     return {
#         "available_models": list(YOLOTableDetector.AVAILABLE_MODELS.keys()),
#         "cached_models": list(_detector_cache.keys()),
#         "coordinate_systems": ["image_pixels", "pdf_points"],
#         "endpoints": {
#             "/detect/pdf": "PDF detection (Detectron-compatible interface)",
#             "/health": "Health check",
#             "/models/info": "Model information",
#         },
#     }


# # ────────────────────────────────────────────────────────────────
# # Local dev entry point
# # ────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)