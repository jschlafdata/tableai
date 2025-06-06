from __future__ import annotations

import base64
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4
from typing import Optional, List, Tuple, Union
import fitz
from pydantic import BaseModel, field_validator, model_validator, ValidationError, Field
from io import BytesIO
import base64
from IPython.display import display
from PIL import Image, ImageDraw
from enum import Enum
from dataclasses import dataclass
from typing import Optional
import fitz
import numpy as np

from tableai_tools.pdf.query import (
    FitzTextIndex, 
    LineTextIndex
)

from tableai_tools.pdf.query_engine import QueryEngine
from tableai_tools.readers.files import FileReader  # Import your FileReader

__all__ = ["PDFModel"]

class PDFModel(BaseModel):
    path: Union[str, Path, bytes, bytearray]
    s3_client: Optional[object] = None
    doc: Optional[fitz.Document] = None
    combined_doc: Optional[fitz.Document] = None
    name: Optional[str] = None
    index: Optional[FitzTextIndex] = None 
    line_index: Optional[LineTextIndex] = None 
    page_metadata: Optional[dict] = None
    query_engine: Optional[QueryEngine] = None
    pdf_core_metadata: Optional[dict] = None
    pdf_combined_doc_coords: Optional[dict] = None
    meta_tag: Optional[str] = None
    _limit: Optional[int] = None
    _context_store: Optional[dict] = None

    @model_validator(mode="after")
    def load_document(self):
        """Load the PDF document using FileReader and validate it's a fitz.Document."""
        try:
            # Use FileReader to load the document
            self.doc = FileReader.pdf(self.path, s3_client=self.s3_client)
            
            # Validate it's a fitz.Document
            if not isinstance(self.doc, fitz.Document):
                raise ValueError(f"FileReader returned {type(self.doc)}, expected fitz.Document")
            
            # Set name if it's a local path
            if isinstance(self.path, (str, Path)) and not str(self.path).startswith(('s3://', 'http://', 'https://')):
                self.name = Path(self.path).stem
            elif isinstance(self.path, str) and self.path.startswith('s3://'):
                # Extract name from S3 path
                self.name = Path(self.path).stem
                
        except Exception as e:
            raise ValueError(f"Could not load PDF document: {e}")
        
        return self

    @model_validator(mode="after")
    def build_search_indexes(self):
        """Build search indexes from the loaded document."""
        if not self.doc:
            raise ValueError("Document not loaded")
            
        self.index = FitzTextIndex.from_document(self.doc)
        all_pages_text_index = self.index.query(
            **{"blocks[*].lines[*].spans[*].text": "*"}, restrict=["text", "font", "bbox"]
        )
        self.page_metadata = self.index.page_metadata
        self.line_index = LineTextIndex(all_pages_text_index, page_metadata=self.page_metadata)
        self.query_engine = QueryEngine(self.line_index)
        return self

    class Config:
        arbitrary_types_allowed = True  # To allow fitz.Document and s3_client

    @property
    def pages(self) -> int:
        """Number of pages in the PDF."""
        if not self.doc:
            return 0
        return self.doc.page_count

    @property
    def page_sizes(self) -> List[Tuple[float, float]]:
        """List of (width, height) tuples for all pages."""
        if not self.doc:
            return []
        return [(page.rect.width, page.rect.height) for page in self.doc]

    @property
    def first_page_size(self) -> Optional[Tuple[float, float]]:
        """(width, height) tuple for the first page, or None if empty."""
        if not self.doc or self.doc.page_count == 0:
            return None
        first_page = self.doc[0]
        return (first_page.rect.width, first_page.rect.height)

    def get_page(self, page_number: int, combined=False) -> fitz.Page:
        """Get the fitz.Page object for a given page number (0-based)."""
        if combined == True and not self.combined_doc:
            raise ValueError("Combined PDF document not loaded")
        if combined == False and not self.doc:
            raise ValueError("PDF document not loaded")
        if combined == False and page_number < 0 or page_number >= self.doc.page_count:
            raise IndexError(f"Page number {page_number} out of range")
        if combined == True:
            return self.combined_doc[0]
        else:
            return self.doc[page_number]

    @staticmethod
    def convert_pdf_page_to_base64(
        page: fitz.Page,
        zoom: float = 2.0,
        clip: Optional[fitz.Rect] = None
    ) -> str:
        """
        Render a PDF page (or a clipped region) to a base64-encoded PNG.
        Args:
            page (fitz.Page): The PyMuPDF page object
            zoom (float): Zoom factor for rendering
            clip (fitz.Rect, optional): Region to clip before rendering
        Returns:
            str: Base64-encoded PNG image as string
        """
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        img_bytes = pix.tobytes("png")
        return base64.b64encode(img_bytes).decode("utf-8")

    def get_page_base64(
        self,
        page_number: int,
        zoom: float = 2.0,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        combined=False
    ) -> str:
        """
        Render the given page (or a clipped region of it) as base64 PNG.

        Args:
            page_number (int): zero-based page index
            zoom (float): resolution multiplier
            bounds (tuple of four floats, optional):
                (x0, y0, x1, y1) in page coordinates to clip. 
                If omitted, the full page is rendered.

        Returns:
            str: Base64-encoded PNG image.
        """
        page = self.get_page(page_number, combined=combined)
        clip_rect = fitz.Rect(*bounds) if bounds is not None else None
        return self.convert_pdf_page_to_base64(page, zoom=zoom, clip=clip_rect)

    # Iterable: yields (page_number, fitz.Page) for allowed pages
    def __iter__(self):
        n_pages = self.pages
        limit = self._limit or n_pages
        for i in range(min(n_pages, limit)):
            yield i, self.get_page(i)

    def set_limit(self, limit: int):
        """Set max number of pages to iterate."""
        self._limit = limit

    def clear_limit(self):
        """Remove page limit."""
        self._limit = None

    # Optionally, a property to show current effective limit
    @property
    def effective_limit(self):
        return self._limit or self.pages
    
    @model_validator(mode="after")
    def metadata(self):
        """
        Fetch high-level PDF metadata (pages, coords, meta_tag, etc.).
        If corrupted, return details from error_message.
        """
        if not self.doc:
            return self
            
        md = self.doc.metadata  # merged Info + XMP
        meta_fields = {
            key: md.get(key, None)
            for key in ["creator", "title", "author", "subject", "keywords"]
        }
        pdf_trailer = self.doc.pdf_trailer()
        if isinstance(pdf_trailer, dict) and "Root" in pdf_trailer:
            version = pdf_trailer.get("Version", None)
        else:
            version = None

        # Build a unique-ish hash of core PDF metadata (optional usage)
        meta_hash = '|'.join(
            f"{k}|{''.join(v.split())}" for k, v in meta_fields.items() if v
        ).strip()
        self.meta_tag = meta_hash

        coords = []
        pages = len(self.doc)
        for i, page in enumerate(self.doc):
            rect = page.rect
            coords.append({
                i: {
                    "page_box": [rect.x0, rect.y0, rect.x1, rect.y1],
                    "width": rect.width,
                    "height": rect.height
                }
            })

        width = self.doc[0].rect.width if pages > 0 else None
        height = self.doc[0].rect.height if pages > 0 else None

        self.pdf_core_metadata = {
            "pages": pages,
            "coords": coords,
            "meta_tag": self.meta_tag if self.meta_tag else 'NONE',
            "pdf_version": version,
            "width": width,
            "height": height
        }
        return self

    def page_to_numpy(
        self,
        page_number: int,
        zoom: float = 2.0,
        as_rgba: bool = False
    ) -> Tuple[int, np.ndarray, int, int]:
        """
        Render a single page from a fitz.Document into a NumPy array.

        Args:
            page_number (int): zero‐based index of the page to render.
            zoom (float): zoom factor for rendering (default=2.0).
            as_rgba (bool): if True, include alpha channel. Otherwise, RGB.

        Returns:
            (page_number, np_image, image_width, image_height)
              - page_number (int): same index you passed in.
              - np_image (np.ndarray): shape = (H, W, 3) or (H, W, 4) depending on as_rgba.
              - image_width (int): width in pixels after zoom.
              - image_height (int): height in pixels after zoom.
        """
        # 1) Grab the requested page (raises IndexError if out of range)
        page = self.doc[page_number]

        # 2) Build a transformation matrix for the given zoom
        mat = fitz.Matrix(zoom, zoom)

        # 3) Render the page into a Pixmap. If as_rgba=True, request alpha channel.
        pix: fitz.Pixmap = page.get_pixmap(matrix=mat, alpha=as_rgba)

        # 4) Convert Pixmap to a PIL Image
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(
            mode,
            (pix.width, pix.height),
            pix.samples  # raw pixel bytes
        )

        # 5) Convert PIL Image → NumPy array
        np_image = np.asarray(img)

        # 6) Return the tuple
        return page_number, np_image, pix.width, pix.height
    
    @model_validator(mode="after")
    def combine_pages_into_one(self):
        """
        Combines all pages of input_pdf_path into one tall PDF,
        then extracts recurring text blocks (keeping only top/bottom strips).
        Returns a metadata dict.
        """
        if not self.doc:
            return self
            
        zoom=1
        self.combined_doc = fitz.open()
        original_page_count = self.pages
        page_coords = self.pdf_core_metadata['coords']

        total_height = 0.0
        max_width = 0.0
        page_heights: List[float] = []
        for entry in page_coords:
            page_idx, meta = next(iter(entry.items()))
            h = meta["height"] * zoom
            w = meta["width"]  * zoom
            page_heights.append(h)
            total_height += h
            max_width = max(max_width, w)

        combined_page = self.combined_doc.new_page(width=max_width, height=total_height)

        page_breaks = []
        current_y = 0.0

        # Place each original page
        for page_index in range(original_page_count):
            page_breaks.append(current_y)
            page = self.doc[page_index]
            width, height = page.rect.width, page.rect.height

            target_rect = fitz.Rect(0, current_y, width, current_y + height)
            combined_page.show_pdf_page(target_rect, self.doc, page_index)
            current_y += height
        
        self.pdf_combined_doc_coords = {
            'rect': tuple(self.combined_doc[0].rect), 
            'original_pages': original_page_count, 
            'virtual_page_breaks': page_breaks
        }
        
        return self

    def show(self, page_number, crop_box=None, highlight_boxes=None, box_color="red", box_width=2, combined=False):
        """
        Display a rendered PDF page with optional crop and highlight boxes.

        Args:
            page_number (int): Page to render.
            crop_box (tuple): Optional crop area (PDF coordinates: x0, y0, x1, y1).
            highlight_boxes (list): Optional list of PDF-coordinate boxes [(x0, y0, x1, y1)].
            box_color (str): Outline color of highlight boxes.
            box_width (int): Outline width.
        """
        # Render base64 image
        img_base64 = self.get_page_base64(page_number=page_number, bounds=crop_box, combined=combined)
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Determine scaling from PDF points to image pixels
        if combined == True:
            page_height = self.combined_doc[page_number].rect.height
            page_width = self.combined_doc[page_number].rect.width
        else:
            page_height = self.doc[page_number].rect.height
            page_width = self.doc[page_number].rect.width
        zoom = 1.0

        if crop_box:
            crop_width = crop_box[2] - crop_box[0]
            crop_height = crop_box[3] - crop_box[1]
        else:
            crop_width = page_width
            crop_height = page_height

        scale_x = img.width / crop_width
        scale_y = img.height / crop_height
        offset_x = crop_box[0] if crop_box else 0
        offset_y = crop_box[1] if crop_box else 0

        if highlight_boxes:
            for x0, y0, x1, y1 in highlight_boxes:
                box = [
                    (x0 - offset_x) * scale_x,
                    (y0 - offset_y) * scale_y,
                    (x1 - offset_x) * scale_x,
                    (y1 - offset_y) * scale_y,
                ]
                draw.rectangle(box, outline=box_color, width=box_width)

        display(img)
    
    def transform_bbox_coordinates_simple(
        self,
        bbox: tuple,
        y_offset: float,
        scale_x: float,
        scale_y: float
    ) -> tuple:
        """
        Transform a bounding box from original page coordinates to combined page coordinates.

        Args:
            bbox: Original bbox as (x0, y0, x1, y1)
            y_offset: Y offset for this page in combined PDF
            scale_x: X scale factor
            scale_y: Y scale factor

        Returns:
            Transformed bbox as (x0, y0, x1, y1)
        """
        x0, y0, x1, y1 = bbox

        # Apply scaling and offset
        new_x0 = x0 * scale_x
        new_x1 = x1 * scale_x
        new_y0 = (y0 * scale_y) + y_offset
        new_y1 = (y1 * scale_y) + y_offset

        return (new_x0, new_y0, new_x1, new_y1)
    
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

        page_w, page_h = self.page_sizes[page_number]       # PDF pts
        sx, sy = page_w / img_w, page_h / img_h

        return (x1_px * sx, y1_px * sy, x2_px * sx, y2_px * sy)

    def page_bbox_to_combined(
        self,
        page_number: int,
        page_bbox: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """
        Lift a page-local bbox into the single *tall* page that
        `combine_pages_into_one()` built.
        """
        if not self.pdf_combined_doc_coords:
            raise ValueError("Combined PDF not yet built")

        y_offset = self.pdf_combined_doc_coords["virtual_page_breaks"][page_number]
        # 1:1 scale within the individual page → combined
        return self.transform_bbox_coordinates_simple(
            page_bbox, y_offset=y_offset, scale_x=1.0, scale_y=1.0
        )