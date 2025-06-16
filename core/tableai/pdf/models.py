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

from tableai.pdf.query import (
    LineTextIndex, 
    GroupbyTransform
)

from tableai.pdf.query_engine import QueryEngine
from tableai.readers.files import FileReader  # Import your FileReader

__all__ = ["PDFModel"]

try:
    from IPython.display import display
    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False

class PDFModel(BaseModel):
    path: Union[str, Path, bytes, bytearray]
    s3_client: Optional[object] = None
    doc: Optional[fitz.Document] = None
    name: Optional[str] = None
    line_index: Optional['LineTextIndex'] = None
    query_engine: Optional[QueryEngine] = None
    meta_version: Optional[str] = None
    meta_tag: Optional[str] = None
    virtual_page_metadata: Optional[dict] = None
    
    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def initialize_and_metadata(self) -> "PDFModel":
        """Single Pydantic validator that loads, combines, indexes, and extracts metadata."""
        # 1. Load the original multi-page PDF
        try:
            temp_doc = FileReader.pdf(self.path, s3_client=self.s3_client)
            if not isinstance(temp_doc, fitz.Document):
                raise ValueError(f"FileReader returned {type(temp_doc)}, expected fitz.Document")
            
            if isinstance(self.path, (str, Path)):
                self.name = Path(self.path).stem

        except Exception as e:
            raise ValueError(f"Could not load temporary PDF document: {e}")

        # 2. Combine into a single tall document and gather virtual metadata
        self.doc, self.virtual_page_metadata = self._combine_pages_and_get_metadata(temp_doc)
        
        # 3. Build the "virtually aware" LineTextIndex
        self.line_index = LineTextIndex.from_document(
            self.doc, 
            virtual_page_metadata=self.virtual_page_metadata
        )

        # 4. Initialize the Query Engine
        self.query_engine = QueryEngine(self.line_index)
        
        # 5. Extract Document Metadata
        md = self.doc.metadata
        meta_fields = {k: md.get(k) for k in ["creator", "title", "author", "subject", "keywords"]}
        pdf_trailer = self.doc.pdf_trailer()
        self.meta_version = pdf_trailer.get("Version") if isinstance(pdf_trailer, dict) else None
        self.meta_tag = '|'.join(f"{k}|{''.join(v.split())}" for k, v in meta_fields.items() if v).strip()
        temp_doc.close()
        
        return self

    @staticmethod
    def _combine_pages_and_get_metadata(original_doc: fitz.Document) -> Tuple[fitz.Document, dict]:
        """
        Combines pages, creates virtual page metadata, and calculates the
        translated content area for each virtual page.
        """
        combined_doc = fitz.open()
        total_height, max_width = 0.0, 0.0
        
        # Get dimensions for all original pages
        original_page_dims = []
        for page in original_doc:
            dims = {"width": page.rect.width, "height": page.rect.height}
            original_page_dims.append(dims)
            total_height += dims["height"]
            max_width = max(max_width, dims["width"])
    
        # Create the combined page
        combined_page = combined_doc.new_page(width=max_width, height=total_height)

        virtual_page_bounds = {}
        virtual_page_breaks = []
        virtual_page_content_areas = {
            "margin_bboxes": [],   # Flat list of all margin boxes in *absolute* coordinates
            "content_bboxes": [],  # Flat list of all content boxes in *absolute* coordinates
            "pages": {}            # Per-page dictionary with *relative* coordinates
        }
        current_y = 0.0
        
        for i, page in enumerate(original_doc):
            virtual_page_breaks.append((current_y, i))
            
            target_rect = fitz.Rect(0, current_y, page.rect.width, current_y + page.rect.height)
            virtual_page_bounds[i] = tuple(target_rect) 
            
            combined_page.show_pdf_page(target_rect, original_doc, i)
    
            # --- NEW LOGIC FOR POPULATING THE DATA STRUCTURE ---
            
            # 1. Calculate the content area on the original page (relative coordinates)
            local_content_area_rect = fitz.Rect()
            for b in page.get_text("blocks"):
                local_content_area_rect |= b[:4]
    
            # 2. Process only if the page has content
            if not local_content_area_rect.is_empty:
                # A. Convert to a plain tuple for storing (relative coords)
                content_bbox_rel = tuple(local_content_area_rect)
                # B. Calculate margin boxes (relative coords)
                margin_bboxes_rel = Map.inverse_page_blocks({i: {'bboxes': [content_bbox_rel], 'page_width': page.rect.width, 'page_height': page.rect.height}})
    
                # C. Translate to absolute coordinates for the combined document
                content_bbox_abs = Map.scale_y(bbox=content_bbox_rel, y_offset=current_y)
                margin_bboxes_abs = [Map.scale_y(bbox=bbox, y_offset=current_y) for bbox in margin_bboxes_rel]
    
                # D. Populate the main dictionary
                virtual_page_content_areas["content_bboxes"].append(content_bbox_abs)
                virtual_page_content_areas["margin_bboxes"].extend(margin_bboxes_abs) # Use extend for lists
                
                # E. Populate the page-specific dictionary with relative data
                virtual_page_content_areas["pages"][i] = {
                    'content_bbox(rel)': content_bbox_rel, 
                    'margin_bboxes(rel)': margin_bboxes_rel
                }
            else:
                # Handle blank pages explicitly for consistency
                virtual_page_content_areas["pages"][i] = {
                    'content_bbox(rel)': None,
                    'margin_bboxes(rel)': []
                }
            # Increment the vertical offset for the next page
            current_y += page.rect.height
        
        # Sort breaks by y-coordinate for efficient lookup
        virtual_page_breaks.sort()
        
        metadata = {
            "page_count": len(original_doc),
            "page_bounds": virtual_page_bounds,
            "page_breaks": virtual_page_breaks,
            "page_content_areas": virtual_page_content_areas, 
            "combined_doc_width": max_width,
            "combined_doc_height": total_height,
            "original_page_dims": original_page_dims,
        }
        
        return combined_doc, metadata
    
    @property
    def pages(self) -> int:
        """Returns the number of virtual pages."""
        return self.virtual_page_metadata.get("page_count", 0) if self.virtual_page_metadata else 0

    def get_page(self, page_number: Optional[int] = None) -> fitz.Page:
        """Always returns the combined document page."""
        if not self.doc:
            raise ValueError("PDF document not loaded")
        if page_number is not None and (page_number < 0 or page_number >= self.pages):
            raise IndexError(f"Virtual page number {page_number} out of range")
        return self.doc[0]

    def _render_combined_crops(self, page, crop_boxes: List, zoom: Optional[float]=1, spacing: Optional[int] = 10) -> str:
        """
        Version with spacing between cropped regions.
        """
        from PIL import Image
        import io
        
        mat = fitz.Matrix(zoom, zoom)
        crop_images = []
        total_height = 0
        max_width = 0
        
        # Render each crop box
        for bbox in crop_boxes:
            clip_rect = fitz.Rect(*bbox)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            crop_images.append(img)
            
            total_height += img.height
            max_width = max(max_width, img.width)
        
        # Add spacing between images (except after last one)
        if len(crop_images) > 1:
            total_height += spacing * (len(crop_images) - 1)
        
        # Create combined image with white background
        combined = Image.new('RGB', (max_width, total_height), 'white')
        
        # Paste each crop with spacing
        y_offset = 0
        for i, img in enumerate(crop_images):
            combined.paste(img, (0, y_offset))
            y_offset += img.height
            if i < len(crop_images) - 1:  # Add spacing except after last image
                y_offset += spacing
        
        # Convert back to base64
        buffer = io.BytesIO()
        combined.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _filter_by_page_limit(self, page_limit: int, highlight_boxes=None, crop_boxes=None):
        """
        Unified function to filter all bbox-based inputs by page limit.
        
        Args:
            page_limit: Maximum virtual page number to include
            highlight_boxes: Optional highlight boxes (list or dict format)
            crop_boxes: Optional crop boxes (list format)
        
        Returns:
            Tuple of (filtered_highlight_boxes, filtered_crop_boxes)
        """
        if not hasattr(self, 'line_index') or not hasattr(self.line_index, '_get_virtual_page_num'):
            # Fallback: return original inputs if we can't determine page numbers
            return highlight_boxes, crop_boxes
        
        def is_bbox_within_limit(bbox):
            """Helper to check if a bbox is within the page limit."""
            page_num = self.line_index._get_virtual_page_num(bbox[1])  # y0 coordinate
            return page_num <= page_limit
        
        # Filter crop_boxes
        filtered_crop_boxes = None
        if crop_boxes:
            filtered_crop_boxes = [bbox for bbox in crop_boxes if is_bbox_within_limit(bbox)]
        
        # Filter highlight_boxes
        filtered_highlight_boxes = None
        if highlight_boxes:
            if isinstance(highlight_boxes, list):
                filtered_highlight_boxes = [bbox for bbox in highlight_boxes if is_bbox_within_limit(bbox)]
            
            elif isinstance(highlight_boxes, dict):
                filtered_highlight_boxes = {}
                for label, data in highlight_boxes.items():
                    boxes = data.get("boxes", [])
                    filtered_boxes = [bbox for bbox in boxes if is_bbox_within_limit(bbox)]
                    
                    if filtered_boxes:  # Only include groups that have boxes within the limit
                        filtered_highlight_boxes[label] = {**data, "boxes": filtered_boxes}
            else:
                filtered_highlight_boxes = highlight_boxes
        
        return filtered_highlight_boxes, filtered_crop_boxes
        

    def get_page_base64(self, page_number: Optional[int] = None, zoom: Optional[float] = 1, 
                       crop_boxes: Optional[List] = None, spacing: Optional[int] = 10, 
                       page_limit: Optional[int] = None,
                       highlight_boxes: Optional[Union[List[tuple], Dict]] = None,
                       box_color: str = "red", box_width: int = 2, font_size: int = 12) -> str:
        """
        Renders a virtual page or entire document as base64 with optional annotations.
        
        Args:
            page_number: Virtual page number to render (None for entire document)
            zoom: Zoom factor for rendering
            crop_boxes: Optional list of bboxes to crop and combine into one image
                       Format: [(x0, y0, x1, y1), ...]
            spacing: Pixels of white space between cropped regions
            page_limit: Maximum virtual page number to include (stops at this page)
            highlight_boxes: Optional highlight boxes to draw on the image (same format as show())
            box_color: Default color for highlight boxes
            box_width: Width of highlight box outlines
            font_size: Font size for highlight box labels
        
        Returns:
            Base64 encoded PNG image with annotations
        """
        page = self.get_page(page_number)
        
        # Apply page_limit filtering at the start
        if page_limit is not None:
            highlight_boxes, crop_boxes = self._filter_by_page_limit(
                page_limit, 
                highlight_boxes=highlight_boxes, 
                crop_boxes=crop_boxes
            )
        
        # If crop_boxes provided, render each crop and combine them
        if crop_boxes:
            base64_img = self._render_combined_crops(page, crop_boxes, zoom, spacing)
        else:
            # Original logic for single page/document rendering
            clip_rect = None
            if page_number is not None:
                # Clip to the specific virtual page bounds
                page_bounds = self.virtual_page_metadata["page_bounds"].get(page_number)
                if not page_bounds:
                    raise KeyError(f"Bounding box for virtual page {page_number} not found.")
                clip_rect = fitz.Rect(*page_bounds)
            else:
                # Handle page_limit for entire document rendering
                if page_limit is not None:
                    clip_rect = self._get_document_bounds_with_limit(page_limit)
            
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, clip=clip_rect)
            img_bytes = pix.tobytes("png")
            base64_img = base64.b64encode(img_bytes).decode("utf-8")
        
        # If highlight_boxes provided, draw them on the image
        if highlight_boxes:
            base64_img = self._add_highlight_boxes_to_base64(
                base64_img, highlight_boxes, page_number, page_limit, 
                box_color, box_width, font_size, zoom
            )
        
        return base64_img
    
    
    def _add_highlight_boxes_to_base64(self, base64_img: str, highlight_boxes, page_number, 
                                      page_limit, box_color, box_width, font_size, zoom) -> str:
        """
        Draws highlight boxes on a base64 image and returns the annotated base64 image.
        """
        from PIL import Image, ImageDraw, ImageFont
        import io
        import base64
        
        # Convert base64 to PIL Image
        img_bytes = base64.b64decode(base64_img)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Load font
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        
        # Calculate coordinate transformation (same logic as show())
        if page_number is not None:
            vp_bounds = self.virtual_page_metadata["page_bounds"][page_number]
            offset_x, offset_y = vp_bounds[0], vp_bounds[1]
            view_width, view_height = vp_bounds[2] - offset_x, vp_bounds[3] - offset_y
        else:
            offset_x, offset_y = 0, 0
            if page_limit is not None:
                view_width, view_height = self._get_limited_view_dimensions(page_limit)
            else:
                view_width = self.virtual_page_metadata["combined_doc_width"]
                view_height = self.virtual_page_metadata["combined_doc_height"]
        
        if view_width == 0 or view_height == 0:
            return base64_img  # Return original if no valid dimensions
        
        scale_x = img.width / view_width
        scale_y = img.height / view_height
        
        # Draw highlight boxes (same logic as show())
        if isinstance(highlight_boxes, dict):
            for label, data in highlight_boxes.items():
                color = data.get("color", box_color)
                boxes = data.get("boxes", [])
                
                for individual_box in boxes:
                    x0, y0, x1, y1 = individual_box
                    
                    # Draw the bounding box
                    box_in_pixels = [
                        (x0 - offset_x) * scale_x,
                        (y0 - offset_y) * scale_y,
                        (x1 - offset_x) * scale_x,
                        (y1 - offset_y) * scale_y,
                    ]
                    if not (box_in_pixels[2] > box_in_pixels[0] and box_in_pixels[3] > box_in_pixels[1]):
                        continue
                    
                    draw.rectangle(box_in_pixels, outline=color, width=box_width)
    
                    # Draw the label
                    px0, py0 = box_in_pixels[0], box_in_pixels[1]
                    text_anchor = (px0, py0 - box_width)
                    text_bbox = draw.textbbox(text_anchor, label, font=font, anchor="lb")
                    
                    bg_padding = 3
                    bg_rect = (
                        text_bbox[0] - bg_padding,
                        text_bbox[1] - bg_padding,
                        text_bbox[2] + bg_padding,
                        text_bbox[3] + bg_padding
                    )
                    
                    draw.rectangle(bg_rect, fill=color)
                    draw.text(text_anchor, label, font=font, fill="white", anchor="lb")
    
        elif isinstance(highlight_boxes, list):
            for x0, y0, x1, y1 in highlight_boxes:
                box_in_pixels = [
                    (x0 - offset_x) * scale_x,
                    (y0 - offset_y) * scale_y,
                    (x1 - offset_x) * scale_x,
                    (y1 - offset_y) * scale_y,
                ]
                if (box_in_pixels[2] > box_in_pixels[0] and box_in_pixels[3] > box_in_pixels[1]):
                    draw.rectangle(box_in_pixels, outline=box_color, width=box_width)
        
        # Convert back to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    
    def _get_limited_view_dimensions(self, page_limit: int) -> tuple:
        """Gets view dimensions up to the specified page limit."""
        page_bounds = self.virtual_page_metadata.get("page_bounds", {})
        
        max_x = 0
        max_y = 0
        
        for page_num, bounds in page_bounds.items():
            if page_num <= page_limit:
                x0, y0, x1, y1 = bounds
                max_x = max(max_x, x1)
                max_y = max(max_y, y1)
        
        return max_x, max_y
    
    
    def _get_document_bounds_with_limit(self, page_limit: int) -> Optional[fitz.Rect]:
        """Gets document bounds up to the specified page limit."""
        page_bounds = self.virtual_page_metadata.get("page_bounds", {})
        
        if not page_bounds or page_limit < 0:
            return None
        
        max_y = 0
        max_x = 0
        
        for page_num, bounds in page_bounds.items():
            if page_num <= page_limit:
                x0, y0, x1, y1 = bounds
                max_x = max(max_x, x1)
                max_y = max(max_y, y1)
        
        if max_y > 0:
            return fitz.Rect(0, 0, max_x, max_y)
        
        return None
    
    
    def show(
        self, 
        page_number: Optional[int] = None, 
        highlight_boxes: Optional[Union[List[tuple], Dict]] = None, 
        crop_boxes: Optional[List] = None,
        box_color: str = "red", 
        box_width: int = 2,
        font_size: int = 12,
        grid=False,
        zoom: Optional[float]=1, 
        spacing: Optional[int] = 10,
        page_limit: Optional[int] = None
    ):
        """
        Displays a rendered PDF page with optional highlight boxes.
    
        Args:
            page_number (Optional[int]): The virtual page to display. If None, displays the entire document.
            highlight_boxes (Optional[Union[List[tuple], Dict]]): Boxes to draw.
            crop_boxes (Optional[List]): Optional list of bboxes to crop and combine
            box_color (str): The default color for boxes if not specified in the dict schema.
            box_width (int): The width of the box outlines.
            font_size (int): The font size for labels when using the dict schema.
            grid (bool): Whether to show y-coordinate grid lines
            zoom (Optional[float]): Zoom factor for rendering
            spacing (Optional[int]): Pixels of white space between cropped regions
            page_limit (Optional[int]): Maximum virtual page number to include (stops at this page)
        """
        if not _IPYTHON_AVAILABLE:
            print("IPython/Jupyter is required to display images.")
            return
    
        # UNIFIED FILTERING: Apply page_limit to all inputs at the start
        if page_limit is not None:
            highlight_boxes, crop_boxes = self._filter_by_page_limit(
                page_limit, 
                highlight_boxes=highlight_boxes, 
                crop_boxes=crop_boxes
            )
    
        img_base64 = self.get_page_base64(
            page_number=page_number, 
            crop_boxes=crop_boxes, 
            spacing=spacing, 
            zoom=zoom,
            page_limit=page_limit  # Still pass it for document bounds calculation
        )
        img_bytes = base64.b64decode(img_base64)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
    
        try:
            font = ImageFont.truetype("Arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
    
        if page_number is not None:
            vp_bounds = self.virtual_page_metadata["page_bounds"][page_number]
            offset_x, offset_y = vp_bounds[0], vp_bounds[1]
            view_width, view_height = vp_bounds[2] - offset_x, vp_bounds[3] - offset_y
        else:
            offset_x, offset_y = 0, 0
            if page_limit is not None:
                # Calculate view dimensions based on page_limit
                view_width, view_height = self._get_limited_view_dimensions(page_limit)
            else:
                view_width = self.virtual_page_metadata["combined_doc_width"]
                view_height = self.virtual_page_metadata["combined_doc_height"]
    
        if view_width == 0 or view_height == 0:
            display(img)
            return
    
        scale_x = img.width / view_width
        scale_y = img.height / view_height
    
        # Handle highlight boxes (already filtered by this point)
        if isinstance(highlight_boxes, dict):
            for label, data in highlight_boxes.items():
                color = data.get("color", box_color)
                boxes = data.get("boxes", [])
                
                for individual_box in boxes:
                    x0, y0, x1, y1 = individual_box
                    
                    # Draw the bounding box
                    box_in_pixels = [
                        (x0 - offset_x) * scale_x,
                        (y0 - offset_y) * scale_y,
                        (x1 - offset_x) * scale_x,
                        (y1 - offset_y) * scale_y,
                    ]
                    if not (box_in_pixels[2] > box_in_pixels[0] and box_in_pixels[3] > box_in_pixels[1]):
                        continue
                    
                    draw.rectangle(box_in_pixels, outline=color, width=box_width)
    
                    # Draw the label
                    px0, py0 = box_in_pixels[0], box_in_pixels[1]
                    text_anchor = (px0, py0 - box_width)
                    text_bbox = draw.textbbox(text_anchor, label, font=font, anchor="lb")
                    
                    bg_padding = 3
                    bg_rect = (
                        text_bbox[0] - bg_padding,
                        text_bbox[1] - bg_padding,
                        text_bbox[2] + bg_padding,
                        text_bbox[3] + bg_padding
                    )
                    
                    draw.rectangle(bg_rect, fill=color)
                    draw.text(text_anchor, label, font=font, fill="white", anchor="lb")
    
        elif isinstance(highlight_boxes, list):
            for x0, y0, x1, y1 in highlight_boxes:
                box_in_pixels = [
                    (x0 - offset_x) * scale_x,
                    (y0 - offset_y) * scale_y,
                    (x1 - offset_x) * scale_x,
                    (y1 - offset_y) * scale_y,
                ]
                if (box_in_pixels[2] > box_in_pixels[0] and box_in_pixels[3] > box_in_pixels[1]):
                    draw.rectangle(box_in_pixels, outline=box_color, width=box_width)
    
        # Grid logic (unchanged)
        if grid:
            try:
                grid_font_size = max(10, int(font_size * 0.8))
                grid_font = ImageFont.truetype("Arial.ttf", grid_font_size)
            except IOError:
                grid_font = ImageFont.load_default()
            
            grid_color = (255, 20, 147, 150)
            start_y_pdf = offset_y
            end_y_pdf = offset_y + view_height
            tick_y_pdf = (int(start_y_pdf) // 100) * 100
            
            while tick_y_pdf < end_y_pdf:
                if tick_y_pdf >= start_y_pdf:
                    y_pixel = (tick_y_pdf - offset_y) * scale_y
                    draw.line([(0, y_pixel), (img.width, y_pixel)], fill=grid_color, width=1)
                    
                    label_text = str(tick_y_pdf)
                    text_anchor = (5, y_pixel + 2)
                    text_bbox = draw.textbbox(text_anchor, label_text, font=grid_font, anchor="lt")
                    draw.rectangle(text_bbox, fill="white")
                    draw.text(text_anchor, label_text, font=grid_font, fill="black", anchor="lt")
                tick_y_pdf += 100
        
        display(img)
            
    def __iter__(self):
        """Iterates through virtual pages."""
        for i in range(self.pages):
            yield i
            
    def page_bbox_to_combined(self, page_number: int, page_relative_bbox: tuple) -> tuple:
        """Converts a page-relative bbox to combined document coordinates."""
        if not self.virtual_page_metadata:
            raise ValueError("Document not initialized")

        vp_bounds = self.virtual_page_metadata["page_bounds"][page_number]
        offset_x, offset_y = vp_bounds[0], vp_bounds[1]
        
        pr_x0, pr_y0, pr_x1, pr_y1 = page_relative_bbox
        
        return (pr_x0 + offset_x, pr_y0 + offset_y, pr_x1 + offset_x, pr_y1 + offset_y)
