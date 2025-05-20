import base64
from enum import Enum
from pathlib import Path
from uuid import UUID, uuid4
from typing import Optional, List, Tuple

import fitz
from pydantic import BaseModel, field_validator, model_validator, ValidationError, Field

class DocType(str, Enum):
    PDF = ".pdf"
    XLSX = ".xlsx"
    METADATA = "metadata"

class Source(str, Enum):
    S3 = "s3"
    LOCAL = "local"

class PDFModel(BaseModel):
    path: Path
    source: Source
    id: UUID = Field(default_factory=uuid4)
    name: Optional[str] = None
    type: Optional[DocType] = None
    doc: Optional[fitz.Document] = None
    _limit: Optional[int] = None

    @field_validator('path')
    @classmethod
    def validate_path(cls, v):
        if not v.exists():
            raise ValueError(f"File not found: {v}")
        if v.suffix.lower() != DocType.PDF.value:
            raise ValueError(f"File extension '{v.suffix}' is not a PDF: {v}")
        return v

    @model_validator(mode="after")
    def set_fields(self):
        self.name = self.path.stem
        self.type = DocType.PDF
        try:
            self.doc = fitz.open(str(self.path))
        except Exception as e:
            raise ValueError(f"Could not open PDF: {e}")
        return self

    class Config:
        arbitrary_types_allowed = True  # To allow fitz.Document

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

    def get_page(self, page_number: int) -> fitz.Page:
        """Get the fitz.Page object for a given page number (0-based)."""
        if not self.doc:
            raise ValueError("PDF document not loaded")
        if page_number < 0 or page_number >= self.doc.page_count:
            raise IndexError(f"Page number {page_number} out of range")
        return self.doc[page_number]

    @staticmethod
    def convert_pdf_page_to_base64(page: fitz.Page, zoom: float = 2.0) -> str:
        """
        Render a PDF page to a base64-encoded PNG image.
        Args:
            page (fitz.Page): The PyMuPDF page object
            zoom (float): Zoom factor for rendering (higher = higher resolution)
        Returns:
            str: Base64-encoded PNG image as string
        """
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        return base64.b64encode(img_bytes).decode("utf-8")

    def get_page_base64(self, page_number: int, zoom: float = 2.0) -> str:
        """Render the given page (by number) as base64 PNG."""
        page = self.get_page(page_number)
        return self.convert_pdf_page_to_base64(page, zoom=zoom)

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