from importlib.metadata import version as _metadata_version
from .pdf import PDFDocument, DocumentIdentity, Page
from .exceptions import (
    PdfProcessingError, PdfPathDoesNotExist, PdfHasNoPagesError, PdfHasNoSizeError,
)

__all__ = (
    '__version__',
    'PDFDocument',
    "DocumentIdentity", 
    "Page",
    "PdfProcessingError", 
    "PdfPathDoesNotExist", 
    "PdfHasNoPagesError", 
    "PdfHasNoSizeError",
)

__version__ = _metadata_version('tableai_slim')