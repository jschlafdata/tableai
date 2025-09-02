from importlib.metadata import version as _metadata_version
from .pdf import PDFDocument
from ._exceptions import PdfProcessingError, PdfPathDoesNotExist, PdfHasNoPagesError, PdfHasNoSizeError

__all__ = (
    '__version__',
    'PDFDocument',
    'PdfProcessingError', 
    'PdfPathDoesNotExist', 
    'PdfHasNoPagesError', 
    'PdfHasNoSizeError'
)

__version__ = _metadata_version('tableai_slim')