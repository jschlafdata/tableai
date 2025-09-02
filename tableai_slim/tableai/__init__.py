from importlib.metadata import version as _metadata_version
from .pdf import PDFDocument

__all__ = (
    '__version__',
    'PDFDocument'
)

__version__ = _metadata_version('tableai_slim')