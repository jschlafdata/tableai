from ultralytics import YOLO, YOLOv10
from .vision import Detect

from importlib.metadata import version as _metadata_version


__all__ = (
    '__version__',
    'YOLOv10',
    'Detect'
)

__version__ = _metadata_version('tableai_plugins')
