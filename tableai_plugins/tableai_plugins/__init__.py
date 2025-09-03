from importlib.metadata import version as _metadata_version
from ultralytics import YOLO, YOLOv10
from .vision import Detect
from .dit import (
    first_page_image,
    dit_embed,
    cluster_dir
)

__all__ = (
    '__version__',
    'YOLOv10',
    'Detect',
    'first_page_image',
    'dit_embed',
    'cluster_dir'
)


__version__ = _metadata_version('tableai_plugins')