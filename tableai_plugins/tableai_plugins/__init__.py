from importlib.metadata import version as _metadata_version
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _metadata_version("tableai_plugins")  # distribution name
except PackageNotFoundError:
    __version__ = "0.0.0"

from .loader import get_plugin, has_plugin, list_plugins, MissingOptionalDependency

__all__ = ("__version__", "get_plugin", "has_plugin", "list_plugins", "MissingOptionalDependency")