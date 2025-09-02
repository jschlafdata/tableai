from importlib.metadata import version as _metadata_version


__all__ = (
    '__version__'
)

__version__ = _metadata_version('tableai_slim')