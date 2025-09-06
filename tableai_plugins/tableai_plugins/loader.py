# tableai_plugins/loader.py
from importlib.metadata import entry_points
from typing import Dict, Type

class MissingOptionalDependency(ImportError):
    pass

def _installed_plugin_entry_points() -> Dict[str, object]:
    # Entry points declared under [project.entry-points."tableai.plugins"]
    eps = entry_points(group="tableai.plugins")
    return {ep.name: ep for ep in eps}

def list_plugins() -> Dict[str, str]:
    """Return {name: 'module:object'} for installed plugins."""
    out = {}
    for name, ep in _installed_plugin_entry_points().items():
        out[name] = f"{ep.module}:{ep.attr}" if ep.attr else ep.module
    return out

def has_plugin(name: str) -> bool:
    return name in _installed_plugin_entry_points()

def get_plugin(name: str):
    """
    Load and return the plugin *class* (not an instance).
    Raise MissingOptionalDependency with a helpful pip hint.
    """
    eps = _installed_plugin_entry_points()
    if name not in eps:
        raise MissingOptionalDependency(
            f"Plugin '{name}' is not installed.\n"
            f"Install with: pip install 'tableai-plugins[{name}]'"
        )
    return eps[name].load()