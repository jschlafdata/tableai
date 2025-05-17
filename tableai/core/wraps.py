from functools import wraps
import fitz
from typing import Callable, Any, Optional
from tableai.data_loaders.files import FileReader
# from tableai.nodes.file_node import DirectoryFileNode  # adjust to your import path

def fitzAwrap(func):
    @wraps(func)
    def decorator(*args, **kwargs):
        # Handle both class methods and standalone functions
        if hasattr(args[0], '__class__'):
            node = args[1]  # self is args[0], node is next
        else:
            node = args[0]

        path = node.stage_paths[node.current_stage].get("abs_path")
        print(f"running load for path: {path}")
        recovery = node.stage_paths[node.current_stage].get("recovery_path")

        doc = fitz.open(path)
        try:
            return func(*args, doc=doc, **kwargs)
        finally:
            doc.close()

    return decorator


def FitzRUN(method):
    @wraps(method)
    def wrapper(self, fn, *args, **kwargs):
        stage = self.current_stage
        path = str(self.local_path)
        recovery = self.stage_paths[stage].get("recovery_path")
        
        doc = fitz.open(path)
        try:
            return method(self, fn, doc, *args, **kwargs)
        finally:
            doc.close()
    return wrapper