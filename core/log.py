import logging
import inspect
import functools
from typing import Callable, Union, Any
import time 
import asyncio

### logging most critital for production runs & application development.

import warnings
warnings.filterwarnings("ignore")

def log_execution_time(level: Union[str, int] = "DEBUG") -> Callable:
    """
    A decorator that logs function execution time and details using the global logger.
    Only logs if the logger's level is less than or equal to the specified level.
    
    Args:
        level: Logging level as string ("DEBUG", "INFO", etc.) or integer (logging.DEBUG, logging.INFO, etc.)
        
    Usage:
        @log_execution_time("DEBUG")  # Only log if logger level <= DEBUG
        def my_function():
            pass
            
        @log_execution_time("INFO")   # Only log if logger level <= INFO
        async def my_async_function():
            pass
    """
    # Convert string level to integer if needed
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper())
    else:
        numeric_level = level
        
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            # Get logger from the main module
            logger = logging.getLogger('dbx_rag_logger')
            
            # Check if we should log based on levels
            if logger.getEffectiveLevel() > numeric_level:
                return await func(*args, **kwargs)
                
            # Get function details
            func_name = func.__name__
            
            # Determine the method type and class name
            if args:
                first_arg = args[0]
                if isinstance(first_arg, type):
                    class_name = first_arg.__name__
                    func_qualified_name = f"{class_name}.{func_name} (classmethod)"
                elif hasattr(first_arg, '__class__') and not isinstance(first_arg, (str, int, float, bool)):
                    class_name = first_arg.__class__.__name__
                    func_qualified_name = f"{class_name}.{func_name}"
                else:
                    qualname_parts = func.__qualname__.split('.')
                    if len(qualname_parts) > 1:
                        class_name = qualname_parts[-2]
                        func_qualified_name = f"{class_name}.{func_name} (static)"
                    else:
                        func_qualified_name = func_name
            else:
                qualname_parts = func.__qualname__.split('.')
                if len(qualname_parts) > 1:
                    class_name = qualname_parts[-2]
                    func_qualified_name = f"{class_name}.{func_name} (static)"
                else:
                    func_qualified_name = func_name
            
            # Log function entry
            logger.log(numeric_level, f"Entering {func_qualified_name}")
            
            start_time = time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
                
                execution_time = time.perf_counter() - start_time
                
                logger.log(
                    numeric_level,
                    f"Successfully completed {func_qualified_name} - "
                    f"Execution time: {execution_time:.3f} seconds"
                )
                
                return result
                
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                
                logger.error(
                    f"Error in {func_qualified_name} after {execution_time:.3f} seconds - "
                    f"Error: {str(e)}"
                )
                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            # Get logger from the main module
            logger = logging.getLogger('databricks_application')
            
            # Check if we should log based on levels
            if logger.getEffectiveLevel() > numeric_level:
                return func(*args, **kwargs)
            
            func_name = func.__name__
            
            if args:
                first_arg = args[0]
                if isinstance(first_arg, type):
                    class_name = first_arg.__name__
                    func_qualified_name = f"{class_name}.{func_name} (classmethod)"
                elif hasattr(first_arg, '__class__') and not isinstance(first_arg, (str, int, float, bool)):
                    class_name = first_arg.__class__.__name__
                    func_qualified_name = f"{class_name}.{func_name}"
                else:
                    qualname_parts = func.__qualname__.split('.')
                    if len(qualname_parts) > 1:
                        class_name = qualname_parts[-2]
                        func_qualified_name = f"{class_name}.{func_name} (static)"
                    else:
                        func_qualified_name = func_name
            else:
                qualname_parts = func.__qualname__.split('.')
                if len(qualname_parts) > 1:
                    class_name = qualname_parts[-2]
                    func_qualified_name = f"{class_name}.{func_name} (static)"
                else:
                    func_qualified_name = func_name
            
            logger.log(numeric_level, f"Entering {func_qualified_name}")
            
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                
                execution_time = time.perf_counter() - start_time
                
                logger.log(
                    numeric_level,
                    f"Successfully completed {func_qualified_name} - "
                    f"Execution time: {execution_time:.3f} seconds"
                )
                
                return result
                
            except Exception as e:
                execution_time = time.perf_counter() - start_time
                
                logger.error(
                    f"Error in {func_qualified_name} after {execution_time:.3f} seconds - "
                    f"Error: {str(e)}"
                )
                raise
        
        # Return appropriate wrapper based on whether the function is async or not
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
            
    return decorator

class ContextFilter(logging.Filter):
    """
    A filter that adds context information to log records including:
    - script name
    - class name (if applicable)
    - function name
    """
    def filter(self, record):
        try:
            # Get the caller's frame
            current_frame = inspect.currentframe()
            
            # Walk up the stack until we find the caller's frame
            caller_frame = None
            frame = current_frame
            while frame:
                # Skip frames from this module and logging module
                if (frame.f_code.co_filename != __file__ and 
                    'logging' not in frame.f_code.co_filename and
                    'logger.py' not in frame.f_code.co_filename):
                    caller_frame = frame
                    break
                frame = frame.f_back

            if caller_frame:
                # Get context details
                code = caller_frame.f_code
                # Default module name for notebooks
                module_name = "notebook"
                
                # Try to get function name
                func_name = code.co_name
                if func_name == '<module>':
                    func_name = 'global'
                    
                # Try to get class name
                class_name = None
                if 'self' in caller_frame.f_locals:
                    instance = caller_frame.f_locals['self']
                    class_name = instance.__class__.__name__
                elif 'cls' in caller_frame.f_locals:
                    class_name = caller_frame.f_locals['cls'].__name__
                    
                # Build context string
                if class_name:
                    record.context = f"[{class_name}.{func_name}]"
                else:
                    record.context = f"[{func_name}]"
                    
            else:
                record.context = "[unknown]"
                
        except Exception as e:
            record.context = "[unknown]"
        finally:
            # Clean up frames
            if 'current_frame' in locals():
                del current_frame
            if 'caller_frame' in locals():
                del caller_frame
            if 'frame' in locals():
                del frame
                
        return True

class TableaiLogger:
    """
    Custom logger for Databricks that uses notebook's display() function
    """
    _instance = None
    _logger = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TableaiLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._logger is None:
            self._setup_logger()

    def _setup_logger(self):
        """Initialize the logger with custom formatting and filtering."""
        self._logger = logging.getLogger('table_ai_logger')
        
        if not self._logger.handlers:
            # Create custom handler for Databricks
            handler = LogHandler()
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s %(context)s %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)
            self._logger.addFilter(ContextFilter())

    def get_logger(self) -> logging.Logger:
        """Return the configured logger instance."""
        return self._logger

class LogHandler(logging.Handler):
    """
    Custom handler that uses Databricks' display() function
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            print(msg, flush=True)
        except Exception:
            self.handleError(record)

# Create a global logger instance
logger = TableaiLogger().get_logger()

# Convenience functions
def debug(msg: str, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)

def info(msg: str, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def warning(msg: str, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs):
    logger.error(msg, *args, **kwargs)

def critical(msg: str, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)
