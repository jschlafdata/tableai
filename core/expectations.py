import functools
import json
from typing import Optional, Callable
from json import JSONDecodeError

class BaseExpectation:
    def __init__(self, ok: bool = False, return_type=None, error_class=None, handler=None):
        """
        :param ok: If True, failures are 'ok' and return a fallback value.
        :param return_type: A type or value to return on failure if ok=True.
        :param error_class: A custom error class or callable to instantiate on failure.
        :param handler: A callable or class with a handle_failure() method used on failure.
        """
        self.ok = ok
        self.return_type = return_type
        self.error_class = error_class
        self.handler = handler

    def on_failure(self, e: Exception, *args, **kwargs):
        """
        Determine what to do when the wrapped function fails.
        1. If a handler is provided and defines handle_failure(), use it.
        2. Else if a handler is a simple callable (function), call it.
        3. If ok=True, return return_type fallback.
        4. If error_class is provided, return or instantiate it.
        5. Otherwise re-raise the original exception.
        """
        # Check if handler is provided
        if self.handler is not None:
            # If handler has a handle_failure method, call that
            if hasattr(self.handler, 'handle_failure') and callable(getattr(self.handler, 'handle_failure')):
                return self.handler.handle_failure(e, *args, **kwargs)
            # If handler is a callable but not a class with handle_failure,
            # assume it's a function or a callable object that can be called directly
            elif callable(self.handler):
                return self.handler(e, *args, **kwargs)

        # If no handler or handler didn't resolve the issue:
        if self.ok:
            if isinstance(self.return_type, type):
                if self.return_type is bool:
                    return False
                if self.return_type is str:
                    return ""
                return self.return_type()
            return self.return_type

        # If not ok and we have an error_class
        if self.error_class:
            if callable(self.error_class):
                return self._instantiate_error_class(e)
            else:
                return self.error_class

        # Else re-raise original exception
        raise e

    def _instantiate_error_class(self, original_exception):
        # Special handling if your error_class is something like JSONDecodeError
        # which requires specific arguments. Supply defaults if needed.
        if self.error_class is JSONDecodeError:
            # JSONDecodeError requires (msg, doc, pos)
            return JSONDecodeError(str(original_exception), "", 0)
        return self.error_class(str(original_exception))


def expects(type=None, error_class=None, handler=None):
    """
    A decorator that:
    - Tries to run the function.
    - Validates the result type if `type` is given.
    - On failure, tries `handler` if provided, else tries error_class or re-raises.
    """
    expectation = BaseExpectation(ok=False, error_class=error_class, handler=handler)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                # Validate the type if provided
                if type is not None:
                    expected_types = type if isinstance(type, (list, tuple)) else [type]
                    if not any(isinstance(result, t) for t in expected_types):
                        raise TypeError(f"Expected {expected_types}, got {type(result)}")
                return result
            except Exception as e:
                return expectation.on_failure(e, *args, **kwargs)
        return wrapper
    return decorator