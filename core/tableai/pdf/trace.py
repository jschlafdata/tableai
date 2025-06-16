from typing import List, Dict, Any, Tuple, Optional, Callable
from pydantic import BaseModel
import inspect
import textwrap
import hashlib
from tableai.pdf.coordinates import (
    Geometry,
    CoordinateMapping
)

from __future__ import annotations

import inspect
import textwrap
from typing import Any, Callable, Dict, List, Optional

try:
    # Optional import – only needed when you pass Pydantic models
    from pydantic import BaseModel
except ImportError:   # pragma: no cover
    BaseModel = object    # type: ignore


# ---------- helper ---------------------------------------------------------- #
def _callable_payload(fn: Callable) -> Dict[str, Any]:
    """
    Represent a callable as a serialisable payload.

    Returned structure is intentionally simple so it can be carried around in
    the trace dictionary and later re‑hydrated / pretty‑printed by a renderer.
    """
    label = getattr(fn, "__qualname__", getattr(fn, "__name__", str(fn)))
    src = None
    try:
        src = textwrap.dedent(inspect.getsource(fn)).strip()
    except (OSError, TypeError):  # built‑ins, C‑extensions, lambdas built in REPL
        pass

    closure_vars: Dict[str, Any] | None = None
    if fn.__closure__:
        closure_vars = {
            cellvar: cell.cell_contents
            for cellvar, cell in zip(fn.__code__.co_freevars, fn.__closure__)
        }

    return {
        "label": label,
        "source": src,
        "closure_vars": closure_vars,
        "is_lambda": fn.__name__ == "<lambda>",
    }


# ---------- core TraceLog --------------------------------------------------- #
class TraceLog:
    """
    Lightweight execution trace.

    • Records every step (name, callable, parameters, result length, description)
    • Stores arbitrary metadata (e.g. images) via `add_metadata`
    • Extracts the unique user‑defined functions involved in a run
    • Returns all data as plain Python structures (dictionaries / lists)
    """

    def __init__(self) -> None:
        self.steps: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

    # --------------------------------------------------------------------- #
    # public helpers
    # --------------------------------------------------------------------- #
    def add_metadata(self, metadata: Dict[str, Any]) -> None:
        """Merge arbitrary metadata into the trace (e.g. base64 images)."""
        self.metadata.update(metadata)

    def run_and_log_step(
        self,
        step_name: str,
        function: Callable,
        *,
        log_function: Optional[Callable] = None,
        params: Optional[Dict[str, Any] | BaseModel] = None,
        description: Optional[str] = None,
    ) -> Any:
        """
        Execute `function`, record a step, and return the result.

        `log_function` lets you separate “execution callable” from the callable
        that should be captured in the trace (useful when executing lambdas that
        delegate to real functions).
        """
        result = function()

        if log_function:
            func_to_log = log_function
            composite_name = (
                f"lambda → {getattr(log_function, '__qualname__', log_function.__name__)}"
            )
        else:
            func_to_log = getattr(function, "func", function)  # unwrap functools.partial
            composite_name = None

        self.add_step(
            step_name=step_name,
            function=func_to_log,
            params=params,
            result=result,
            description=description,
            composite_name=composite_name,
        )
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Retrieve the complete, serialisable trace."""
        return {"steps": self.steps, "metadata": self.metadata}

    # --------------------------------------------------------------------- #
    # core recording
    # --------------------------------------------------------------------- #
    def add_step(
        self,
        *,
        step_name: str,
        function: Callable,
        params: Optional[Dict[str, Any] | BaseModel] = None,
        result: Any = None,
        description: Optional[str] = None,
        composite_name: Optional[str] = None,
    ) -> None:
        """Append a step entry to `self.steps`."""
        raw_params = (
            params.model_dump(mode="json") if isinstance(params, BaseModel) else params
        )
        serialised_params = self._serialize_parameters(raw_params)

        if isinstance(serialised_params, dict):
            # Drop keys with value None to keep the trace concise
            serialised_params = {
                k: v for k, v in serialised_params.items() if v is not None
            }

        self.steps.append(
            {
                "step_name": step_name,
                "function_obj": function,  # keep the live object for later
                "function_name": composite_name
                or getattr(function, "__name__", str(function)),
                "parameters": serialised_params,
                "description": description
                or textwrap.dedent(function.__doc__ or "").strip(),
                "output_count": len(result)
                if hasattr(result, "__len__")
                else (1 if result is not None else 0),
            }
        )

    # --------------------------------------------------------------------- #
    # parameter serialisation
    # --------------------------------------------------------------------- #
    def _serialize_parameters(self, obj: Any) -> Any:
        """Recursively walk `obj`, replacing callables with a payload dict."""
        if callable(obj):
            return _callable_payload(obj)

        if isinstance(obj, dict):
            return {k: self._serialize_parameters(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._serialize_parameters(v) for v in obj]
        return obj

    # --------------------------------------------------------------------- #
    # analysis helpers
    # --------------------------------------------------------------------- #
    def _extract_global_functions(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a mapping {function_name: {description, sample_params}} extracted
        from all recorded steps.  Lambdas used *directly* are ignored – but
        lambdas that delegate (via `lambda → func_name`) are resolved to
        `func_name`.
        """
        functions: Dict[str, Dict[str, Any]] = {}

        for step in self.steps:
            func_obj = step.get("function_obj")
            if not callable(func_obj):
                continue

            func_name = step["function_name"]

            # Skip pure lambdas (no right‑hand side)
            if func_name.startswith("<lambda"):
                continue
            if func_name.startswith("lambda") and "→" not in func_name:
                continue

            # Use right‑hand side of composite names (lambda → real_fn)
            display_name = func_name.split("→")[-1].strip()

            if display_name not in functions:
                functions[display_name] = {
                    "description": textwrap.dedent(func_obj.__doc__ or "").strip(),
                    "sample_params": step["parameters"],
                }

        return functions
    
class TraceableWorkflow:
    """
    A generic runner that executes any compatible workflow function and manages
    the lifecycle of the TraceLog for that run.
    """
    def __init__(self, pdf_model: 'PDFModel'):
        self.pdf_model = pdf_model
        self.last_trace: Optional[TraceLog] = None

    def run(
        self, 
        workflow_function: Callable, 
        params: Optional[BaseModel] = None, 
        **kwargs
    ) -> Any:
        """
        Runs a given workflow function, injecting the pdf_model and a new trace log.

        Args:
            workflow_function: The pluggable workflow function to execute. It must accept
                               (pdf_model, trace, params, **kwargs).
            params: A Pydantic model of parameters for the workflow.
            **kwargs: Any additional keyword arguments for the workflow.
        
        Returns:
            The final data result of the workflow function.
        """
        trace = TraceLog()
        
        # Execute the workflow, passing it the context it needs
        result = workflow_function(
            self.pdf_model,
            trace=trace,
            params=params,
            **kwargs
        )
        
        self.last_trace = trace
        return result

    def get_last_trace(self) -> Optional[TraceLog]:
        """Returns the TraceLog from the most recent run."""
        if not self.last_trace:
            print("No workflow has been run yet.")
        return self.last_trace

# def _describe_callable(fn: Callable) -> str:
#     """The same logic used by TraceLog, made standalone for reuse."""
#     try:
#         name = fn.__qualname__
#     except AttributeError:
#         name = type(fn).__qualname__

#     if name == '<lambda>':
#         try:
#             src = inspect.getsource(fn).strip()
#             one_line = ' '.join(src.split())
#             sig = hashlib.sha1(one_line.encode()).hexdigest()[:6]
#             return f"lambda[{sig}]: {one_line}"
#         except (OSError, TypeError):
#             return "lambda[unknown-src]"
#     return name

# def _callable_payload(fn: Callable) -> dict:
#     """Return JSON‑serialisable metadata for any callable."""
#     payload = {"label": _describe_callable(fn)}
    
#     # Analyze ALL lambdas, not just those with closure variables
#     if getattr(fn, '__name__', '') == '<lambda>':
#         # Handle closure variables if they exist
#         if fn.__closure__:
#             closure_vars = {}
#             used_vars = list(fn.__code__.co_freevars)
            
#             for var, cell in zip(fn.__code__.co_freevars, fn.__closure__):
#                 closure_vars[var] = repr(cell.cell_contents)
            
#             payload["closure_vars"] = closure_vars
#             if used_vars:
#                 payload["uses_vars"] = used_vars
        
#         # Try to get the lambda body for ALL lambdas
#         try:
#             import inspect
#             source = inspect.getsource(fn).strip()
#             # Extract just the lambda body (after the colon)
#             if ':' in source:
#                 lambda_body = source.split(':', 1)[1]
                
#                 # Clean up the lambda body more carefully
#                 lambda_body = lambda_body.strip()
                
#                 # Remove trailing comma if it exists (from being in a list/dict)
#                 if lambda_body.endswith(','):
#                     lambda_body = lambda_body[:-1].strip()
                
#                 # Remove trailing closing brackets/parens that might be from the containing structure
#                 while lambda_body.endswith(('\n', ' ', '\t')):
#                     lambda_body = lambda_body.rstrip('\n \t')
                
#                 # Check for proper parentheses balance
#                 open_parens = lambda_body.count('(')
#                 close_parens = lambda_body.count(')')
                
#                 # If we have unbalanced parens and the body doesn't end with ')', 
#                 # it might be cut off
#                 if open_parens > close_parens and not lambda_body.rstrip().endswith(')'):
#                     # Try to get more context - look for the complete lambda expression
#                     lines = source.split('\n')
#                     lambda_found = False
#                     complete_body = ""
#                     paren_count = 0
                    
#                     for line in lines:
#                         if ':' in line and 'lambda' in line:
#                             lambda_found = True
#                             # Start from after the colon
#                             start_idx = line.find(':') + 1
#                             complete_body = line[start_idx:]
#                             # Count parentheses in this first line
#                             paren_count = complete_body.count('(') - complete_body.count(')')
#                         elif lambda_found:
#                             # Continue collecting lines until parentheses are balanced
#                             complete_body += '\n' + line
#                             paren_count += line.count('(') - line.count(')')
                            
#                             # Stop when we have balanced parentheses or hit certain endings
#                             if paren_count <= 0:
#                                 break
                    
#                     if complete_body.strip():
#                         lambda_body = complete_body.strip()
#                         # Remove trailing artifacts
#                         if lambda_body.endswith(','):
#                             lambda_body = lambda_body[:-1].strip()
                
#                 payload["lambda_body"] = lambda_body
                
#                 # Extract specific attribute access patterns (for lambdas with closure vars)
#                 if fn.__closure__:
#                     used_vars = list(fn.__code__.co_freevars)
#                     import re
#                     attribute_accesses = []
#                     for var in used_vars:
#                         # Find patterns like p.HEADER_BOUND, p.FOOTER_BOUND, etc.
#                         pattern = rf'{re.escape(var)}\.(\w+)'
#                         matches = re.findall(pattern, lambda_body)
#                         for match in matches:
#                             attribute_accesses.append(f"{var}.{match}")
                    
#                     if attribute_accesses:
#                         payload["accesses"] = list(set(attribute_accesses))  # Remove duplicates
                    
#         except (OSError, TypeError, inspect.IndentationError):
#             # If we can't get the source, try to at least show what variables it uses
#             if fn.__closure__:
#                 used_vars = list(fn.__code__.co_freevars)
#                 payload["lambda_body"] = f"<lambda using: {', '.join(used_vars)}>"
#             else:
#                 payload["lambda_body"] = "<lambda: source unavailable>"
    
#     return payload

# class TraceLog:
#     """
#     An active logger that both executes and records the provenance of a workflow,
#     fully surfacing named and anonymous callables along with their captured data.
#     """
#     def __init__(self):
#         self.steps: List[Dict[str, Any]] = []
#         self.metadata: dict = {}

#     def add_metadata(self, metadata: Dict[str, Any]):
#         """Add metadata to the trace log (e.g., for storing images)."""
#         self.metadata.update(metadata)

#     def run_and_log_step(self, step_name: str, function: Callable, log_function: Optional[Callable] = None, params: Optional[Dict[str, Any]] = None, description: Optional[str] = None) -> Any:
#         """
#         Executes a function as a traceable step, logs the details, and returns its result.
#         """
#         # Execute the function to get the result
#         result = function()
        
#         # Use the explicit log_function for metadata if provided, 
#         # otherwise fall back to inspecting the lambda.
#         if log_function:
#             # Store both the execution function and the log function
#             func_to_log = log_function
#             # Create a composite name showing the relationship
#             composite_name = f"lambda → {getattr(log_function, '__qualname__', getattr(log_function, '__name__', str(log_function)))}"
#         else:
#             func_to_log = function.func if hasattr(function, 'func') else function
#             composite_name = None
        
#         self.add_step(step_name, func_to_log, params, result, description, composite_name)
#         return result

#     def add_step(self, step_name: str, function: Callable, params: Optional[BaseModel] = None, result: Any = None, description: Optional[str] = None, composite_name: Optional[str] = None):
#         """Adds a record of a processing step to the log."""
#         # Extract raw params: preserve callables if BaseModel
#         if isinstance(params, BaseModel):
#             raw = params.model_dump(mode='json')  # use mode='json' so field_serializers fire
#         else:
#             raw = params

#         # Recursively serialize parameters, replacing callables with descriptive source labels
#         parameters = self._serialize_parameters(raw)

#         # Drop None values from resulting dict
#         if isinstance(parameters, dict):
#             parameters = {k: v for k, v in parameters.items() if v is not None}

#         step_info = {
#             "step_name": step_name,
#             "function_obj": function,  # Store the actual function object
#             "function_name": composite_name or getattr(function, '__name__', str(function)),
#             "parameters": parameters,
#             "description": description or textwrap.dedent(function.__doc__ or "").strip(),
#             "output_count": len(result) if hasattr(result, '__len__') else 1 if result is not None else 0
#         }
#         self.steps.append(step_info)

#     def _serialize_parameters(self, obj: Any) -> Any:
#         """
#         Recursively replace callables in the parameters structure with their descriptive labels.
#         """
#         if callable(obj):
#             return _callable_payload(obj)  # Use the enhanced version with closure inspection
#         if isinstance(obj, dict):
#             return {k: self._serialize_parameters(v) for k, v in obj.items()}
#         if isinstance(obj, list):
#             return [self._serialize_parameters(v) for v in obj]
#         return obj

#     def _format_lambda_body(self, body: str) -> str:
#         """Format lambda body with Black-style formatting"""
#         try:
#             # Try to use Black if available
#             import black
#             # Wrap in a dummy function for Black to parse
#             dummy_code = f"lambda d: {body}"
#             formatted = black.format_str(dummy_code, mode=black.FileMode(line_length=80))
#             # Extract just the lambda part
#             if formatted.startswith("lambda d: "):
#                 return formatted[10:].strip()
#             return formatted.strip()
#         except ImportError:
#             # Fallback to manual formatting if Black isn't available
#             return self._manual_format_lambda(body)
    
#     def _manual_format_lambda(self, body: str) -> str:
#         """Manual Black-style formatting for lambda bodies"""
#         # Clean up whitespace
#         body = body.strip()
        
#         # Add spaces around operators
#         import re
#         body = re.sub(r'([<>=!]+)', r' \1 ', body)
#         body = re.sub(r'\s+', ' ', body)  # Collapse multiple spaces
        
#         # Format parentheses with proper spacing
#         body = re.sub(r'\(\s*', '(', body)
#         body = re.sub(r'\s*\)', ')', body)
        
#         # Format logical operators
#         body = body.replace(' and ', ' and ')
#         body = body.replace(' or ', ' or ')
        
#         # Break long lines
#         if len(body) > 80:
#             # Find good break points (after 'and', 'or')
#             parts = []
#             current = ""
            
#             for part in re.split(r'(\s+(?:and|or)\s+)', body):
#                 if current and len(current + part) > 80:
#                     parts.append(current.strip())
#                     current = part
#                 else:
#                     current += part
            
#             if current:
#                 parts.append(current.strip())
            
#             if len(parts) > 1:
#                 # Multi-line format
#                 formatted = "(\n"
#                 for i, part in enumerate(parts):
#                     if i > 0:
#                         formatted += "\n    "
#                     formatted += f"    {part}"
#                 formatted += "\n)"
#                 return formatted
        
#         return body

#     def format_json_outline(self, data: Any, prefix: str = "", max_str_len: int = 100) -> List[str]:
#         """Outline-style dump of nested dict/list for LLM consumption with enhanced lambda formatting."""
#         lines: List[str] = []
#         if isinstance(data, dict):
#             for key, value in data.items():
#                 cur = f"{prefix}.{key}" if prefix else key
#                 if isinstance(value, dict):
#                     # Special handling for lambda payloads
#                     if 'label' in value and 'lambda_body' in value:
#                         lines.append(f"{cur} (lambda function):")
                        
#                         # Show closure variables in full (only if they exist)
#                         if 'closure_vars' in value and value['closure_vars']:
#                             for var_name, var_value in value['closure_vars'].items():
#                                 lines.append(f"{cur}.closure_vars.{var_name}: {var_value}")
                        
#                         # Show lambda body with Python formatting
#                         if 'lambda_body' in value:
#                             body = value['lambda_body']
#                             lines.append(f"{cur}.lambda_body:")
#                             lines.append("```python")
#                             # Format with Black-style formatting
#                             formatted_body = self._format_lambda_body(body)
#                             lines.append(f"lambda d: {formatted_body}")
#                             lines.append("```")
                        
#                         # Show accesses in a clean list
#                         if 'accesses' in value and value['accesses']:
#                             lines.append(f"{cur}.accesses: {', '.join(value['accesses'])}")
#                     else:
#                         lines.append(f"{cur} (object with {len(value)} properties)")
#                         lines.extend(self.format_json_outline(value, cur, max_str_len))
#                 elif isinstance(value, list):
#                     # Don't show the redundant array header, just show the contents
#                     if value:
#                         # Show all items for small arrays, or first few for large ones
#                         if len(value) <= 5:
#                             for i, item in enumerate(value):
#                                 if isinstance(item, dict) and 'label' in item and 'lambda_body' in item:
#                                     # Special lambda formatting
#                                     lines.append(f"{cur}[{i}] (lambda function):")
                                    
#                                     # Show closure variables in full (only if they exist)
#                                     if 'closure_vars' in item and item['closure_vars']:
#                                         for var_name, var_value in item['closure_vars'].items():
#                                             lines.append(f"{cur}[{i}].closure_vars.{var_name}: {var_value}")
                                    
#                                     # Show lambda body with Python formatting
#                                     if 'lambda_body' in item:
#                                         body = item['lambda_body']
#                                         lines.append(f"{cur}[{i}].lambda_body:")
#                                         lines.append("```python")
#                                         # Format with Black-style formatting
#                                         formatted_body = self._format_lambda_body(body)
#                                         lines.append(f"lambda d: {formatted_body}")
#                                         lines.append("```")
                                    
#                                     # Show accesses in a clean list
#                                     if 'accesses' in item and item['accesses']:
#                                         lines.append(f"{cur}[{i}].accesses: {', '.join(item['accesses'])}")
                                        
#                                 elif isinstance(item, (dict, list)):
#                                     lines.extend(self.format_json_outline(item, f"{cur}[{i}]", max_str_len))
#                                 else:
#                                     item_str = str(item)
#                                     if len(item_str) > max_str_len:
#                                         lines.append(f"{cur}[{i}]: \"{item_str[:max_str_len]}...\" (string, {len(item_str)} chars)")
#                                     else:
#                                         lines.append(f"{cur}[{i}]: {item_str}")
#                         else:
#                             # For large arrays, show first few items
#                             first = value[0]
#                             if isinstance(first, dict):
#                                 lines.extend(self.format_json_outline(first, f"{cur}[0]", max_str_len))
#                             else:
#                                 preview = ", ".join(
#                                     (str(item)[:20] + '...' if len(str(item))>20 else str(item))
#                                     for item in value[:3]
#                                 )
#                                 lines.append(f"{cur}: {preview}")
#                 else:
#                     s = str(value)
#                     if len(s) > max_str_len:
#                         lines.append(f"{cur}: \"{s[:max_str_len]}...\" (string, {len(s)} chars)")
#                     else:
#                         lines.append(f"{cur}: {s}")
#         elif isinstance(data, list):
#             for i, item in enumerate(data[:10]):
#                 cur = f"{prefix}[{i}]" if prefix else f"[{i}]"
#                 if isinstance(item, (dict, list)):
#                     lines.extend(self.format_json_outline(item, cur, max_str_len))
#                 else:
#                     s = str(item)
#                     if len(s) > max_str_len:
#                         lines.append(f"{cur}: \"{s[:max_str_len]}...\" (string, {len(s)} chars)")
#                     else:
#                         lines.append(f"{cur}: {s}")
#             if len(data) > 10:
#                 lines.append(f"{prefix}[...] ({len(data)-10} more items)")
#         return lines

#     def _extract_global_functions(self) -> Dict[str, Dict[str, Any]]:
#         """Extract unique functions and their documentation from all steps."""
#         functions = {}
        
#         for step in self.steps:
#             func_obj = step.get('function_obj')
#             if not callable(func_obj):
#                 continue
                
#             func_name = step['function_name']
            
#             # Skip pure lambda functions
#             if 'lambda[' in func_name and '→' not in func_name:
#                 continue
#             if func_name == '<lambda>' or func_name.startswith('<lambda'):
#                 continue
            
#             # For composite names (lambda → real_function), extract the real function name
#             if '→' in func_name:
#                 real_func_name = func_name.split('→')[-1].strip()
#                 display_name = real_func_name
#             else:
#                 display_name = func_name
                
#             if display_name not in functions:
#                 functions[display_name] = {
#                     'description': textwrap.dedent(func_obj.__doc__ or "").strip(),
#                     'sample_params': step['parameters']
#                 }
        
#         return functions

#     def display_images(self):
#         """Display base64 images stored in metadata."""
#         if 'sample_images' not in self.metadata:
#             print("No sample images available in trace.")
#             return
        
#         sample_images = self.metadata['sample_images']
#         image_config = self.metadata.get('image_config', {})
        
#         print(f"\n📸 SAMPLE IMAGES (Zoom: {image_config.get('zoom', 1.0)}x, "
#               f"Pages: 0-{image_config.get('page_limit', 'all')})")
#         print("="*60)
        
#         # Display metadata
#         if 'metadata' in sample_images:
#             meta = sample_images['metadata']
#             print(f"📊 Results: {sample_images.get('noise_regions_count', 0)} noise regions, "
#                   f"{sample_images.get('inverse_regions_count', 0)} content regions")
#             print(f"🎨 Colors: Noise={meta.get('noise_color', 'red')}, "
#                   f"Content={meta.get('inverse_color', 'blue')}")
        
#         # Display images using IPython if available
#         try:
#             from IPython.display import display, Image as IPImage, HTML
#             import base64
            
#             # Display original image
#             if 'original_pdf_sample' in sample_images:
#                 print(f"\n🗄️ Original PDF Sample:")
#                 display(HTML("<h4>Original PDF</h4>"))
#                 img_bytes = base64.b64decode(sample_images['original_pdf_sample'])
#                 display(IPImage(data=img_bytes, width=800))
            
#             # Display annotated image  
#             if 'annotated_pdf_sample' in sample_images:
#                 print(f"\n🎯 Annotated PDF Sample:")
#                 display(HTML("<h4>Noise Detection Results</h4>"))
#                 img_bytes = base64.b64decode(sample_images['annotated_pdf_sample'])
#                 display(IPImage(data=img_bytes, width=800))
                
#         except ImportError:
#             # Fallback for non-Jupyter environments
#             print("📷 Original PDF: Base64 image available")
#             print("📷 Annotated PDF: Base64 image available")
#             print("   To view: Use in Jupyter environment or decode base64 manually")

#     def get_formatted_trace_with_images(self, use_json_outline=False, max_str_len=100, include_global_functions=True, include_images=False) -> str:
#         """
#         Returns the trace as a formatted string for LLM consumption.
#         Images are excluded by default (too large for text processing).
#         """
#         lines = ["EXECUTION TRACE SUMMARY"]
#         lines.append("=" * 50)
        
#         if not self.steps:
#             lines.append("No steps were recorded.")
#             return '\n'.join(lines)
        
#         # Add summary if sample images exist
#         if 'sample_images' in self.metadata:
#             sample_images = self.metadata['sample_images']
#             lines.append(f"\nRESULT SUMMARY:")
#             lines.append(f"- Noise regions detected: {sample_images.get('noise_regions_count', 0)}")
#             lines.append(f"- Content regions identified: {sample_images.get('inverse_regions_count', 0)}")
#             lines.append(f"- Pages analyzed: {sample_images.get('pages_included', 'unknown')}")
            
#             if not include_images:
#                 lines.append(f"- Sample images: Available (excluded from text report)")
#             elif include_images:
#                 lines.append(f"- Original PDF image: {len(sample_images.get('original_pdf_sample', ''))} base64 chars")
#                 lines.append(f"- Annotated PDF image: {len(sample_images.get('annotated_pdf_sample', ''))} base64 chars")
        
#         # Use existing formatting logic
#         return self.get_formatted_trace(use_json_outline, max_str_len, include_global_functions)

#     def _print_result_focused_summary(self, include_images=True, use_json_outline=False, max_str_len=100, include_global_functions=True, noise_detection_result: 'NoiseDetectionResult'=None):
#         """Print summary combining result-focused format with complete execution details."""
#         try:
#             from rich.console import Console
#             from rich.panel import Panel
#             from rich.table import Table
#             from rich import box
            
#             console = Console()
            
#             if not self.steps:
#                 console.print(Panel("❌ No steps were recorded.", title="Execution Trace", style="red"))
#                 return
            
#             # Create the result object using the new method
#             result = self.create_noise_detection_result(noise_detection_result)
            
#             # === PART 1: RESULT-FOCUSED OVERVIEW ===
#             console.print(Panel.fit("🎯 NOISE DETECTION RESULT SUMMARY", style="bold cyan"))
            
#             # 1. OVERVIEW
#             console.print(Panel(result.overview, title="📋 OVERVIEW", border_style="blue"))
            
#             # 2. GOAL  
#             console.print(Panel(result.goal, title="🎯 GOAL", border_style="green"))
            
#             # 3. RESULT STATISTICS
#             stats_content = f"""[bold green]✅ PROCESSING COMPLETE[/bold green]

# [cyan]Noise Regions Detected:[/cyan] {result.noise_regions_count}
# [blue]Content Regions Identified:[/blue] {result.content_regions_count}  
# [yellow]Pages Analyzed:[/yellow] {result.pages_analyzed}
# [dim]Processing Time:[/dim] {result.processing_timestamp.strftime('%Y-%m-%d %H:%M:%S')}"""

#             if result.image_config:
#                 stats_content += f"""

# [dim]Image Settings:[/dim]
#   Zoom: {result.image_config.get('zoom', 1.0)}x
#   Colors: Noise={result.image_config.get('noise_color', 'red')}, Content={result.image_config.get('inverse_color', 'blue')}"""
            
#             console.print(Panel(stats_content, title="📊 RESULT STATISTICS", border_style="green"))
            
#             # 4. IMAGES
#             if include_images:
#                 console.print(Panel.fit("📸 RESULT IMAGES", style="bold magenta"))
#                 result.display_images()
            
#             # === PART 2: DETAILED EXECUTION ANALYSIS ===
#             console.print(Panel.fit("🔧 DETAILED EXECUTION ANALYSIS", style="bold blue"))
            
#             # Print Global Functions section if requested
#             if include_global_functions:
#                 import json
                
#                 global_functions = self._extract_global_functions()
                
#                 console.print(Panel.fit("🔧 GLOBAL FUNCTIONS USED", style="bold blue"))
                
#                 for func_name, func_info in global_functions.items():
#                     # Build the complete panel content
#                     panel_content = f"[bold green]{func_name}()[/]\n\n"
#                     panel_content += f"[dim]Description:[/] {func_info['description']}\n\n"
                    
#                     if func_info['sample_params']:
#                         panel_content += "[dim]Parameters:[/]\n\n"
                        
#                         if use_json_outline:
#                             param_lines = self.format_json_outline(func_info['sample_params'], max_str_len=max_str_len)
#                             if param_lines:
#                                 for line in param_lines:
#                                     panel_content += f"  {line}\n"
#                         else:
#                             # Clean JSON formatting without background highlighting
#                             try:
#                                 json_str = json.dumps(func_info['sample_params'], indent=2, ensure_ascii=False)
#                                 panel_content += json_str
#                             except (TypeError, ValueError):
#                                 panel_content += str(func_info['sample_params'])
                    
#                     console.print(Panel(panel_content, border_style="blue"))
#                     console.print()  # Add spacing
                
#                 console.print()  # Add spacing
            
#             # Print execution steps table
#             table = Table(title="📋 STEP-BY-STEP EXECUTION", box=box.ROUNDED)
#             table.add_column("Step", style="cyan", no_wrap=True)
#             table.add_column("Overview", style="magenta")
#             table.add_column("Function", style="green")
#             table.add_column("Description", style="blue")
#             table.add_column("Parameters Used", style="yellow")
#             table.add_column("Output Count", style="red")
            
#             for i, step in enumerate(self.steps, 1):
#                 # Format parameters using JSON outline if requested
#                 if use_json_outline and step['parameters']:
#                     param_lines = self.format_json_outline(step['parameters'], max_str_len=max_str_len)
#                     formatted_params = '\n'.join(param_lines) if param_lines else str(step['parameters'])
#                 elif step['parameters']:
#                     # Use pretty JSON formatting for parameters
#                     try:
#                         import json
#                         formatted_params = json.dumps(step['parameters'], indent=2, ensure_ascii=False)
#                     except (TypeError, ValueError):
#                         formatted_params = str(step['parameters'])
#                 else:
#                     formatted_params = str(step['parameters'])
                
#                 # Get description, truncate if too long for table display
#                 description = step.get('description', '').strip()
#                 if len(description) > 100:
#                     description = description[:100] + "..."
                
#                 table.add_row(
#                     str(i),
#                     step['step_name'],
#                     f"{step['function_name']}()",
#                     description,
#                     formatted_params,
#                     str(step['output_count'])
#                 )
            
#             console.print(table)
            
#             # === PART 3: OPTIONAL PARAMETERS FOR REFINEMENT ===
#             console.print(Panel(result.process_optional_parameters, 
#                               title="⚙️ PROCESS OPTIONAL PARAMETERS (for refinement)", 
#                               border_style="yellow", 
#                               expand=False))
            
#         except ImportError:
#             # Fallback for environments without Rich
#             self._print_basic_summary_with_complete_details(include_images, use_json_outline, max_str_len, include_global_functions)

#     def create_noise_detection_result(self, noise_detection_result: 'NoiseDetectionResult'):
#         """
#         Creates a NoiseDetectionResult from this TraceLog instance.
#         """
#         # Create and populate the result
#         result = noise_detection_result()
        
#         # Extract image data if available
#         if 'sample_images' in self.metadata:
#             sample_images = self.metadata['sample_images']
            
#             result.result_image = sample_images.get('annotated_pdf_sample')
#             result.original_image = sample_images.get('original_pdf_sample')
#             result.noise_regions_count = sample_images.get('noise_regions_count', 0)
#             result.content_regions_count = sample_images.get('inverse_regions_count', 0)
#             result.pages_analyzed = sample_images.get('pages_included', 0)
#             result.image_config = sample_images.get('metadata')
        
#         # Extract parameters used from first step if available
#         if self.steps:
#             first_step = self.steps[0]
#             result.parameters_used = first_step.get('parameters', {})
        
#         return result
    
#     def _print_basic_summary_with_complete_details(self, include_images=True, use_json_outline=False, max_str_len=100, include_global_functions=True):
#         """Fallback method for complete summary without Rich."""
#         result = self.create_noise_detection_result()
        
#         print("="*80)
#         print("🎯 NOISE DETECTION RESULT SUMMARY")
#         print("="*80)
        
#         # === PART 1: RESULT OVERVIEW ===
#         print("\n" + result.overview)
#         print("\n" + "="*80)
#         print(result.goal)
#         print("\n" + "="*80)
        
#         print(f"\n📊 RESULT STATISTICS:")
#         print(f"   ✅ Noise Regions: {result.noise_regions_count}")
#         print(f"   ✅ Content Regions: {result.content_regions_count}")
#         print(f"   ✅ Pages Analyzed: {result.pages_analyzed}")
#         print(f"   ⏰ Processing Time: {result.processing_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
#         if include_images:
#             print(f"\n📸 RESULT IMAGES:")
#             result.display_images()
        
#         # === PART 2: DETAILED EXECUTION ANALYSIS ===
#         print(f"\n{'='*80}")
#         print("🔧 DETAILED EXECUTION ANALYSIS")
#         print("="*80)
        
#         # Print Global Functions section
#         if include_global_functions:
#             print("\n🔧 GLOBAL FUNCTIONS USED [defaults and descriptors]")
#             print("="*60)
            
#             global_functions = self._extract_global_functions()
            
#             for func_name, func_info in global_functions.items():
#                 print(f"\n{func_name}():")
#                 print(f"  Description: {func_info['description']}")
                
#                 if func_info['sample_params']:
#                     if use_json_outline:
#                         param_lines = self.format_json_outline(func_info['sample_params'], max_str_len=max_str_len)
#                         if param_lines:
#                             print("  Parameter Structure:")
#                             for line in param_lines:
#                                 print(f"    {line}")
#                     else:
#                         # Use pretty JSON formatting
#                         try:
#                             import json
#                             json_str = json.dumps(func_info['sample_params'], indent=4, ensure_ascii=False)
#                             print("  Parameters:")
#                             for line in json_str.split('\n'):
#                                 print(f"    {line}")
#                         except (TypeError, ValueError):
#                             print(f"  Parameters: {func_info['sample_params']}")
            
#             print(f"\n{'='*60}")
        
#         # Print step execution
#         print("\n📋 STEP-BY-STEP EXECUTION")
#         print("="*60)
        
#         for i, step in enumerate(self.steps, 1):
#             print(f"\n┌─ Step {i} " + "─" * 40)
#             print(f"│ Overview.......: {step['step_name']}")
#             print(f"│ Function.......: {step['function_name']}()")
            
#             # Add description if it exists
#             if step.get('description') and step['description'].strip():
#                 # Wrap long descriptions
#                 description = step['description']
#                 if len(description) > 80:
#                     # Break long descriptions into multiple lines
#                     words = description.split()
#                     lines = []
#                     current_line = ""
#                     for word in words:
#                         if len(current_line + " " + word) <= 80:
#                             current_line += (" " if current_line else "") + word
#                         else:
#                             lines.append(current_line)
#                             current_line = word
#                     if current_line:
#                         lines.append(current_line)
                    
#                     print(f"│ Description....: {lines[0]}")
#                     for line in lines[1:]:
#                         print(f"│                 {line}")
#                 else:
#                     print(f"│ Description....: {description}")
            
#             # Format parameters using JSON outline if requested
#             if use_json_outline and step['parameters']:
#                 param_lines = self.format_json_outline(step['parameters'], max_str_len=max_str_len)
#                 if param_lines:
#                     print(f"│ Parameters....:")
#                     for line in param_lines:
#                         print(f"│   {line}")
#                 else:
#                     print(f"│ Parameters....: {step['parameters']}")
#             elif step['parameters']:
#                 # Use pretty JSON formatting
#                 try:
#                     import json
#                     json_str = json.dumps(step['parameters'], indent=2, ensure_ascii=False)
#                     print(f"│ Parameters....:")
#                     for line in json_str.split('\n'):
#                         print(f"│   {line}")
#                 except (TypeError, ValueError):
#                     print(f"│ Parameters....: {step['parameters']}")
#             else:
#                 print(f"│ Parameters....: {step['parameters']}")
            
#             print(f"│ Output Count...: {step['output_count']}")
#             print("└" + "─" * 50)
        
#         # === PART 3: OPTIONAL PARAMETERS ===
#         print(f"\n{'='*80}")
#         print("⚙️ PROCESS OPTIONAL PARAMETERS (for refinement):")
#         print("="*80)
#         print(result.process_optional_parameters)
        
#         print(f"\n{'='*80}")
#         print("✅ Complete Analysis Finished")
    
    
#     def print_summary(self, use_json_outline=False, max_str_len=100, include_global_functions=True, 
#                      include_images=True, result_format=True, noise_detection_result: 'NoiseDetectionResult'=None):
#         """
#         Enhanced print_summary with result-focused formatting plus complete execution details.
        
#         Args:
#             result_format: If True, displays in NoiseDetectionResult format with execution details
#         """
        
#         if result_format and hasattr(self, 'metadata') and 'sample_images' in self.metadata:
#             # Use the new comprehensive format (result overview + execution details)
#             self._print_result_focused_summary(include_images, use_json_outline, max_str_len, include_global_functions, noise_detection_result)

#     def _print_basic_summary_with_images(self, use_json_outline=False, max_str_len=100, include_global_functions=True, include_images=True):
#         """Fallback method for basic pretty printing with images."""
#         print("🔍 " + "="*60)
#         print("               EXECUTION TRACE SUMMARY")
#         print("="*60)
        
#         if not self.steps:
#             print("❌ No steps were recorded.")
#             return
        
#         # === NEW: Display images first ===
#         if include_images and hasattr(self, 'metadata') and 'sample_images' in self.metadata:
#             print("\n📸 VISUAL RESULTS")
#             print("="*60)
#             self.display_images()
#             print(f"\n{'='*60}")
        
#         # Print Global Functions section
#         if include_global_functions:
#             print("\n🔧 GLOBAL FUNCTIONS USED [defaults and descriptors]")
#             print("="*60)
            
#             global_functions = self._extract_global_functions()
            
#             for func_name, func_info in global_functions.items():
#                 print(f"\n{func_name}():")
#                 print(f"  Description: {func_info['description']}")
                
#                 if func_info['sample_params']:
#                     if use_json_outline:
#                         param_lines = self.format_json_outline(func_info['sample_params'], max_str_len=max_str_len)
#                         if param_lines:
#                             print("  Parameter Structure:")
#                             for line in param_lines:
#                                 print(f"    {line}")
#                     else:
#                         # Use pretty JSON formatting
#                         try:
#                             import json
#                             json_str = json.dumps(func_info['sample_params'], indent=4, ensure_ascii=False)
#                             print("  Parameters:")
#                             for line in json_str.split('\n'):
#                                 print(f"    {line}")
#                         except (TypeError, ValueError):
#                             print(f"  Parameters: {func_info['sample_params']}")
            
#             print(f"\n{'='*60}")
        
#         # Print step execution
#         print("\n📋 STEP-BY-STEP EXECUTION")
#         print("="*60)
        
#         for i, step in enumerate(self.steps, 1):
#             print(f"\n┌─ Step {i} " + "─" * 40)
#             print(f"│ Overview.......: {step['step_name']}")
#             print(f"│ Function.......: {step['function_name']}()")
            
#             # Add description if it exists
#             if step.get('description') and step['description'].strip():
#                 # Wrap long descriptions
#                 description = step['description']
#                 if len(description) > 80:
#                     # Break long descriptions into multiple lines
#                     words = description.split()
#                     lines = []
#                     current_line = ""
#                     for word in words:
#                         if len(current_line + " " + word) <= 80:
#                             current_line += (" " if current_line else "") + word
#                         else:
#                             lines.append(current_line)
#                             current_line = word
#                     if current_line:
#                         lines.append(current_line)
                    
#                     print(f"│ Description....: {lines[0]}")
#                     for line in lines[1:]:
#                         print(f"│                 {line}")
#                 else:
#                     print(f"│ Description....: {description}")
            
#             # Format parameters using JSON outline if requested
#             if use_json_outline and step['parameters']:
#                 param_lines = self.format_json_outline(step['parameters'], max_str_len=max_str_len)
#                 if param_lines:
#                     print(f"│ Parameters....:")
#                     for line in param_lines:
#                         print(f"│   {line}")
#                 else:
#                     print(f"│ Parameters....: {step['parameters']}")
#             elif step['parameters']:
#                 # Use pretty JSON formatting
#                 try:
#                     import json
#                     json_str = json.dumps(step['parameters'], indent=2, ensure_ascii=False)
#                     print(f"│ Parameters....:")
#                     for line in json_str.split('\n'):
#                         print(f"│   {line}")
#                 except (TypeError, ValueError):
#                     print(f"│ Parameters....: {step['parameters']}")
#             else:
#                 print(f"│ Parameters....: {step['parameters']}")
            
#             print(f"│ Output Count...: {step['output_count']}")
#             print("└" + "─" * 50)
        
#         # === NEW: Add results summary ===
#         if hasattr(self, 'metadata') and 'sample_images' in self.metadata:
#             sample_images = self.metadata['sample_images']
#             print(f"\n{'='*60}")
#             print("📊 FINAL RESULTS")
#             print("="*60)
#             print(f"✅ Noise Regions Detected: {sample_images.get('noise_regions_count', 0)}")
#             print(f"✅ Content Regions Identified: {sample_images.get('inverse_regions_count', 0)}")
#             print(f"✅ Pages Analyzed: {sample_images.get('pages_included', 'unknown')}")
            
#             if 'metadata' in sample_images:
#                 meta = sample_images['metadata']
#                 print(f"\n🎨 Visual Settings:")
#                 print(f"   Zoom: {meta.get('zoom', 1.0)}x")
#                 print(f"   Colors: Noise={meta.get('noise_color', 'red')}, Content={meta.get('inverse_color', 'blue')}")
        
#         print(f"\n{'='*60}")
#         print("✅ Trace Complete")
    
    
#     # Updated method signature for existing methods
#     def print_json_outline_summary(self, max_str_len=100, include_global_functions=True, include_images=True):
#         """Print summary with JSON outline formatting for parameters and optional images."""
#         self.print_summary(use_json_outline=True, max_str_len=max_str_len, 
#                           include_global_functions=include_global_functions, include_images=include_images)


#     def get_formatted_trace(self, use_json_outline=False, max_str_len=100, include_global_functions=True) -> str:
#         """Returns the trace as a formatted string for LLM consumption."""
#         lines = ["EXECUTION TRACE SUMMARY"]
#         lines.append("=" * 50)
        
#         if not self.steps:
#             lines.append("No steps were recorded.")
#             return '\n'.join(lines)
        
#         # Add Global Functions Used section
#         if include_global_functions:
#             lines.append("\nGLOBAL FUNCTIONS USED [defaults and descriptors]")
#             lines.append("=" * 50)
            
#             global_functions = self._extract_global_functions()
            
#             for func_name, func_info in global_functions.items():
#                 lines.append(f"\n{func_name}():")
#                 lines.append(f"  Description: {func_info['description']}")
                
#                 # Show parameter structure from sample usage
#                 if use_json_outline and func_info['sample_params']:
#                     param_lines = self.format_json_outline(func_info['sample_params'], max_str_len=max_str_len)
#                     if param_lines:
#                         lines.append("  Parameter Structure:")
#                         for line in param_lines:
#                             lines.append(f"    {line}")
#                 elif func_info['sample_params']:
#                     lines.append(f"  Parameter Structure: {func_info['sample_params']}")
            
#             lines.append(f"\n{'='*50}")
        
#         # Add step-by-step execution
#         lines.append("\nSTEP-BY-STEP EXECUTION")
#         lines.append("=" * 50)
        
#         for i, step in enumerate(self.steps, 1):
#             lines.append(f"\nStep {i}:")
#             lines.append(f"Overview: {step['step_name']}")
#             lines.append(f"Function: {step['function_name']}()")
            
#             # Add description if it exists and is meaningful
#             if step.get('description') and step['description'].strip():
#                 lines.append(f"Description: {step['description']}")
            
#             # Format parameters used (values only, not structure)
#             if use_json_outline and step['parameters']:
#                 param_lines = self.format_json_outline(step['parameters'], max_str_len=max_str_len)
#                 if param_lines:
#                     lines.append("Parameters:")
#                     for line in param_lines:
#                         lines.append(f"  {line}")
#                 else:
#                     lines.append(f"Parameters: {step['parameters']}")
#             else:
#                 lines.append(f"Parameters: {step['parameters']}")
            
#             lines.append(f"Output Count: {step['output_count']}")
        
#         return '\n'.join(lines)

#     def to_dict(self) -> List[Dict[str, Any]]:
#         """Returns the trace as a list of dictionaries."""
#         return self.steps
        
# class TraceableWorkflow:
#     """
#     A generic runner that executes any compatible workflow function and manages
#     the lifecycle of the TraceLog for that run.
#     """
#     def __init__(self, pdf_model: 'PDFModel'):
#         self.pdf_model = pdf_model
#         self.last_trace: Optional[TraceLog] = None

#     def run(
#         self, 
#         workflow_function: Callable, 
#         params: Optional[BaseModel] = None, 
#         **kwargs
#     ) -> Any:
#         """
#         Runs a given workflow function, injecting the pdf_model and a new trace log.

#         Args:
#             workflow_function: The pluggable workflow function to execute. It must accept
#                                (pdf_model, trace, params, **kwargs).
#             params: A Pydantic model of parameters for the workflow.
#             **kwargs: Any additional keyword arguments for the workflow.
        
#         Returns:
#             The final data result of the workflow function.
#         """
#         trace = TraceLog()
        
#         # Execute the workflow, passing it the context it needs
#         result = workflow_function(
#             self.pdf_model,
#             trace=trace,
#             params=params,
#             **kwargs
#         )
        
#         self.last_trace = trace
#         return result

#     def get_last_trace(self) -> Optional[TraceLog]:
#         """Returns the TraceLog from the most recent run."""
#         if not self.last_trace:
#             print("No workflow has been run yet.")
#         return self.last_trace


# from __future__ import annotations

# import inspect
# import textwrap
# from typing import Any, Callable, Dict, List, Optional

# try:
#     # Optional import – only needed when you pass Pydantic models
#     from pydantic import BaseModel
# except ImportError:   # pragma: no cover
#     BaseModel = object    # type: ignore


# # ---------- helper ---------------------------------------------------------- #
# def _callable_payload(fn: Callable) -> Dict[str, Any]:
#     """
#     Represent a callable as a serialisable payload.

#     Returned structure is intentionally simple so it can be carried around in
#     the trace dictionary and later re‑hydrated / pretty‑printed by a renderer.
#     """
#     label = getattr(fn, "__qualname__", getattr(fn, "__name__", str(fn)))
#     src = None
#     try:
#         src = textwrap.dedent(inspect.getsource(fn)).strip()
#     except (OSError, TypeError):  # built‑ins, C‑extensions, lambdas built in REPL
#         pass

#     closure_vars: Dict[str, Any] | None = None
#     if fn.__closure__:
#         closure_vars = {
#             cellvar: cell.cell_contents
#             for cellvar, cell in zip(fn.__code__.co_freevars, fn.__closure__)
#         }

#     return {
#         "label": label,
#         "source": src,
#         "closure_vars": closure_vars,
#         "is_lambda": fn.__name__ == "<lambda>",
#     }


# # ---------- core TraceLog --------------------------------------------------- #
# class TraceLog:
#     """
#     Lightweight execution trace.

#     • Records every step (name, callable, parameters, result length, description)
#     • Stores arbitrary metadata (e.g. images) via `add_metadata`
#     • Extracts the unique user‑defined functions involved in a run
#     • Returns all data as plain Python structures (dictionaries / lists)
#     """

#     def __init__(self) -> None:
#         self.steps: List[Dict[str, Any]] = []
#         self.metadata: Dict[str, Any] = {}

#     # --------------------------------------------------------------------- #
#     # public helpers
#     # --------------------------------------------------------------------- #
#     def add_metadata(self, metadata: Dict[str, Any]) -> None:
#         """Merge arbitrary metadata into the trace (e.g. base64 images)."""
#         self.metadata.update(metadata)

#     def run_and_log_step(
#         self,
#         step_name: str,
#         function: Callable,
#         *,
#         log_function: Optional[Callable] = None,
#         params: Optional[Dict[str, Any] | BaseModel] = None,
#         description: Optional[str] = None,
#     ) -> Any:
#         """
#         Execute `function`, record a step, and return the result.

#         `log_function` lets you separate “execution callable” from the callable
#         that should be captured in the trace (useful when executing lambdas that
#         delegate to real functions).
#         """
#         result = function()

#         if log_function:
#             func_to_log = log_function
#             composite_name = (
#                 f"lambda → {getattr(log_function, '__qualname__', log_function.__name__)}"
#             )
#         else:
#             func_to_log = getattr(function, "func", function)  # unwrap functools.partial
#             composite_name = None

#         self.add_step(
#             step_name=step_name,
#             function=func_to_log,
#             params=params,
#             result=result,
#             description=description,
#             composite_name=composite_name,
#         )
#         return result

#     def to_dict(self) -> Dict[str, Any]:
#         """Retrieve the complete, serialisable trace."""
#         return {"steps": self.steps, "metadata": self.metadata}

#     # --------------------------------------------------------------------- #
#     # core recording
#     # --------------------------------------------------------------------- #
#     def add_step(
#         self,
#         *,
#         step_name: str,
#         function: Callable,
#         params: Optional[Dict[str, Any] | BaseModel] = None,
#         result: Any = None,
#         description: Optional[str] = None,
#         composite_name: Optional[str] = None,
#     ) -> None:
#         """Append a step entry to `self.steps`."""
#         raw_params = (
#             params.model_dump(mode="json") if isinstance(params, BaseModel) else params
#         )
#         serialised_params = self._serialize_parameters(raw_params)

#         if isinstance(serialised_params, dict):
#             # Drop keys with value None to keep the trace concise
#             serialised_params = {
#                 k: v for k, v in serialised_params.items() if v is not None
#             }

#         self.steps.append(
#             {
#                 "step_name": step_name,
#                 "function_obj": function,  # keep the live object for later
#                 "function_name": composite_name
#                 or getattr(function, "__name__", str(function)),
#                 "parameters": serialised_params,
#                 "description": description
#                 or textwrap.dedent(function.__doc__ or "").strip(),
#                 "output_count": len(result)
#                 if hasattr(result, "__len__")
#                 else (1 if result is not None else 0),
#             }
#         )

#     # --------------------------------------------------------------------- #
#     # parameter serialisation
#     # --------------------------------------------------------------------- #
#     def _serialize_parameters(self, obj: Any) -> Any:
#         """Recursively walk `obj`, replacing callables with a payload dict."""
#         if callable(obj):
#             return _callable_payload(obj)

#         if isinstance(obj, dict):
#             return {k: self._serialize_parameters(v) for k, v in obj.items()}
#         if isinstance(obj, list):
#             return [self._serialize_parameters(v) for v in obj]
#         return obj

#     # --------------------------------------------------------------------- #
#     # analysis helpers
#     # --------------------------------------------------------------------- #
#     def _extract_global_functions(self) -> Dict[str, Dict[str, Any]]:
#         """
#         Return a mapping {function_name: {description, sample_params}} extracted
#         from all recorded steps.  Lambdas used *directly* are ignored – but
#         lambdas that delegate (via `lambda → func_name`) are resolved to
#         `func_name`.
#         """
#         functions: Dict[str, Dict[str, Any]] = {}

#         for step in self.steps:
#             func_obj = step.get("function_obj")
#             if not callable(func_obj):
#                 continue

#             func_name = step["function_name"]

#             # Skip pure lambdas (no right‑hand side)
#             if func_name.startswith("<lambda"):
#                 continue
#             if func_name.startswith("lambda") and "→" not in func_name:
#                 continue

#             # Use right‑hand side of composite names (lambda → real_fn)
#             display_name = func_name.split("→")[-1].strip()

#             if display_name not in functions:
#                 functions[display_name] = {
#                     "description": textwrap.dedent(func_obj.__doc__ or "").strip(),
#                     "sample_params": step["parameters"],
#                 }

#         return functions