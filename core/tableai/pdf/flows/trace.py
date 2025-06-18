import inspect
import ast
from textwrap import dedent
from pydantic import Field, BaseModel, ConfigDict
from typing import Optional, Any, List, Callable, Dict
import inspect
import json
from tableai.pdf.flows.generic_context import NodeContext, RunContext
from tableai.pdf.flows.trace_log import _serialize_safely

TRACE_IGNORE = True

class FieldSpecification(BaseModel):
    """Describes a single field within a Pydantic model."""
    name: str
    type: str
    description: Optional[str] = None
    default: Any = 'REQUIRED'
    impact: Optional[str] = None

class ModelSpecification(BaseModel):
    """Describes a Pydantic model used as an input_model."""
    name: str
    description: Optional[str] = None
    fields: List[FieldSpecification]

class CallSpecification(BaseModel):
    """Describes a function call made within a node with rich metadata."""
    name: str
    type: str
    docstring: Optional[str] = None
    module: Optional[str] = None
    signature: Optional[str] = None
    source_code: Optional[str] = Field(default=None, repr=False) # Exclude from default repr

class NodeSpecification(BaseModel):
    """
    A complete, self-contained specification of a single node in the flow graph.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # --- Core Info ---
    name: str
    description: str
    
    # --- Dependencies & Configuration ---
    
    # =============================================================
    # THE CORE FIX: Provide a default value to make the field optional.
    # =============================================================
    dependencies: List[str] = Field(default_factory=list, description="A list of node names this node depends on.")
    
    decorator_declaration: str = Field(description="The exact decorator call used to define this node.")
    input_model_spec: Optional[ModelSpecification] = Field(default=None, description="Specification of the Pydantic model used for parameters.")
    current_params: Dict[str, Any] = Field(description="The actual parameter values for this node at registration time.")
    
    # --- Function Analysis ---
    return_type: str
    source_code: str
    called_functions: List[CallSpecification]


# Constants
TRACE_IGNORE = True

def should_ignore_trace(obj: Any) -> bool:
    """
    Comprehensive check to determine if an object should be ignored for tracing.
    """
    # 1. Direct attribute check on the object
    if hasattr(obj, '__trace_ignore__') and getattr(obj, '__trace_ignore__', False):
        return True
    
    # 2. Check the class if this is a method/bound method
    if hasattr(obj, '__self__'):
        instance_class = obj.__self__.__class__
        if hasattr(instance_class, '__trace_ignore__') and getattr(instance_class, '__trace_ignore__', False):
            return True
        
        # Also check the method itself on the class
        method_name = obj.__name__
        if hasattr(instance_class, method_name):
            method_obj = getattr(instance_class, method_name)
            if hasattr(method_obj, '__trace_ignore__') and getattr(method_obj, '__trace_ignore__', False):
                return True
    
    # 3. Check unbound methods (for static/class methods)
    if hasattr(obj, '__func__'):
        if hasattr(obj.__func__, '__trace_ignore__') and getattr(obj.__func__, '__trace_ignore__', False):
            return True
    
    # 4. Check wrapped functions
    if hasattr(obj, '__wrapped__'):
        if hasattr(obj.__wrapped__, '__trace_ignore__') and getattr(obj.__wrapped__, '__trace_ignore__', False):
            return True
    
    # 5. For functions, also check if they have trace_ignore in their source
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        try:
            source = inspect.getsource(obj)
            # Look for __trace_ignore__ = True in the source
            if '__trace_ignore__ = True' in source or '__trace_ignore__=True' in source:
                return True
        except (OSError, TypeError):
            pass
    
    return False

def has_trace_ignore_in_source(func_or_method) -> bool:
    """
    Check if a function has __trace_ignore__ = True in its source code.
    This handles cases where the flag is set inside the function body.
    """
    try:
        source = inspect.getsource(func_or_method)
        # Look for various forms of trace ignore assignment
        patterns = [
            '__trace_ignore__ = True',
            '__trace_ignore__=True', 
            '__trace_ignore__ = TRACE_IGNORE',
            '__trace_ignore__=TRACE_IGNORE'
        ]
        return any(pattern in source for pattern in patterns)
    except (OSError, TypeError):
        return False

def _generate_node_specification(
    func: Callable,
    flow_instance,
    context: 'NodeContext',
) -> 'NodeSpecification':
    """
    Fixed version of your _generate_node_specification that properly handles trace ignore flags.
    """
    
    __trace_ignore__ = False  # This function itself should be ignored
    
    # --- 1. Basic Info ---
    node_name = func.__name__
    description = context.description if hasattr(context, 'description') and context.description else (func.__doc__ or f"No description for {node_name}.").strip()
    
    # === 2. Decorator Declaration String ===
    flow_var_name = "flow"
    try:
        for name, var in inspect.currentframe().f_back.f_globals.items():
            if var is flow_instance:
                flow_var_name = name
                break
    except Exception: 
        pass

    if context.context_type == 'root':
        param_repr = f"{type(context.params).__name__}(...)"
        context_repr = f"NodeContext[{param_repr}]"
    else: # dependency or context_only
        deps_str = ", ".join(context.wait_for_nodes or [])
        context_repr = f"NodeContext[{deps_str}]"

    kwargs_from_context = context.extra_params.copy()
    if context.input_model:
        kwargs_from_context['input_model'] = context.input_model

    kwargs_repr = ", ".join([f"{k}={type(v).__name__ if not isinstance(v, str) else repr(v)}" for k, v in kwargs_from_context.items()])
    decorator_repr = f"@{flow_var_name}.step({context_repr}({kwargs_repr}))"
    
    # === 3. Input Model Specification ===
    input_model_spec = None
    input_model = context.input_model
    if input_model:
        model_class = input_model if inspect.isclass(input_model) else type(input_model)
        
        # Only proceed if the model should not be ignored
        if not should_ignore_trace(model_class):
            # try:
            if hasattr(model_class, 'model_fields'):  # Pydantic v2
                fields_spec = []
                for field_name, field_info in model_class.model_fields.items():
                    impact_level = None
                    if field_info.json_schema_extra and isinstance(field_info.json_schema_extra, dict):
                        impact_level = field_info.json_schema_extra.get('impact')

                    fields_spec.append({
                        'name': field_name,
                        'type': str(field_info.annotation),
                        'description': field_info.description,
                        'default': field_info.default if field_info.default is not ... else 'REQUIRED',
                        'impact': impact_level
                    })
                input_model_spec = {
                    'name': model_class.__name__,
                    'description': (model_class.__doc__ or "No description provided.").strip(),
                    'fields': fields_spec
                }
            # except Exception:
            #     pass

    # === 4. Current Parameters ===
    current_params = context.extra_params.copy()
    if input_model and not isinstance(input_model, type):
        try:
            current_params.update(input_model.model_dump())
        except Exception:
            pass

    annotations = func.__annotations__
    return_type = str(annotations.get('return', 'Any'))
    
    try:
        source_code = inspect.getsource(func)
    except (OSError, TypeError):
        source_code = "Source code not available"
    
    # === 5. Called Functions Analysis (THE KEY FIX) ===
    try:
        tree = ast.parse(source_code)
        raw_call_names = sorted(list({name for node in ast.walk(tree) if isinstance(node, ast.Call) and (name := _get_call_name(node))}))
    except Exception:
        raw_call_names = []
    
    called_functions_spec = []
    
    # Filter built-ins and common methods
    flow_var_name = "flow" 
    PYTHON_BUILTINS = set(dir(__builtins__))
    COMMON_METHODS = {'get', 'update', 'append', 'items', 'keys', 'values', 'pop', 'clear'}

    for name in raw_call_names:
        if name.startswith(f"{flow_var_name}.step"): 
            continue
            
        first_part = name.split('.')[0]
        last_part = name.split('.')[-1]
        
        if first_part in PYTHON_BUILTINS or last_part in COMMON_METHODS: 
            continue
        
        # Try to resolve the name to a live object
        callable_obj = func.__globals__.get(first_part)
            
        if callable_obj:
            # try:
            # Navigate to the final callable
            for attr in name.split('.')[1:]:
                callable_obj = getattr(callable_obj, attr)
            
            # === THE CRITICAL FIX: Check multiple ways ===
            should_skip = False
            
            # 1. Check the __trace_ignore__ attribute directly
            if getattr(callable_obj, '__trace_ignore__', False):
                should_skip = True
            
            # 2. Use our comprehensive ignore check
            if should_ignore_trace(callable_obj):
                should_skip = True
            
            # 3. Check source code for trace ignore (for functions like merge_all_boxes)
            if has_trace_ignore_in_source(callable_obj):
                should_skip = True
            
            # 4. For methods on classes, check if the class is ignored
            if hasattr(callable_obj, '__self__'):
                if should_ignore_trace(callable_obj.__self__.__class__):
                    should_skip = True
            
            if should_skip:
                continue  # Skip this call
            
            # Get metadata for non-ignored functions
            metadata = _get_callable_metadata(callable_obj)
            if metadata:
                called_functions_spec.append({
                    'type': 'function_call',
                    **metadata
                })

    # --- Assemble Final Specification ---
    return NodeSpecification(
        name=node_name,
        description=description,
        dependencies=context.wait_for_nodes or [],
        decorator_declaration=decorator_repr,
        input_model_spec=input_model_spec,
        current_params=current_params,
        return_type=return_type,
        source_code=source_code,
        called_functions=called_functions_spec
    )

def _get_call_name(node: ast.Call) -> Optional[str]:
    """Extract the call name from an AST Call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        parts = []
        current = node.func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return '.'.join(reversed(parts))
    return None

def _get_callable_metadata(callable_obj):
    """Enhanced version that respects trace ignore flags."""
    # try:
    name = getattr(callable_obj, '__name__', str(callable_obj))
    docstring = getattr(callable_obj, '__doc__', None) or "No documentation available."
    
    module_name = None
    if hasattr(callable_obj, '__module__'):
        module_name = callable_obj.__module__
    
    signature = None
    try:
        signature = str(inspect.signature(callable_obj))
    except (ValueError, TypeError):
        signature = None
    
    source_code = None
    try:
        source_code = inspect.getsource(callable_obj)
    except (OSError, TypeError):
        source_code = None
    
    return {
        'name': name,
        'docstring': docstring.strip() if docstring else "No documentation available.",
        'module': module_name,
        'signature': signature,
        'source_code': source_code
    }
    # except Exception:
    #     return None


async def trace_call(ctx: 'RunContext', target_object: Any, method_name: str, **kwargs: Any) -> Any:
    """
    Traces and executes a method call, recording the details via the context's
    injected recorder. This is the new way nodes will trace calls.
    """
    __trace_ignore__ = True
    ctx._record_sub_call({
        "target": type(target_object).__name__,
        "method": method_name,
        "params": {k: _serialize_safely(v) for k, v in kwargs.items()}
    })
    
    method_to_call = getattr(target_object, method_name)
    if inspect.iscoroutinefunction(method_to_call):
        return await method_to_call(**kwargs)
    else:
        return method_to_call(**kwargs)
    
