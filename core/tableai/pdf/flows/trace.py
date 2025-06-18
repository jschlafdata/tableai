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
    
    Args:
        obj: The object to check (function, class, method, etc.)
    
    Returns:
        bool: True if the object should be ignored, False otherwise
    """
    
    # 1. Direct attribute check on the object
    if hasattr(obj, '__trace_ignore__') and getattr(obj, '__trace_ignore__', False):
        return True
    
    # 2. Check the class if this is a method/bound method
    if hasattr(obj, '__self__'):
        # Bound method - check the instance's class
        instance_class = obj.__self__.__class__
        if hasattr(instance_class, '__trace_ignore__') and getattr(instance_class, '__trace_ignore__', False):
            return True
        
        # Also check the method itself
        method_name = obj.__name__
        if hasattr(instance_class, method_name):
            method_obj = getattr(instance_class, method_name)
            if hasattr(method_obj, '__trace_ignore__') and getattr(method_obj, '__trace_ignore__', False):
                return True
    
    # 3. Check unbound methods (for static/class methods)
    if hasattr(obj, '__func__'):
        if hasattr(obj.__func__, '__trace_ignore__') and getattr(obj.__func__, '__trace_ignore__', False):
            return True
    
    # 4. For classes, check the class itself
    if inspect.isclass(obj):
        if hasattr(obj, '__trace_ignore__') and getattr(obj, '__trace_ignore__', False):
            return True
    
    # 5. For functions that might be wrapped by decorators
    if hasattr(obj, '__wrapped__'):
        # Check the wrapped function
        if hasattr(obj.__wrapped__, '__trace_ignore__') and getattr(obj.__wrapped__, '__trace_ignore__', False):
            return True
    
    # 6. Check the module level
    if hasattr(obj, '__module__'):
        try:
            module = inspect.getmodule(obj)
            if module and hasattr(module, '__trace_ignore__') and getattr(module, '__trace_ignore__', False):
                return True
        except Exception:
            pass
    
    return False


def trace_ignore_decorator(func_or_class):
    """
    Decorator to mark functions or classes as trace-ignored.
    
    Usage:
        @trace_ignore_decorator
        def my_function():
            pass
        
        @trace_ignore_decorator
        class MyClass:
            pass
    """
    func_or_class.__trace_ignore__ = True
    return func_or_class


def _get_callable_metadata(callable_obj):
    """
    Enhanced version that respects trace ignore flags.
    """
    if should_ignore_trace(callable_obj):
        return None  # Signal that this should be ignored
    
    try:
        name = getattr(callable_obj, '__name__', str(callable_obj))
        docstring = getattr(callable_obj, '__doc__', None) or "No documentation available."
        
        # Get module info
        module_name = None
        if hasattr(callable_obj, '__module__'):
            module_name = callable_obj.__module__
        
        # Get signature if possible
        signature = None
        try:
            signature = str(inspect.signature(callable_obj))
        except (ValueError, TypeError):
            signature = None
        
        # Get source code if available
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
    except Exception:
        return None


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


def _generate_node_specification(
    func: Callable,
    flow_instance,
    context: 'NodeContext',
) -> 'NodeSpecification':
    """
    Enhanced version of _generate_node_specification that properly handles trace ignore flags.
    """
    
    # Set trace ignore on this function itself
    __trace_ignore__ = TRACE_IGNORE
    
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

    # Build decorator representation
    if context.context_type == 'root':
        param_repr = f"{type(context.params).__name__}(...)"
        context_repr = f"NodeContext[{param_repr}]"
    else:
        deps_str = ", ".join(context.wait_for_nodes or [])
        context_repr = f"NodeContext[{deps_str}]"

    kwargs_from_context = context.extra_params.copy()
    if context.input_model:
        kwargs_from_context['input_model'] = context.input_model

    kwargs_repr = ", ".join([f"{k}={type(v).__name__ if not isinstance(v, str) else repr(v)}" 
                            for k, v in kwargs_from_context.items()])
    decorator_repr = f"@{flow_var_name}.step({context_repr}({kwargs_repr}))"
    
    # === 3. Input Model Specification ===
    input_model_spec = None
    input_model = context.input_model
    if input_model and not should_ignore_trace(input_model):
        model_class = input_model if inspect.isclass(input_model) else type(input_model)
        
        # Check if the model class should be ignored
        if not should_ignore_trace(model_class):
            try:
                # Assuming BaseModel from pydantic
                if hasattr(model_class, 'model_fields'):
                    fields_spec = []
                    for field_name, field_info in model_class.model_fields.items():
                        impact_level = None
                        if (hasattr(field_info, 'json_schema_extra') and 
                            field_info.json_schema_extra and 
                            isinstance(field_info.json_schema_extra, dict)):
                            impact_level = field_info.json_schema_extra.get('impact')

                        fields_spec.append({
                            'name': field_name,
                            'type': str(field_info.annotation),
                            'description': getattr(field_info, 'description', None),
                            'default': field_info.default if field_info.default is not ... else 'REQUIRED',
                            'impact': impact_level
                        })
                    
                    input_model_spec = {
                        'name': model_class.__name__,
                        'description': (model_class.__doc__ or "No description provided.").strip(),
                        'fields': fields_spec
                    }
            except Exception:
                pass

    # === 4. Current Parameters ===
    current_params = context.extra_params.copy()
    if input_model and not isinstance(input_model, type):
        try:
            if hasattr(input_model, 'model_dump'):
                current_params.update(input_model.model_dump())
            elif hasattr(input_model, 'dict'):
                current_params.update(input_model.dict())
        except Exception:
            pass

    # === 5. Return Type ===
    annotations = func.__annotations__
    return_type = str(annotations.get('return', 'Any'))
    
    # === 6. Source Code ===
    try:
        source_code = inspect.getsource(func)
    except (OSError, TypeError):
        source_code = "Source code not available."
    
    # === 7. Called Functions Analysis ===
    called_functions_spec = []
    
    # try:
    tree = ast.parse(source_code)
    raw_call_names = sorted(list({
        name for node in ast.walk(tree) 
        if isinstance(node, ast.Call) and (name := _get_call_name(node))
    }))
    
    # Filter and analyze calls
    PYTHON_BUILTINS = set(dir(__builtins__))
    COMMON_METHODS = {'get', 'update', 'append', 'items', 'keys', 'values', 'pop', 'clear'}
    
    for name in raw_call_names:
        # Skip flow step decorators
        if name.startswith(f"{flow_var_name}.step"): 
            continue
            
        first_part = name.split('.')[0]
        last_part = name.split('.')[-1]
        
        # Skip builtins and common methods
        if first_part in PYTHON_BUILTINS or last_part in COMMON_METHODS: 
            continue
        
        # Try to resolve the callable
        callable_obj = func.__globals__.get(first_part)
        
        if callable_obj:
            try:
                # Navigate to the actual callable
                for attr in name.split('.')[1:]:
                    callable_obj = getattr(callable_obj, attr)
                
                # Check if this callable should be ignored
                if should_ignore_trace(callable_obj):
                    continue
                
                # Get metadata
                metadata = _get_callable_metadata(callable_obj)
                if metadata:  # Only add if not ignored
                    called_functions_spec.append({
                        'type': 'function_call',
                        **metadata
                    })
                    
            except AttributeError:
                # If we can't resolve it and it's not ignored, add as method call
                called_functions_spec.append({
                    'name': name,
                    'type': "method_call",
                    'docstring': "Method on a local or instance variable.",
                    'module': None,
                    'signature': None,
                    'source_code': None
                })
        else:
            # Unresolved name - add as method call unless it should be ignored
            called_functions_spec.append({
                'name': name,
                'type': "method_call", 
                'docstring': "Method on a local or instance variable.",
                'module': None,
                'signature': None,
                'source_code': None
            })
                
    # except Exception:
    #     # If AST parsing fails, continue with empty called_functions_spec
    #     pass
    
    # === 8. Assemble Final Specification ===
    return {
        'name': node_name,
        'description': description,
        'decorator_declaration': decorator_repr,
        'input_model_spec': input_model_spec,
        'current_params': current_params,
        'return_type': return_type,
        'source_code': source_code,
        'called_functions': called_functions_spec
    }


async def trace_call(ctx: 'RunContext', target_object: Any, method_name: str, **kwargs: Any) -> Any:
    """
    Traces and executes a method call, recording the details via the context's
    injected recorder. This is the new way nodes will trace calls.
    """
    __trace_ignore__ = TRACE_IGNORE
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
    
