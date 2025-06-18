import inspect
import ast
from textwrap import dedent
from pydantic import Field, BaseModel, ConfigDict
from typing import Optional, Any, List, Callable, Dict
import inspect
import json
from tableai.pdf.flows.generic_context import NodeContext, RunContext
from tableai.pdf.flows.trace_log import _serialize_safely

class FieldSpecification(BaseModel):
    """Describes a single field within a Pydantic model."""
    name: str
    type: str
    description: Optional[str] = None
    default: Any = 'REQUIRED'

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

def _generate_node_specification(
    func: Callable,
    flow_instance,
    context: 'NodeContext'
) -> NodeSpecification:
    """The master function that assembles the NodeSpecification from a materialized context."""
    
    # --- 1. Basic Info ---
    node_name = func.__name__
    description = context.description if hasattr(context, 'description') and context.description else dedent(func.__doc__ or f"No description for {node_name}.").strip()
    
    # === 2. Decorator Declaration String ===
    flow_var_name = "flow"
    try:
        for name, var in inspect.currentframe().f_back.f_globals.items():
            if var is flow_instance:
                flow_var_name = name
                break
    except Exception: pass

    if context.context_type == 'root':
        param_repr = f"{type(context.params).__name__}(...)"
        context_repr = f"NodeContext[{param_repr}]"
    else: # dependency or context_only
        deps_str = ", ".join(context.wait_for_nodes or [])
        context_repr = f"NodeContext[{deps_str}]"

    # Reconstruct the kwargs from the context's extra params for the repr
    kwargs_from_context = context.extra_params.copy()
    if context.input_model:
        kwargs_from_context['input_model'] = context.input_model

    kwargs_repr = ", ".join([f"{k}={type(v).__name__ if not isinstance(v, str) else repr(v)}" for k, v in kwargs_from_context.items()])
    decorator_repr = f"@{flow_var_name}.step({context_repr}({kwargs_repr}))"
    
    # =============================================================
    # 3. Input Model Specification (Restored and Corrected)
    # =============================================================
    input_model_spec = None
    input_model = context.input_model
    if input_model:
        # Determine if we have the class or an instance of it
        model_class = input_model if inspect.isclass(input_model) and issubclass(input_model, BaseModel) else type(input_model)
        
        if issubclass(model_class, BaseModel):
            fields_spec = []
            for field_name, field_info in model_class.model_fields.items():
                fields_spec.append(FieldSpecification(
                    name=field_name,
                    type=str(field_info.annotation),
                    description=field_info.description,
                    default=field_info.default if field_info.default is not ... else 'REQUIRED'
                ))
            input_model_spec = ModelSpecification(
                name=model_class.__name__,
                description=dedent(model_class.__doc__ or "No description provided.").strip(),
                fields=fields_spec
            )

    # =============================================================
    # 4. Current Parameters (Corrected Logic)
    # =============================================================
    current_params = context.extra_params.copy()
    if input_model and not isinstance(input_model, type):
        current_params.update(input_model.model_dump())

    annotations = func.__annotations__
    return_type = str(annotations.get('return', 'Any'))
    source_code = dedent(inspect.getsource(func))
    # --- Call Analysis ---
    # (Using the helper functions from the previous step)
    tree = ast.parse(source_code)
    raw_call_names = sorted(list({name for node in ast.walk(tree) if isinstance(node, ast.Call) and (name := _get_call_name(node))}))
    
    called_functions_spec = []
    # (Filtering logic for built-ins and common methods is the same)
    flow_var_name = "flow" # Get this properly as before
    PYTHON_BUILTINS, COMMON_METHODS = set(dir(__builtins__)), {'get', 'update', 'append'}

    for name in raw_call_names:
        if name.startswith(f"{flow_var_name}.step"): continue
        first_part, last_part = name.split('.')[0], name.split('.')[-1]
        if first_part in PYTHON_BUILTINS or last_part in COMMON_METHODS: continue
        
        # Try to resolve the name to a live object in the function's globals
        callable_obj = func.__globals__.get(first_part)
        
        if callable_obj:
            try:
                # Traverse nested attributes (e.g., data.group.chain)
                for attr in name.split('.')[1:]:
                    callable_obj = getattr(callable_obj, attr)
                
                # Get the rich metadata and create the spec object
                metadata = _get_callable_metadata(callable_obj)
                called_functions_spec.append(CallSpecification(type='function_call', **metadata))

            except AttributeError:
                # Couldn't resolve the full path, treat as a method call
                called_functions_spec.append(CallSpecification(name=name, type="method_call", docstring="Method on a local or instance variable."))
        else:
            # Cannot resolve, likely a method on a local variable
             called_functions_spec.append(CallSpecification(name=name, type="method_call", docstring="Method on a local or instance variable."))

    # --- Assemble Final Specification ---
    return NodeSpecification(
        name=node_name,
        description=description,
        decorator_declaration=decorator_repr,
        input_model_spec=input_model_spec,
        current_params=current_params,
        return_type=return_type,
        source_code=source_code,
        called_functions=called_functions_spec
    )

def _get_call_name(call_node: ast.Call) -> Optional[str]:
    """
    Recursively reconstructs the full name of a called function from an AST Call node.
    Handles simple calls `func()`, attribute calls `obj.method()`, and nested calls `a.b.c()`.
    """
    func = call_node.func
    parts = []
    while isinstance(func, ast.Attribute):
        parts.append(func.attr)
        func = func.value
    if isinstance(func, ast.Name):
        parts.append(func.id)
        return ".".join(reversed(parts))
    return None # For complex calls like `(lambda: x)()`, we can't get a simple name.


def _get_callable_metadata(callable_obj: Callable) -> Dict[str, Any]:
    """Inspects a live callable object and extracts rich metadata."""
    metadata = {
        "name": getattr(callable_obj, '__name__', '<unknown>'),
        "docstring": dedent(inspect.getdoc(callable_obj) or "No docstring.").strip(),
        "module": None,
        "signature": None,
        "source_code": "<source not available>"
    }
    try:
        metadata['signature'] = str(inspect.signature(callable_obj))
    except (ValueError, TypeError):
        metadata['signature'] = "()" # Fallback for built-ins
        
    try:
        module = inspect.getmodule(callable_obj)
        if module:
            metadata['module'] = module.__name__
    except TypeError:
        pass

    try:
        metadata['source_code'] = dedent(inspect.getsource(callable_obj))
    except (TypeError, OSError):
        pass
        
    return metadata


async def trace_call(ctx: 'RunContext', target_object: Any, method_name: str, **kwargs: Any) -> Any:
    """
    Traces and executes a method call, recording the details via the context's
    injected recorder. This is the new way nodes will trace calls.
    """
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
    
