from pydantic import BaseModel, Field, computed_field, ConfigDict
from typing import TypeVar, Dict, Any, Generic, Union, Optional, List, Type, Callable, Tuple
from dataclasses import is_dataclass
from tableai.pdf.flows.validations import ValidationRule
from tableai.pdf.flows.generic_dependencies import D

InputT = TypeVar('InputT')

class StepInput(BaseModel, Generic[InputT]):
    """
    A unified, generic container passed to ALL nodes in the flow.
    It packages the node's primary data payload and its specific configuration.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # The primary data payload for the step.
    # For a root node, this will be a StepParams object.
    # For a dependent node, this will be the result of the upstream node.
    data: InputT
    
    # The extra configuration for this specific node.
    config: Dict[str, Any] = Field(default_factory=dict)
    input_model: Optional[Union[BaseModel, object]] = None

class RunContext(Generic[D]):
    def __init__(self, deps: D, state: Dict[str, Any]):
        self._deps = deps
        self._state = state
    @property
    def deps(self) -> D: return self._deps
    @property
    def state(self) -> Dict[str, Any]: return self._state

class BaseContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    description: Optional[str] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    validation_rule: Optional[ValidationRule] = None
    on_validation_failure: str = 'raise'

    def __init__(self, **data: Any):
        known_fields = self.model_fields.keys()
        init_data = {k: v for k, v in data.items() if k in known_fields}
        extra_params = {k: v for k, v in data.items() if k not in known_fields}
        
        # Merge any ad-hoc kwargs into the extra_params dict
        init_data.setdefault('extra_params', {}).update(extra_params)
        super().__init__(**init_data)

ExpectedDataT = TypeVar('ExpectedDataT')

class NodeContext(BaseContext, Generic[ExpectedDataT]):
    params: Optional[Union[BaseModel, object]] = None
    wait_for_nodes: Optional[List[str]] = None
    expected_return_type: Optional[Type[ExpectedDataT]] = Field(default=None, exclude=True)
    input_model: Optional[Union[Type[BaseModel], BaseModel]] = None
    
    @computed_field
    @property
    def expected_type_name(self) -> Optional[str]:
        if self.expected_return_type and hasattr(self.expected_return_type, '__name__'):
            return self.expected_return_type.__name__
        return None
    
    @computed_field
    @property
    def context_type(self) -> str:
        """Returns the context type - 'root' or 'dependency'."""
        # A node is a dependency node if it waits for any other nodes.
        if self.wait_for_nodes:
            return "dependency"
        elif self.params is not None:
            return "root"
        else:
            # This case can be for simple nodes that only use context.
            return "context_only"
            
    @classmethod
    def __class_getitem__(cls, items: Union[Callable, object, Tuple[Callable, ...]]):
        """
        Unified factory that handles single params, single dependencies,
        or a tuple of multiple dependencies.
        """
        # Case 1: Multiple Dependencies (a tuple of functions)
        if isinstance(items, tuple):
            node_funcs = items
            for func in node_funcs:
                if not callable(func): # or not cls._is_flow_graph_function(func):
                    raise TypeError("All items in a dependency tuple must be callable flow graph functions.")
            
            def multi_dependency_factory(**kwargs):
                kwargs['wait_for_nodes'] = [f.__name__ for f in node_funcs]
                return cls(**kwargs)
            return multi_dependency_factory

        # Case 2: Single Parameter object (root node)
        elif isinstance(items, BaseModel) or is_dataclass(items):
            def param_factory(**kwargs):
                kwargs['params'] = items
                return cls(**kwargs)
            return param_factory
        
        # Case 3: Single Dependency (a single function)
        elif callable(items):
            # if not cls._is_flow_graph_function(items): raise TypeError(...)
            def single_dependency_factory(**kwargs):
                # We store it as a list for consistency.
                kwargs['wait_for_nodes'] = [items.__name__]
                return cls(**kwargs)
            return single_dependency_factory
        
        # Case 4: Invalid input
        else:
            raise TypeError("Item in NodeContext[...] must be a param object, a single function, or a tuple of functions.")
