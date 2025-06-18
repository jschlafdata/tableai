from typing import Tuple, Generic, TypeVar, List, Dict, Any, Optional, Union, Callable, Type
from pydantic import BaseModel, Field, computed_field
from dataclasses import dataclass, is_dataclass
from pydantic import ValidationError, ConfigDict
from collections import defaultdict
from tableai.pdf.flows.trace import _generate_node_specification
from tableai.pdf.flows.validations import ValidationRule
from tableai.pdf.generic_tools import BaseChain
from tableai.pdf.flows.generic_results import R
from tableai.pdf.flows.generic_dependencies import D
from tableai.pdf.flows.generic_context import NodeContext, RunContext, StepInput
from tableai.pdf.flows.trace import NodeSpecification

class Flow(Generic[D, R]):
    """
    A graph-based flow executor using unified NodeContext for all step configurations.
    """
    class NodeAccessor:
        """
        A proxy object providing a clean API to inspect and dynamically
        update a node's configuration.
        """
        def __init__(self, flow: 'Flow', node_name: str):
            if node_name not in flow.nodes: raise KeyError(f"Node '{node_name}' not registered.")
            self._node_name = node_name
            self._node_def = flow.nodes[node_name]

        def set_config(self, **kwargs_to_update: Any) -> 'Flow.NodeAccessor':
            """
            Updates the node's configuration. It intelligently updates fields on
            the NodeContext object or its extra_params.
            """
            context = self._node_def['context_config']
            for key, value in kwargs_to_update.items():
                if hasattr(context, key) and key != 'extra_params':
                    setattr(context, key, value)
                else:
                    context.extra_params[key] = value
            print(f"Updated config for node '{self._node_name}': {kwargs_to_update}")
            return self

        def get_config(self) -> 'NodeContext':
            """Returns the current configuration object for the node."""
            return self._node_def['context_config']
            
        @property
        def specification(self) -> 'NodeSpecification':
            """Returns the complete, rich specification for this node."""
            # The spec is now generated once at registration time and stored.
            return self._node_def['specification']
        
        def __repr__(self):
            return f"<NodeAccessor for '{self._node_name}'>"
    
    def __init__(self, deps_type: Type[D], result_type: Type[R], overview: str, goal: str, auto_clear_on_failure: bool = True):
        self.deps_type = deps_type
        self.result_type = result_type
        self.overview = overview
        self.goal = goal
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self._run_cache: Dict[str, Any] = {}
        self.auto_clear_on_failure = auto_clear_on_failure

    # =============================================================
    # The `step` decorator also needs a final simplification
    # =============================================================
    def step(self, context: 'NodeContext') -> Callable[[Callable], Callable]:
        """
        Registers a function as a node. It receives a fully-formed NodeContext
        object and generates the node's specification immediately.
        """
        def decorator(func: Callable) -> Callable:
            node_name = func.__name__
            if node_name in self.nodes:
                raise NameError(f"A step with the name '{node_name}' is already registered.")

            # Generate the full specification at registration time.
            spec = _generate_node_specification(func, self, context)

            self.nodes[node_name] = {
                "function": func,
                "context_config": context,
                "specification": spec,
                "description": spec.description
            }
            print(f"Registered node '{node_name}' (type: {context.context_type})")
            return func
        return decorator

    def get_dag(self) -> Dict[str, List[str]]:
        """
        Calculates and returns the dependency graph as an adjacency list.
        
        Returns:
            A dictionary where keys are node names and values are a list
            of nodes that depend on that key.
        """
        adj = defaultdict(list)
        # Also track nodes that are depended upon
        dependents = set()

        for name, config in self.nodes.items():
            context_config = config['context_config']
            if context_config.context_type == "dependency" and context_config.wait_for_nodes:
                for dep_name in context_config.wait_for_nodes:
                    if dep_name not in self.nodes:
                        raise ValueError(f"Node '{name}' has an undefined dependency: '{dep_name}'")
                    adj[dep_name].append(name)
                    dependents.add(name)
        
        # Ensure all nodes are in the adjacency list, even if they have no children
        for name in self.nodes:
            if name not in adj:
                adj[name] = []
        
        return dict(adj)

    def _get_execution_order(self, dag: Dict[str, List[str]]) -> List[str]:
        """
        Performs a topological sort on the provided DAG to find the execution order.
        """
        in_degree = {name: 0 for name in self.nodes}
        # Calculate in-degrees from the DAG
        for parent, children in dag.items():
            for child in children:
                in_degree[child] += 1
        
        queue = [name for name in self.nodes if in_degree[name] == 0]
        order = []
        
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in dag.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected in the dependency graph.")
        return order

    def clear_steps(self):
        """Clears all registered nodes from the flow."""
        print(f"--- Clearing {len(self.nodes)} registered nodes from the flow ---")
        self.nodes.clear()

    def clear_cache(self):
        """Clears the internal run cache."""
        print("--- Clearing flow run cache ---")
        self._run_cache.clear()

    def get_node(self, node_name: str) -> 'NodeAccessor':
        """
        Gets an accessor object for a specific node, allowing for dynamic
        inspection and updates to its configuration.
        """
        return self.NodeAccessor(self, node_name)
    
    def _get_execution_order(self) -> List[str]:
        # This method uses `self.nodes` directly, no materialization needed.
        adj = defaultdict(list)
        in_degree = {name: 0 for name in self.nodes}
        for name, config in self.nodes.items():
            context_config = config['context_config']
            if context_config.context_type == "dependency" and context_config.wait_for_nodes:
                for dep_name in context_config.wait_for_nodes:
                    if dep_name not in self.nodes:
                        raise ValueError(f"Node '{name}' has an undefined dependency: '{dep_name}'")
                    adj[dep_name].append(name)
                    in_degree[name] += 1
        
        queue = [name for name in self.nodes if in_degree[name] == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0: queue.append(neighbor)
        if len(order) != len(self.nodes): raise ValueError("Cycle detected in dependency graph.")
        return order
    
    def _prepare_step_input(self, context_config: 'NodeContext', flow_state: Dict[str, Any]) -> 'StepInput':
        extra_params = context_config.extra_params
        input_model = context_config.input_model
        data_payload: Any = None

        if context_config.context_type == "root":
            data_payload = context_config.params
        elif context_config.context_type == "dependency":
            if len(context_config.wait_for_nodes) == 1:
                dep_name = context_config.wait_for_nodes[0]
                data_payload = flow_state.get(dep_name)
            else:
                data_payload = {dep_name: flow_state.get(dep_name) for dep_name in context_config.wait_for_nodes}
        elif context_config.context_type == "context_only":
            data_payload = None
        else:
            raise ValueError(f"Unknown context type: {context_config.context_type}")
            
        return StepInput(data=data_payload, config=extra_params, input_model=input_model)
        
    async def run(self, deps: D, use_cache: bool = False) -> R:
        """
        Asynchronously executes the flow. The node configurations are already
        materialized, so it can proceed directly to execution.
        """
        if not isinstance(deps, self.deps_type):
            raise TypeError(f"Dependencies must be of type {self.deps_type.__name__}")

        deps.trace.clear()
        
        try:
            # 1. First, calculate the graph structure (the DAG).
            dag = self.get_dag()
            # 2. Then, calculate a valid execution order from the DAG.
            execution_order = self._get_execution_order(dag)
            print(f"\nExecution Order: {' -> '.join(execution_order)}\n")
        except (ValueError, KeyError) as e:
            # ... (error handling is the same)
            pass

        flow_state = self._run_cache if use_cache else {}
        if use_cache: print("--- Running with cache enabled ---")

        try:
            for node_name in execution_order:
                if use_cache and node_name in flow_state:
                    print(f"--- Skipping Node (found in cache): {node_name} ---\n")
                    continue
        
                node_config = self.nodes[node_name]
                context = RunContext(deps=deps, state=flow_state)
                context_config = node_config["context_config"]
                sub_calls = []
                
                def record_sub_call(log_entry: Dict): sub_calls.append(log_entry)
                context._record_sub_call = record_sub_call
        
                step_input = self._prepare_step_input(context_config, flow_state)
                
                # Execute the node's function and get the initial result
                result = await node_config["function"](context, step_input)
    
                # "Live" runner logic: Finalize chain if needed
                if isinstance(result, BaseChain):
                    print(f"Node '{node_name}' returned a chain. Automatically executing...")
                    final_result_from_chain = await result.as_chain_result()
                    
                    # Log this finalization as a traced sub-call
                    sub_calls.append({"target": type(result).__name__, "method": "as_chain_result", "params": {}})
                    result = final_result_from_chain
                
                # Log the complete, successful execution to the trace history
                deps.trace.log_node_execution(
                    name=node_name, 
                    description=node_config["description"], 
                    config=context_config,
                    func_input=step_input, 
                    sub_calls=sub_calls, 
                    output=result
                )
                
                # Perform post-run validation if a rule is defined
                if context_config.validation_rule:
                    if not context_config.validation_rule.validate(result):
                        error_message = f"Validation failed for node '{node_name}': {context_config.validation_rule.description}."
                        if context_config.on_validation_failure == 'raise':
                            raise ValidationError(error_message)
                        elif context_config.on_validation_failure == 'warn':
                            print(f"WARNING: {error_message}")
                
                # Update the flow's state for downstream nodes
                flow_state[node_name] = result
                if use_cache: 
                    self._run_cache[node_name] = result
    
        except Exception as e:
            # Full error handling logic
            print(f"\nFlow execution failed at node '{node_name if 'node_name' in locals() else 'initialization'}': {e}")
            if self.auto_clear_on_failure:
                print("Auto-clearing steps and cache due to execution failure...")
                self.clear_steps()
                self.clear_cache()
            raise e

        node_specs=[]
        for n in list(self.nodes.keys()):
            try:
                _n = self.get_node(n)
                nod_spec = _n.specification.model_dump()
                node_specs.append(nod_spec)
            except Exception as e:
                node_specs.append({n: 'Node spec collection failed. {e}'})

        flow_state['_metadata'] = {
            'dag': dag,
            'execution_order': execution_order, 
            'func_trace': node_specs, 
            'mermaid': self.to_mermaid()
        }
        return flow_state

    def to_mermaid(self, orientation: str = "TD") -> str:
        """
        Generates a Mermaid graph syntax string representing the flow's
        Directed Acyclic Graph (DAG).
        
        Args:
            orientation (str): The graph orientation ('TD' for top-down,
                               'LR' for left-to-right, etc.).
                               
        Returns:
            A string containing the complete Mermaid graph definition.
        """
        if not self.nodes:
            return "graph TD;\n    subgraph Empty Flow\n        A[No nodes registered]\n    end"

        # Start the graph definition with the specified orientation
        mermaid_string = f"graph {orientation};\n"
        
        # Define styles for different node types
        mermaid_string += "    classDef root fill:#D5E8D4,stroke:#82B366,stroke-width:2px;\n"
        mermaid_string += "    classDef dependency fill:#DAE8FC,stroke:#6C8EBF,stroke-width:2px;\n"
        mermaid_string += "    classDef context_only fill:#F8CECC,stroke:#B85450,stroke-width:2px;\n"
        
        # A set to keep track of nodes already defined to avoid duplicates
        defined_nodes = set()

        # Iterate through all nodes to define their connections and styles
        for name, config in self.nodes.items():
            context = config['context_config']
            
            # Define the current node with its name as the label
            node_label = name.replace("_", " ")
            mermaid_string += f'    {name}["{node_label}"]\n'
            
            # Apply the appropriate CSS class based on the context type
            if context.context_type == "root":
                mermaid_string += f"    class {name} root;\n"
            elif context.context_type == "dependency":
                 mermaid_string += f"    class {name} dependency;\n"
            else: # context_only
                 mermaid_string += f"    class {name} context_only;\n"

            # Define the edges (arrows) from dependencies to the current node
            if context.context_type == "dependency" and context.wait_for_nodes:
                for dep_name in context.wait_for_nodes:
                    mermaid_string += f"    {dep_name} --> {name}\n"

        return mermaid_string
    
    async def __call__(self, deps: D, use_cache: bool = False) -> R:
        """Convenience method to run the flow."""
        return await self.run(deps, use_cache=use_cache)