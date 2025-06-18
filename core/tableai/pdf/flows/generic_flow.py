from pydantic import BaseModel, Field, ConfigDict, create_model, PrivateAttr, model_validator
from typing import Dict, List, Any, Optional, Union, get_type_hints, Type
from datetime import datetime
from abc import ABC, abstractmethod
import inspect
from tableai.pdf.flows.generic_dependencies import FlowDependencies
from tableai.pdf.generic_params import TextNormalizer, WhitespaceGenerator
from tableai.pdf.pdf_model import LoadType, PDFModel
from tableai.pdf.flows.trace_log import EnhancedTraceLog
from tableai.pdf.flows.base_flow import Flow
from tableai.pdf.flows.generic_results import FlowResult
from tableai.pdf.flows.generic_params import FlowParams
from tableai.pdf.flows.generic_results import FlowResultStage

import textwrap
import json

class FlowReportGenerator:
    """
    Generates a human-readable Markdown report from a Flow's pre-computed
    node specifications and its execution trace log.
    """
    def __init__(self, flow: 'Flow', trace_log: 'EnhancedTraceLog'):
        self.flow = flow
        self.trace_log = trace_log

    def _format_params(self, params: Dict[str, Any]) -> str:
        """Formats a dictionary of parameters into a nice markdown block."""
        if not params:
            return "`None`"
        # Use pretty-printed JSON for a clean, readable dictionary format
        return f"```json\n{json.dumps(params, indent=2)}\n```"

    def _format_input_model(self, model_spec: Optional['ModelSpecification']) -> str:
        """Formats an input model's specification into a markdown table."""
        if not model_spec:
            return "`None`"
        
        header = f"#### Model: `{model_spec.name}`\n*_{model_spec.description or 'No description.'}_*\n\n"
        table = "| Parameter | Type | Default | Description |\n"
        table += "|:---|:---|:---|:---|\n"
        for field in model_spec.fields:
            # Clean up the type string for readability
            field_type = field.type.replace("<class '", "").replace("'>", "").replace("typing.", "")
            # Format default value for display
            default_val = f"`{field.default!r}`" if field.default != 'REQUIRED' else "**REQUIRED**"
            table += f"| `{field.name}` | `{field_type}` | {default_val} | {field.description or ''} |\n"
        return header + table

    def _format_called_functions(self, called_funcs: List['CallSpecification']) -> str:
        """Formats the list of called functions into detailed sections."""
        if not called_funcs:
            return "_No significant internal function calls were traced._"
        
        report = ""
        for call in called_funcs:
            # We now use the rich 'name' from the CallSpecification directly
            report += f"\n- **`{call.name}`**\n"
            if call.module:
                report += f"  - **Module:** `{call.module}`\n"
            if call.signature:
                report += f"  - **Signature:** `{call.signature}`\n"
            if call.docstring and call.docstring != "No docstring.":
                report += f"  - **Description:** *{call.docstring.replace(chr(10), ' ')}*\n"
        return report

    def generate_report(self) -> str:
        """
        Generates the full, human-readable markdown report for the entire flow.
        """
        # --- 1. Overall Flow Summary ---
        report = f"# Flow Report: {self.flow.overview}\n\n"
        report += f"**Goal:** *{self.flow.goal}*\n\n"
        
        # --- 2. Mermaid Diagram ---
        report += "## Workflow Graph\n\n"
        report += "```mermaid\n"
        report += self.flow.to_mermaid()
        report += "\n```\n\n"
        
        # --- 3. Detailed Node Breakdown ---
        report += "## Node Breakdown\n\n"
        
        # Use the execution order from the trace log for correct ordering
        execution_log = self.trace_log.get_log()
        
        for key, log_entry in execution_log.items():
            node_name = log_entry['name']
            
            # =============================================================
            # THE CORE FIX: Get the specification directly from the node
            # =============================================================
            try:
                node_spec = self.flow.get_node(node_name).specification
            except KeyError:
                report += f"### {key}\n\n> Could not retrieve specification for this node.\n\n---\n\n"
                continue

            report += f"### {key}\n\n"
            report += f"**Description:**\n> {node_spec.description.replace(chr(10), ' ')}\n\n"
            
            # --- Dependencies ---
            if node_spec.dependencies:
                deps_str = ", ".join([f"`{d}`" for d in node_spec.dependencies])
                report += f"**Depends On:** {deps_str}\n\n"
            
            # --- Parameters ---
            report += "**Parameters & Configuration:**\n\n"
            report += "**Input Model:**\n"
            report += f"{self._format_input_model(node_spec.input_model_spec)}\n\n"
            report += "**Effective Parameters Used at Registration:**\n"
            report += f"{self._format_params(node_spec.current_params)}\n\n"
            report += "*Guidance: These parameters can be modified by updating the `NodeContext` object for this step.*\n\n"

            # --- Internals ---
            report += "**Internals & Called Functions:**\n\n"
            report += "**Declaration:**\n"
            report += f"```python\n{node_spec.decorator_declaration}\n```\n"
            report += "**Called Functions:**\n"
            report += f"{self._format_called_functions(node_spec.called_functions)}\n\n"

            report += "---\n\n"

        return report

class GenericPDFFlowContext(BaseModel):
    """
    The ultimate, all-in-one flow runner. It encapsulates the flow's
    definition and can be executed with runtime-specific data like a pdf_path.
    """
    # --- Private Attributes for internal state ---
    # The Flow object is created once and held for the lifetime of the runner.
    _flow: Optional['Flow'] = PrivateAttr(default=None)
    _pdf_model: Optional['PDFModel'] = PrivateAttr(default=None)

    # --- Configuration Fields (provided by the user at creation) ---
    flow_params: 'FlowParams'
    
    # --- Optional Overrides & Default Configurations ---
    # These can be set at definition time to configure the runner's behavior.
    final_result_: Optional['FlowResult'] = None
    text_normalizer: TextNormalizer = TextNormalizer(
        patterns={
            r'page\s*\d+\s*of\s*\d+': 'page xx of xx',
            r'page\s*\d+': 'page xx'
        }
    )
    whitespace_generator: WhitespaceGenerator = WhitespaceGenerator(min_gap=5.0)
    load_type: 'LoadType' = LoadType.FULL
    trace_logger: 'EnhancedTraceLog' = Field(default_factory=EnhancedTraceLog)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode='after')
    def _init_flow_instance(self) -> 'GenericPDFFlowContext':
        """Creates the internal Flow instance from the provided params."""
        if self._flow is None:
            self._flow = Flow(
                deps_type=self.flow_params.deps_type,
                result_type=self.flow_params.result_type,
                overview=self.flow_params.overview,
                goal=self.flow_params.goal
            )
        return self

    @property
    def flow(self) -> 'Flow':
        """Provides access to the managed Flow instance for node definition."""
        if self._flow is None:
            # This should not happen due to the model_validator, but it's a safe check.
            raise RuntimeError("Flow instance has not been initialized.")
        return self._flow

    def _build_dependencies(self, pdf_path: str) -> 'FlowDependencies':
        """
        Builds the FlowDependencies object for a specific run, using the
        provided pdf_path. This is called "just-in-time" by the run method.
        """
        print(f"--- Initializing Dependencies for: {pdf_path} ---")
        self._pdf_model = PDFModel(
            path=pdf_path, 
            text_normalizer=self.text_normalizer,
            whitespace_generator=self.whitespace_generator,
            load_type=self.load_type
        )
        # We always use the runner's single trace_logger instance.
        return FlowDependencies(
            pdf_model=self._pdf_model, 
            trace=self.trace_logger
        )

    # =============================================================
    # The `run` method now accepts the required runtime arguments.
    # =============================================================
    async def run(self, pdf_path: str, inclue_pdf_model: bool = True) -> Dict[str, Any]:
        """
        Executes the encapsulated flow for a specific pdf_path.
        """
        # 1. Build the dependencies for this specific run.
        dependencies = self._build_dependencies(pdf_path)
        
        # 2. Get the flow to run from our internal state.
        flow_to_run = self.flow
        
        print(f"\n>>> Asynchronously Running flow: {flow_to_run.overview} <<<")
        
        # 3. Execute the flow with the just-in-time dependencies.
        result = await flow_to_run.run(deps=dependencies)
        self.final_result_ = result
        report = FlowReportGenerator(flow=self.flow, trace_log=self.trace_logger)
        return result, self.flow, report