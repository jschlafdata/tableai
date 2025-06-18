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
import pprint

class FlowReportGenerator:
    """
    Generates a human-readable and prescriptive Markdown report from a Flow's
    statically analyzed node specifications.
    """
    def __init__(self, flow: 'Flow'):
        """
        Initializes the report generator with the flow definition and
        pre-computes execution order.
        """
        self.flow = flow
        try:
            self.execution_order = flow._get_execution_order(flow.get_dag())
        except (ValueError, KeyError):
            self.execution_order = list(getattr(flow, 'nodes', []))  # fallback ordering

    def _format_params(self, params: Dict[str, Any]) -> str:
        """Formats a dictionary of parameters into a Markdown JSON code block."""
        if not params:
            return "`None`"
        try:
            body = json.dumps(params, indent=2, default=repr)
        except TypeError:
            body = pprint.pformat(params)
        return f"```json\n{body}\n```"

    def _format_input_model(self, model_spec: Optional['ModelSpecification']) -> str:
        """Formats an input model's specification into a Markdown table."""
        if not model_spec or not getattr(model_spec, 'fields', None):
            return "`None`"
        high_impact = [f for f in model_spec.fields if f.impact == 'HIGH']
        other = [f for f in model_spec.fields if f.impact != 'HIGH']

        report = f"#### Model: `{model_spec.name}`\n" \
                 f"*_{model_spec.description or 'No description.'}_*\n\n"
        # All parameters
        report += "<details><summary>All Parameters</summary>\n\n"
        report += "| Parameter | Type | Default | Impact | Description |\n"
        report += "|:---|:---|:---|:---|:---|\n"
        for f in model_spec.fields:
            ftype = getattr(f.type, '__name__', str(f.type))
            default = f"`{f.default!r}`" if f.default != 'REQUIRED' else "**REQUIRED**"
            impact = f"`{f.impact}`" if getattr(f, 'impact', None) else ""
            report += f"| `{f.name}` | `{ftype}` | {default} | {impact} | {f.description or ''} |\n"
        report += "\n</details>\n\n"
        # High impact section
        if high_impact:
            report += "**High-Impact Tuning Parameters:**\n\n"
            report += "| Parameter | Type | Default | Description |\n"
            report += "|:---|:---|:---|:---|\n"
            for f in high_impact:
                ftype = getattr(f.type, '__name__', str(f.type))
                default = f"`{f.default!r}`" if f.default != 'REQUIRED' else "**REQUIRED**"
                report += f"| `{f.name}` | `{ftype}` | {default} | {f.description or ''} |\n"
            report += "\n"
        return report

    def _format_called_functions(self, called_funcs: List['CallSpecification']) -> str:
        """Formats the list of called functions, including their module, signature, docstring, and source."""
        if not called_funcs:
            return "_No significant internal function calls were traced._"
        report = ""
        for call in called_funcs:
            report += f"- **`{call.name}`**\n"
            if call.module:
                report += f"  - **Module:** `{call.module}`\n"
            if call.signature:
                report += f"  - **Signature:** `{call.signature}`\n"
            if getattr(call, 'docstring', None) and call.docstring.lower() != "no docstring":
                doc = call.docstring.replace('\n', ' ')
                report += f"  - **Description:** *{doc}*\n"
            if getattr(call, 'source_code', None) and call.source_code != "<source not available>":
                src = textwrap.indent(textwrap.dedent(call.source_code), '    ')
                report += (
                    "  <details>\n"
                    "  <summary>View Source</summary>\n\n"
                    "  ```python\n"
                    f"{src}\n"
                    "  ```\n"
                    "  </details>\n"
                )
        return report

    def generate_report(self) -> str:
        """
        Generates the full, human-readable Markdown report for the entire flow.
        """
        report = f"# Flow Report: {self.flow.overview}\n\n"
        report += f"**Goal:** *{self.flow.goal}*\n\n"
        # Mermaid diagram
        report += "## Workflow Graph\n\n"
        report += "```mermaid\n" + self.flow.to_mermaid() + "\n```\n\n"
        # Node breakdown
        report += "## Node Breakdown\n\n"
        for i, node_name in enumerate(self.execution_order, start=1):
            try:
                node = self.flow.get_node(node_name)
                spec = node.specification
            except (KeyError, AttributeError):
                report += (
                    f"### {i}. {node_name}\n\n"
                    "> Could not retrieve specification for this node.\n\n---\n\n"
                )
                continue
            report += f"### {i}. {node_name}\n\n"
            report += f"**Description:**\n> {spec.description or 'No description.'}\n\n"
            if getattr(spec, 'decorator_declaration', None):
                report += "```python\n" + spec.decorator_declaration + "\n```\n\n"
            report += "**Input Model:**\n\n" + self._format_input_model(spec.input_model_spec) + "\n"
            if getattr(spec, 'current_params', None):
                report += "**Effective Parameters at Definition:**\n\n"
                report += self._format_params(spec.current_params) + "\n\n"
            report += "**Internals & Called Functions:**\n\n"
            report += self._format_called_functions(spec.called_functions) + "\n\n---\n\n"
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
    _path: Optional[str] = PrivateAttr(default=None)

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
        self._path = pdf_path
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
        
        # 3. Execute the flow with the just-in-time dependencies.
        result = await flow_to_run.run(deps=dependencies)
        self.final_result_ = result
        report = FlowReportGenerator(flow=self.flow)
        gen = report.generate_report()
        # print(f"generate_report: {report.generate_report()}")
        return result, self.flow, report, self._path