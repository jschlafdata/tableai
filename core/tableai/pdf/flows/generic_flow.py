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
    async def run(self, pdf_path: str, inclue_pdf_model: bool = True) -> 'FlowResult':
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
        if inclue_pdf_model:
            result.pdf_model = self._pdf_model
        
        # 4. Store the result and return it.
        self.final_result_ = result
        return result