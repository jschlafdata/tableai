from collections import UserList
from typing import Optional, List,  Dict, Any, Callable
import re
from typing import List, Dict, Any, Optional, Union, Type
from tableai.pdf.generic_tools import GroupbyTransform
from pydantic import BaseModel, Field, field_serializer, create_model

TRACE_IGNORE = False

class QueryParams(BaseModel):
    """
    A comprehensive, self-documenting model for all parameters
    used by the FitzSearchIndex.query method.
    """
    class Config:
        # Pydantic needs this to allow non-serializable types like functions.
        arbitrary_types_allowed = True

    # --- Basic Filters ---
    page: Optional[int] = Field(
        default=None,
        description="Filter results to a single virtual page number."
    )
    line: Optional[int] = Field(
        default=None,
        description="Filter results to a single line number within a block."
    )
    key: Optional[str] = Field(
        default=None,
        description="Filter for rows with this specific key. Common keys are 'text', 'normalized_text', and 'full_width_v_whitespace'."
    )

    # --- Spatial & Custom Filters ---
    exclude_bounds: Optional[str] = Field(
        default=None,
        description="The string key of a pre-defined boundary set in the FitzSearchIndex's restriction_store. Any row whose bbox overlaps with these zones will be EXCLUDED from the results."
    )
    bounds_filter: Optional[Callable[[dict], bool]] = Field(
        default=None,
        description="A custom, dynamic filter function (e.g., a lambda) that receives a complete row dictionary and must return True to keep it. This is applied AFTER the 'exclude_bounds' filter, making it ideal for fine-grained spatial logic (e.g., checking x_span, y0_rel, etc.)."
    )

    groupby: Optional[GroupbyTransform] = Field(
        default=None,
        description="A GroupbyTransform object that defines how to group the filtered results."
    )

    # --- Post-Processing & Metadata ---
    transform: Optional[Callable[[list], list]] = Field(
        default=None,
        description="A function that takes the entire list of filtered results and reshapes it. Primarily used for grouping operations like the `groupby()` transform."
    )
    query_label: Optional[str] = Field(
        default=None,
        description="An optional label to attach to each result object for tracking and identification purposes."
    )
    
    @field_serializer('groupby', 'transform', 'bounds_filter', when_used='json-unless-none')
    def serialize_special_types(self, value: Any) -> Any:
        """Intelligently serializes special types for readable logs."""
        if isinstance(value, GroupbyTransform):
            return value.to_dict() # Use our new descriptive method
        
        if callable(value):
            func_name = getattr(value, '__name__', '<lambda>')
            if hasattr(value, 'func'):
                func_name = f"partial({getattr(value.func, '__name__', 'unknown')})"
            return f"<function: {func_name}>"
            
        return value

class TextNormalizer:
    """A callable object that normalizes text based on a set of regex patterns."""
    def __init__(self, patterns: Dict[str, str], description: Optional[str] = None, output_key: Optional[str]='normalized_text'):
        self.patterns = patterns
        self.output_key = output_key or 'normalized_text'
        self.description = description or f"Normalizes text using a set of regex substitutions. Used to build index items with the key=[{output_key}]."

    def __call__(self, text: str) -> str:
        """Makes the object callable to perform the normalization."""
        t = text.lower().strip()
        for pattern, replacement in self.patterns.items():
            t = re.sub(pattern, replacement, t)
        return t

    def to_dict(self) -> dict:
        """Creates a human-readable dictionary for logging."""
        return {
            "type": "TextNormalizer",
            "patterns": self.patterns,
            "description": self.description
        }

class WhitespaceGenerator:
    """A callable object that computes full-width vertical whitespace."""
    def __init__(self, min_gap: float = 5.0, description: Optional[str] = None, output_key: Optional[str]='full_width_v_whitespace'):
        self.min_gap = min_gap
        self.output_key = output_key or 'full_width_v_whitespace'
        self.description = description or f"Detects vertical whitespace regions spanning the page width. Used to build index items with the key=[{output_key}]."

    def __call__(self, by_page: Dict, page_metadata: Dict) -> List[Dict[str, Any]]:
        """Makes the object callable to perform the calculation."""
        # The entire logic from your old compute_full_width_v_whitespace function goes here.
        # ... just use self.min_gap instead of the hardcoded value.
        results = []
        for page_num, rows in by_page.items():
            spans = [r for r in rows if r.get("key") == "text" and r.get("bbox")]
            spans = sorted(spans, key=lambda r: r["y0"])
            page_width = page_metadata.get(page_num, {}).get("width", 612.0)  # fallback default A4 width
    
            for i in range(len(spans) - 1):
                a, b = spans[i], spans[i + 1]
                gap = b["y0"] - a["y1"]
                if gap >= self.min_gap:
                    y0 = a["y1"]
                    y1 = b["y0"]
                    results.append({
                        "page": page_num,
                        "block": -1,
                        "line": -1,
                        "span": -1,
                        "index": -1,
                        "key": "full_width_v_whitespace",
                        "value": "",
                        "path": None,
                        "gap": gap,
                        "bbox": (0.0, y0, page_width, y1),
                        "x0": 0.0,
                        "y0": y0,
                        "x1": page_width,
                        "y1": y1,
                        "x_span": page_width,
                        "y_span": y1 - y0,
                        "meta": {"gap_class": "large" if gap > 20 else "small"}
                    })
        return results

    def to_dict(self) -> dict:
        """Creates a human-readable dictionary for logging."""
        return {
            "type": "WhitespaceGenerator",
            "min_gap": self.min_gap,
            "description": self.description
        }


class GenericFunctionParams(BaseModel):
    """
    A base model for query parameters. All dynamically created parameter
    models will inherit from this class. You can place truly universal
    parameters here.
    """
    __trace_ignore__ = TRACE_IGNORE
    # Example of a truly universal parameter that all models will inherit
    query_label: Optional[str] = Field(
        default=None,
        description="An optional label to attach to the query results for tracking."
    )
    @classmethod
    def create_custom_model(
        cls, 
        model_name: str, 
        custom_fields: Dict[str, Dict[str, Any]],
        instantiate: bool = True,
        **instance_values
    ) -> Union[Type[BaseModel], BaseModel]:
        """
        Dynamically creates a new Pydantic model class, optionally returning an instance.
        
        Args:
            model_name (str): The name for the new Pydantic model class
            custom_fields (Dict): Field definitions for the new model  
            instantiate (bool): If True, return an instance; if False, return the class
            **instance_values: Values to populate the instance with (only used if instantiate=True)
            
        Returns:
            Either the model class or an instance of it
        """
        # Create the model class (same as before)
        __trace_ignore__ = TRACE_IGNORE
        pydantic_fields: Dict[str, Any] = {}
        
        for field_name, config in custom_fields.items():
            field_type = config.get("type", Any)
            default_value = config.get("default", ...)
            description = config.get("description", None)
            impact = config.get("impact", None)
            
            pydantic_fields[field_name] = (
                field_type,
                Field(default=default_value, description=description, impact=impact)
            )
    
        new_model_class = create_model(
            model_name,
            __base__=cls,
            **pydantic_fields
        )
        
        if instantiate:
            return new_model_class(**instance_values)
        else:
            return new_model_class