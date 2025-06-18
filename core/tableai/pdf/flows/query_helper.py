from typing import List, Dict, Any, Optional, Callable, ClassVar
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
from tableai.pdf.generic_params import QueryParams
from tableai.pdf.generic_tools import GroupbyTransform
from tableai.pdf.generic_tools import groupby

class QueryType(str, Enum):
    """Supported query types with different available fields."""
    STANDARD = "standard"
    WHITESPACE = "whitespace"

class GenericQueryBase(BaseModel):
    """
    High-level parameters for querying the LineTextIndex with validation and restrictions.
    Provides clear guidance on available options and enforces proper usage patterns.
    """

    __trace_ignore__ = True
    
    # =================== FIELD MAPPINGS & CONSTANTS ===================
    # Use ClassVar to make these accessible in validators
    
    # This mapping shows what the GROUP RESULT fields will be named AFTER grouping
    # The keys here are the original groupby keys, values are the resulting field names
    GROUPBY_RESULT_MAPPING: ClassVar[Dict[str, str]] = {
        "page": "group_page",
        "block": "group_block", 
        "line": "group_line",
        "span": "group_span",
        "font_meta": "group_font_meta",
        "bbox": "group_bboxes",
        "x_span": "group_x_spans",
        "y_span": "group_y_spans",
        "region": "group_regions",
        "normalized_value": "group_normalized_values",
        "text": "group_text",  
        "normalized_text": "group_normalized_text",
        "path": "group_paths",
        "bbox(rel)": "group_bboxes_rel",
        "size": "group_sizes",
        "color": "group_colors",
        "font": "group_fonts"
    }
    
    # Valid keys that can be used for groupby operations
    VALID_GROUPBY_KEYS: ClassVar[List[str]] = list(GROUPBY_RESULT_MAPPING.keys())
    
    ALL_AVAILABLE_KEYS: ClassVar[List[str]] = [
        'alpha', 'origin', 'normalized_text', 'ascender', 'size', 'flags', 
        'bbox', 'descender', 'bidi', 'full_width_v_whitespace', 'color', 
        'font', 'text', 'char_flags'
    ]
    
    CORE_INDEX_RESULT_VALUES: ClassVar[List[str]] = [
        'physical_page', 'block', 'line', 'span', 'font_meta', 'index', 
        'page', 'bbox', 'x0', 'y0', 'x1', 'y1', 'x_span', 'y_span', 
        'region', 'x0(rel)', 'y0(rel)', 'x1(rel)', 'y1(rel)', 'bbox(rel)', 
        'page_height(rel)', 'page_width(rel)', 'physical_page_bounds', 
        'normalized_value', 'key', 'value', 'path', 'text', 'normalized_text'
    ]
    
    FULL_WIDTH_WHITESPACE_VALUES: ClassVar[List[str]] = [
        'page', 'block', 'line', 'span', 'index', 'key', 'value', 'path', 
        'gap', 'bbox', 'x0', 'y0', 'x1', 'y1', 'x_span', 'y_span', 'meta', 
        'font_meta', 'physical_page_bounds'
    ]
    
    # =================== RESTRICTED COMBINATIONS ===================
    
    INVALID_KEY_VALUE_COMBINATIONS: ClassVar[Dict[str, List[str]]] = {
        'full_width_v_whitespace': ['text'],  # Whitespace entries don't have text values
        'gap': ['text', 'font', 'size'],      # Gap entries don't have font properties
    }
    
    GROUPBY_RESTRICTED_KEYS: ClassVar[List[str]] = [
        'full_width_v_whitespace',  # Don't group by whitespace type
        'gap',                      # Don't group by gap values
        'index'                     # Don't group by unique identifiers
    ]
    
    RECOMMENDED_GROUPBY_KEYS: ClassVar[List[str]] = [
        'block', 'line', 'span', 'page', 'font', 'size', 'color', 'region'
    ]
    
    # =================== QUERY PARAMETERS ===================
    
    query_type: QueryType = Field(
        default=QueryType.STANDARD,
        description="Type of query - determines available fields and restrictions"
    )
    
    key: Optional[str] = Field(
        default=None,
        description="Filter by specific key from ALL_AVAILABLE_KEYS"
    )
    
    page: Optional[int] = Field(
        default=None,
        description="Filter by virtual page number"
    )
    
    groupby_keys: Optional[List[str]] = Field(
        default=None,
        description="Keys to group by - these are the ORIGINAL field names before grouping"
    )
    
    include_fields: Optional[List[str]] = Field(
        default=None,
        description="Fields to include in results - must be valid for query_type"
    )
    
    exclude_bounds: Optional[str] = Field(
        default=None,
        description="Named restriction bounds to exclude from results"
    )
    
    query_label: Optional[str] = Field(
        default=None,
        description="Label for tracking and debugging queries"
    )
    
    # =================== VALIDATION ===================
    
    @field_validator('key')
    @classmethod
    def validate_key(cls, v):
        if v is not None and v not in cls.ALL_AVAILABLE_KEYS:
            raise ValueError(f"Key '{v}' not in available keys: {cls.ALL_AVAILABLE_KEYS}")
        return v
    
    @field_validator('groupby_keys')
    @classmethod
    def validate_groupby_keys(cls, v):
        if v is None:
            return v
        
        for key in v:
            if key in cls.GROUPBY_RESTRICTED_KEYS:
                raise ValueError(f"Key '{key}' is restricted for groupby operations. "
                               f"Restricted keys: {cls.GROUPBY_RESTRICTED_KEYS}")
            
            if key not in cls.VALID_GROUPBY_KEYS:
                raise ValueError(f"Key '{key}' is not a valid groupby key. "
                               f"Valid groupby keys: {cls.VALID_GROUPBY_KEYS}")
        return v
    
    @field_validator('include_fields')
    @classmethod
    def validate_include_fields(cls, v, info):
        if v is None:
            return v
        
        # Get query_type from context
        query_type = info.data.get('query_type', QueryType.STANDARD)
        
        if query_type == QueryType.STANDARD:
            valid_fields = cls.CORE_INDEX_RESULT_VALUES
        elif query_type == QueryType.WHITESPACE:
            valid_fields = cls.FULL_WIDTH_WHITESPACE_VALUES
        else:
            valid_fields = cls.CORE_INDEX_RESULT_VALUES
        
        for field in v:
            if field not in valid_fields:
                raise ValueError(f"Field '{field}' not valid for query_type '{query_type}'. "
                               f"Valid fields: {valid_fields}")
        return v
    
    @model_validator(mode='after')
    def validate_key_value_combinations(self):
        """Validate that key and value combinations make sense."""
        if self.key and self.key in self.INVALID_KEY_VALUE_COMBINATIONS:
            restricted_values = self.INVALID_KEY_VALUE_COMBINATIONS[self.key]
            if self.include_fields:
                invalid_includes = [f for f in self.include_fields if f in restricted_values]
                if invalid_includes:
                    raise ValueError(f"Cannot include {invalid_includes} when key='{self.key}'. "
                                   f"Key '{self.key}' is incompatible with: {restricted_values}")
        return self
    
    def get_groupby_result_fields(self) -> List[str]:
        """Get the field names that will be available AFTER grouping."""
        if not self.groupby_keys:
            return []
        return [self.GROUPBY_RESULT_MAPPING[key] for key in self.groupby_keys]
    
    def get_available_fields(self) -> List[str]:
        """Get available fields based on query type."""
        if self.query_type == QueryType.WHITESPACE:
            return self.FULL_WIDTH_WHITESPACE_VALUES
        return self.CORE_INDEX_RESULT_VALUES
    
    def get_suggested_includes(self) -> List[str]:
        """Get suggested include fields for common use cases."""
        if self.query_type == QueryType.WHITESPACE:
            return ['bbox', 'gap', 'physical_page_bounds']
        else:
            return ['bbox', 'path', 'text', 'normalized_text', 'bbox(rel)']
    
    def to_query_params(self) -> Dict[str, Any]:
        """Convert to dictionary suitable for LineTextIndex.query() WITHOUT groupby."""
        params = {}
        
        if self.key:
            params['key'] = self.key
        if self.page is not None:
            params['page'] = self.page
        if self.exclude_bounds:
            params['exclude_bounds'] = self.exclude_bounds
        if self.query_label:
            params['query_label'] = self.query_label

        if not set({k for k,v in params.items() if v}) - set(['key']):
            return params

    def to_groupby_params(self) -> Optional[Dict[str, Any]]:
        """Convert to groupby() function parameters using ORIGINAL keys."""
        if not self.groupby_keys:
            return None
        
        # Build parameters for groupby function
        params = {}
        
        # Use the ORIGINAL keys for groupby (not the mapped result names)
        params['groupby_keys'] = self.groupby_keys.copy()
        
        # Add include fields if specified
        if self.include_fields:
            params['include'] = self.include_fields
        else:
            params['include'] = self.get_suggested_includes()
        
        return groupby(*self.groupby_keys.copy(), include=params['include'], query_label=self.query_label)
    
    def to_groupby_transform(self, filterby: Optional[Callable] = None, description: Optional[str] = None) -> 'GroupbyTransform':
        """Convert to GroupbyTransform object with optional filtering."""
        if not self.groupby_keys:
            raise ValueError("Cannot create GroupbyTransform without groupby_keys")
        include_fields = self.include_fields or self.get_suggested_includes()
        
        # Create GroupbyTransform object using ORIGINAL keys
        return GroupbyTransform(
            *self.groupby_keys,  # Unpack the original keys
            filterby=filterby,
            include=include_fields,
            query_label=self.query_label,
            description=description
        )
    
    def create_base_query_params(self) -> 'QueryParams':
        """Create a QueryParams object for the base query (before grouping)."""        
        return QueryParams(
            key=self.key,
            page=self.page,
            exclude_bounds=self.exclude_bounds,
            query_label=self.query_label
        )
    
    def build_query(self, filterby: Optional[Callable] = None, description: Optional[str] = None) -> 'QueryParams':
        """Create a complete QueryParams object with groupby if specified."""
        # Start with base query params
        __trace_ignore__ = True
        base_params = self.create_base_query_params()
        
        # Add groupby if specified
        if self.groupby_keys:
            if filterby:
                groupby_transform = self.to_groupby_transform(filterby=filterby, description=description)
            else:
                groupby_transform = self.to_groupby_transform(filterby=None, description=description)
            base_params.groupby = groupby_transform
        
        return base_params


class QueryBuilder:
    """
    A simple builder that constructs the actual query configuration object.
    Its only job is to provide a clean, keyword-based interface.
    """
    @staticmethod
    def build(**kwargs) -> GenericQueryBase:
        """
        Creates a GenericQueryBase object directly from keyword arguments.
        
        Example:
            QueryBuilder.build(key="text", groupby_keys=["block"])
        """
        __trace_ignore__ = True
        # We can add validation or default logic here if needed in the future.
        return GenericQueryBase(**kwargs)