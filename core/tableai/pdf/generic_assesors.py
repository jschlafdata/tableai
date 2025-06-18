from typing import Optional, List,  Dict, Any, Callable, Tuple, TypeVar, Generic
from tableai.pdf.generic_results import GroupbyQueryResult, DefaultQueryResult
from tableai.pdf.generic_tools import GroupChain # Assuming this is its correct location

T = TypeVar('T')

class BaseAccessor(Generic[T]):
    """Base class for type-aware accessors."""
    def __init__(self, data: List[T]):
        self._data = data
        self._validate()

    def _validate(self):
        # Optional: Add runtime validation
        pass

    def apply(self, func: Callable[[T], any]) -> list:
        """Applies a function to each item."""
        return [func(item) for item in self._data]

# --- Accessor for DefaultQueryResult ---
class DefaultAccessor(BaseAccessor[DefaultQueryResult]):
    """

    Provides type-aware methods for a ResultSet of DefaultQueryResult objects.
    """
    def get_values(self) -> list:
        """Extracts the 'value' from each result."""
        return self.apply(lambda item: item.value)

    def get_bboxes(self) -> list[tuple | None]:
        """Extracts the 'bbox' from each result."""
        return self.apply(lambda item: item.bbox)

# --- Updated GroupAccessor with chaining support ---
class GroupAccessor(BaseAccessor[GroupbyQueryResult]):
    
    @property
    def chain(self) -> GroupChain:
        """Access the chainable interface."""
        return GroupChain(self._data)
    
    def process(
        self,
        aggregations: Optional[Dict[str, Callable]] = None,
        filters: Optional[List[Callable]] = None,
        include: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Legacy method for backward compatibility.
        Consider using the chain interface instead.
        """
        # Use provided parameters or defaults
        aggregations = aggregations or {}
        filters = filters or []
        include = include or []

        results = []
        for group in self._data:
            # Build the initial dictionary for this group
            output_dict = {'group_id': group.group_id}

            # Add requested fields from the group object
            for field_name in include:
                if hasattr(group, field_name):
                    output_dict[field_name] = getattr(group, field_name)

            # Apply aggregation functions to compute new fields
            for new_field_name, agg_func in aggregations.items():
                output_dict[new_field_name] = agg_func(group)

            # Apply all filter functions - keep group only if all return True
            keep_group = all(filter_func(output_dict) for filter_func in filters)
            
            if keep_group:
                results.append(output_dict)
            
        return results