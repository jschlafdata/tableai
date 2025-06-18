from typing import List, Union, Dict, Any, Callable, Optional
from typing import Dict, Any, Callable, Union
from copy import deepcopy
from tableai.pdf.generic_results import ChainResult, GroupbyQueryResult
import itertools
from collections import defaultdict

class BaseChain:
    """
    An empty base class to mark an object as a 'chain' that needs
    to be executed to produce a final result.
    """
    # The abstract method now correctly returns a ChainResult
    async def as_chain_result(self) -> ChainResult:
        raise NotImplementedError

class GroupChain(BaseChain):
    """Chainable processor for GroupbyQueryResult objects with pandas-style API."""
    
    def __init__(self, data: List[GroupbyQueryResult]):
        self._data = data
        self._result_data = None  # Will hold processed dictionaries
        
    def include(self, fields: Union[str, List[str]]) -> 'GroupChain':
        """
        Include fields from the original group objects.
        
        Args:
            fields: Field name(s) to include from group objects
            
        Returns:
            New GroupChain instance for continued chaining
        """
        if isinstance(fields, str):
            fields = [fields]
            
        # Initialize result data if not already done
        if self._result_data is None:
            self._result_data = [{'group_id': group.group_id} for group in self._data]
        
        # Create new chain with copied data
        new_chain = GroupChain(self._data)
        new_chain._result_data = deepcopy(self._result_data)
        
        # Add requested fields
        for i, group in enumerate(self._data):
            for field_name in fields:
                if hasattr(group, field_name):
                    new_chain._result_data[i][field_name] = getattr(group, field_name)
                    
        return new_chain
    
    def agg(self, aggregations: Dict[str, Callable]) -> 'GroupChain':
        """
        Apply aggregation functions to compute new fields.
        
        Args:
            aggregations: Dictionary of {field_name: aggregation_function}
            
        Returns:
            New GroupChain instance for continued chaining
        """
        # Initialize result data if not already done
        if self._result_data is None:
            self._result_data = [{'group_id': group.group_id} for group in self._data]
            
        # Create new chain with copied data
        new_chain = GroupChain(self._data)
        new_chain._result_data = deepcopy(self._result_data)
        
        # Apply aggregations
        for i, group in enumerate(self._data):
            for field_name, agg_func in aggregations.items():
                new_chain._result_data[i][field_name] = agg_func(group)
                
        return new_chain
    
    def filter(self, condition: Callable[[Dict[str, Any]], bool]) -> 'GroupChain':
        """
        Filter groups based on a condition function.
        
        Args:
            condition: Function that takes a result dict and returns bool
            
        Returns:
            New GroupChain instance for continued chaining
        """
        if self._result_data is None:
            raise ValueError("Must call include() or agg() before filter()")
            
        # Create new chain with filtered data
        new_chain = GroupChain([])
        new_chain._result_data = []
        
        filtered_groups = []
        for i, result_dict in enumerate(self._result_data):
            if condition(result_dict):
                new_chain._result_data.append(deepcopy(result_dict))
                filtered_groups.append(self._data[i])
                
        new_chain._data = filtered_groups
        return new_chain
    
    def query(self, condition: str) -> 'GroupChain':
        """
        Filter using a pandas-style query string (simplified version).
        
        Args:
            condition: Query condition as string (e.g., "merged_bbox_rel_y < 100")
            
        Returns:
            New GroupChain instance for continued chaining
        """
        # This is a simplified implementation - you could use pandas.eval for full functionality
        def eval_condition(row_dict):
            # Create a safe evaluation context with the row data
            local_vars = row_dict.copy()
            try:
                return eval(condition, {"__builtins__": {}}, local_vars)
            except:
                return False
                
        return self.filter(eval_condition)
    
    def assign(self, **kwargs) -> 'GroupChain':
        """
        Create new columns using keyword arguments (pandas-style).
        
        Args:
            **kwargs: Column_name=function pairs
            
        Returns:
            New GroupChain instance for continued chaining
        """
        return self.agg(kwargs)
    
    def to_list(self) -> List[Dict[str, Any]]:
        """
        Return the processed data as a list of dictionaries.
        
        Returns:
            List of processed group dictionaries
        """
        if self._result_data is None:
            # If no processing was done, return basic group info
            return [{'group_id': group.group_id} for group in self._data]
        return self._result_data.copy()
    
    def to_dict(self, orient: str = 'records') -> Union[List[Dict], Dict[str, List]]:
        """
        Convert to dictionary in various formats (pandas-style).
        
        Args:
            orient: 'records' (list of dicts) or 'dict' (dict of lists)
            
        Returns:
            Data in requested format
        """
        data = self.to_list()
        
        if orient == 'records':
            return data
        elif orient == 'dict':
            if not data:
                return {}
            result = {}
            for key in data[0].keys():
                result[key] = [row.get(key) for row in data]
            return result
        else:
            raise ValueError(f"Unsupported orient: {orient}")
    
    def head(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return first n results (pandas-style)."""
        return self.to_list()[:n]

    async def as_chain_result(self) -> ChainResult[Dict[str, Any]]:
        """
        Executes the chain and wraps the resulting list of dictionaries
        in a Chainable container, preserving the fluent interface.
        """
        result_list = self.to_list()
        return ChainResult(result_list)
    
    def __len__(self) -> int:
        """Return number of groups after filtering."""
        if self._result_data is None:
            return len(self._data)
        return len(self._result_data)
    
    def __repr__(self) -> str:
        """String representation showing current state."""
        return f"GroupChain({len(self)} groups)"
    
# Add this new class to your file
class GroupbyTransform:
    """A callable object that encapsulates the logic and parameters of a groupby operation."""
    def __init__(self, *keys, filterby=None, include=None, query_label=None, description=None):
        self.keys = keys
        self.group_id_field: str = "group_id"
        self.filterby = filterby or (lambda g: True)
        self.include = include if include is not None else ["bbox", "path", "text"]
        self.query_label = query_label
        self.description = description or f"Groups data by the following keys: {keys}"
        
        # --- Pre-define the mappings here for clarity ---
        self.AGGREGATE_MAPPING = {
            "page": "group_page",
            "block": "group_block",
            "line": "group_line",
            "span": "group_span",
            "font_meta": "group_font_meta",
            "bbox": "group_bboxes",
            "x_span": "group_x_spans",
            "y_span": "group_y_spans",
            "region": "group_regions",
            "text": "group_text",
            "normalized_text": "group_normalized_text",  
            "normalized_value": "group_normalized_values",
            "path": "group_paths",
            "bbox(rel)": "group_bboxes_rel"
        }
        self.CONSISTENT_FIELDS = [
            "page",
            "index", 
            "region",
            "physical_page",
            "physical_page_bounds",
            "meta",
        ]

    def to_dict(self) -> dict:
        """Creates a human-readable dictionary representation for logging."""
        return {
            "type": "groupby",
            "keys": self.keys,
            "include": self.include,
            "filterby": getattr(self.filterby, '__name__', '<lambda>'),
            "description": self.description
        }

    def __call__(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """This makes instances of the class callable, just like a function."""
        # --- The entire logic from your old _transform function goes here ---
        grouped = defaultdict(list)
        for row in rows:
            key_tuple = tuple(row.get(k) for k in self.keys)
            grouped[key_tuple].append(row)
            
        output_summaries = []
        for group_key, group_rows in grouped.items():
            if not self.filterby(group_rows):
                continue
            
            first_member = group_rows[0]

            # 3. Create the base summary dictionary with consistent fields
            summary = {
                self.group_id_field: group_key,
                "groupby_keys": self.keys,
                "member_count": len(group_rows),
            }
            if self.query_label:
                summary['query_label'] = self.query_label
            
            # Copy over the consistent fields from the first member of the group
            for field in self.CONSISTENT_FIELDS:
                if field in first_member:
                    summary[field] = first_member[field]
            
            # Dynamically add the key-value pairs used for grouping (e.g., summary['region'] = 'header')
            for i, key_name in enumerate(self.keys):
                summary[key_name] = group_key[i]

            # 4. Initialize and populate aggregated lists
            
            # Initialize lists for all fields that will be aggregated
            for item_key in self.include:
                if item_key in self.AGGREGATE_MAPPING:
                    summary[self.AGGREGATE_MAPPING[item_key]] = []
            # Iterate through rows once to populate the lists
            for row in group_rows:
                # Handle standard aggregations
                for item_key in self.include:
                    if item_key in self.AGGREGATE_MAPPING and item_key in row:
                        dest_key = self.AGGREGATE_MAPPING[item_key]
                        summary[dest_key].append(row[item_key])
            
            output_summaries.append(summary)
            
        return output_summaries

def groupby(*keys, **kwargs) -> GroupbyTransform:
    """
    Factory function that creates a self-describing, callable GroupbyTransform object.
    
    All arguments (filterby, include, query_label, description) are passed
    directly to the GroupbyTransform constructor.
    """
    return GroupbyTransform(*keys, **kwargs)


def regroup_by_key(
    data: List[Dict[str, Any]], 
    key: str, 
    min_count: int,
    return_list: Optional[bool] = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Regroups a list of dictionaries by a specified key and filters for groups of a minimum size.

    This is useful for performing a second-level aggregation on data that has already
    been processed once.

    Args:
        data: A list of dictionaries to process (e.g., your 'processed_data').
        key: The dictionary key to group by (e.g., 'full_text').
        min_count: The minimum number of items a group must have to be included
                   in the final result.

    Returns:
        A dictionary where keys are the unique values from the grouping 'key',
        and values are lists of the original dictionaries that belong to that group.
    """
    # Step 1: Group all items by the specified key.
    # The defaultdict makes this easy: if a key doesn't exist, it creates a new list for it.
    groups = defaultdict(list)
    for item in data:
        if key in item:
            group_key = item[key]
            groups[group_key].append(item)

    # Step 2: Filter the groups, keeping only those with enough members.
    # A dictionary comprehension is a clean way to build the final result.
    if return_list:
        filtered_groups = [
            items for group_key, items in groups.items()
            if len(items) >= min_count
        ]
        if filtered_groups:
            return list(itertools.chain(*filtered_groups))
        else:
            return []
    else:
        # Return a dictionary containing only the groups that meet the min_count
        return {
            group_key: items
            for group_key, items in groups.items()
            if len(items) >= min_count
        }