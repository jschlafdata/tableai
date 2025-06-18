import operator
import re
from typing import Any, Callable
from pydantic import ConfigDict, BaseModel

# A dictionary to map string operators to their function counterparts
OPERATOR_MAP = {
    '>': operator.gt,
    '>=': operator.ge,
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne,
}

def callable_or_type_serializer(v: Any) -> Any:
    """
    A robust JSON serializer for callables and types.
    It converts them to their string name if possible, otherwise returns a string representation.
    """
    if hasattr(v, '__name__'):
        return v.__name__
    return str(v)

class ValidationRule(BaseModel):
    """A formal model for a validation rule to be applied to a step's result."""
    attribute_name: str
    operator: Callable
    expected_value: Any
    description: str

    model_config = ConfigDict(
        json_encoders={
            Callable: callable_or_type_serializer,
        }
    )
    def validate(self, result_obj: Any) -> bool:
        """Runs the validation against a given object."""
        if not hasattr(result_obj, self.attribute_name):
            return False # Or raise an error if the attribute must exist
        
        actual_value = getattr(result_obj, self.attribute_name)
        # If the attribute is a method (like __len__ for len()), call it
        if callable(actual_value):
            actual_value = actual_value()
            
        return self.operator(actual_value, self.expected_value)

class Assert:
    """A user-friendly factory for creating common ValidationRule instances."""
    
    @classmethod
    def count(cls, expression: str) -> ValidationRule:
        """
        Creates a validation rule for the count (length) of a result.
        
        Args:
            expression (str): A string like " > 0", "== 5", or "<= 10".
        """
        # Safely parse the expression without using eval()
        match = re.match(r'\s*(>|>=|==|!=|<=|<)\s*(-?\d+)\s*', expression)
        if not match:
            raise ValueError(
                f"Invalid count expression: '{expression}'. "
                "Expected format like '> 0' or '== 5'."
            )
        
        op_str, val_str = match.groups()
        op_func = OPERATOR_MAP.get(op_str)
        value = int(val_str)
        
        return ValidationRule(
            attribute_name='__len__',  # `len(obj)` calls `obj.__len__()`
            operator=op_func,
            expected_value=value,
            description=f"Assert count {op_str} {value}"
        )