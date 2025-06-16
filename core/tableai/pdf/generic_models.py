from typing import Optional
from typing import Optional, List, Tuple, Union, Dict, Any, Callable, TYPE_CHECKING, TypeVar, Generic, Type
from pydantic import BaseModel, field_validator, model_validator, ValidationError, Field, create_model, field_serializer


class GenericFunctionParams(BaseModel):
    """
    A base model for query parameters. All dynamically created parameter
    models will inherit from this class. You can place truly universal
    parameters here.
    """
    # Example of a truly universal parameter that all models will inherit
    query_label: Optional[str] = Field(
        default=None,
        description="An optional label to attach to the query results for tracking."
    )

    @classmethod
    def create_custom_model(
        cls, 
        model_name: str, 
        custom_fields: Dict[str, Dict[str, Any]]
    ) -> Type[BaseModel]:
        """
        Dynamically creates a new Pydantic model that inherits from this base class.

        Args:
            model_name (str): The name for the new Pydantic model class (e.g., "HorizontalWhitespaceParams").
            custom_fields (Dict): A dictionary defining the custom fields for the new model.
                The format is:
                {
                    "field_name": {
                        "type": field_type (e.g., int, str),
                        "default": default_value,
                        "description": "A helpful description."
                    },
                    ...
                }

        Returns:
            A new Pydantic model class, ready to be instantiated.
        """
        # Prepare a dictionary of field definitions in the format Pydantic's create_model expects
        pydantic_fields: Dict[str, Any] = {}
        
        for field_name, config in custom_fields.items():
            field_type = config.get("type", Any)
            default_value = config.get("default", ...) # ... means the field is required if no default
            description = config.get("description", None)
            
            # The value in the dict must be a tuple: (type, Field_instance)
            pydantic_fields[field_name] = (
                field_type,
                Field(default=default_value, description=description)
            )

        # Use Pydantic's built-in function to create the new model class
        # __base__=cls ensures it inherits from GenericQueryParams
        new_model_class = create_model(
            model_name,
            __base__=cls,
            **pydantic_fields
        )
        
        return new_model_class


# For horizontal_whitespace
HorizontalWhitespaceParams = GenericFunctionParams.create_custom_model(
    "HorizontalWhitespaceParams", {
        'page_number': { 'type': Optional[int], 'default': None, 'description': "Optional page number to search within." },
        'y_tolerance': { 'type': int, 'default': 10, 'description': "Minimum vertical gap to be considered whitespace." }
    }
)

# For group_vertically_touching_bboxes
GroupTouchingBoxesParams = GenericFunctionParams.create_custom_model(
    "GroupTouchingBoxesParams", {
        'y_tolerance': { 'type': float, 'default': 2.0, 'description': "Max vertical distance between boxes to be considered 'touching'." }
    }
)

# For paragraphs -> find_paragraph_blocks
ParagraphsParams = GenericFunctionParams.create_custom_model(
    "ParagraphsParams", {
        'width_threshold': { 'type': float, 'default': 0.5, 'description': "Minimum relative width (0.0-1.0) for a line to be a paragraph seed." },
        'x0_tol': { 'type': float, 'default': 2.0, 'description': "Tolerance for x0 alignment between paragraph lines." },
        'font_size_tol': { 'type': float, 'default': 0.2, 'description': "Tolerance for font size similarity between lines." },
        'y_gap_max': { 'type': float, 'default': 7.0, 'description': "Maximum vertical gap allowed between lines in a paragraph." }
    }
)
