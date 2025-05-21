# from typing import List, Dict, Optional
# from pydantic import BaseModel, Field, ValidationError, Extra

# class ColumnHeaderLevels(BaseModel):
#     level0: str
#     # Allow any level1, level2, ... keys, but only with string values
#     class Config:
#         extra = Extra.allow  # Accept dynamic levelN keys

# class TableMetadata(BaseModel):
#     table_index: int
#     title: Optional[str]  # Can be None
#     columns: List[str]
#     column_metadata: Dict[str, ColumnHeaderLevels]

# class TableStructures(BaseModel):
#     number_of_tables: int
#     tables: List[TableMetadata]
