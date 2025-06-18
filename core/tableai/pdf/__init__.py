# --- CHANGE: Import the core types and the accessors ---
from tableai.pdf.generic_results import ResultSet, ChainResult, GroupbyQueryResult, DefaultQueryResult
from tableai.pdf.generic_assesors import GroupAccessor, DefaultAccessor

# =============================================================
# 3. ENRICH THE ResultSet CLASS WITH ACCESSOR PROPERTIES
# =============================================================
# By adding these properties here, we break the circular import.
# `generic_types.py` doesn't know about accessors, and `generic_assesors.py`
# doesn't know about this file.

@property
def group(self: ResultSet) -> GroupAccessor:
    """
    Accessor for operations on a ResultSet of GroupbyQueryResult objects.
    """
    if not self.data or not isinstance(self.data[0], GroupbyQueryResult):
        raise AttributeError("'.group' accessor is only available for ResultSets of GroupbyQueryResult.")
    return GroupAccessor(self.data)

@property
def default(self: ResultSet) -> DefaultAccessor:
    """
    Accessor for operations on a ResultSet of DefaultQueryResult objects.
    """
    if not self.data or not isinstance(self.data[0], DefaultQueryResult):
        raise AttributeError("'.default' accessor is only available for ResultSets of DefaultQueryResult.")
    return DefaultAccessor(self.data)

# --- Monkey-patch the properties onto the imported ResultSet class ---
ResultSet.group = group
ResultSet.default = default