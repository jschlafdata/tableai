from typing import Optional
from collections import defaultdict
from tableai.pdf.query_funcs import (
    try_convert_float, 
    try_convert_percent, 
    find_paragraph_blocks,
    try_convert_date,
    patterns, 
    expand_bounds
)

from tableai.pdf.query import (
    groupby
)

class QueryEngine:
    def __init__(self, line_index):
        self.line_index = line_index
        self.y_tolerance = 10
        self.HEADER_BOUND = 100
        self.FOOTER_BOUND = 75
        self.MIN_OCCURRENCES = 2

        self.queries = {
            "Page.Indicators":       self.page_indicators,
            "Numbers":               self.numbers,
            "Percentages":           self.percentages,
            "Dates":                 self.dates,
            "Toll.Free.#":           self.toll_free,
            "Horizontal.Whitespace": self.horizontal_whitespace,
            "Paragraphs":            self.paragraphs,
            "Header.Footer.Blocks":  self.header_footer_bounds
        }

    def get(self, label, *args, **kwargs):
        if label not in self.queries:
            raise ValueError(f"No such query: {label}")
        return self.queries[label](*args, **kwargs)

    # def page_indicators(self, **kwargs):
    #     """Find common page spans like 'Page X of Y'."""
    #     def transform(rows):
    #         return filterby(
    #             lambda t: t and t.lower().startswith("page"),
    #             field="normalized_value",
    #             test=bool
    #         )(rows)

    #     return self.line_index.query(key="text", transform=transform)

    # def numbers(self, **kwargs):
    #     """Return all blocks that parse as floats or ints."""
    #     def transform(rows):
    #         return filterby(try_convert_float, "value", test=lambda x: x is not None)(rows)

    #     return self.line_index.query(key="text", transform=transform)

    # def percentages(self, **kwargs):
    #     """Return all blocks that parse as percentages."""
    #     def transform(rows):
    #         return filterby(try_convert_percent, "value", test=lambda x: x is not None)(rows)

    #     return self.line_index.query(key="text", transform=transform)

    # def dates(self, **kwargs):
    #     """Return all date fields."""
    #     def transform(rows):
    #         return filterby(try_convert_date, "value", test=bool)(rows)

    #     return self.line_index.query(key="text", transform=transform)

    # def toll_free(self, **kwargs):
    #     """Return all toll-free phone number patterns."""
    #     def transform(rows):
    #         return filterby(lambda t: patterns(t, pattern_name="toll_free"), field="value", test=bool)(rows)

    #     return self.line_index.query(key="text", transform=transform)

    # def horizontal_whitespace(self, y_tolerance: Optional[int] = None, **kwargs):
    #     """Find full-width whitespace blocks above a given gap threshold."""
    #     y_tol = y_tolerance or kwargs.get("y_tolerance", self.y_tolerance)

    #     def transform(rows):
    #         return [r for r in rows if r["gap"] >= y_tol]

    #     return self.line_index.query(key="full_width_v_whitespace", transform=transform)

    # def paragraphs(self, **kwargs):
    #     """Return large merged paragraph blocks spanning most of the page width."""
    #     def transform(rows):
    #         return find_paragraph_blocks(self.line_index, rows)

    #     return self.line_index.query(
    #         key="text",
    #         bounds_filter=lambda r: (
    #             r["x_span"] is not None and
    #             r["x_span"] > (self.line_index.page_text_bounds[r["page"]]["max_x"] -
    #                            self.line_index.page_text_bounds[r["page"]]["min_x"]) * 0.5
    #         ),
    #         transform=transform
    #     )

    # def header_footer_bounds(
    #     self,
    #     HEADER_BOUND: Optional[int] = None,
    #     FOOTER_BOUND: Optional[int] = None,
    #     MIN_OCCURRENCES: Optional[int] = None,
    #     **kwargs
    # ):
    #     """Group whitespace blocks above/below page bounds into header/footer candidates."""
    #     header = HEADER_BOUND if HEADER_BOUND is not None else kwargs.get("HEADER_BOUND", self.HEADER_BOUND)
    #     footer = FOOTER_BOUND if FOOTER_BOUND is not None else kwargs.get("FOOTER_BOUND", self.FOOTER_BOUND)
    #     min_occ = MIN_OCCURRENCES if MIN_OCCURRENCES is not None else kwargs.get("MIN_OCCURRENCES", self.MIN_OCCURRENCES)

    #     # core transforms
    #     transforms = [
    #         groupby(
    #             "region", "page",
    #             filterby=lambda grp: len(grp) >= min_occ,
    #             group_id_field="group_id"
    #         ),
    #         merge_result_group_bounds(query_label="Header.Footer.Blocks"),
    #         expand_bounds(x_mode="full", y_mode="auto")
    #     ]

    #     combined = chain_transform(*transforms)

    #     return self.line_index.query(
    #         key="text",
    #         bounds_filter=lambda r: (
    #             r["y0"] < header or
    #             r["y1"] > (r["meta"]["height"] - footer)
    #         ),
    #         transform=combined
    #     )