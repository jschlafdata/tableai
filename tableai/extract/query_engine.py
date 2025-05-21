from tableai.extract.index_search import  (
     FitzTextIndex, 
     LineTextIndex, 
     groupby, 
     filterby, 
     chain_transform,
     render_api, 
     search_normalized_text,
     normalize_word,
     normalize_recurring_text, 
     patterns, 
     try_convert_float, 
     identify_currency_symbols,
     try_convert_date,
     find_paragraph_blocks,
     try_convert_percent,
     merge_result_group_bounds,
     expand_bounds,
     PDFTools
 )


class QueryEngine:
    def __init__(self, line_index, 
                 y_tolerance=10, HEADER_BOUND=100, 
                 FOOTER_BOUND=75, MIN_OCCURRENCES=2):
        self.line_index = line_index
        self.y_tolerance = y_tolerance
        self.HEADER_BOUND = HEADER_BOUND
        self.FOOTER_BOUND = FOOTER_BOUND
        self.MIN_OCCURRENCES = MIN_OCCURRENCES

        # Registry of available query types
        self.queries = {
            "Page.Indicators": self.page_indicators,
            "Numbers": self.numbers,
            "Percentages": self.percentages,
            "Dates": self.dates,
            "Toll.Free.#": self.toll_free,
            "Horizontal.Whitespace": self.horizontal_whitespace,
            "Paragraphs": self.paragraphs,
            "Header.Footer.Blocks": self.header_footer_bounds
        }

    def get(self, label, *args, **kwargs):
        if label not in self.queries:
            raise ValueError(f"No such query: {label}")
        return self.queries[label](*args, **kwargs)

    # Query implementations
    def page_indicators(self):
        return self.line_index.query(
            key="text",
            transform=lambda rows: render_api(
                query_label="Page.Indicators",
                description="find common page spans",
                pdf_metadata={},
                include=[]
            )(filterby(
                lambda t: t and t.lower() == "page xx of xx",
                field="normalized_value",
                test=bool
            )(rows))
        )

    def numbers(self):
        return self.line_index.query(
            key="text",
            transform=lambda rows: render_api(
                query_label="Numbers",
                description="Return all blocks that successfully evaluate as Floats or Ints",
                pdf_metadata={}
            )(filterby(try_convert_float, "value", test=lambda x: x is not None)(rows))
        )

    def percentages(self):
        return self.line_index.query(
            key="text",
            transform=lambda rows: render_api(
                query_label="Percentages",
                description="Return all blocks that successfully evaluate as %pct",
                pdf_metadata={}
            )(filterby(try_convert_percent, "value", test=lambda x: x is not None)(rows))
        )

    def dates(self):
        return self.line_index.query(
            key="text",
            transform=lambda rows: render_api(
                query_label="Dates",
                description="Return all date fields.",
                pdf_metadata={}
            )(filterby(try_convert_date, "value", test=bool)(rows))
        )

    def toll_free(self):
        return self.line_index.query(
            key="text",
            transform=lambda rows: render_api(
                query_label="Toll.Free.#",
                description="Return all toll-free patterns.",
                pdf_metadata={}
            )(filterby(lambda t: patterns(t, pattern_name="toll_free"), field="value", test=bool)(rows))
        )

    def horizontal_whitespace(self, y_tolerance=None):
        if y_tolerance:
            self.y_tolerance = y_tolerance
        return self.line_index.query(
            key="full_width_v_whitespace",
            transform=lambda rows: render_api(
                query_label="Horizontal.Whitespace",
                description=f"Find full width spanning whitespace with Y tolerance {self.y_tolerance}",
                pdf_metadata={},
                include=[]
            )([r for r in rows if r["gap"] >= self.y_tolerance])
        )

    def paragraphs(self):
        return self.line_index.query(
            key="text",
            bounds_filter=lambda r: (
                r["x_span"] is not None and
                r["x_span"] > (self.line_index.page_text_bounds[r["page"]]["max_x"] -
                               self.line_index.page_text_bounds[r["page"]]["min_x"]) * 0.5
            ),
            transform=lambda rows: render_api(
                query_label="Paragraphs",
                description="Return large X spanning merged paragraphs",
                pdf_metadata={}
            )(find_paragraph_blocks(self.line_index, rows))
        )

    def header_footer_bounds(self):
        return self.line_index.query(
            key="text",
            bounds_filter=lambda r: (
                r["y0"] < self.HEADER_BOUND or
                r["y1"] > (r["meta"]["height"] - self.FOOTER_BOUND)
            ),
            transform=chain_transform(
                groupby(
                    "region", "page",
                    filterby=lambda group: len(group) >= self.MIN_OCCURRENCES,
                    group_id_field="group_id"
                ),
                merge_result_group_bounds(query_label="Header.Footer.Blocks"),
                expand_bounds(x_mode="full", y_mode="auto"),
                render_api(
                    query_label="Header.Footer.Blocks",
                    description="Return probable header and footer bounds for pages",
                    pdf_metadata={}
                )
            )
        )