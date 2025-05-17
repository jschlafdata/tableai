TABLE_STRUCTURE_PROMPT_V2 = """
You are given an image of a PDF page that may contain one or more tables.

Definition of a table:
- A table is a rectangular structure with rows and columns, often including one or more header rows at the top.
- Tables are visually distinct when separated by whitespace, horizontal lines, or section titles.
- If two table-like structures share the same layout but are visually separated, treat them as **distinct tables**.
- If a new column layout appears, it indicates the start of a **new table**.

Instructions:
1. Count how many **complete, visually distinct tables** appear in the image and include that count as `"number_of_tables"`.

2. For each table you find:
   - Assign a unique `"table_index"` starting from 0 (increment for each table).
   - If a **title** or heading appears directly above the table (e.g. bold, centered, or all-caps text), include it as `"title"`; otherwise, use `null`.

3. Extract the **leaf-level (lowest-level)** column headers from left to right, and place them in the `"columns"` array.

4. For each lowest-level column, return a `"column_metadata"` entry that shows all **visual header levels** above it:
   - Use `"level0"` for the topmost header group
   - Use `"level1"` for the next lower group, and so on
   - The final level (deepest) should be the actual column name

Hierarchy rules:
- Only include levels that are visually distinct (e.g. stacked headers)
- Do **not repeat** a group header for unrelated columns
- Do **not fabricate hierarchy levels** unless they are visually aligned above the column

---

### ✅ Example (abstract visual layout)

If the table header visually looks like:

    Group A       | Group B
    ------------- | --------------
    Col 1 | Col 2 | Col 3 | Col 4

Then your JSON might include:

{
  "columns": ["Col 1", "Col 2", "Col 3", "Col 4"],
  "column_metadata": {
    "Col 1": { "level0": "Group A", "level1": "Col 1" },
    "Col 2": { "level0": "Group A", "level1": "Col 2" },
    "Col 3": { "level0": "Group B", "level1": "Col 3" },
    "Col 4": { "level0": "Group B", "level1": "Col 4" }
  }
}

If no grouping is visually present (a single header row), then use:

{
  "columns": ["Col A", "Col B"],
  "column_metadata": {
    "Col A": { "level0": "Col A" },
    "Col B": { "level0": "Col B" }
  }
}

---

### ✅ Output Format

Respond with only a **valid JSON object** using the following structure:

{
  "number_of_tables": <integer>,
  "tables": [
    {
      "table_index": <integer>,
      "title": <string or null>,
      "columns": ["col1", "col2", ...],
      "column_metadata": {
        "col1": {
          "level0": "<header text>",
          "level1": "<header text if applicable>",
          ...
        },
        "col2": {
          "level0": "<header text>",
          ...
        }
      }
    },
    ...
  ]
}
"""


TABLE_STRUCTURE_PROMPT = """
You are given an image of a PDF page that may contain one or more tables.

Definition of a table:
- A table is defined as a grid-like structure with data organized in rows and columns, usually with a header row.
- Tables are visually distinct from each other (separated by whitespace, lines, or headings).
- Each table has its own column structure; if a new column layout appears, it is a separate table.
- If two tables share the same column layout but are clearly separated visually or by spacing/lines, they count as two tables.

Important rules:
1. Identify how many complete tables you see in the image (an integer count).
2. Pay attention to complex (“tree-like”) header structures. For instance, there may be multiple levels of headers:
   - Level 0 (top-level header)
   - Level 1 (child of top-level header)
   - Level 2 (child of Level 1), etc.
3. If a table has a title or heading (some text clearly above the table), include it in the field `"title"`; otherwise, use `null`.

For each table, provide the following in a JSON response:
1. The table’s index (zero-based).
2. The table’s title (string or `null`).
3. A flat list of the **lowest-level** column headers in the order they appear. (For example, if the table has a 2-level header with “Location” spanning two sub-headers “City” and “State,” then the “columns” array might be `["City", "State"]`).
4. A `"column_metadata"` object that captures the hierarchy of headers for each lowest-level column. 
   - The keys of `"column_metadata"` should be the exact lowest-level column names.
   - The values should be dictionaries of levels: for example, if “City” is under top-level “Location,” you might have:

     ```json
     {
       "City": {
         "level0": "Location",
         "level1": "City"
       }
     }
     ```

   - If there are three levels, use `"level0"`, `"level1"`, `"level2"`, etc.

Output only a **valid JSON** object and nothing else. Use the following structure exactly:

```json
{
  "number_of_tables": <integer>,
  "tables": [
    {
      "table_index": <integer>,
      "title": <string or null>,
      "columns": ["col1", "col2", ...],
      "column_metadata": {
        "col1": {
          "level0": "<header text>",
          "level1": "<header text if applicable>",
          ...
        },
        "col2": {
          "level0": "<header text>",
          "level1": "<header text if applicable>",
          ...
        }
      }
    },
    ...
  ]
}
"""