TABLE_STRUCTURE_PROMPT_V5 = """
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

4. For each column header in the `"columns"` array, create a matching entry in `"column_metadata"` with:
   - `"level0"` containing the exact column header text (same as in the columns array)
   - `"level1"` containing any parent header text directly above it (if applicable)
   - `"level2"` containing any header above level1 (if applicable)
   - And so on for additional levels

IMPORTANT FORMATTING RULES:
- The output must be a single, valid JSON object - not an array.
- Each value in column_metadata must be a simple text string - never an object or array.
- Ensure every column listed in the "columns" array has a corresponding entry in "column_metadata".
- Level values (level0, level1, etc.) must always be strings, never objects or arrays.

---

### ✅ Example (abstract visual layout, bottom-up counting)

If the table header visually looks like:

    Group A       | Group B
    ------------- | --------------
    Col 1 | Col 2 | Col 3 | Col 4

Then your JSON should include:

{
  "number_of_tables": 1,
  "tables": [
    {
      "table_index": 0,
      "title": null,
      "columns": ["Col 1", "Col 2", "Col 3", "Col 4"],
      "column_metadata": {
        "Col 1": { "level0": "Col 1", "level1": "Group A" },
        "Col 2": { "level0": "Col 2", "level1": "Group A" },
        "Col 3": { "level0": "Col 3", "level1": "Group B" },
        "Col 4": { "level0": "Col 4", "level1": "Group B" }
      }
    }
  ]
}

---

### ✅ Output Format

Return ONLY the following valid JSON object structure with no additional text:

{
  "number_of_tables": <integer>,
  "tables": [
    {
      "table_index": <integer>,
      "title": <string or null>,
      "columns": ["col1", "col2", ...],
      "column_metadata": {
        "col1": { 
          "level0": "<leaf header text (exact column name)>",
          "level1": "<parent header above, if applicable>",
          "level2": "<header above level1, if applicable>"
        },
        "col2": {
          "level0": "<leaf header text (exact column name)>",
          "level1": "<parent header above, if applicable>"
        },
        "col3": {
          "level0": "<leaf header text (exact column name)>",
        },
        ...additional columns...
      }
    },
    ...additional tables...
  ]
}

FINAL VERIFICATION:
1. Ensure column_metadata has an entry for EVERY column in the columns array.
2. Verify that all level values are simple strings (never objects or arrays).
3. The output must be a single JSON object, not an array of objects.
4. Ensure there are no nested JSON structures inside any level values.
"""

TABLE_STRUCTURE_PROMPT_V4 = """
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

4. For each column, return a `"column_metadata"` entry that shows **all visual header levels above it, starting from the leaf header row**:

   - Use `"level0"` for the **leaf header row** (the actual column name).
   - Use `"level1"` for any parent header **directly above** it.
   - Use `"level2"` for any header above that parent, and so on.
   - IMPORTANT: Include ALL parent headers that visually span above each column.

Hierarchy rules:
- **Start from the leaf header (the actual column name) as level0, and work upward.**
- Include ALL header levels that visually span above each column.
- A header spans above a column if it appears to be centered above or aligned with that column.
- Do **not repeat** a group header for unrelated columns.
- Do **not fabricate hierarchy levels** that aren't visually present.

---

### ✅ Example (abstract visual layout, bottom-up counting)

If the table header visually looks like:

    Group A       | Group B
    ------------- | --------------
    Col 1 | Col 2 | Col 3 | Col 4

Then your JSON should include:

{
  "columns": ["Col 1", "Col 2", "Col 3", "Col 4"],
  "column_metadata": {
    "Col 1": { "level0": "Col 1", "level1": "Group A" },
    "Col 2": { "level0": "Col 2", "level1": "Group A" },
    "Col 3": { "level0": "Col 3", "level1": "Group B" },
    "Col 4": { "level0": "Col 4", "level1": "Group B" }
  }
}

---

For a more complex example, if the table header looks like:

    Products                | Financial Summary
    ----------------------- | ---------------------------
    Clothing | Electronics  | Revenue | Cost | Profit
    
Then your JSON should include:

{
  "columns": ["Clothing", "Electronics", "Revenue", "Cost", "Profit"],
  "column_metadata": {
    "Clothing": { "level0": "Clothing", "level1": "Products" },
    "Electronics": { "level0": "Electronics", "level1": "Products" },
    "Revenue": { "level0": "Revenue", "level1": "Financial Summary" },
    "Cost": { "level0": "Cost", "level1": "Financial Summary" },
    "Profit": { "level0": "Profit", "level1": "Financial Summary" }
  }
}

---

Special Cases:

- If there is only one table and no title, use `"title": null`.

Example:
{
  "number_of_tables": 1,
  "tables": [
    {
      "table_index": 0,
      "title": null,
      "columns": ["Header1", "Header2"],
      "column_metadata": {
        "Header1": { "level0": "Header1" },
        "Header2": { "level0": "Header2" }
      }
    }
  ]
}

- If something looks like a table, but it has no recognizable column names or headers, ignore that table
and continue on.

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
          "level0": "<leaf header text (column name)>",
          "level1": "<parent header above, if applicable>",
          "level2": "<header above level1, if applicable>",
          ...
        },
        ...
      }
    },
    ...
  ]
}

If there are no visible tables or column headers: 
Return this structure:

{
  "number_of_tables": 0,
  "tables": []
}

"""

TABLE_STRUCTURE_PROMPT_V3 = """
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

4. For each column, return a `"column_metadata"` entry that shows **all visual header levels above it, starting from the lowest header row**:

   - Use `"level0"` for the **bottom-most (leaf) header row** (the actual column name).
   - Use `"level1"` for the next header row **above** it.
   - Use `"level2"` for the row above that, and so on.
   - Continue incrementing the level for each visually distinct header row upward.
   - The highest `"levelN"` should be the topmost header group (if any).

Hierarchy rules:
- **Start from the bottom header row (closest to the data), and count upward.**
- Only include levels that are visually distinct (e.g. stacked headers).
- Do **not repeat** a group header for unrelated columns.
- Do **not fabricate hierarchy levels** unless they are visually aligned above the column.

---

### ✅ Example (abstract visual layout, counted bottom-up)

If the table header visually looks like:

    Group A       | Group B
    ------------- | --------------
    Col 1 | Col 2 | Col 3 | Col 4

Then your JSON should include:

{
  "columns": ["Col 1", "Col 2", "Col 3", "Col 4"],
  "column_metadata": {
    "Col 1": { "level0": "Col 1", "level1": "Group A" },
    "Col 2": { "level0": "Col 2", "level1": "Group A" },
    "Col 3": { "level0": "Col 3", "level1": "Group B" },
    "Col 4": { "level0": "Col 4", "level1": "Group B" }
  }
}

---

Special Cases:

- If there is only one table and no title, use `"title": null`.

Example:
{
  "number_of_tables": 1,
  "tables": [
    {
      "table_index": 0,
      "title": null,
      "columns": ["Header1", "Header2"],
      "column_metadata": {
        "Header1": { "level0": "Header1" },
        "Header2": { "level0": "Header2" }
      }
    }
  ]
}

- If something looks like a table, but it has no recognizable column names or headers, ignore that table
and continue on.

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
          "level0": "<bottom-most header text (column name)>",
          "level1": "<header above, if applicable>",
          "level2": "<header above level1, if applicable>",
          ...
        },
        ...
      }
    },
    ...
  ]
}
"""

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