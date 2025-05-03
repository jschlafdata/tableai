url = 'http://embeddings-ollama.gpu.svc.cluster.local:11434/api/generate'
image_path = '/data/client_docs/controlled_products/2024/april/process/abq_2024_04-PAGE_1.png'

table_metadata_prompt = f"""
You are given an image of a PDF page that contains one or more tables. 
Your tasks:
1) Tell me how many tables you see in the image (an integer count).
2) For each table:
   - Provide a table index (starting at 0).
   - Provide the table's label or name, or null if unknown.
   - Provide a list of the table's column headers (if any).
   - Provide the **bottom-right** cell value in this table (i.e., the value in the last row, last column).

Output only a JSON object, with no extra text or explanation. 
Use the following structure exactly:

{{
  "number_of_tables": <integer>,
  "tables": [
    {{
      "table_index": <integer>,
      "table_name": "<string or null>",
      "columns": ["col1", "col2", ...],
      "bottom_right_value": <string>
    }},
    ...
  ]
}}
"""

table_totals_prompt = f"""
You are given an image of a PDF page that contains one or more tables.

Your task is to find all **table rows or cells** that represent **totals or summations** of other rows or columns.

These total indicators may include, but are not limited to:  
"Total", "TOTAL", "Totals", "Grand Total", "Sum", "Subtotal", "Net Total", "Amount Due", "Balance", or other common variations that imply a final or aggregated row/column value.

Identify all such total **values** and return them along with their **row and column indexes** (as they appear in the table structure, starting from 0).

Do NOT include headings, intermediate group totals, or decorative text.

Output only a JSON object, with no extra text or explanation.  
Use the following structure exactly:

{{
  "totals": [
    {{
      "row_index": <integer>,
      "column_index": <integer>, 
      "value": <string>
    }},
    ...
  ]
}}
"""

pdf_classification = """
Which comany issued this credit statement?
"""


def vision_inference(
    prompt, 
    pdf_path='/data/client_docs/controlled_products/2024/april/process/abq_2024_04-PAGE_1.png'
):
    # Define the question and model information as JSON
    data = {
        "model": "llama3.2-vision",  # Example model name
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "images": [
            encode_image_to_base64(pdf_path)
        ],
        "options": 
        {
          "temperature": 0,
          "top_k": 1,
          "top_p": 1
        }
    }

    # Send the POST request with the JSON payload
    response = requests.post(url, json=data)

    # Parse the response (Assuming the response is JSON)
    response_data = response.json()
    return json.loads(response_data['response'])


import requests
import base64
import json

# Function to encode an image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')  # Convert to base64
    return encoded_string

# Define the API endpoint
url = 'http://embeddings-ollama.gpu.svc.cluster.local:11434/api/generate'

prompt = """
Identify any total, totals, subtotal, monthly total or summary rows for each table in the provided image.

Return a strict JSON object in the following format (with no extra text):
{
  "table_index": [
    {
      "row_index": <integer or "unknown">,
      "col_index": <integer or "unknown">,
      "cell_value": "<string>"
    },
    ...
  ]
}
"""

# Define the request payload
data = {
    "model": "llama3.2-vision",  # Example model name
    "prompt": prompt,
    "stream": False,
    # "format": "json",
    "images": [
        encode_image_to_base64('/data/client_docs/controlled_products/2024/april/process/abq_2024_04-PAGE_1.png')
    ]
}

# Send the POST request with the JSON payload
response = requests.post(url, json=data)

# Parse the response (Assuming the response is JSON)
response_data = response.json()


import requests
import base64
import json

# Function to encode an image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')  # Convert to base64
    return encoded_string

# Define the API endpoint
url = 'http://embeddings-ollama.gpu.svc.cluster.local:11434/api/generate'

# Define the question and model information as JSON
data = {
    "model": "llama3.2-vision",  # Example model name
    "prompt": "You will be provided with an image. Your task is to identify every table in the image, extract each table’s content, and return the results in JSON. For each table, include",  # Your question here
    "stream": False,
    # "format": "json",
    "images": [encode_image_to_base64('/data/client_docs/controlled_products/2024/april/process/abq_2024_04-PAGE_1.png')]  # Encode the image to base64
}

# Send the POST request with the JSON payload
response = requests.post(url, json=data)

# Parse the response (Assuming the response is JSON)
response_data = response.json()



def get_nth_table_from_image(image_path, n, nname):
    """
    Send a request to the LLM to identify and return the nth table in the image.
    If there are fewer than n tables in the image, the model should return an empty array or indicate no such table.
    """
    # Define the API endpoint
    url = 'http://embeddings-ollama.gpu.svc.cluster.local:11434/api/generate'
    
    # Create a parameterized prompt
    # The prompt instructs the model to focus on only the nth table.
    prompt = f"""
You will be provided with an image. Your task is to identify the {nname} table in the image, extract the table’s content, 
and return the result in JSON. For that table, include:

1. An index or identifier (e.g., "table_{n}").
2. The bounding box coordinates of the table (x1, y1, x2, y2).
3. The table’s content in a rows-and-columns structure.

Use the following JSON format (no additional text or commentary):

{{
  "tables": [
    {{
      "table_index": {n},
      "bounding_box": [x1, y1, x2, y2],
      "table_data": [
        ["row1_col1", "row1_col2", ...],
        ["row2_col1", "row2_col2", ...],
        ...
      ]
    }}
  ]
}}

If you do not see at least {n} tables, return an empty "tables" array.
"""

    # Define the request payload
    data = {
        "model": "granite3.2-vision",
        "prompt": prompt,
        "stream": False,
        "images": [encode_image_to_base64(image_path)]
    }

    # Send the POST request
    response = requests.post(url, json=data)

    # Parse and return the JSON response
    response_data = response.json()
    return response_data