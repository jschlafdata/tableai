// components/pdf/services/pdfApi.js

const API_BASE = "http://localhost:8000";

const handleResponse = async (response) => {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP error! Status: ${response.status}`);
  }
  return response.json();
};

// --- PDF Processing Requests ---
export const fetchMetadata = async () =>
  fetch(`${API_BASE}/query/records?merge=true`).then(handleResponse);

export const processPdf = async ({ fileId }) =>
  fetch(`${API_BASE}/tableai/extract/doc_query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: fileId }),
  }).then(handleResponse);

  export const runVisionInference = async ({
    fileId,
    stage,
    classificationLabel,
    visionOptions // <--- add this
  }) =>
    fetch(`${API_BASE}/tableai/vision/find_table_columns`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        file_id: fileId,
        stage,
        classification_label: classificationLabel,
        ...visionOptions // <-- merge the options in
      }),
    }).then(handleResponse);

// Example: TableHeaderBounds endpoint
export const fetchTableHeaders = async ({ fileId, stage, classificationLabel }) =>
  fetch(`${API_BASE}/tableai/extract/vision/table_headers`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ file_id: fileId, stage: stage, classification_label: classificationLabel }),
  }).then(handleResponse);

// Add more endpoints as needed
