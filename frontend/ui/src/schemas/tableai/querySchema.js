// pdfQuerySchema.js
export const PDF_QUERY_SCHEMA = {
    query_label: 'string',
    description: 'string',
    pdf_metadata: {
      uuid: 'string',
      source: 'string',
      file_name: 'string',
      local_path: 'string',
      fastapi_url: 'string',
      stored_stages: 'array',
      completed_stages: 'array',
      pdf_metadata: {
        pages: 'number',
        meta_tag: 'string',
        recovered: 'boolean',
        recovery_path: 'string',
        coords: 'array'
      }
    },
    results: {
      pages: 'object' // key: page number, value: array of result dicts
    }
  };