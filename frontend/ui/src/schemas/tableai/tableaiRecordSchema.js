export const TABLEAI_RECORD_SCHEMA = {
    dropbox_sync_records: [
      {
        dropbox_safe_id: 'string',
        dropbox_id: 'string',
        file_name: 'string',
        name: 'string',
        path_display: 'string',
        path_lower: 'string',
        path_categories: {
          '*': 'string' // dynamic keys like client, year, month
        },
        directories: 'array<string>',
        type: 'string',
        client_modified: 'string',       // ISO timestamp
        server_modified: 'string',       // ISO timestamp
        content_hash: 'string',
        is_downloadable: 'boolean',
        size: 'number',
        rev: 'string'
      }
    ],
  
    extraction_metadata: {
      '*': {
        stage0: 'object',
        stage1: 'object',
        stage2: 'object',
        stage3: 'object',
        fastapi_url: 'string',
        pdf_metadata: {
          pages: 'number',
          meta_tag: 'string',
          recovered: 'boolean',
          recovery_path: 'string',
          coords: 'array<object>'
        }
      }
    }
  };