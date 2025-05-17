export const TABLEAI_METADATA_SCHEMA = {
    files: [
      {
        uuid: 'string',
        source: 'string',
        source_type: 'string',
        file_name: 'string',
        dropbox_safe_id: 'string',
        local_path: 'string',
        input_dir: 'string',
        output_dir: 'string',
        classification: 'string',
        fastapi_url: 'string',
  
        directories: 'array<string>',
        path_categories: {
          '*': 'string' // dynamic keys like client, year, etc.
        },
  
        completed_stages: 'array<number>',
        stage_paths: {
          '*': 'string'
        },
  
        stage0: 'object',
        stage1: 'object',
        stage2: 'object',
        stage3: 'object',
  
        pdf_metadata: {
          pages: 'number',
          meta_tag: 'string',
          recovered: 'boolean',
          recovery_path: 'string',
          coords: 'array<object>'
        }
      }
    ],
  
    filters: {
      subDirectory: 'string',
      searchQuery: 'string',
      fileIdQuery: 'string',
      selectedFileIndex: 'number',
      selectedStage: 'string', // e.g., 'stage0', 'stage1'
      categoryFilters: {
        '*': 'string' // dynamic keys from path_categories
      },
      classificationFilter: 'string'
    },
  
    uiOptions: {
      availableClassifications: 'array<string>',
      showClassificationFilter: 'boolean'
    },
  
    metadataOptions: {
      subDirectories: 'array<string>',
      categoryKeys: 'array<string>'
    }
  };
  