// stageConfig.js
// Configuration file for PDF viewer stages

/**
 * Stage configuration that defines:
 * - pathKey: The metadata key for the PDF path
 * - dimensionKeys: The metadata keys for page dimensions
 * - overlayConfigs: Configuration for various overlays
 * - pageBreaksKey: The metadata key for page breaks (if any)
 */
const stageConfig = {
  // Original PDF configuration (stage0)
  original: {
    pathKey: 'metadata.stage0.rel_path', // Updated to reference stage0
    dimensionKeys: {
      // Updated to use the new page-specific dimensions
      width: (metadata, pageIndex = 0) => {
        if (!metadata?.metadata?.stage0?.coords?.[pageIndex]?.[pageIndex]) {
          return 612; // Default fallback width
        }
        return metadata.metadata.stage0.coords[pageIndex][pageIndex].width;
      },
      height: (metadata, pageIndex = 0) => {
        if (!metadata?.metadata?.stage0?.coords?.[pageIndex]?.[pageIndex]) {
          return 792; // Default fallback height
        }
        return metadata.metadata.stage0.coords[pageIndex][pageIndex].height;
      }
    },
    overlayConfigs: [],
    pageBreaksKey: null,
    // Add a pages count reference
    pagesCountKey: 'metadata.stage0.pages'
  },
  
  // Stage 1 configuration
  stage1: {
    pathKey: 'metadata.stage1.rel_path', // Updated path key
    dimensionKeys: {
      width: 'metadata.stage1.page_width',
      height: 'metadata.stage1.page_height'
    },
    overlayConfigs: [
      {
        type: 'recurring_blocks',
        dataKey: 'metadata.stage1.recurring_blocks',
        color: 'rgba(255, 0, 0, 0.2)',
        toggleKey: 'showRecurringBlocks',
        label: 'Recurring Blocks'
      },
      {
        type: 'data_blocks',
        dataKey: 'metadata.stage1.data_blocks',
        color: 'rgba(94, 169, 254, 0.2)',
        toggleKey: 'showDataBlocks',
        label: 'Data Blocks'
      }
    ],
    pageBreaksKey: 'metadata.stage1.page_breaks'
  },
  
  // Stage 2 configuration
  stage2: {
    pathKey: 'metadata.stage2.rel_path', // All stages use the same PDF path
    dimensionKeys: {
      width: 'metadata.stage2.page_width',
      height: 'metadata.stage2.page_height'
    },
    overlayConfigs: [
      {
        type: 'table_bounds',
        dataKey: 'metadata.stage2.tables',
        dataTransform: (tables) => tables
          .filter(table => table.table_bounds)
          .map(table => table.table_bounds),
        color: 'rgba(0, 0, 255, 0.2)',
        toggleKey: 'showTableBounds',
        label: 'Table Bounds'
      },
      {
        type: 'table_names',
        dataKey: 'metadata.stage2.tables',
        dataTransform: (tables) => tables
          .filter(table => table.table_name)
          .map(table => table.table_name),
        color: 'rgba(0, 128, 0, 0.2)',
        toggleKey: 'showTableNames',
        label: 'Table Names'
      },
      {
        type: 'table_headers',
        dataKey: 'metadata.stage2.tables',
        dataTransform: (tables) => tables
          .filter(table => table.columns)
          .map(table => table.columns),
        color: 'rgba(75, 227, 75, 0.28)',
        toggleKey: 'showTableHeaders',
        label: 'Table Headers'
      },
      {
        type: 'table_totals',
        dataKey: 'metadata.stage2.tables',
        dataTransform: (tables) => tables
          .filter(table => table.totals)
          .map(table => table.totals),
        color: 'rgba(153, 115, 235, 0.39)',
        toggleKey: 'showTableTotals',
        label: 'Table Totals'
      },
      {
        type: 'whitespace_blocks',
        dataKey: 'metadata.stage2.whitespace_blocks',
        color: 'rgba(255, 0, 0, 0.2)',
        toggleKey: 'showWhitespaceBlocks',
        label: 'Whitespace Blocks'
      },
      {
        type: 'inverse_tbl_bounds',
        dataKey: 'metadata.stage2.inverse_tbl_bounds',
        color: 'rgba(255, 0, 0, 0.2)',
        toggleKey: 'showInverseTables',
        label: 'Inverse Table Bounds'
      },
      {
        type: 'internal_headers',
        dataKey: 'metadata.stage2.tables',
        dataTransform: (tables, allData) => {
          // Get page width using the getValue function
          const pageWidth = getValue(allData, 'metadata.stage2.page_width', 612);
          
          // Filter tables with internal_headers_y and create boxes
          return tables
            .filter(table => table.internal_headers_y && table.internal_headers_y.length > 0)
            .flatMap(table => {
              // Map each Y coordinate to a rectangle
              return table.internal_headers_y.map(headerY => {
                return [
                  0,               // x0 - start from left edge
                  headerY - 1,     // y0 - slightly above header line
                  pageWidth,       // x1 - use the page width from data
                  headerY + 1      // y1 - slightly below header line
                ];
              });
            });
        },
        color: 'rgba(255, 165, 0, 0.5)',
        toggleKey: 'showInternalHeaders',
        label: 'Internal Headers'
      },
      {
        type: 'alignment_lines',
        dataKey: 'metadata.stage2.tables',
        dataTransform: (tables) => {
          return tables
            .filter(table => table.table_row_meta?.alignment_line_data)
            .flatMap(table => {
              return table.table_row_meta.alignment_line_data.map(line => {
                // Convert to [x0, y0, x1, y1] format expected by viewer
                return [line.x0, line.y0, line.x1, line.y1];
              });
            });
        },
        color: 'rgba(0, 128, 255, 0.4)',
        toggleKey: 'showAlignmentLines',
        label: 'Column Alignment Lines'
      }, 
      {
        type: 'multi_tables',
        dataKey: 'metadata.stage2.merged_tables',
        dataTransform: (mergedTablesObj, metadata) => {
          if (!mergedTablesObj || typeof mergedTablesObj !== 'object') {
            console.warn('No merged_tables data found');
            return [];
          }
      
          const allSubTables = Object.entries(mergedTablesObj).flatMap(([key, table]) => {
            if (table && Array.isArray(table.merged_sub_tables)) {
              return table.merged_sub_tables.map(bounds => {
                const [x0, y0, x1, y1] = bounds;
                return [x0, y0, x1, y1];  // <- important!
              });
            }
            return [];
          });
      
          console.log(`Found ${allSubTables.length} merged_sub_tables`);
          return allSubTables;
        },
        color: 'rgba(255, 165, 0, 0.5)',
        toggleKey: 'showMultiTables',
        label: 'Multi Tables'
      }
    ],
    pageBreaksKey: null
  },
  
  // Stage 3 configuration
  stage3: {
    pathKey: 'metadata.stage1.rel_path', // All stages use the same PDF path
    dimensionKeys: {
      width: 'metadata.stage3.metadata.page_width',
      height: 'metadata.stage3.metadata.page_height'
    },
    overlayConfigs: [
      {
        type: 'table_bounds',
        dataKey: 'metadata.stage3.metadata.tables',
        dataTransform: (tables) => tables
          .filter(table => table.snippet_coordinates_on_new_pdf)
          .map(table => table.snippet_coordinates_on_new_pdf),
        color: 'rgba(0, 0, 255, 0.2)',
        toggleKey: 'showTableBounds',
        label: 'Table Bounds'
      },
      {
        type: 'table_names',
        dataKey: 'metadata.stage3.metadata.tables',
        dataTransform: (tables) =>
          tables
            .filter(table => table?.sub_rects?.table_name)
            .map(table => table.sub_rects.table_name),
        color: 'rgba(0, 128, 0, 0.2)',
        toggleKey: 'showTableNames',
        label: 'Table Names'
      },
      {
        type: 'table_continued',
        dataKey: 'metadata.stage3.metadata.tables',
        dataTransform: (tables) =>
          tables
            .filter(table => table?.sub_rects?.table_bounds_continued)
            .map(table => table.sub_rects.table_bounds_continued),
        color: 'rgba(0, 128, 0, 0.2)',
        toggleKey: 'showTableContinued',
        label: 'Table Continued'
      }
    ],
    pageBreaksKey: 'metadata.stage3.metadata.page_breaks'
  }

  // Add more stages as needed following the same pattern
  // stage4: { ... }
};

/**
 * Helper function to get value from a nested object using dot notation
 * For example: getValue(obj, 'metadata.stage1.pdf_path') will return obj.metadata.stage1.pdf_path
 * Also supports function-based values from the config
 */
export const getValue = (obj, path, defaultValue = null) => {
  if (!obj || !path) return defaultValue;

  // Check if path is a function (for dimension functions)
  if (typeof path === 'function') {
    try {
      return path(obj) ?? defaultValue;
    } catch (e) {
      console.error('Error getting value using function:', e);
      return defaultValue;
    }
  }

  const keys = path.split('.');
  let result = obj;

  for (const key of keys) {
    if (result === null || result === undefined || !Object.prototype.hasOwnProperty.call(result, key)) {
      return defaultValue;
    }
    result = result[key];
  }

  return result ?? defaultValue;
};

/**
 * Function to get the available toggle controls for a stage
 */
export const getStageToggles = (stageName) => {
  const stage = stageConfig[stageName];
  if (!stage) return [];
  
  // Get only the toggles that are relevant to this stage
  return stage.overlayConfigs
    .filter(config => config.toggleKey)
    .map(config => ({
      key: config.toggleKey,
      label: config.label
    }));
};

/**
 * Get color map for a stage
 */
export const getColorMap = (stageName) => {
  const stage = stageConfig[stageName];
  if (!stage || !stage.overlayConfigs) return {};
  
  const colorMap = {};
  
  // Build a map of types to colors
  stage.overlayConfigs.forEach(config => {
    colorMap[config.type] = config.color;
  });
  
  return colorMap;
};

/**
 * Process bounding boxes with the updated metadata structure
 */
export const processBoundingBoxes = (metadata, stageName, toggles) => {
  if (!metadata || !stageName) return null;
  
  const stage = stageConfig[stageName];
  if (!stage || !stage.overlayConfigs) return null;
  
  console.log('Processing boxes for stage:', stageName);
  console.log('Toggle states:', toggles);
  console.log('Overlay configs:', stage.overlayConfigs);
  
  // Create result object with structure expected by BoundingBoxes
  const result = {};
  
  for (const config of stage.overlayConfigs) {
    // Skip this overlay if its toggle is disabled
    if (config.toggleKey && toggles && toggles[config.toggleKey] === false) {
      console.log(`Skipping ${config.type} because toggle ${config.toggleKey} is disabled`);
      continue;
    }
    
    // Get the data from metadata using the dataKey
    let data = getValue(metadata, config.dataKey, []);
    if (!data || (Array.isArray(data) && data.length === 0)) {
      console.log(`No data found for ${config.type} using key ${config.dataKey}`);
      continue;
    }
    
    console.log(`Found data for ${config.type}:`, data);
    
    // Apply the data transform if provided
    if (config.dataTransform) {
      data = config.dataTransform(data, metadata);
      console.log(`After transform:`, data);
    }
    
    // Ensure data is an array
    if (!Array.isArray(data)) data = [data];
    if (data.length === 0) continue;
    
    // Add to result with the structure expected by BoundingBoxes
    result[config.type] = data;
    console.log(`Added ${data.length} boxes for ${config.type}`);
  }
  
  console.log('Final result:', result);
  return Object.keys(result).length > 0 ? result : null;
};

/**
 * Get all available stage names
 */
export const getStageNames = () => Object.keys(stageConfig);

export default stageConfig;