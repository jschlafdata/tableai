// stageConfig.js
export const getStageNames = () => Object.keys(stageConfig)

function generateHSLColors(count, opacity = 0.3) {
  const colors = [];
  const jump = 137; // use a prime number for max dispersion (like golden angle)
  for (let i = 0; i < count; i++) {
    const hue = (i * jump) % 360;
    colors.push(`hsla(${hue}, 80%, 55%, ${opacity})`);
  }
  return colors;
}

function formatColumnDataForDisplay(pagesData) {
  if (!pagesData || typeof pagesData !== 'object') {
    console.warn('No stage3 pages data found');
    return [];
  }

  // Step 1: Flatten all columns from all pages into one array
  const allCols = Object.entries(pagesData).flatMap(([_, pageData]) =>
    Array.isArray(pageData.col_bbox) ? pageData.col_bbox : []
  );

  // Step 2: Generate a color for each column index
  const colors = generateHSLColors(allCols.length);

  // Step 3: Re-map with assigned colors
  let globalIndex = 0;

  return Object.entries(pagesData).flatMap(([pageIndex, pageData]) => {
    if (!Array.isArray(pageData.col_bbox)) return [];

    return pageData.col_bbox.map((colObj) => {
      const color = colors[globalIndex++];
      return {
        coords: colObj.coords || colObj.bbox,
        pageIndex: parseInt(pageIndex, 10),
        columnName: colObj.columnName,
        uniqueKey: colObj.columnName,
        columnIndex: globalIndex - 1,
        color
      };
    });
  });
}

function transformSpanningText(pagesObj, metadata) {
  if (!pagesObj || typeof pagesObj !== 'object') {
    console.warn('No spanning_text data found');
    return [];
  }

  return Object.entries(pagesObj).flatMap(([pageIndex, pageData]) => {
    // Check if spanning_text exists
    if (!pageData.spanning_text || !Array.isArray(pageData.spanning_text)) {
      return [];
    }
    
    // Transform each spanning text item
    return pageData.spanning_text.map((item, idx) => ({
      coords: item.bbox,
      pageIndex: parseInt(pageIndex, 10),
      text: item.text,
      coverage: item.coverage,
      itemIndex: idx,
      type: 'spanning_text'
    }));
  });
}

const stageConfig = {
  stage0: {
    pathKey: 'stage_paths.0.abs_path',
    dimensionKeys: {
      width: 'metadata.stage0.page_width',
      height: 'metadata.stage0.page_height'
    },
    overlayConfigs: [
    ],
    pageBreaksKey: ''
  }, 
  stage1: {
    pathKey: 'stage_paths.1.abs_path',
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
  stage3: {
    pathKey: 'metadata.stage3.rel_path', // All stages use the same PDF path
    dimensionKeys: {
      width: '',
      height: ''
    },
    overlayConfigs: [
      {
        type: 'stage3_new_bboxes',
        dataKey: 'metadata.stage3.pages',
        dataTransform: (pagesObj, metadata) => {
          if (!pagesObj || typeof pagesObj !== 'object') {
            console.warn('No stage3 pages data found');
            return [];
          }
      
          return Object.entries(pagesObj).flatMap(([pageIndex, pageData]) => {
            return (pageData.placed_items || []).map((item, idx) => ({
              coords: item.new_bbox,
              pageIndex: parseInt(pageIndex),  // important for dynamic scaling
            }));
          });
        },
        color: 'rgba(30, 144, 255, 0.4)',
        toggleKey: 'showMultiTables',
        label: 'Full Tables'
      },
      {
        type: 'stage3_clean_bboxes',
        dataKey: 'metadata.stage3.clean_stage3',
        dataTransform: (pagesObj, metadata) => {
          if (!pagesObj || typeof pagesObj !== 'object') {
            console.warn('No clean_stage3 data found');
            return [];
          }
      
          return Object.entries(pagesObj).map(([pageIndex, pageData]) => {
            const bbox = pageData.new_tables;
            return {
              coords: bbox,
              pageIndex: parseInt(pageIndex),
              scaled_totY0: pageData.scaled_totY0,  // optional if you want it later
              table_key: pageData.table_key         // optional
            };
          });
        },
        color: 'rgba(175, 96, 239, 0.4)',
        toggleKey: 'showTableBounds',
        label: 'Clean Tables'
      },
      {
        type: 'stage3_column_names',
        dataKey: 'metadata.stage3.clean_stage3',
        dataTransform: (pagesObj, metadata) => formatColumnDataForDisplay(pagesObj),
        toggleKey: 'showTableHeaders',
        label: 'Table Columns',
        colorBy: 'color'  // Support for coloring columns
      },
      {
        type: 'stage3_spanning_text',
        dataKey: 'metadata.stage3.clean_stage3',
        dataTransform: transformSpanningText,
        color: 'rgba(255, 140, 0, 0.4)',  // Orange color to distinguish from tables
        toggleKey: 'showSpanningText',
        label: 'Spanning Text'
      },
      {
        type: 'stage3_table_cells',
        dataKey: 'metadata.stage3.clean_stage3',
        dataTransform: (pagesObj, metadata) => {
          if (!pagesObj || typeof pagesObj !== 'object') {
            console.warn('No stage3 pages data found');
            return [];
          }
      
          return Object.entries(pagesObj).flatMap(([pageIndex, pageData]) => {
            const tableCells = pageData.table_cells;
            if (!tableCells || typeof tableCells !== 'object') return [];
      
            return Object.entries(tableCells).flatMap(([columnName, cells]) => {
              return cells.map(([text, coords], rowIndex) => ({
                coords,
                text,
                pageIndex: parseInt(pageIndex, 10),
                columnName,
                rowIndex,
              }));
            });
          });
        },
        color: 'rgba(0, 120, 255, 0.3)',  // translucent blue
        toggleKey: 'showTableCells',
        label: 'Table Cells'
      },
      {
        type: 'stage3_row_bounds',
        dataKey: 'metadata.stage3.clean_stage3',
        dataTransform: (pagesObj, metadata) => {
          if (!pagesObj || typeof pagesObj !== 'object') {
            console.warn('No stage3 pages data found');
            return [];
          }
      
          return Object.entries(pagesObj).flatMap(([pageIndex, pageData]) => {
            const rowBounds = pageData.row_bounds;
            if (!Array.isArray(rowBounds)) return [];
      
            return rowBounds.map((row, idx) => ({
              coords: row,
              pageIndex: parseInt(pageIndex, 10),
              rowIndex: idx,
              color: idx % 2 === 0
                ? 'rgba(84, 236, 64, 0.3)'  // light orange
                : 'rgba(99, 121, 246, 0.15)', // slightly different shade
            }));
          });
        },
        toggleKey: 'showRowBounds',
        label: 'Row Bounds'
      },
      {
        type: 'stage3_cell_bounds',
        dataKey: 'metadata.stage3.clean_stage3',
        dataTransform: (pagesObj) => {
          if (!pagesObj || typeof pagesObj !== 'object') {
            console.warn('No stage3 pages data found');
            return [];
          }
      
          return Object.entries(pagesObj).flatMap(([pageIndex, pageData]) => {
            const cells = pageData.value_cells;
            if (!Array.isArray(cells)) {
              console.warn(`No table_cells array found on page ${pageIndex}`);
              return [];
            }
      
            return cells.map((cell) => {
              const [x0, y0, x1, y1] = cell.bbox || [];
              const colLabel = cell.columnName?.split('_')[0] ?? 'Col';
      
              return {
                coords: [x0, y0, x1, y1],
                pageIndex: parseInt(pageIndex, 10),
                rowIndex: cell.row_index,
                colIndex: cell.col_index,
                borderColor: 'black',
                borderWidth: 1,
                color: cell.col_index % 2 === 0
                  ? 'rgba(0, 128, 255, 0.41)'
                  : 'rgba(0, 255, 128, 0.35)',
                label: `${colLabel} (${cell.row_index}, ${cell.col_index})`
              };
            });
          });
        },
        toggleKey: 'showCellBounds',
        label: 'Cell Bounds'
      },
      {
        type: 'stage3_totals_box',
        dataKey: 'metadata.stage3.clean_stage3',
        dataTransform: (pagesObj, metadata) => {
          if (!pagesObj || typeof pagesObj !== 'object') {
            console.warn('No stage3 pages data found');
            return [];
          }
      
          return Object.entries(pagesObj).flatMap(([pageIndex, pageData]) => {
            const bbox = pageData.totals_bbox;
            if (!Array.isArray(bbox) || bbox.length !== 4) return [];
      
            return [{
              coords: bbox,
              pageIndex: parseInt(pageIndex, 10),
              label: `Totals Box`,
              color: 'rgba(255, 0, 0, 0.3)',
              borderColor: 'black',
              borderWidth: 1
            }];
          });
        },
        toggleKey: 'showTotalsBox',
        label: 'Totals Box'
      },
      {
        type: 'stage3_totals_cells',
        dataKey: 'metadata.stage3.clean_stage3',
        dataTransform: (pagesObj) => {
          if (!pagesObj || typeof pagesObj !== 'object') {
            console.warn('No stage3 pages data found');
            return [];
          }
      
          return Object.entries(pagesObj).flatMap(([pageIndex, pageData]) => {
            const cells = pageData.totals_cells;
            if (!Array.isArray(cells)) {
              console.warn(`No totals_cells array found on page ${pageIndex}`);
              return [];
            }
      
            return cells.map((cell) => {
              const [x0, y0, x1, y1] = cell.bbox || [];
              const colLabel = cell.columnName?.split('_')[0] ?? 'Col';
      
              return {
                coords: [x0, y0, x1, y1],
                pageIndex: parseInt(pageIndex, 10),
                colIndex: cell.col_index,
                columnName: cell.columnName,
                color: cell.col_index % 2 === 0
                  ? 'rgba(255, 215, 0, 0.25)' // light gold
                  : 'rgba(255, 165, 0, 0.25)', // soft orange
                borderColor: 'black',
                borderWidth: 1,
                label: `Total: ${colLabel} (${cell.col_index})`
              };
            });
          });
        },
        toggleKey: 'showTotalsCells',
        label: 'Totals Cells'
      }
    ],
    pageBreaksKey: ''
  },
}

export const getValue = (obj, path, defaultValue = null) => {
  if (!obj || !path) return defaultValue
  return path.split('.').reduce((acc, key) =>
    acc && Object.prototype.hasOwnProperty.call(acc, key) ? acc[key] : defaultValue
  , obj)
}

export const getStageToggles = (stageName) => {
  const stage = stageConfig[stageName]
  if (!stage) return []
  return stage.overlayConfigs
    .filter(config => config.toggleKey)
    .map(config => ({ key: config.toggleKey, label: config.label }))
}

export const getColorMap = (stageName) => {
  const stage = stageConfig[stageName]
  if (!stage || !stage.overlayConfigs) return {}
  return Object.fromEntries(
    stage.overlayConfigs.map(config => [config.type, config.color])
  )
}

export const processBoundingBoxes = (metadata, stageName, toggles) => {
  if (!metadata || !stageName) return null
  const stage = stageConfig[stageName]
  if (!stage || !stage.overlayConfigs) return null

  const result = {}

  for (const config of stage.overlayConfigs) {
    if (config.toggleKey && toggles?.[config.toggleKey] === false) continue
    let data = getValue(metadata, config.dataKey, [])
    if (!data || (Array.isArray(data) && data.length === 0)) continue
    if (config.dataTransform) data = config.dataTransform(data, metadata)
    if (!Array.isArray(data)) data = [data]
    if (data.length > 0) result[config.type] = data
  }

  return Object.keys(result).length > 0 ? result : null
}

export default stageConfig
