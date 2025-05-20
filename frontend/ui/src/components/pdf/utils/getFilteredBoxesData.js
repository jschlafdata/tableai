/**
 * Gets formatted bounding box data for a specific page
 * 
 * @param {Object} options - Configuration options
 * @param {Array|Object} options.data - Results data (array for PdfProcessingResults, object for BoundingBoxesPortal)
 * @param {number} options.pageNumber - Current page number (1-based)
 * @param {boolean} options.showResults - Whether to show results
 * @param {Object} [options.toggles={}] - Optional toggles to filter results by label
 * @param {function} [options.formatItem] - Optional function to customize item formatting
 * @param {boolean} [options.includeMetadata=false] - Whether to include metadata items
 * @param {string} [options.labelPrefix=''] - Optional prefix for labels
 * @returns {Object|null} Formatted boxes data grouped by label, or null if no data
 */
function getFilteredBoxesData(options) {
    const {
      data,
      pageNumber,
      showResults,
      toggles = {},
      formatItem = null,
      includeMetadata = false,
      labelPrefix = ''
    } = options;
    
    // Early return conditions
    if (!data || !showResults) return null;
    
    const pageIndex = String(pageNumber - 1);
    const formattedData = {};
    
    try {
      // Handle array of results (PdfProcessingResults style)
      if (Array.isArray(data)) {
        data.forEach(result => {
          const queryLabel = result.query_label;
          
          // Skip if toggle is false for this label
          if (toggles[queryLabel] === false) return;
          
          const pages = result.results?.pages;
          if (!pages) return;
          
          const pageBoxes = pages[pageIndex];
          if (!pageBoxes || !pageBoxes.length) return;
          
          formattedData[queryLabel] = pageBoxes.map(box => {
            const coords = box.bbox || [box.x0, box.y0, box.x1, box.y1];
            const baseItem = { 
              ...box, 
              coords, 
              text: box.value || '', 
              queryLabel 
            };
            
            return formatItem ? formatItem(baseItem) : baseItem;
          });
        });
      } 
      // Handle single result object (BoundingBoxesPortal style)
      else {
        const pages = data.results?.pages;
        if (!pages) return null;
        
        const pageItems = pages[pageIndex];
        if (!pageItems || !pageItems.length) return null;
        
        pageItems.forEach(item => {
          const tableIndex = item.table_index;
          const label = `${labelPrefix}${tableIndex}`;
          
          // Skip if toggle is false for this label
          if (toggles[label] === false) return;
          
          if (!formattedData[label]) {
            formattedData[label] = [];
          }
          
          // Add the main bounding box
          const coords = item.bbox || [item.x0, item.y0, item.x1, item.y1];
          const baseItem = {
            ...item,
            coords,
            text: item.table_title ?? `${labelPrefix}${tableIndex}`,
          };
          
          formattedData[label].push(
            formatItem ? formatItem(baseItem) : baseItem
          );
          
          // Handle metadata items if requested
          if (includeMetadata && item.table_metadata) {
            Object.entries(item.table_metadata).forEach(([metaKey, metaObj]) => {
              if (!metaObj.bbox) return;
              
              const metaItem = {
                ...metaObj,
                coords: metaObj.bbox,
                text: metaKey,
                isMetadata: true,
              };
              
              formattedData[label].push(
                formatItem ? formatItem(metaItem) : metaItem
              );
            });
          }
        });
      }
      
      return Object.keys(formattedData).length > 0 ? formattedData : null;
    } catch (error) {
      console.error('Error formatting boxes data:', error);
      return null;
    }
  }
  
  export default getFilteredBoxesData;