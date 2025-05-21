/**
 * Retrieves PDF dimensions for a specific page from processing results
 * @param {Object} results - The PDF processing results object
 * @param {number} pageNumber - The page number (1-based index)
 * @param {Object} [fallbackDimensions=null] - Optional fallback dimensions if none found
 * @returns {Object|null} - Object with width and height properties, or null if dimensions not found
 */
function getPdfDimensions(results, pageNumber, fallbackDimensions = null) {
    if (!results || pageNumber < 1) {
      return fallbackDimensions;
    }
  
    const pageIndex = pageNumber - 1;
    
    // Method 1: Check for top-level pdf_metadata
    if (results.pdf_metadata && results.pdf_metadata[pageIndex]) {
      return {
        width: results.pdf_metadata[pageIndex].width,
        height: results.pdf_metadata[pageIndex].height,
      };
    }
    
    // Method 2: Check in results.pages structure
    const pages = results.results?.pages;
    if (pages) {
      const pageItems = pages[String(pageIndex)] || pages[pageIndex];
      if (pageItems && pageItems.length > 0) {
        for (const item of pageItems) {
          if (item.meta && item.meta.width && item.meta.height) {
            return {
              width: item.meta.width,
              height: item.meta.height,
            };
          }
        }
      }
    }
    
    // Method 3: Handle array of results (from first snippet)
    if (Array.isArray(results)) {
      for (const result of results) {
        const dimensions = getPdfDimensions(result, pageNumber);
        if (dimensions) {
          return dimensions;
        }
      }
    }
    
    // No dimensions found
    return fallbackDimensions;
  }
  
  export default getPdfDimensions;