
// src/services/pdfProcessingService.js
/**
 * Service for PDF processing API calls
 */

/**
 * Process a PDF file by calling the API
 * @param {string} fileId - The ID of the file to process
 * @param {number} stage - The processing stage
 * @returns {Promise} - Promise resolving to processing results
 * @throws {Error} - If the API call fails
 */
export const processPdfDocument = async (fileId, stage) => {
    if (!fileId) {
      throw new Error("No file selected");
    }
  
    const response = await fetch(`http://localhost:8000/tableai/extract/doc_query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ file_id: fileId, stage: stage }),
    });
  
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
  
    return response.json();
  };
  
  /**
   * Extracts PDF metadata from processing results
   * @param {Array} results - Processing results from the API
   * @param {number} pageIndex - Page index (0-based)
   * @returns {Object} - PDF dimensions
   */
  export const extractPdfMetadata = (results, pageIndex) => {
    if (results && results.length > 0 && results[0].pdf_metadata) {
      const pageIndexStr = String(pageIndex);
      const pdfMetadata = results[0].pdf_metadata;
      
      if (pdfMetadata[pageIndexStr]) {
        return pdfMetadata[pageIndexStr];
      }
    }
    
    return { width: 612, height: 792 }; // Default PDF dimensions
  };