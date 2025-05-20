/**
 * transformCoordWithContainer:
 * A function to compute bounding box coordinates based on the rendered PDF container.
 */
export const transformCoordWithContainer = (
    coords,
    color,
    pageDimensions,
    tableHeaders,
    currentPage
  ) => {
    // For example, get the PDF’s actual dimensions:
    const pdfDimensions = getPdfDimensions(tableHeaders, currentPage) || {};
  
    // Fallback to provided pageDimensions if needed
    const renderedWidth = pageDimensions.width;
    const renderedHeight = pageDimensions.height;
  
    const pdfWidth = pdfDimensions.width || 612.0;
    const pdfHeight = pdfDimensions.height || 792.0;
  
    let [x0, y0, x1, y1] = coords.map(Number);
    if (x1 < x0) [x0, x1] = [x1, x0];
    if (y1 < y0) [y0, y1] = [y1, y0];
  
    const widthRatio = renderedWidth / pdfWidth;
    const heightRatio = renderedHeight / pdfHeight;
  
    const x = x0 * widthRatio;
    const y = y0 * heightRatio;
    const width = (x1 - x0) * widthRatio;
    const height = (y1 - y0) * heightRatio;
  
    return {
      position: 'absolute',
      left: `${x}px`,
      top: `${y}px`,
      width: `${width}px`,
      height: `${height}px`,
      border: `2px solid ${color}`,
      backgroundColor: color.replace('1)', '0.2)').replace(')', ', 0.2)'),
      zIndex: 2001,
      pointerEvents: 'auto',
      cursor: 'pointer',
      opacity: 1
    };
  };
  
  function getPdfDimensions(tableHeaders, currentPage) {
    // Try from pdf_metadata
    if (tableHeaders?.pdf_metadata && tableHeaders.pdf_metadata[currentPage - 1]) {
      return {
        width: tableHeaders.pdf_metadata[currentPage - 1].width,
        height: tableHeaders.pdf_metadata[currentPage - 1].height
      };
    }
    // Or from the first page item meta
    if (tableHeaders?.results?.pages) {
      const pageIndex = String(currentPage - 1);
      const pageItems = tableHeaders.results.pages[pageIndex];
      if (pageItems && pageItems.length > 0 && pageItems[0].meta) {
        return {
          width: pageItems[0].meta.width,
          height: pageItems[0].meta.height
        };
      }
    }
    return null;
  }