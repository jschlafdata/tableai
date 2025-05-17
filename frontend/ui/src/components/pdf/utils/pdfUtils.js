// Helper functions for PDF utils
// Save this to src/components/pdf/utils/pdfUtils.js

/**
 * Transforms coordinates from PDF coordinate system to screen coordinates
 * @param {Array} coords Array of coordinates [x0, y0, x1, y1]
 * @param {string} color CSS color string
 * @param {number} pageIndex Page index (0-based)
 * @param {Object} context Additional context data
 * @returns {Object} CSS style object with positioning
 */
export const transformCoord = (coords, color, pageIndex = 0, context = {}) => {
  // Extract context data or use defaults
  const { pageDimensions = { width: 0, height: 0 }, scale = 1.5 } = context;
  
  if (!coords || coords.length !== 4) {
    console.error('Invalid coordinates:', coords);
    return {
      position: 'absolute',
      left: 0,
      top: 0,
      width: 0,
      height: 0,
      border: `2px solid ${color}`,
      backgroundColor: color.replace('1)', '0.2)'),
      boxSizing: 'border-box',
      pointerEvents: 'auto'
    };
  }
  
  // Parse coordinates
  let [x0, y0, x1, y1] = coords;
  
  // Convert to numbers if they're strings
  x0 = parseFloat(x0);
  y0 = parseFloat(y0);
  x1 = parseFloat(x1);
  y1 = parseFloat(y1);
  
  // Get page dimensions
  const { width, height } = pageDimensions;
  
  if (!width || !height) {
    console.warn('Page dimensions not available, using default positioning');
    return {
      position: 'absolute',
      left: 0,
      top: 0,
      width: 0,
      height: 0,
      border: `2px solid ${color}`,
      backgroundColor: color.replace('1)', '0.2)'),
      boxSizing: 'border-box',
      pointerEvents: 'auto'
    };
  }
  
  // Log conversion for debugging
  console.log('Converting coordinates:', { coords, pageDimensions, scale });
  
  // Apply scaling factor to convert PDF coordinates to screen coordinates
  const scaledWidth = width * scale;
  const scaledHeight = height * scale;
  
  // Calculate position and dimensions
  const x = (x0 / width) * scaledWidth;
  const y = (y0 / height) * scaledHeight;
  const w = ((x1 - x0) / width) * scaledWidth;
  const h = ((y1 - y0) / height) * scaledHeight;
  
  // Log resulting values
  console.log('Transformed to:', { x, y, w, h, scaledWidth, scaledHeight });
  
  // Return CSS style object
  return {
    position: 'absolute',
    left: `${x}px`,
    top: `${y}px`,
    width: `${w}px`,
    height: `${h}px`,
    border: `2px solid ${color}`,
    backgroundColor: color.replace('1)', '0.2)'),
    boxSizing: 'border-box',
    pointerEvents: 'auto'
  };
};

/**
 * Gets a color map for different types of bounding boxes
 * @returns {Object} Map of query labels to colors
 */
export const getColorMap = () => ({
  pageIndicators: 'rgba(255, 99, 71, 0.7)', // Red-ish
  headers: 'rgba(65, 105, 225, 0.7)',       // Blue-ish
  footers: 'rgba(60, 179, 113, 0.7)',       // Green-ish
  tables: 'rgba(255, 165, 0, 0.7)',         // Orange
  paragraphs: 'rgba(238, 130, 238, 0.7)',   // Purple
  title: 'rgba(255, 215, 0, 0.7)',          // Gold
  default: 'rgba(200, 200, 200, 0.7)'       // Gray
});

/**
 * Formats query results into a structure suitable for BoundingBoxes component
 * @param {Array} results Query results from API
 * @param {number} currentPage Current page (0-based index)
 * @param {boolean} showResults Whether to show results
 * @returns {Object|null} Formatted boxes data or null
 */
export const getFormattedBoxesData = (results, currentPage, showResults) => {
  if (!results || !showResults) return null;
  
  // Convert page index to string for object keys
  const pageStr = String(currentPage);
  const formattedData = {};
  
  // Process each query result
  results.forEach(result => {
    // Skip if no pages data or no data for current page
    if (!result.results?.pages?.[pageStr]) return;
    
    const boxes = result.results.pages[pageStr];
    if (!boxes || !boxes.length) return;
    
    // Format boxes for this query type
    formattedData[result.query_label] = boxes.map(box => {
      // Extract coordinates with fallbacks for different formats
      const coords = box.bbox || [box.x0, box.y0, box.x1, box.y1];
      
      return {
        ...box,
        coords: coords,
        text: box.value || box.text || '',
        queryLabel: result.query_label,
        description: result.description || ''
      };
    });
  });
  
  return Object.keys(formattedData).length > 0 ? formattedData : null;
};

/**
 * Counts total number of results across all query types
 * @param {Object} boxesData Formatted boxes data
 * @returns {number} Total count of results
 */
export const getResultsCount = (boxesData) => {
  if (!boxesData) return 0;
  
  return Object.values(boxesData).reduce((total, boxes) => {
    return total + (Array.isArray(boxes) ? boxes.length : 0);
  }, 0);
};

/**
 * Debug function to check if coordinates are valid
 * @param {Array} coords Coordinates to check [x0, y0, x1, y1]
 * @returns {boolean} Whether coordinates are valid
 */
export const areValidCoords = (coords) => {
  if (!Array.isArray(coords) || coords.length !== 4) {
    return false;
  }
  
  const [x0, y0, x1, y1] = coords;
  
  return (
    typeof x0 === 'number' && !isNaN(x0) &&
    typeof y0 === 'number' && !isNaN(y0) &&
    typeof x1 === 'number' && !isNaN(x1) &&
    typeof y1 === 'number' && !isNaN(y1) &&
    x1 >= x0 && y1 >= y0
  );
};

// // Helper functions for PDF utils
// // Save this to src/components/pdf/utils/pdfUtils.js

// /**
//  * Transforms coordinates from PDF coordinate system to screen coordinates
//  * @param {Array} coords Array of coordinates [x0, y0, x1, y1]
//  * @param {string} color CSS color string
//  * @param {number} pageIndex Page index (0-based)
//  * @param {Object} context Additional context data
//  * @returns {Object} CSS style object with positioning
//  */
// export const transformCoord = (coords, color, pageIndex = 0, context = {}) => {
//   // Extract context data or use defaults
//   const { pageDimensions = { width: 0, height: 0 }, scale = 1.5 } = context;
  
//   if (!coords || coords.length !== 4) {
//     console.error('Invalid coordinates:', coords);
//     return {
//       position: 'absolute',
//       left: 0,
//       top: 0,
//       width: 0,
//       height: 0,
//       border: `2px solid ${color}`,
//       backgroundColor: color.replace('1)', '0.2)'),
//       boxSizing: 'border-box',
//       pointerEvents: 'auto'
//     };
//   }
  
//   // Parse coordinates
//   const [x0, y0, x1, y1] = coords;
  
//   // Get page dimensions
//   const { width, height } = pageDimensions;
  
//   if (!width || !height) {
//     console.warn('Page dimensions not available, using default positioning');
//     return {
//       position: 'absolute',
//       left: 0,
//       top: 0,
//       width: 0,
//       height: 0,
//       border: `2px solid ${color}`,
//       backgroundColor: color.replace('1)', '0.2)'),
//       boxSizing: 'border-box',
//       pointerEvents: 'auto'
//     };
//   }
  
//   // Apply scaling factor to convert PDF coordinates to screen coordinates
//   const scaledWidth = width * scale;
//   const scaledHeight = height * scale;
  
//   // Calculate position and dimensions
//   const x = (x0 / width) * scaledWidth;
//   const y = (y0 / height) * scaledHeight;
//   const w = ((x1 - x0) / width) * scaledWidth;
//   const h = ((y1 - y0) / height) * scaledHeight;
  
//   // Return CSS style object
//   return {
//     position: 'absolute',
//     left: `${x}px`,
//     top: `${y}px`,
//     width: `${w}px`,
//     height: `${h}px`,
//     border: `2px solid ${color}`,
//     backgroundColor: color.replace('1)', '0.2)'),
//     boxSizing: 'border-box',
//     pointerEvents: 'auto'
//   };
// };

// /**
//  * Gets a color map for different types of bounding boxes
//  * @returns {Object} Map of query labels to colors
//  */
// export const getColorMap = () => ({
//   pageIndicators: 'rgba(255, 99, 71, 0.7)', // Red-ish
//   headers: 'rgba(65, 105, 225, 0.7)',       // Blue-ish
//   footers: 'rgba(60, 179, 113, 0.7)',       // Green-ish
//   tables: 'rgba(255, 165, 0, 0.7)',         // Orange
//   paragraphs: 'rgba(238, 130, 238, 0.7)',   // Purple
//   title: 'rgba(255, 215, 0, 0.7)',          // Gold
//   default: 'rgba(200, 200, 200, 0.7)'       // Gray
// });

// /**
//  * Formats query results into a structure suitable for BoundingBoxes component
//  * @param {Array} results Query results from API
//  * @param {number} currentPage Current page (0-based index)
//  * @param {boolean} showResults Whether to show results
//  * @returns {Object|null} Formatted boxes data or null
//  */
// export const getFormattedBoxesData = (results, currentPage, showResults) => {
//   if (!results || !showResults) return null;
  
//   // Convert page index to string for object keys
//   const pageStr = String(currentPage);
//   const formattedData = {};
  
//   // Process each query result
//   results.forEach(result => {
//     // Skip if no pages data or no data for current page
//     if (!result.results?.pages?.[pageStr]) return;
    
//     const boxes = result.results.pages[pageStr];
//     if (!boxes || !boxes.length) return;
    
//     // Format boxes for this query type
//     formattedData[result.query_label] = boxes.map(box => {
//       // Extract coordinates with fallbacks for different formats
//       const coords = box.bbox || [box.x0, box.y0, box.x1, box.y1];
      
//       return {
//         ...box,
//         coords: coords,
//         text: box.value || box.text || '',
//         queryLabel: result.query_label,
//         description: result.description || ''
//       };
//     });
//   });
  
//   return Object.keys(formattedData).length > 0 ? formattedData : null;
// };

// /**
//  * Counts total number of results across all query types
//  * @param {Object} boxesData Formatted boxes data
//  * @returns {number} Total count of results
//  */
// export const getResultsCount = (boxesData) => {
//   if (!boxesData) return 0;
  
//   return Object.values(boxesData).reduce((total, boxes) => {
//     return total + (Array.isArray(boxes) ? boxes.length : 0);
//   }, 0);
// };

// /**
//  * Debug function to check if coordinates are valid
//  * @param {Array} coords Coordinates to check [x0, y0, x1, y1]
//  * @returns {boolean} Whether coordinates are valid
//  */
// export const areValidCoords = (coords) => {
//   if (!Array.isArray(coords) || coords.length !== 4) {
//     return false;
//   }
  
//   const [x0, y0, x1, y1] = coords;
  
//   return (
//     typeof x0 === 'number' && !isNaN(x0) &&
//     typeof y0 === 'number' && !isNaN(y0) &&
//     typeof x1 === 'number' && !isNaN(x1) &&
//     typeof y1 === 'number' && !isNaN(y1) &&
//     x1 >= x0 && y1 >= y0
//   );
// };