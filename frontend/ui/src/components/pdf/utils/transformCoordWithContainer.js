/**
 * Transforms PDF coordinates to fit a rendered container with styling options
 * 
 * @param {Array<number>} coords - Array of coordinates [x0, y0, x1, y1]
 * @param {Object} options - Configuration options
 * @param {string} options.color - Border color (CSS color string)
 * @param {Object} options.containerDimensions - Dimensions of the container element
 * @param {number} options.containerDimensions.width - Container width in pixels
 * @param {number} options.containerDimensions.height - Container height in pixels
 * @param {Object} options.pdfData - PDF data object containing dimensions info
 * @param {number} options.pageNumber - Current page number (1-based)
 * @param {Object} [options.customStyles={}] - Additional CSS styles to apply
 * @param {number} [options.opacity=1] - Opacity for the box
 * @param {number} [options.zIndex=2000] - Z-index for the box
 * @param {number} [options.backgroundOpacity=0.2] - Background opacity
 * @returns {Object} CSS style object for the bounding box
 */
function transformCoordWithContainer(coords, options) {
    const {
      color,
      containerDimensions,
      pdfData,
      pageNumber,
      customStyles = {},
      opacity = 1,
      zIndex = 2000,
      backgroundOpacity = 0.2,
      pointerEvents = 'auto',
      cursor = 'pointer'
    } = options;
  
    // Import getPdfDimensions function
    const getPdfDimensions = require('./getPdfDimensions').default;
    
    // Get PDF dimensions
    const pdfDimensions = getPdfDimensions(pdfData, pageNumber, containerDimensions);
    
    // Early return if container dimensions not available
    if (!containerDimensions?.width || !containerDimensions?.height) {
      return { 
        position: 'absolute', 
        left: '0px', 
        top: '0px', 
        width: '0px', 
        height: '0px', 
        opacity: 0 
      };
    }
  
    // Parse and normalize coordinates
    let [x0, y0, x1, y1] = coords.map(Number);
    if (x1 < x0) [x0, x1] = [x1, x0];
    if (y1 < y0) [y0, y1] = [y1, y0];
  
    // Calculate scaling ratios
    const renderedWidth = containerDimensions.width;
    const renderedHeight = containerDimensions.height;
    const pdfWidth = pdfDimensions?.width || 612.0;  // Default US Letter width
    const pdfHeight = pdfDimensions?.height || 792.0; // Default US Letter height
  
    const widthRatio = renderedWidth / pdfWidth;
    const heightRatio = renderedHeight / pdfHeight;
  
    // Transform coordinates
    const x = x0 * widthRatio;
    const y = y0 * heightRatio;
    const width = (x1 - x0) * widthRatio;
    const height = (y1 - y0) * heightRatio;
  
    // Create background color with opacity
    let backgroundColor = color;
    if (color.includes('rgba')) {
      backgroundColor = color.replace(/rgba\((.+?),[^,]+\)/, `rgba($1,${backgroundOpacity})`);
    } else if (color.includes('rgb')) {
      backgroundColor = color.replace(/rgb\((.+?)\)/, `rgba($1,${backgroundOpacity})`);
    } else {
      backgroundColor = `${color}${backgroundOpacity * 100}%`;
    }
  
    // Return the style object
    return {
      position: 'absolute',
      left: `${x}px`,
      top: `${y}px`,
      width: `${width}px`,
      height: `${height}px`,
      border: `2px solid ${color}`,
      backgroundColor,
      zIndex,
      pointerEvents,
      cursor,
      opacity,
      ...customStyles
    };
  }
  
  export default transformCoordWithContainer;