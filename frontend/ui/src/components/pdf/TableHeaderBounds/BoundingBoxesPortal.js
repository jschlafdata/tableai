
import React, { useEffect } from 'react';
import ReactDOM from 'react-dom';

import BoundingBoxes from '../utils/BoundingBoxes';
import transformCoordWithContainer from '../utils/transformCoordWithContainer';
// import getFilteredBoxesData from '../utils/getFilteredBoxesData';

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
 * @param {Object} [options.metadataToggles=null] - Optional metadata toggles to control field visibility
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
    labelPrefix = '',
    metadataToggles = null
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
        if (tableIndex == null) return;
        
        const label = `${labelPrefix}${tableIndex}`;
        
        // Skip if toggle is false for this label
        if (toggles[label] === false) return;
        
        if (!formattedData[label]) {
          formattedData[label] = [];
        }
        
        // Check if main bbox should be included based on metadata toggles
        const shouldIncludeMainBox = !metadataToggles || 
                                    metadataToggles[label]?.main_bbox !== false;
        
        if (shouldIncludeMainBox) {
          // Add the main bounding box
          const coords = item.bbox || [item.x0, item.y0, item.x1, item.y1];
          const baseItem = {
            ...item,
            coords,
            text: item.table_title ?? `${labelPrefix}${tableIndex}`,
            isMainBox: true,
            tooltipText: `${label}: Main Box`
          };
          
          formattedData[label].push(
            formatItem ? formatItem(baseItem) : baseItem
          );
        }
        
        // Handle metadata items if requested
        if (includeMetadata && item.table_metadata) {
          Object.entries(item.table_metadata).forEach(([metaKey, metaObj]) => {
            if (!metaObj.bbox) return;
            
            // Check if this metadata field should be included based on metadata toggles
            const shouldIncludeMetaField = !metadataToggles || 
                                         metadataToggles[label]?.[metaKey] !== false;
            
            if (shouldIncludeMetaField) {
              const metaItem = {
                ...metaObj,
                coords: metaObj.bbox,
                text: metaKey,
                isMetadata: true,
                metadataKey: metaKey,
                tooltipText: `${label}: ${metaKey}`,
                dashed: true,
                alpha: 0.3
              };
              
              formattedData[label].push(
                formatItem ? formatItem(metaItem) : metaItem
              );
            }
          });
        }
        
        // Add other fields with bbox info if they exist
        if (includeMetadata) {
          ['bounds_index', 'hierarchy', 'col_hash'].forEach(attr => {
            if (!item[attr]?.bbox) return;
            
            // Check if this field should be included based on metadata toggles
            const shouldIncludeField = !metadataToggles || 
                                    metadataToggles[label]?.[attr] !== false;
            
            if (shouldIncludeField) {
              const attrItem = {
                ...item[attr],
                coords: item[attr].bbox,
                text: attr,
                isMetadata: true,
                metadataKey: attr,
                tooltipText: `${label}: ${attr}`,
                dashed: true,
                alpha: 0.3
              };
              
              formattedData[label].push(
                formatItem ? formatItem(attrItem) : attrItem
              );
            }
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

const BoundingBoxesPortal = ({
  pdfContainerRef,
  tableHeaders,
  tableToggles,
  metadataToggles,
  boundingBoxesVisible,
  pageDimensions,
  currentPage,
  colorMap
}) => {
  // Let's define "showResults" consistently
  const showResults = boundingBoxesVisible;

  /**
   * Setup and cleanup for the bounding boxes container
   */
  useEffect(() => {
    // Cleanup on unmount
    return () => {
      const container = document.getElementById('table-header-box-container');
      if (container && container.parentNode) {
        container.parentNode.removeChild(container);
      }
    };
  }, []);

  const canRenderBoxes = () => {
    return (
      pdfContainerRef?.current &&
      pageDimensions.width > 0 &&
      pageDimensions.height > 0 &&
      showResults &&
      tableHeaders?.results?.pages
    );
  };

  // If conditions fail, return null early
  if (!canRenderBoxes()) {
    return null;
  }

  /**
   * Filter bounding boxes for the current page
   */
  const boxesData = getFilteredBoxesData({
    data: tableHeaders,
    pageNumber: currentPage,
    showResults,
    toggles: tableToggles,
    includeMetadata: true,
    labelPrefix: 'Table ',
    metadataToggles: metadataToggles
  });

  // If no boxes to display, return null
  if (!boxesData) {
    return null;
  }

  // Ensure we have the container for the bounding boxes
  let boxContainer = document.getElementById('table-header-box-container');
  if (!boxContainer) {
    boxContainer = document.createElement('div');
    boxContainer.id = 'table-header-box-container';
    boxContainer.style.position = 'absolute';
    boxContainer.style.top = '0';
    boxContainer.style.left = '0';
    boxContainer.style.width = '100%';
    boxContainer.style.height = '100%';
    boxContainer.style.pointerEvents = 'none';
    boxContainer.style.zIndex = '1001';
    boxContainer.style.overflow = 'hidden';

    if (pdfContainerRef?.current) {
      pdfContainerRef.current.style.position = 'relative';
      pdfContainerRef.current.appendChild(boxContainer);
    }
  }

  const transformCoord = (coords, label) =>
    transformCoordWithContainer(coords, {
      color: colorMap[label] || 'rgba(255, 0, 0, 1)',
      containerDimensions: pageDimensions,
      pdfData: tableHeaders,
      pageNumber: currentPage,
      zIndex: 2001
    });

  // Render the bounding boxes into a portal
  return ReactDOM.createPortal(
    <BoundingBoxes
      boxesData={boxesData}
      transformCoord={transformCoord}
      colorMap={colorMap}
      showTooltips={true}
      metadataToggles={metadataToggles}
    />,
    boxContainer
  );
};

export default BoundingBoxesPortal;

// export default getFilteredBoxesData;

// const BoundingBoxesPortal = ({
//   pdfContainerRef,
//   tableHeaders,
//   tableToggles,
//   metadataToggles, // New prop for metadata field toggles
//   boundingBoxesVisible,
//   pageDimensions,
//   currentPage,
//   colorMap
// }) => {
//   // Let's define "showResults" consistently
//   const showResults = boundingBoxesVisible;

//   const canRenderBoxes = () => {
//     return (
//       pdfContainerRef?.current &&
//       pageDimensions.width > 0 &&
//       pageDimensions.height > 0 &&
//       showResults &&
//       tableHeaders?.results?.pages
//     );
//   };

//   /**
//    * Setup and cleanup for the bounding boxes container
//    */
//   useEffect(() => {
//     // Create container if needed
//     let boxContainer = document.getElementById('table-header-box-container');
//     if (!boxContainer && pdfContainerRef?.current) {
//       boxContainer = document.createElement('div');
//       boxContainer.id = 'table-header-box-container';
//       boxContainer.style.position = 'absolute';
//       boxContainer.style.top = '0';
//       boxContainer.style.left = '0';
//       boxContainer.style.width = '100%';
//       boxContainer.style.height = '100%';
//       boxContainer.style.pointerEvents = 'none';
//       boxContainer.style.zIndex = '1001';
//       boxContainer.style.overflow = 'hidden';

//       pdfContainerRef.current.style.position = 'relative';
//       pdfContainerRef.current.appendChild(boxContainer);
//     }

//     // Cleanup on unmount
//     return () => {
//       const container = document.getElementById('table-header-box-container');
//       if (container && container.parentNode) {
//         container.parentNode.removeChild(container);
//       }
//     };
//   }, [pdfContainerRef]);

//   // If conditions fail, return null early
//   if (!canRenderBoxes()) {
//     return null;
//   }

//   /**
//    * Enhanced version of getFilteredBoxesData that respects metadata toggles
//    */
//   const getEnhancedBoxesData = () => {
//     // Start with the standard filtered boxes
//     let baseBoxes = getFilteredBoxesData({
//       data: tableHeaders,
//       pageNumber: currentPage,
//       showResults,
//       includeMetadata: true,
//       labelPrefix: 'Table '
//     });

//     // If no metadata toggles are provided, return the base boxes
//     if (!metadataToggles) {
//       return baseBoxes;
//     }

//     // Filter boxes based on metadata toggles
//     const enhancedBoxes = [];

//     // Process each box and check if it should be included based on metadata toggles
//     baseBoxes.forEach(box => {
//       const tableLabel = box.label;
      
//       // Skip if the table is toggled off
//       if (!tableToggles[tableLabel]) {
//         return;
//       }

//       // Main box should be included if main_bbox toggle is on
//       if (box.isMainBox && metadataToggles[tableLabel]?.main_bbox) {
//         enhancedBoxes.push(box);
//       } 
//       // Metadata boxes should be included if their specific toggle is on
//       else if (box.metadataKey && metadataToggles[tableLabel]?.[box.metadataKey]) {
//         enhancedBoxes.push(box);
//       }
//       // Include any box that isn't specifically controlled by a toggle
//       else if (!box.metadataKey && !box.isMainBox) {
//         enhancedBoxes.push(box);
//       }
//     });

//     return enhancedBoxes;
//   };

//   // If no metadata toggles are provided, use the original implementation
//   const boxesData = metadataToggles ? getEnhancedBoxesData() : getFilteredBoxesData({
//     data: tableHeaders,
//     pageNumber: currentPage,
//     showResults,
//     includeMetadata: true,
//     labelPrefix: 'Table '
//   });

//   // If no boxes to display, return null
//   if (!boxesData || boxesData.length === 0) {
//     return null;
//   }

//   // Get the container for the bounding boxes
//   const boxContainer = document.getElementById('table-header-box-container');
//   if (!boxContainer) {
//     return null;
//   }

//   const transformCoord = (coords, label) =>
//     transformCoordWithContainer(coords, {
//       color: colorMap[label] || 'rgba(255, 0, 0, 1)',
//       containerDimensions: pageDimensions,
//       pdfData: tableHeaders,
//       pageNumber: currentPage,
//       zIndex: 2001
//     });

//   // Render the bounding boxes into a portal
//   return ReactDOM.createPortal(
//     <BoundingBoxes
//       boxesData={boxesData}
//       transformCoord={transformCoord}
//       colorMap={colorMap}
//       showTooltips={true}
//       metadataToggles={metadataToggles} // Pass the toggles down
//     />,
//     boxContainer
//   );
// };

// export default BoundingBoxesPortal;

// import React, { useEffect } from 'react';
// import ReactDOM from 'react-dom';

// import BoundingBoxes from '../utils/BoundingBoxes';
// import transformCoordWithContainer from '../utils/transformCoordWithContainer';
// import getFilteredBoxesData from '../utils/getFilteredBoxesData';

// const BoundingBoxesPortal = ({
//   pdfContainerRef,
//   tableHeaders,
//   tableToggles,
//   boundingBoxesVisible,
//   pageDimensions,
//   currentPage,
//   colorMap
// }) => {

//   // Let's define "showResults" consistently
//   const showResults = boundingBoxesVisible;

//   const canRenderBoxes = () => {
//     return (
//       pdfContainerRef?.current &&
//       pageDimensions.width > 0 &&
//       pageDimensions.height > 0 &&
//       showResults &&
//       tableHeaders?.results?.pages
//     );
//   };

//   /**
//    * Filter your bounding boxes for the current page (currentPage - 1 if your app is 1-based).
//    */
//   const boxesData = getFilteredBoxesData({
//     data: tableHeaders,
//     pageNumber: currentPage,
//     showResults,
//     includeMetadata: true,
//     labelPrefix: 'Table '
//   });

//   // Correctly call with arguments:

//   // If data is null (no bounding boxes), or conditions fail, return null early
//   if (!canRenderBoxes() || !boxesData) {
//     return null;
//   }

//   // This ensures we have the absolute container for the bounding boxes
//   let boxContainer = document.getElementById('table-header-box-container');
//   if (!boxContainer) {
//     boxContainer = document.createElement('div');
//     boxContainer.id = 'table-header-box-container';
//     boxContainer.style.position = 'absolute';
//     boxContainer.style.top = '0';
//     boxContainer.style.left = '0';
//     boxContainer.style.width = '100%';
//     boxContainer.style.height = '100%';
//     boxContainer.style.pointerEvents = 'none';
//     boxContainer.style.zIndex = '1001';
//     boxContainer.style.overflow = 'hidden';

//     pdfContainerRef.current.style.position = 'relative';
//     pdfContainerRef.current.appendChild(boxContainer);
//   }

//   const transformCoord = (coords, label) =>
//     transformCoordWithContainer(coords, {
//       color: colorMap[label] || 'rgba(255, 0, 0, 1)',
//       containerDimensions: pageDimensions,
//       pdfData: tableHeaders,
//       pageNumber: currentPage,
//       zIndex: 2001
//     });

//   // Render the bounding boxes into a portal
//   return ReactDOM.createPortal(
//     <BoundingBoxes
//       boxesData={boxesData}
//       transformCoord={transformCoord}
//       colorMap={colorMap}
//       showTooltips={true}
//     />,
//     boxContainer
//   );
// };

// export default BoundingBoxesPortal;