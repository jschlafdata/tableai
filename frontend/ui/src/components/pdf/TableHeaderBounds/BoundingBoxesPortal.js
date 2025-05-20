import React, { useEffect } from 'react';
import ReactDOM from 'react-dom';

import BoundingBoxes from '../BoundingBoxes';
import { transformCoordWithContainer } from './useBoundingBoxTransform';

const BoundingBoxesPortal = ({
  pdfContainerRef,
  tableHeaders,
  tableToggles,
  boundingBoxesVisible,
  pageDimensions,
  currentPage,
  colorMap
}) => {

  // Let's define "showResults" consistently
  const showResults = boundingBoxesVisible;

  const canRenderBoxes = () => {
    return (
      pdfContainerRef?.current &&
      pageDimensions.width > 0 &&
      pageDimensions.height > 0 &&
      showResults &&
      tableHeaders?.results?.pages
    );
  };

  /**
   * Filter your bounding boxes for the current page (currentPage - 1 if your app is 1-based).
   */
  function getFilteredBoxesData(tableHeaders, showResults) {
    if (!tableHeaders?.results?.pages || !showResults) return null;

    // Suppose your pages are 1-based in the UI, so you do -1 to get a zero-based index.
    const pageIndex = String(currentPage - 1); 
    const pageItems = tableHeaders.results.pages[pageIndex];
    if (!pageItems || !pageItems.length) return null;

    const formattedData = {};
  
    pageItems.forEach(item => {
      const label = `Table ${item.table_index}`;
      if (!formattedData[label]) {
        formattedData[label] = [];
      }
      // Add the main bounding box
      formattedData[label].push({
        ...item,
        coords: item.bbox,
        text: item.table_title ?? `Table ${item.table_index}`,
      });
  
      // Flatten the sub-metadata bounding boxes
      if (item.table_metadata) {
        Object.entries(item.table_metadata).forEach(([metaKey, metaObj]) => {
          if (!metaObj.bbox) return;
          formattedData[label].push({
            ...metaObj,
            coords: metaObj.bbox,
            text: metaKey,
            isMetadata: true,
          });
        });
      }
    });
  
    return formattedData;
  }

  // Correctly call with arguments:
  const boxesData = getFilteredBoxesData(tableHeaders, showResults);

  // If data is null (no bounding boxes), or conditions fail, return null early
  if (!canRenderBoxes() || !boxesData) {
    return null;
  }

  // This ensures we have the absolute container for the bounding boxes
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

    pdfContainerRef.current.style.position = 'relative';
    pdfContainerRef.current.appendChild(boxContainer);
  }

  const transformCoord = (coords, label) =>
    transformCoordWithContainer(
      coords,
      colorMap[label] || 'rgba(255, 0, 0, 1)',  // 2nd arg: color from colorMap
      pageDimensions,                            // 3rd arg
      tableHeaders,                              // 4th arg
      currentPage                                // 5th arg
    );

  // Render the bounding boxes into a portal
  return ReactDOM.createPortal(
    <BoundingBoxes
      boxesData={boxesData}
      transformCoord={transformCoord}
      colorMap={colorMap}
      showTooltips={true}
    />,
    boxContainer
  );
};

export default BoundingBoxesPortal;

// import React, { useEffect } from 'react';
// import ReactDOM from 'react-dom';

// import BoundingBoxes from '../BoundingBoxes';
// import { transformCoordWithContainer } from './useBoundingBoxTransform';

// const BoundingBoxesPortal = ({
//   pdfContainerRef,
//   tableHeaders,
//   tableToggles,
//   boundingBoxesVisible,
//   pageDimensions,
//   currentPage,
//   colorMap
// }) => {
//   /**
//    * Only show bounding boxes if we have valid data, container refs, etc.
//    */
//   const canRenderBoxes = () => {
//     return (
//       pdfContainerRef?.current &&
//       pageDimensions.width > 0 &&
//       pageDimensions.height > 0 &&
//       boundingBoxesVisible &&
//       tableHeaders?.results?.pages
//     );
//   };

// function getFilteredBoxesData(tableHeaders, showResults) {
//     if (!tableHeaders?.results?.pages || !showResults) return null;
  
//     const pageIndex = '0'; // or dynamic
//     const pageItems = tableHeaders.results.pages[pageIndex];
//     if (!pageItems || !pageItems.length) return null;
  
//     // This object will hold arrays of bounding boxes keyed by a label
//     const formattedData = {};
  
//     pageItems.forEach(item => {
//       // The main bounding box
//       const label = `Table ${item.table_index}`;
//       if (!formattedData[label]) {
//         formattedData[label] = [];
//       }
//       // Add the main bounding box
//       formattedData[label].push({
//         ...item,
//         coords: item.bbox,
//         text: item.table_title ?? `Table ${item.table_index}`,
//         // any other custom fields you like
//       });
  
//       // **Now flatten the sub-metadata bounding boxes**:
//       if (item.table_metadata) {
//         Object.entries(item.table_metadata).forEach(([metaKey, metaObj]) => {
//           if (!metaObj.bbox) return; // skip if no bounding box
//           formattedData[label].push({
//             ...metaObj,             // if it has coords, etc.
//             coords: metaObj.bbox,
//             text: metaKey,         // or any display label you want
//             isMetadata: true       // a flag you can check in boundingBoxes if you like
//           });
//         });
//       }
//     });
  
//     return formattedData;
//   }

//   const transformCoord = coords =>
//     transformCoordWithContainer(
//       coords,
//       pdfContainerRef.current,
//       pageDimensions
//     );

//   const boxesData = getFilteredBoxesData();

//   if (!canRenderBoxes() || !boxesData) {
//     return null;
//   }

//   // Ensure the container exists
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

//     // Ensure pdf container is position: relative
//     pdfContainerRef.current.style.position = 'relative';
//     pdfContainerRef.current.appendChild(boxContainer);
//   }

//   const showResults = boundingBoxesVisible;

//   return ReactDOM.createPortal(
//     <BoundingBoxes
//         boxesData={getFilteredBoxesData(tableHeaders, showResults)}
//         transformCoord={transformCoord}
//         colorMap={colorMap}
//         showTooltips={true}
//     />,
//     boxContainer
//   );
// };

// export default BoundingBoxesPortal;