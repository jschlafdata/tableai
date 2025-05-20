import React, { useEffect } from 'react';
import ReactDOM from 'react-dom';

import BoundingBoxes from '../utils/BoundingBoxes';
import transformCoordWithContainer from '../utils/transformCoordWithContainer';
import getFilteredBoxesData from '../utils/getFilteredBoxesData';

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
  const boxesData = getFilteredBoxesData({
    data: tableHeaders,
    pageNumber: currentPage,
    showResults,
    includeMetadata: true,
    labelPrefix: 'Table '
  });

  // Correctly call with arguments:

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
    />,
    boxContainer
  );
};

export default BoundingBoxesPortal;