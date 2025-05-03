import React from 'react';

/**
 * Draws horizontal lines at each 'y' in pageBreaks,
 * using the ratio of the displayed PDF height to the
 * PDF's original coordinate system.
 */
function PageBreakLines({ pageBreaks, docWidth, docHeight, pdfHeight }) {
  if (!pageBreaks || pageBreaks.length === 0) {
    return null;
  }

  // We'll map each break to a horizontal line
  return (
    <>
      {pageBreaks.map((yVal, index) => {
        const ratioY = docHeight / pdfHeight; // how many screen-px per 1 PDF pt
        const top = yVal * ratioY;            // screen coordinate

        return (
          <div
            key={index}
            style={{
              position: 'absolute',
              left: 0,
              top: `${top}px`,
              width: `${docWidth}px`,
              height: '2px',
              backgroundColor: 'red',
              opacity: 0.6
            }}
          />
        );
      })}
    </>
  );
}

export default PageBreakLines;
