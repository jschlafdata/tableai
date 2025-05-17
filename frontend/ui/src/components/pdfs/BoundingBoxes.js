// BoundingBoxes.js
import React from 'react';

/**
 * Renders bounding box overlays on a PDF
 * 
 * @param {Object} props
 * @param {Object} props.boxesData - Object with box data by type
 * @param {Function} props.transformCoord - Function to transform coordinates to CSS
 * @param {Object} props.colorMap - Map of box types to colors
 */
function BoundingBoxes({ boxesData, transformCoord, colorMap = {} }) {
  if (!boxesData) return null;

  // Log boxes data for debugging
  console.log("Rendering boxes data:", boxesData);
  console.log("Color map:", colorMap);

  // Render all box types
  return (
    <div className="bounding-boxes-container">
      {Object.entries(boxesData).map(([boxType, boxes]) => {
        const color = colorMap[boxType] || 'rgba(255, 0, 0, 0.2)';
        
        if (!Array.isArray(boxes)) {
          console.error(`Boxes for type ${boxType} is not an array`, boxes);
          return null;
        }
        
        return (
          <div key={boxType} className={`box-type-${boxType}`}>
            {boxes.map((box, idx) => {
              const coords = Array.isArray(box) ? box : box.coords;
              const pageIndex = box?.pageIndex ?? 0;

              if (!Array.isArray(coords) || coords.length !== 4) {
                console.error(`Invalid coords for ${boxType} at index ${idx}:`, coords);
                return null;
              }

              // 👇 Check per-box override color first
              const color = box.color || colorMap[boxType] || 'rgba(255, 0, 0, 0.2)';
              const boxStyle = {
                ...transformCoord(coords, color, pageIndex),
                ...(box.borderColor && { border: `${box.borderWidth ?? 1}px solid ${box.borderColor}` })
              };
              return (
                <div 
                  key={`${boxType}-${idx}`} 
                  className={`bounding-box ${boxType}`}
                  style={boxStyle}
                  title={boxType}
                />
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

export default BoundingBoxes;