// Updated BoundingBoxes component with debugging
import React, { useEffect, useRef, useState } from 'react';

/**
 * Component to render bounding boxes over a PDF
 * 
 * @param {Object} props Component props
 * @param {Object} props.boxesData Object with box data by query label
 * @param {Function} props.transformCoord Function to transform coordinates
 * @param {Object} props.colorMap Map of query labels to colors
 * @param {Boolean} props.showTooltips Whether to show tooltips on hover
 */
const BoundingBoxes = ({
  boxesData,
  transformCoord,
  colorMap = {},
  showTooltips = true,
}) => {
  // Create a ref for the container of all bounding boxes
  const boxesContainerRef = useRef(null);
  const [hovered, setHovered] = useState(null);
  
  // useEffect must be called at the top level, before any conditional returns
  useEffect(() => {
    // Log on mount for debugging
    console.log('BoundingBoxes mounted, container ref:', boxesContainerRef.current);
    console.log('Render with data:', boxesData ? Object.keys(boxesData).join(', ') : 'none');
    
    return () => {
      console.log('BoundingBoxes unmounted');
    };
  }, [boxesData]);
  
  // Check if we have valid data to render
  if (!boxesData || Object.keys(boxesData).length === 0) {
    console.log('No boxesData provided to BoundingBoxes or empty data');
    return null;
  }
  
  // Log rendering for debugging
  console.log('Rendering BoundingBoxes with data for:', Object.keys(boxesData).join(', '));
  
  return (
    <div 
      ref={boxesContainerRef}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 100
      }}
      className="bounding-boxes-overlay"
    >
      {Object.entries(boxesData).map(([queryLabel, boxes]) => {
        if (!Array.isArray(boxes) || boxes.length === 0) {
          console.log(`No boxes to render for ${queryLabel}`);
          return null;
        }
        
        console.log(`Rendering ${boxes.length} boxes for ${queryLabel}`);
        
        return (
          <React.Fragment key={queryLabel}>
            {boxes.map((box, index) => {
              // Get coordinates from the box
              const coords = box.bbox || box.coords || [box.x0, box.y0, box.x1, box.y1];
              
              if (!coords || coords.length !== 4) {
                console.error(`Invalid coords for box in ${queryLabel}:`, coords);
                return null;
              }
              
              // Get color from the color map or use default
              const color = box.color || colorMap[queryLabel] || colorMap.default || 'rgba(255, 0, 0, 0.5)';
              
              // Apply the transformation to get pixel coordinates
              const style = transformCoord(coords, color);
              
              // Generate a unique key for this box
              const boxKey = `${queryLabel}-${index}`;
              const isHovered = hovered === boxKey;
              
              return (
                <div key={boxKey} className={`box-container ${queryLabel}`}>
                  <div
                    style={{
                      ...style,
                      zIndex: isHovered ? 101 : 100,
                      backgroundColor: isHovered 
                        ? `${color.replace(')', ', 0.4)')}` 
                        : `${color.replace(')', ', 0.2)')}`,
                      pointerEvents: 'auto',
                      cursor: 'pointer',
                    }}
                    title={showTooltips ? `${queryLabel}: ${box.value || box.text || ''}` : ''}
                    onMouseEnter={() => setHovered(boxKey)}
                    onMouseLeave={() => setHovered(null)}
                    className="bounding-box"
                  >
                    {/* Small label in top-left corner */}
                    {/* <div
                      style={{
                        position: 'absolute',
                        top: '-18px',
                        left: '0',
                        backgroundColor: color,
                        color: 'white',
                        padding: '1px 3px',
                        fontSize: '10px',
                        borderRadius: '2px',
                        maxWidth: '120px',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        pointerEvents: 'none',
                      }}
                      className="box-label"
                    > */}
                      {/* {queryLabel} */}
                      {/* {box.confidence && ` (${Math.round(box.confidence * 100)}%)`} */}
                    {/* </div> */}
                  </div>
                  
                  {/* Show tooltip on hover if enabled */}
                  {showTooltips && (box.value || box.text) && isHovered && (
                    <div 
                      style={{
                        position: 'absolute',
                        top: parseInt(style.top, 10) + parseInt(style.height, 10),
                        left: style.left,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        color: 'white',
                        padding: '4px 8px',
                        borderRadius: '4px',
                        fontSize: '12px',
                        zIndex: 102,
                        pointerEvents: 'none',
                        maxWidth: '300px',
                        wordBreak: 'break-word',
                        transform: 'translateY(4px)',
                      }}
                      className="box-tooltip"
                    >
                      {box.value || box.text}
                    </div>
                  )}
                </div>
              );
            })}
          </React.Fragment>
        );
      })}
    </div>
  );
};

export default BoundingBoxes;