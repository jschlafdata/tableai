// Updated PdfProcessingResults.jsx with Vision Inference and preserving custom color functions
import React, { useState, useEffect, useRef } from 'react';
import {
  Button,
  CircularProgress,
  Snackbar,
  Alert,
  Box,
  Typography,
  FormGroup,
  FormControlLabel,
  Switch,
  Chip,
  Paper,
  Collapse,
  IconButton,
  Tooltip
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import CancelIcon from '@mui/icons-material/Cancel';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import VisibilityIcon from '@mui/icons-material/Visibility';
import ReactDOM from 'react-dom';

import BoundingBoxes from './BoundingBoxes';
import { getResultsCount } from './utils/pdfUtils';

const PdfProcessingResults = ({
  fileId,
  stage,
  pageDimensions,
  currentPage,
  pdfContainerRef,
  scale = 1.5,
  metadata = {}
}) => {
  const [loading, setLoading] = useState(false);
  const [visionLoading, setVisionLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [showResults, setShowResults] = useState(true);
  const [queryToggles, setQueryToggles] = useState({});
  const [showQueryControls, setShowQueryControls] = useState(true);
  const [boundingBoxesVisible, setBoundingBoxesVisible] = useState(false);

  // Check if file has classification
  const hasClassification = !!metadata?.classification;

  useEffect(() => {
    // Log page dimensions to debug potential issues
    console.log('Page dimensions:', pageDimensions);
    console.log('PDF Container Ref:', pdfContainerRef?.current);
  }, [pageDimensions, pdfContainerRef]);

  // Log metadata for debugging
    useEffect(() => {
    console.log('Current metadata:', metadata);
    console.log('Has classification:', hasClassification, metadata?.classification);
  }, [metadata, hasClassification]);

  // Make sure we re-render when query toggles change
  useEffect(() => {
    if (showResults && Object.keys(queryToggles).length > 0) {
      console.log('Query toggles changed, updating bounding boxes');
      // Force a refresh of bounding boxes when toggles change
      setBoundingBoxesVisible(false);
      const timer = setTimeout(() => setBoundingBoxesVisible(true), 50);
      return () => clearTimeout(timer);
    }
  }, [queryToggles, showResults]);

  const processPdf = async () => {
    if (!fileId) {
      setError("No file selected");
      return;
    }

    setLoading(true);
    setResults([]);
    setError(null);
    setQueryToggles({});
    setBoundingBoxesVisible(false);

    try {
      console.log('Processing PDF with file_id:', fileId, 'stage:', stage);
      const response = await fetch(`http://localhost:8000/tableai/extract/doc_query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ file_id: fileId, stage: stage }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Received query results:', data);
      setResults(data);
      setShowResults(true);

      const initialToggles = {};
      data.forEach(result => {
        initialToggles[result.query_label] = true;
      });
      setQueryToggles(initialToggles);
      setBoundingBoxesVisible(true);

    } catch (err) {
      console.error('Error processing PDF:', err);
      setError(`Failed to process PDF: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };
  
  // Run vision inference
  const runVisionInference = async () => {
    if (!fileId || !hasClassification) {
      setError("Cannot run vision inference - file must have a classification");
      return;
    }

    setVisionLoading(true);
    setError(null);

    try {
      console.log('Running vision inference for file_id:', fileId, 'stage:', stage, 'classification:', metadata.classification);
      
      const response = await fetch(`http://localhost:8000/tableai/extract/vision/structure`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          file_id: fileId, 
          stage: stage,
          classification_label: metadata.classification 
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      const data = await response.json();
      console.log('Vision inference results:', data);
      
      // Show success message
      setError({ message: "Vision inference completed successfully!", severity: "success" });
      
      // Process results if needed - this depends on how you want to handle the results
      // If they are in the same format as the document query results, you could potentially
      // add them to the existing results or replace them

    } catch (err) {
      console.error('Error running vision inference:', err);
      setError(`Failed to run vision inference: ${err.message}`);
    } finally {
      setVisionLoading(false);
    }
  };

  // Custom color generation function
  const getColorForLabel = (() => {
    const cache = {};
    return (label) => {
      if (cache[label]) return cache[label];
      let hash = 0;
      for (let i = 0; i < label.length; i++) {
        hash = label.charCodeAt(i) + ((hash << 5) - hash);
      }
      // Wide hue spread
      const hue = Math.abs(hash) % 360;
      // Vary saturation and lightness for even more separation
      const saturation = 60 + (Math.abs(hash) % 30); // 60% - 89%
      const lightness = 45 + (Math.abs(hash * 7) % 30); // 45% - 74%
      const color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
      cache[label] = color;
      return color;
    };
  })();
  
  // In your main component render:
  const colorMap = {};
  results.forEach(r => {
    colorMap[r.query_label] = getColorForLabel(r.query_label);
  });

  const toggleResults = () => {
    setShowResults(!showResults);
    // Update bounding box visibility when toggling results
    setBoundingBoxesVisible(!showResults);
  };

  const toggleQueryLabel = (queryLabel) => {
    console.log(`Toggling query label ${queryLabel}`, !queryToggles[queryLabel]);
    setQueryToggles(prev => {
      const newToggles = {
        ...prev,
        [queryLabel]: !prev[queryLabel]
      };
      console.log('New query toggles:', newToggles);
      return newToggles;
    });
    
    // Force re-render of bounding boxes
    setBoundingBoxesVisible(prev => {
      if (prev) {
        // Quick toggle to force re-render
        setTimeout(() => setBoundingBoxesVisible(true), 10);
        return false;
      }
      return true;
    });
  };

  const toggleAllQueryLabels = (value) => {
    const updated = {};
    Object.keys(queryToggles).forEach(key => {
      updated[key] = value;
    });
    setQueryToggles(updated);
  };

  const getFilteredBoxesData = () => {
    if (!results || results.length === 0 || !showResults) return null;
    
    // Log current page for debugging
    console.log('Getting boxes for current page:', currentPage);
    console.log('Query toggles:', queryToggles);
    
    try {
      // This is a direct implementation using your response structure
      const formattedData = {};
      
      results.forEach(result => {
        const queryLabel = result.query_label;
        
        // Skip this query type if it's toggled off
        if (queryToggles[queryLabel] === false) {
          console.log(`Query ${queryLabel} is toggled off, skipping`);
          return;
        }
        
        const pages = result.results?.pages;
        
        if (!pages) {
          console.log(`No pages data found for ${queryLabel}`);
          return;
        }
        
        // Convert to zero-based indexing (your currentPage is 1-based)
        const pageIndex = String(currentPage - 1);
        
        console.log(`Looking for page ${pageIndex} in`, Object.keys(pages));
        
        const pageBoxes = pages[pageIndex];
        if (!pageBoxes || !pageBoxes.length) {
          console.log(`No boxes found for page ${pageIndex} in ${queryLabel}`);
          return;
        }
        
        console.log(`Found ${pageBoxes.length} boxes for ${queryLabel} on page ${pageIndex}`);
        
        formattedData[queryLabel] = pageBoxes.map(box => {
          // Check if we have bbox or use x0,y0,x1,y1
          const coords = box.bbox || [box.x0, box.y0, box.x1, box.y1];
          
          return {
            ...box,
            coords,
            text: box.value || '',
            queryLabel
          };
        });
      });
      
      console.log('Filtered boxes data:', formattedData);
      
      // Return null if no boxes found
      return Object.keys(formattedData).length > 0 ? formattedData : null;
    } catch (error) {
      console.error('Error formatting boxes data:', error);
      return null;
    }
  };

  const getQueryLabelResultsCount = (label) => {
    const queryResult = results.find(r => r.query_label === label);
    if (!queryResult || !queryResult.results?.pages) return 0;
    
    // Adjust for zero-based indexing in the results
    const pageIndex = String(currentPage - 1);
    return (queryResult.results.pages[pageIndex] || []).length;
  };

  // Fixed transformCoord function for PDFs
  const transformCoordWithContainer = (coords, color) => {
    // Make sure we're working with the correct page dimensions
    if (!pageDimensions.width || !pageDimensions.height) {
      console.warn('Page dimensions not available:', pageDimensions);
      return { x: 0, y: 0, width: 0, height: 0, color };
    }
    
    // Simple coordinate transformation
    // [x0, y0, x1, y1] to CSS positioning
    let [x0, y0, x1, y1] = coords;
    
    // Convert to numbers if they're strings
    x0 = parseFloat(x0);
    y0 = parseFloat(y0);
    x1 = parseFloat(x1);
    y1 = parseFloat(y1);
    
    // Fix any invalid coordinates (ensure x1 > x0 and y1 > y0)
    if (x1 < x0) [x0, x1] = [x1, x0];
    if (y1 < y0) [y0, y1] = [y1, y0];
    
    console.log('Original coords:', [x0, y0, x1, y1]);
    console.log('Page dimensions:', pageDimensions);
    console.log('Scale:', scale);
    
    // Get the actual rendered dimensions of the PDF
    const renderedWidth = pageDimensions.width;
    const renderedHeight = pageDimensions.height;
    
    // Get the PDF's intrinsic dimensions (from metadata)
    // Find these in the PDF metadata or extract from API response
    const pdfWidth = 612.0; // From your API response metadata
    const pdfHeight = 792.0; // From your API response metadata
    
    // Calculate the ratio between rendered and original PDF dimensions
    const widthRatio = renderedWidth / pdfWidth;
    const heightRatio = renderedHeight / pdfHeight;
    
    // Apply coordinate transformation with correct scaling
    // PDF coordinates start from bottom-left, CSS starts from top-left
    const x = x0 * widthRatio;
    const y = y0 * heightRatio; // No flipping needed based on your coordinates
    const width = (x1 - x0) * widthRatio;
    const height = (y1 - y0) * heightRatio;
    
    const transformedCoords = { x, y, width, height };
    console.log('Transformed coords:', transformedCoords);
    
    return {
      position: 'absolute',
      left: `${x}px`,
      top: `${y}px`,
      width: `${width}px`,
      height: `${height}px`,
      border: `2px solid ${color}`,
      backgroundColor: color.replace('1)', '0.3)'),
      zIndex: 2000,
      pointerEvents: 'auto',
      opacity: 1
    };
  };

  const boxesData = getFilteredBoxesData();
  const resultsCount = boxesData ? getResultsCount(boxesData) : 0;
  
  const canRenderBoxes = pdfContainerRef?.current && 
                        pageDimensions.width > 0 && 
                        pageDimensions.height > 0 && 
                        boundingBoxesVisible &&
                        boxesData;

  // Render bounding boxes directly into the PDF container using a portal
  const renderBoundingBoxesPortal = () => {
    if (!canRenderBoxes) {
      console.log('Cannot render boxes yet:', { 
        containerExists: !!pdfContainerRef?.current,
        pageDimensions,
        boundingBoxesVisible,
        hasBoxesData: !!boxesData
      });
      return null;
    }
    
    console.log('Rendering bounding boxes portal with dimensions:', pageDimensions);
    
    // Create a dedicated container for the bounding boxes if it doesn't exist
    let boxContainer = document.getElementById('bounding-box-container');
    if (!boxContainer) {
      boxContainer = document.createElement('div');
      boxContainer.id = 'bounding-box-container';
      boxContainer.style.position = 'absolute';
      boxContainer.style.top = '0';
      boxContainer.style.left = '0';
      boxContainer.style.width = '100%';
      boxContainer.style.height = '100%';
      boxContainer.style.pointerEvents = 'none';
      boxContainer.style.zIndex = '1000'; // Ensure it's on top
      
      // Add it to the PDF container
      pdfContainerRef.current.appendChild(boxContainer);
    }
    
    return ReactDOM.createPortal(
      <BoundingBoxes
        boxesData={boxesData}
        transformCoord={(coords, color) => transformCoordWithContainer(coords, color)}
        colorMap={colorMap}
        showTooltips={true}
      />,
      boxContainer
    );
  };

  return (
    <>
      {/* Process Controls */}
      <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
        <Button
          variant="contained"
          color="primary"
          onClick={processPdf}
          disabled={loading || !fileId}
          startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
        >
          {loading ? "Processing..." : "Process PDF"}
        </Button>

        <Tooltip title={!hasClassification ? "File must have a classification to run vision inference" : ""}>
          <span>
            <Button
              variant="contained"
              color="secondary"
              onClick={runVisionInference}
              disabled={visionLoading || !fileId || !hasClassification}
              startIcon={visionLoading ? <CircularProgress size={20} color="inherit" /> : <VisibilityIcon />}
            >
              {visionLoading ? "Running..." : "Run Vision Inference"}
            </Button>
          </span>
        </Tooltip>

        {results.length > 0 && (
          <Button
            variant="outlined"
            color="primary"
            onClick={toggleResults}
            startIcon={showResults ? <CancelIcon /> : <SearchIcon />}
          >
            {showResults ? "Hide Results" : "Show Results"}
          </Button>
        )}

        {results.length > 0 && (
          <Typography variant="body2" color="textSecondary">
            Found {resultsCount} results on page {currentPage}
          </Typography>
        )}
      </Box>

      {/* Query Filters */}
      {results.length > 0 && (
        <Paper elevation={1} sx={{
          p: 2, mb: 2,
          bgcolor: '#f5f5f5',
          border: '1px solid #e0e0e0',
          borderRadius: '4px'
        }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="subtitle1" fontWeight="bold">
              Query Types
            </Typography>
            <Box>
              <Button size="small" onClick={() => toggleAllQueryLabels(true)} sx={{ mr: 1 }}>Show All</Button>
              <Button size="small" onClick={() => toggleAllQueryLabels(false)}>Hide All</Button>
              <IconButton size="small" onClick={() => setShowQueryControls(prev => !prev)} sx={{ ml: 1 }}>
                {showQueryControls ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              </IconButton>
            </Box>
          </Box>

          <Collapse in={showQueryControls}>
            <FormGroup sx={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: 1 }}>
              {results.map(result => {
                const label = result.query_label;
                const description = result?.description || '';
                const count = getQueryLabelResultsCount(label);
                const color = getColorForLabel(label);

                return (
                  <Box key={label} sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    bgcolor: 'white',
                    p: 1,
                    borderRadius: '4px',
                    border: '1px solid #e0e0e0',
                    minWidth: '200px'
                  }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={queryToggles[label] !== false}
                            onChange={() => toggleQueryLabel(label)}
                            size="small"
                          />
                        }
                        label={<Typography variant="body2" fontWeight="bold">{label}</Typography>}
                      />
                    <Chip
                        label={count}
                        size="small"
                        sx={{
                            bgcolor: getColorForLabel(label) + '20', // 20 = 12% opacity in hex
                            border: `1px solid ${getColorForLabel(label)}`
                        }}
                        />
                    </Box>
                    {description && (
                      <Typography variant="caption" sx={{ mt: 0.5, color: 'text.secondary' }}>
                        {description}
                      </Typography>
                    )}
                  </Box>
                );
              })}
            </FormGroup>
          </Collapse>
        </Paper>
      )}

      {/* Render the bounding boxes via portal */}
      {renderBoundingBoxesPortal()}

      {/* Error/Success Snackbar */}
      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          severity={error?.severity || "error"} 
          onClose={() => setError(null)}
        >
          {error?.message || error}
        </Alert>
      </Snackbar>
    </>
  );
};

export default PdfProcessingResults;

// // Updated PdfProcessingResults.jsx with Fixed Bounding Box Rendering
// import React, { useState, useEffect, useRef } from 'react';
// import {
//   Button,
//   CircularProgress,
//   Snackbar,
//   Alert,
//   Box,
//   Typography,
//   FormGroup,
//   FormControlLabel,
//   Switch,
//   Chip,
//   Paper,
//   Collapse,
//   IconButton
// } from '@mui/material';
// import SearchIcon from '@mui/icons-material/Search';
// import CancelIcon from '@mui/icons-material/Cancel';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// import ExpandLessIcon from '@mui/icons-material/ExpandLess';
// import ReactDOM from 'react-dom';

// import BoundingBoxes from './BoundingBoxes';

// import {
//   transformCoord,
//   getColorMap,
//   getFormattedBoxesData,
//   getResultsCount
// } from './utils/pdfUtils';

// const PdfProcessingResults = ({
//   fileId,
//   stage,
//   pageDimensions,
//   currentPage,
//   pdfContainerRef,
//   scale = 1.5
// }) => {
//   const [loading, setLoading] = useState(false);
//   const [results, setResults] = useState([]);
//   const [error, setError] = useState(null);
//   const [showResults, setShowResults] = useState(true);
//   const [queryToggles, setQueryToggles] = useState({});
//   const [showQueryControls, setShowQueryControls] = useState(true);
//   const [boundingBoxesVisible, setBoundingBoxesVisible] = useState(false);

//   useEffect(() => {
//     // Log page dimensions to debug potential issues
//     console.log('Page dimensions:', pageDimensions);
//     console.log('PDF Container Ref:', pdfContainerRef?.current);
//   }, [pageDimensions, pdfContainerRef]);

//   const processPdf = async () => {
//     if (!fileId) {
//       setError("No file selected");
//       return;
//     }

//     setLoading(true);
//     setResults([]);
//     setError(null);
//     setQueryToggles({});
//     setBoundingBoxesVisible(false);

//     try {
//       console.log('Processing PDF with file_id:', fileId, 'stage:', stage);
//       const response = await fetch(`http://localhost:8000/tableai/extract/doc_query`, {
//         method: 'POST',
//         headers: { 'Content-Type': 'application/json' },
//         body: JSON.stringify({ file_id: fileId, stage: stage }),
//       });

//       if (!response.ok) {
//         throw new Error(`HTTP error! Status: ${response.status}`);
//       }

//       const data = await response.json();
//       console.log('Received query results:', data);
//       setResults(data);
//       setShowResults(true);

//       const initialToggles = {};
//       data.forEach(result => {
//         initialToggles[result.query_label] = true;
//       });
//       setQueryToggles(initialToggles);
//       setBoundingBoxesVisible(true);

//     } catch (err) {
//       console.error('Error processing PDF:', err);
//       setError(`Failed to process PDF: ${err.message}`);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const getColorForLabel = (() => {
//     const cache = {};
//     return (label) => {
//       if (cache[label]) return cache[label];
//       let hash = 0;
//       for (let i = 0; i < label.length; i++) {
//         hash = label.charCodeAt(i) + ((hash << 5) - hash);
//       }
//       // Wide hue spread
//       const hue = Math.abs(hash) % 360;
//       // Vary saturation and lightness for even more separation
//       const saturation = 60 + (Math.abs(hash) % 30); // 60% - 89%
//       const lightness = 45 + (Math.abs(hash * 7) % 30); // 45% - 74%
//       const color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
//       cache[label] = color;
//       return color;
//     };
//   })();
  
//   // In your main component render:
//   const colorMap = {};
//   results.forEach(r => {
//     colorMap[r.query_label] = getColorForLabel(r.query_label);
//   });

//   const toggleResults = () => {
//     setShowResults(!showResults);
//     // Update bounding box visibility when toggling results
//     setBoundingBoxesVisible(!showResults);
//   };

//   const toggleQueryLabel = (queryLabel) => {
//     setQueryToggles(prev => ({
//       ...prev,
//       [queryLabel]: !prev[queryLabel]
//     }));
//   };

//   const toggleAllQueryLabels = (value) => {
//     const updated = {};
//     Object.keys(queryToggles).forEach(key => {
//       updated[key] = value;
//     });
//     setQueryToggles(updated);
//   };

//   const getFilteredBoxesData = () => {
//     if (!results || results.length === 0 || !showResults) return null;
    
//     // Log current page for debugging
//     console.log('Getting boxes for current page:', currentPage);
    
//     try {
//       // This is a direct implementation using your response structure
//       const formattedData = {};
      
//       results.forEach(result => {
//         const queryLabel = result.query_label;
//         const pages = result.results?.pages;

//         if (queryToggles[queryLabel] === false) {
//             console.log(`Query ${queryLabel} is toggled off, skipping`);
//             return;
//           }
        
//         if (!pages) {
//           console.log(`No pages data found for ${queryLabel}`);
//           return;
//         }
        
//         // Convert to zero-based indexing (your currentPage is 1-based)
//         const pageIndex = String(currentPage - 1);
        
//         console.log(`Looking for page ${pageIndex} in`, Object.keys(pages));
        
//         const pageBoxes = pages[pageIndex];
//         if (!pageBoxes || !pageBoxes.length) {
//           console.log(`No boxes found for page ${pageIndex} in ${queryLabel}`);
//           return;
//         }
        
//         console.log(`Found ${pageBoxes.length} boxes for ${queryLabel} on page ${pageIndex}`);
        
//         formattedData[queryLabel] = pageBoxes.map(box => {
//           // Check if we have bbox or use x0,y0,x1,y1
//           const coords = box.bbox || [box.x0, box.y0, box.x1, box.y1];
          
//           return {
//             ...box,
//             coords,
//             text: box.value || '',
//             queryLabel
//           };
//         });
//       });
      
//       console.log('Formatted boxes data:', formattedData);
      
//       // Return null if no boxes found
//       return Object.keys(formattedData).length > 0 ? formattedData : null;
//     } catch (error) {
//       console.error('Error formatting boxes data:', error);
//       return null;
//     }
//   };

//   const getQueryLabelResultsCount = (label) => {
//     const queryResult = results.find(r => r.query_label === label);
//     if (!queryResult || !queryResult.results?.pages) return 0;
    
//     // Adjust for zero-based indexing in the results
//     const pageIndex = String(currentPage - 1);
//     return (queryResult.results.pages[pageIndex] || []).length;
//   };

// //   const getQueryLabelColor = (queryLabel) => {
// //     const colorMap = getColorMap();
// //     return colorMap[queryLabel] || colorMap.default;
// //   };

//   // Fixed transformCoord function for PDFs
//   const transformCoordWithContainer = (coords, color) => {
//     // Make sure we're working with the correct page dimensions
//     if (!pageDimensions.width || !pageDimensions.height) {
//       console.warn('Page dimensions not available:', pageDimensions);
//       return { x: 0, y: 0, width: 0, height: 0, color };
//     }
    
//     // Simple coordinate transformation
//     // [x0, y0, x1, y1] to CSS positioning
//     let [x0, y0, x1, y1] = coords;
    
//     // Convert to numbers if they're strings
//     x0 = parseFloat(x0);
//     y0 = parseFloat(y0);
//     x1 = parseFloat(x1);
//     y1 = parseFloat(y1);
    
//     // Fix any invalid coordinates (ensure x1 > x0 and y1 > y0)
//     if (x1 < x0) [x0, x1] = [x1, x0];
//     if (y1 < y0) [y0, y1] = [y1, y0];
    
//     console.log('Original coords:', [x0, y0, x1, y1]);
//     console.log('Page dimensions:', pageDimensions);
//     console.log('Scale:', scale);
    
//     // Get the actual rendered dimensions of the PDF
//     const renderedWidth = pageDimensions.width;
//     const renderedHeight = pageDimensions.height;
    
//     // Get the PDF's intrinsic dimensions (from metadata)
//     // Find these in the PDF metadata or extract from API response
//     const pdfWidth = 612.0; // From your API response metadata
//     const pdfHeight = 792.0; // From your API response metadata
    
//     // Calculate the ratio between rendered and original PDF dimensions
//     const widthRatio = renderedWidth / pdfWidth;
//     const heightRatio = renderedHeight / pdfHeight;
    
//     // Apply coordinate transformation with correct scaling
//     // PDF coordinates start from bottom-left, CSS starts from top-left
//     const x = x0 * widthRatio;
//     const y = y0 * heightRatio; // No flipping needed based on your coordinates
//     const width = (x1 - x0) * widthRatio;
//     const height = (y1 - y0) * heightRatio;
    
//     const transformedCoords = { x, y, width, height };
//     console.log('Transformed coords:', transformedCoords);
    
//     return {
//       position: 'absolute',
//       left: `${x}px`,
//       top: `${y}px`,
//       width: `${width}px`,
//       height: `${height}px`,
//       border: `2px solid ${color}`,
//       backgroundColor: color.replace('1)', '0.3)'),
//       zIndex: 2000,
//       pointerEvents: 'auto',
//       opacity: 1
//     };
//   };

// //   const colorMap = getColorMap();
//   const boxesData = getFilteredBoxesData();
//   const resultsCount = boxesData ? getResultsCount(boxesData) : 0;
  
//   const canRenderBoxes = pdfContainerRef?.current && 
//                         pageDimensions.width > 0 && 
//                         pageDimensions.height > 0 && 
//                         boundingBoxesVisible &&
//                         boxesData;

//   // Render bounding boxes directly into the PDF container using a portal
//   const renderBoundingBoxesPortal = () => {
//     if (!canRenderBoxes) {
//       console.log('Cannot render boxes yet:', { 
//         containerExists: !!pdfContainerRef?.current,
//         pageDimensions,
//         boundingBoxesVisible,
//         hasBoxesData: !!boxesData
//       });
//       return null;
//     }
    
//     console.log('Rendering bounding boxes portal with dimensions:', pageDimensions);
    
//     // Create a dedicated container for the bounding boxes if it doesn't exist
//     let boxContainer = document.getElementById('bounding-box-container');
//     if (!boxContainer) {
//       boxContainer = document.createElement('div');
//       boxContainer.id = 'bounding-box-container';
//       boxContainer.style.position = 'absolute';
//       boxContainer.style.top = '0';
//       boxContainer.style.left = '0';
//       boxContainer.style.width = '100%';
//       boxContainer.style.height = '100%';
//       boxContainer.style.pointerEvents = 'none';
//       boxContainer.style.zIndex = '1000'; // Ensure it's on top
      
//       // Add it to the PDF container
//       pdfContainerRef.current.appendChild(boxContainer);
//     }
    
//     return ReactDOM.createPortal(
//       <BoundingBoxes
//         boxesData={boxesData}
//         transformCoord={(coords, color) => transformCoordWithContainer(coords, color)}
//         colorMap={colorMap}
//         showTooltips={true}
//       />,
//       boxContainer
//     );
//   };

//   return (
//     <>
//       {/* Control Panel */}
//       <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
//         <Button
//           variant="contained"
//           color="primary"
//           onClick={processPdf}
//           disabled={loading || !fileId}
//           startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
//         >
//           {loading ? "Processing..." : "Process PDF"}
//         </Button>

//         {results.length > 0 && (
//           <Button
//             variant="outlined"
//             color="primary"
//             onClick={toggleResults}
//             startIcon={showResults ? <CancelIcon /> : <SearchIcon />}
//           >
//             {showResults ? "Hide Results" : "Show Results"}
//           </Button>
//         )}

//         {results.length > 0 && (
//           <Typography variant="body2" color="textSecondary">
//             Found {resultsCount} results on page {currentPage}
//           </Typography>
//         )}
//       </Box>

//       {/* Query Filters */}
//       {results.length > 0 && (
//         <Paper elevation={1} sx={{
//           p: 2, mb: 2,
//           bgcolor: '#f5f5f5',
//           border: '1px solid #e0e0e0',
//           borderRadius: '4px'
//         }}>
//           <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
//             <Typography variant="subtitle1" fontWeight="bold">
//               Query Types
//             </Typography>
//             <Box>
//               <Button size="small" onClick={() => toggleAllQueryLabels(true)} sx={{ mr: 1 }}>Show All</Button>
//               <Button size="small" onClick={() => toggleAllQueryLabels(false)}>Hide All</Button>
//               <IconButton size="small" onClick={() => setShowQueryControls(prev => !prev)} sx={{ ml: 1 }}>
//                 {showQueryControls ? <ExpandLessIcon /> : <ExpandMoreIcon />}
//               </IconButton>
//             </Box>
//           </Box>

//           <Collapse in={showQueryControls}>
//             <FormGroup sx={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: 1 }}>
//               {results.map(result => {
//                 const label = result.query_label;
//                 const description = result?.description || '';
//                 const count = getQueryLabelResultsCount(label);
//                 const color = getColorForLabel(label);

//                 return (
//                   <Box key={label} sx={{
//                     display: 'flex',
//                     flexDirection: 'column',
//                     bgcolor: 'white',
//                     p: 1,
//                     borderRadius: '4px',
//                     border: '1px solid #e0e0e0',
//                     minWidth: '200px'
//                   }}>
//                     <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
//                       <FormControlLabel
//                         control={
//                           <Switch
//                             checked={queryToggles[label] !== false}
//                             onChange={() => toggleQueryLabel(label)}
//                             size="small"
//                           />
//                         }
//                         label={<Typography variant="body2" fontWeight="bold">{label}</Typography>}
//                       />
//                     <Chip
//                         label={count}
//                         size="small"
//                         sx={{
//                             bgcolor: getColorForLabel(label) + '20', // 20 = 12% opacity in hex
//                             border: `1px solid ${getColorForLabel(label)}`
//                         }}
//                         />
//                     </Box>
//                     {description && (
//                       <Typography variant="caption" sx={{ mt: 0.5, color: 'text.secondary' }}>
//                         {description}
//                       </Typography>
//                     )}
//                   </Box>
//                 );
//               })}
//             </FormGroup>
//           </Collapse>
//         </Paper>
//       )}

//       {/* Render the bounding boxes via portal */}
//       {renderBoundingBoxesPortal()}

//       {/* Error Snackbar */}
//       <Snackbar
//         open={!!error}
//         autoHideDuration={6000}
//         onClose={() => setError(null)}
//         anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
//       >
//         <Alert severity="error" onClose={() => setError(null)}>
//           {error}
//         </Alert>
//       </Snackbar>
//     </>
//   );
// };

// export default PdfProcessingResults;