import React, { useState, useEffect } from 'react';
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
  Tooltip,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import CancelIcon from '@mui/icons-material/Cancel';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import VisibilityIcon from '@mui/icons-material/Visibility';
import ReactDOM from 'react-dom';

import BoundingBoxes from './BoundingBoxes';
import { getResultsCount } from './utils/pdfUtils';

import { usePdfDataContext } from '../../context/PdfDataContext';
import FileMetadataDisplay from './FileMetadataDisplay';

// For debugging
const DEBUG = false;

const PdfProcessingResults = ({
  fileId,
  stage, // For backward compatibility
  selectedStage, // New prop to determine which stage results to display
  pageDimensions,
  currentPage,
  pdfContainerRef,
  scale = 1.5,
  metadata = {},
}) => {
  // Use selectedStage if provided, otherwise fall back to stage for backward compatibility
  const effectiveStage = selectedStage !== undefined ? selectedStage : stage;
  
  const {
    processingResults,
    processingLoading,
    visionLoading,
    error,
    handleProcessPdf,
    handleRunVisionInference,
    clearAllResults,
    reloadMetadata,
  } = usePdfDataContext();

  // Local state
  const [showResults, setShowResults] = useState(true);
  const [queryToggles, setQueryToggles] = useState({});
  const [showQueryControls, setShowQueryControls] = useState(true);
  const [boundingBoxesVisible, setBoundingBoxesVisible] = useState(false);
  const [localError, setLocalError] = useState(null);

  // Clean up the container when component unmounts
  useEffect(() => {
    return () => {
      const boxContainer = document.getElementById('bounding-box-container');
      if (boxContainer && boxContainer.parentNode) {
        boxContainer.parentNode.removeChild(boxContainer);
      }
    };
  }, []);

  // On error from hook, show snackbar
  useEffect(() => {
    if (error) setLocalError(error);
  }, [error]);

  // Get the current stage results
  const getCurrentStageResults = () => {
    if (!processingResults) return [];
    
    if (DEBUG) {
      console.log("Processing Results:", processingResults);
      console.log("Effective Stage:", effectiveStage);
    }
    
    // If processingResults is the new dictionary format
    if (processingResults[`stage${effectiveStage}`]) {
      return processingResults[`stage${effectiveStage}`] || [];
    }
    
    // Fallback for backward compatibility
    return Array.isArray(processingResults) ? processingResults : [];
  };

  const currentStageResults = getCurrentStageResults();
  
  if (DEBUG) {
    console.log("Current Stage Results:", currentStageResults);
  }

  // Update toggles/results on new process results or stage change
  useEffect(() => {
    if (!currentStageResults || currentStageResults.length === 0) return;
    
    if (DEBUG) {
      console.log("Setting up toggles for results:", currentStageResults);
    }
    
    const initialToggles = {};
    currentStageResults.forEach((result) => {
      initialToggles[result.query_label] = true;
    });
    setQueryToggles(initialToggles);
    setBoundingBoxesVisible(true);
    setShowResults(true);
  }, [processingResults, effectiveStage]);

  // Force bounding box rerender on toggle change
  useEffect(() => {
    if (showResults && Object.keys(queryToggles).length > 0) {
      setBoundingBoxesVisible(false);
      const timer = setTimeout(() => setBoundingBoxesVisible(true), 50);
      return () => clearTimeout(timer);
    }
  }, [queryToggles, showResults]);

  // Check if file has classification
  const hasClassification = !!metadata?.classification;

  const onProcessPdf = async () => {
    if (!fileId) {
      setLocalError("No file selected");
      return;
    }
    clearAllResults();
    setBoundingBoxesVisible(false);
    try {
      // No longer passing stage parameter
      await handleProcessPdf({ fileId });
      await reloadMetadata(); 
    } catch (err) {
      setLocalError(err.message);
    }
  };
  
  const onRunVisionInference = async () => {
    if (!fileId || !hasClassification) {
      setLocalError("Cannot run vision inference - file must have a classification");
      return;
    }
    try {
      await handleRunVisionInference({
        fileId,
        // No longer passing stage parameter
        classificationLabel: metadata.classification,
      });
      setLocalError({ message: "Vision inference completed successfully!", severity: "success" });
      await reloadMetadata(); 
    } catch (err) {
      setLocalError(err.message);
    }
  };

  // --- Custom color for query label ---
  const getColorForLabel = (() => {
    const cache = {};
    return (label) => {
      if (cache[label]) return cache[label];
      let hash = 0;
      for (let i = 0; i < label.length; i++) {
        hash = label.charCodeAt(i) + ((hash << 5) - hash);
      }
      const hue = Math.abs(hash) % 360;
      const saturation = 60 + (Math.abs(hash) % 30);
      const lightness = 45 + (Math.abs(hash * 7) % 30);
      const color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
      cache[label] = color;
      return color;
    };
  })();

  // Map for label -> color
  const colorMap = {};
  (currentStageResults || []).forEach((r) => {
    colorMap[r.query_label] = getColorForLabel(r.query_label);
  });

  // --- Toggle logic ---
  const toggleResults = () => {
    setShowResults((prev) => !prev);
    setBoundingBoxesVisible((prev) => !prev);
  };

  const toggleQueryLabel = (queryLabel) => {
    setQueryToggles((prev) => ({
      ...prev,
      [queryLabel]: !prev[queryLabel],
    }));
    setBoundingBoxesVisible((prev) => {
      if (prev) {
        setTimeout(() => setBoundingBoxesVisible(true), 10);
        return false;
      }
      return true;
    });
  };

  const toggleAllQueryLabels = (value) => {
    const updated = {};
    Object.keys(queryToggles).forEach((key) => {
      updated[key] = value;
    });
    setQueryToggles(updated);
  };

  // --- Get PDF dimensions from metadata in the results ---
  const getPdfDimensions = () => {
    if (currentStageResults && currentStageResults.length > 0) {
      // Check each result for pdf_metadata or item metadata
      for (const result of currentStageResults) {
        // First, check if there's a top-level pdf_metadata field
        if (result.pdf_metadata && result.pdf_metadata[currentPage - 1]) {
          return {
            width: result.pdf_metadata[currentPage - 1].width,
            height: result.pdf_metadata[currentPage - 1].height
          };
        }
        
        // Then, check page items for metadata
        const pageIndex = String(currentPage - 1);
        const pages = result.results?.pages;
        if (pages && pages[pageIndex] && pages[pageIndex].length > 0) {
          // Check if any box has metadata
          for (const box of pages[pageIndex]) {
            if (box.meta && box.meta.width && box.meta.height) {
              return {
                width: box.meta.width,
                height: box.meta.height
              };
            }
          }
        }
      }
    }
    
    // Fallback to provided dimensions
    return pageDimensions;
  };

  // --- Filtered boxes data for bounding box display ---
  const getFilteredBoxesData = () => {
    if (!currentStageResults || currentStageResults.length === 0 || !showResults) return null;
    try {
      const formattedData = {};
      currentStageResults.forEach((result) => {
        const queryLabel = result.query_label;
        if (queryToggles[queryLabel] === false) return;
        const pages = result.results?.pages;
        if (!pages) return;
        const pageIndex = String(currentPage - 1);
        const pageBoxes = pages[pageIndex];
        if (!pageBoxes || !pageBoxes.length) return;
        formattedData[queryLabel] = pageBoxes.map((box) => {
          const coords = box.bbox || [box.x0, box.y0, box.x1, box.y1];
          return { ...box, coords, text: box.value || '', queryLabel };
        });
      });
      return Object.keys(formattedData).length > 0 ? formattedData : null;
    } catch (error) {
      console.error('Error formatting boxes data:', error);
      return null;
    }
  };

  // --- Results count helper ---
  const getQueryLabelResultsCount = (label) => {
    const queryResult = (currentStageResults || []).find((r) => r.query_label === label);
    if (!queryResult || !queryResult.results?.pages) return 0;
    const pageIndex = String(currentPage - 1);
    return (queryResult.results.pages[pageIndex] || []).length;
  };

  // --- Transform bounding box coordinates ---
  const transformCoordWithContainer = (coords, color) => {
    if (!pageDimensions.width || !pageDimensions.height) {
      return { x: 0, y: 0, width: 0, height: 0, color };
    }
    
    let [x0, y0, x1, y1] = coords.map(parseFloat);
    if (x1 < x0) [x0, x1] = [x1, x0];
    if (y1 < y0) [y0, y1] = [y1, y0];
    
    const renderedWidth = pageDimensions.width;
    const renderedHeight = pageDimensions.height;
    
    // Get PDF dimensions dynamically
    const pdfDimensions = getPdfDimensions();
    const pdfWidth = pdfDimensions.width || 612.0;
    const pdfHeight = pdfDimensions.height || 792.0;
    
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
      backgroundColor: color.replace('1)', '0.3)').replace(')', ', 0.3)'),
      zIndex: 2000,
      pointerEvents: 'auto',
      opacity: 1,
    };
  };

  const boxesData = getFilteredBoxesData();
  const resultsCount = boxesData ? getResultsCount(boxesData) : 0;
  const canRenderBoxes =
    pdfContainerRef?.current &&
    pageDimensions.width > 0 &&
    pageDimensions.height > 0 &&
    boundingBoxesVisible &&
    boxesData;

  // --- Render bounding boxes using a portal ---
  const renderBoundingBoxesPortal = () => {
    if (!canRenderBoxes) {
      if (DEBUG) {
        console.log("Cannot render boxes:", {
          containerExists: !!pdfContainerRef?.current,
          dimensions: pageDimensions,
          boundingBoxesVisible,
          hasBoxesData: !!boxesData
        });
      }
      return null;
    }
    
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
      boxContainer.style.zIndex = '1000';
      boxContainer.style.border = 'none';
      boxContainer.style.background = 'transparent';
      boxContainer.style.boxShadow = 'none';
      boxContainer.style.overflow = 'hidden';
      
      // Ensure parent has position: relative
      pdfContainerRef.current.style.position = 'relative';
      pdfContainerRef.current.appendChild(boxContainer);
    }
    
    return ReactDOM.createPortal(
      <BoundingBoxes
        boxesData={boxesData}
        transformCoord={transformCoordWithContainer}
        colorMap={colorMap}
        showTooltips={true}
      />,
      boxContainer
    );
  };

  // Check if there are results for the current stage
  const hasCurrentStageResults = currentStageResults && currentStageResults.length > 0;

  return (
    <>
      {/* Process Controls */}
      <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
        <FileMetadataDisplay metadata={metadata} />
        {DEBUG && (
          <Typography variant="caption" sx={{ bgcolor: 'yellow' }}>
            Stage: {effectiveStage}, Results: {hasCurrentStageResults ? 'Yes' : 'No'}
          </Typography>
        )}
        <Button
          variant="contained"
          color="primary"
          onClick={onProcessPdf}
          disabled={processingLoading || !fileId}
          startIcon={processingLoading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
        >
          {processingLoading ? "Processing..." : "Process PDF"}
        </Button>

        <Tooltip title={!hasClassification ? "File must have a classification to run vision inference" : ""}>
          <span>
            <Button
              variant="contained"
              color="secondary"
              onClick={onRunVisionInference}
              disabled={visionLoading || !fileId || !hasClassification}
              startIcon={visionLoading ? <CircularProgress size={20} color="inherit" /> : <VisibilityIcon />}
            >
              {visionLoading ? "Running..." : "Run Vision Inference"}
            </Button>
          </span>
        </Tooltip>

        {hasCurrentStageResults && (
          <Button
            variant="outlined"
            color="primary"
            onClick={toggleResults}
            startIcon={showResults ? <CancelIcon /> : <SearchIcon />}
          >
            {showResults ? "Hide Results" : "Show Results"}
          </Button>
        )}

        {hasCurrentStageResults && (
          <Typography variant="body2" color="textSecondary">
            Found {resultsCount} results on page {currentPage} (Stage {effectiveStage})
          </Typography>
        )}
      </Box>

      {/* Query Filters */}
      {hasCurrentStageResults && (
        <Paper
          elevation={1}
          sx={{
            p: 2,
            mb: 2,
            bgcolor: '#f5f5f5',
            border: '1px solid #e0e0e0',
            borderRadius: '4px',
          }}
        >
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="subtitle1" fontWeight="bold">
              Query Types
            </Typography>
            <Box>
              <Button size="small" onClick={() => toggleAllQueryLabels(true)} sx={{ mr: 1 }}>
                Show All
              </Button>
              <Button size="small" onClick={() => toggleAllQueryLabels(false)}>
                Hide All
              </Button>
              <IconButton size="small" onClick={() => setShowQueryControls((prev) => !prev)} sx={{ ml: 1 }}>
                {showQueryControls ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              </IconButton>
            </Box>
          </Box>
          <Collapse in={showQueryControls}>
            <FormGroup sx={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: 1 }}>
              {currentStageResults.map((result) => {
                const label = result.query_label;
                const description = result?.description || '';
                const count = getQueryLabelResultsCount(label);
                const color = getColorForLabel(label);

                return (
                  <Box
                    key={label}
                    sx={{
                      display: 'flex',
                      flexDirection: 'column',
                      bgcolor: 'white',
                      p: 1,
                      borderRadius: '4px',
                      border: '1px solid #e0e0e0',
                      minWidth: '200px',
                    }}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <FormControlLabel
                        control={
                          <Switch
                            checked={queryToggles[label] !== false}
                            onChange={() => toggleQueryLabel(label)}
                            size="small"
                          />
                        }
                        label={
                          <Typography variant="body2" fontWeight="bold">
                            {label}
                          </Typography>
                        }
                      />
                      <Chip
                        label={count}
                        size="small"
                        sx={{
                          bgcolor: getColorForLabel(label) + '20',
                          border: `1px solid ${getColorForLabel(label)}`,
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
        open={!!localError}
        autoHideDuration={6000}
        onClose={() => setLocalError(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert severity={localError?.severity || "error"} onClose={() => setLocalError(null)}>
          {localError?.message || localError}
        </Alert>
      </Snackbar>
    </>
  );
};

export default PdfProcessingResults;

// import React, { useState, useEffect } from 'react';
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
//   IconButton,
//   Tooltip,
// } from '@mui/material';
// import SearchIcon from '@mui/icons-material/Search';
// import CancelIcon from '@mui/icons-material/Cancel';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// import ExpandLessIcon from '@mui/icons-material/ExpandLess';
// import VisibilityIcon from '@mui/icons-material/Visibility';
// import ReactDOM from 'react-dom';

// import BoundingBoxes from './BoundingBoxes';
// import { getResultsCount } from './utils/pdfUtils';

// import { usePdfDataContext } from '../../context/PdfDataContext';
// import FileMetadataDisplay from './FileMetadataDisplay';

// const PdfProcessingResults = ({
//   fileId,
//   selectedStage, // New prop to determine which stage results to display
//   pageDimensions,
//   currentPage,
//   pdfContainerRef,
//   scale = 1.5,
//   metadata = {},
// }) => {
//   const {
//     processingResults,
//     processingLoading,
//     visionLoading,
//     error,
//     handleProcessPdf,
//     handleRunVisionInference,
//     clearAllResults,
//     reloadMetadata,
//   } = usePdfDataContext();

//   // Local state
//   const [showResults, setShowResults] = useState(true);
//   const [queryToggles, setQueryToggles] = useState({});
//   const [showQueryControls, setShowQueryControls] = useState(true);
//   const [boundingBoxesVisible, setBoundingBoxesVisible] = useState(false);
//   const [localError, setLocalError] = useState(null);

//   // Clean up the container when component unmounts
//   useEffect(() => {
//     return () => {
//       const boxContainer = document.getElementById('bounding-box-container');
//       if (boxContainer && boxContainer.parentNode) {
//         boxContainer.parentNode.removeChild(boxContainer);
//       }
//     };
//   }, []);

//   // On error from hook, show snackbar
//   useEffect(() => {
//     if (error) setLocalError(error);
//   }, [error]);

//   // Get the current stage results
//   const getCurrentStageResults = () => {
//     if (!processingResults) return [];
    
//     // If processingResults is the new dictionary format
//     if (processingResults[`stage${selectedStage}`]) {
//       return processingResults[`stage${selectedStage}`] || [];
//     }
    
//     // Fallback for backward compatibility
//     return Array.isArray(processingResults) ? processingResults : [];
//   };

//   const currentStageResults = getCurrentStageResults();

//   // Update toggles/results on new process results or stage change
//   useEffect(() => {
//     if (!currentStageResults || currentStageResults.length === 0) return;
    
//     const initialToggles = {};
//     currentStageResults.forEach((result) => {
//       initialToggles[result.query_label] = true;
//     });
//     setQueryToggles(initialToggles);
//     setBoundingBoxesVisible(true);
//     setShowResults(true);
//   }, [processingResults, selectedStage]);

//   // Force bounding box rerender on toggle change
//   useEffect(() => {
//     if (showResults && Object.keys(queryToggles).length > 0) {
//       setBoundingBoxesVisible(false);
//       const timer = setTimeout(() => setBoundingBoxesVisible(true), 50);
//       return () => clearTimeout(timer);
//     }
//   }, [queryToggles, showResults]);

//   // Check if file has classification
//   const hasClassification = !!metadata?.classification;

//   const onProcessPdf = async () => {
//     if (!fileId) {
//       setLocalError("No file selected");
//       return;
//     }
//     clearAllResults();
//     setBoundingBoxesVisible(false);
//     try {
//       // No longer passing stage parameter
//       await handleProcessPdf({ fileId });
//       await reloadMetadata(); 
//     } catch (err) {
//       setLocalError(err.message);
//     }
//   };
  
//   const onRunVisionInference = async () => {
//     if (!fileId || !hasClassification) {
//       setLocalError("Cannot run vision inference - file must have a classification");
//       return;
//     }
//     try {
//       await handleRunVisionInference({
//         fileId,
//         // No longer passing stage parameter
//         classificationLabel: metadata.classification,
//       });
//       setLocalError({ message: "Vision inference completed successfully!", severity: "success" });
//       await reloadMetadata(); 
//     } catch (err) {
//       setLocalError(err.message);
//     }
//   };

//   // --- Custom color for query label ---
//   const getColorForLabel = (() => {
//     const cache = {};
//     return (label) => {
//       if (cache[label]) return cache[label];
//       let hash = 0;
//       for (let i = 0; i < label.length; i++) {
//         hash = label.charCodeAt(i) + ((hash << 5) - hash);
//       }
//       const hue = Math.abs(hash) % 360;
//       const saturation = 60 + (Math.abs(hash) % 30);
//       const lightness = 45 + (Math.abs(hash * 7) % 30);
//       const color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
//       cache[label] = color;
//       return color;
//     };
//   })();

//   // Map for label -> color
//   const colorMap = {};
//   (currentStageResults || []).forEach((r) => {
//     colorMap[r.query_label] = getColorForLabel(r.query_label);
//   });

//   // --- Toggle logic ---
//   const toggleResults = () => {
//     setShowResults((prev) => !prev);
//     setBoundingBoxesVisible((prev) => !prev);
//   };

//   const toggleQueryLabel = (queryLabel) => {
//     setQueryToggles((prev) => ({
//       ...prev,
//       [queryLabel]: !prev[queryLabel],
//     }));
//     setBoundingBoxesVisible((prev) => {
//       if (prev) {
//         setTimeout(() => setBoundingBoxesVisible(true), 10);
//         return false;
//       }
//       return true;
//     });
//   };

//   const toggleAllQueryLabels = (value) => {
//     const updated = {};
//     Object.keys(queryToggles).forEach((key) => {
//       updated[key] = value;
//     });
//     setQueryToggles(updated);
//   };

//   // --- Get PDF dimensions from metadata in the results ---
//   const getPdfDimensions = () => {
//     if (currentStageResults && currentStageResults.length > 0) {
//       // Check each result for pdf_metadata or item metadata
//       for (const result of currentStageResults) {
//         // First, check if there's a top-level pdf_metadata field
//         if (result.pdf_metadata && result.pdf_metadata[currentPage - 1]) {
//           return {
//             width: result.pdf_metadata[currentPage - 1].width,
//             height: result.pdf_metadata[currentPage - 1].height
//           };
//         }
        
//         // Then, check page items for metadata
//         const pageIndex = String(currentPage - 1);
//         const pages = result.results?.pages;
//         if (pages && pages[pageIndex] && pages[pageIndex].length > 0) {
//           // Check if any box has metadata
//           for (const box of pages[pageIndex]) {
//             if (box.meta && box.meta.width && box.meta.height) {
//               return {
//                 width: box.meta.width,
//                 height: box.meta.height
//               };
//             }
//           }
//         }
//       }
//     }
    
//     // Fallback to provided dimensions
//     return pageDimensions;
//   };

//   // --- Filtered boxes data for bounding box display ---
//   const getFilteredBoxesData = () => {
//     if (!currentStageResults || currentStageResults.length === 0 || !showResults) return null;
//     try {
//       const formattedData = {};
//       currentStageResults.forEach((result) => {
//         const queryLabel = result.query_label;
//         if (queryToggles[queryLabel] === false) return;
//         const pages = result.results?.pages;
//         if (!pages) return;
//         const pageIndex = String(currentPage - 1);
//         const pageBoxes = pages[pageIndex];
//         if (!pageBoxes || !pageBoxes.length) return;
//         formattedData[queryLabel] = pageBoxes.map((box) => {
//           const coords = box.bbox || [box.x0, box.y0, box.x1, box.y1];
//           return { ...box, coords, text: box.value || '', queryLabel };
//         });
//       });
//       return Object.keys(formattedData).length > 0 ? formattedData : null;
//     } catch (error) {
//       console.error('Error formatting boxes data:', error);
//       return null;
//     }
//   };

//   // --- Results count helper ---
//   const getQueryLabelResultsCount = (label) => {
//     const queryResult = (currentStageResults || []).find((r) => r.query_label === label);
//     if (!queryResult || !queryResult.results?.pages) return 0;
//     const pageIndex = String(currentPage - 1);
//     return (queryResult.results.pages[pageIndex] || []).length;
//   };

//   // --- Transform bounding box coordinates ---
//   const transformCoordWithContainer = (coords, color) => {
//     if (!pageDimensions.width || !pageDimensions.height) {
//       return { x: 0, y: 0, width: 0, height: 0, color };
//     }
    
//     let [x0, y0, x1, y1] = coords.map(parseFloat);
//     if (x1 < x0) [x0, x1] = [x1, x0];
//     if (y1 < y0) [y0, y1] = [y1, y0];
    
//     const renderedWidth = pageDimensions.width;
//     const renderedHeight = pageDimensions.height;
    
//     // Get PDF dimensions dynamically
//     const pdfDimensions = getPdfDimensions();
//     const pdfWidth = pdfDimensions.width || 612.0;
//     const pdfHeight = pdfDimensions.height || 792.0;
    
//     const widthRatio = renderedWidth / pdfWidth;
//     const heightRatio = renderedHeight / pdfHeight;
    
//     const x = x0 * widthRatio;
//     const y = y0 * heightRatio;
//     const width = (x1 - x0) * widthRatio;
//     const height = (y1 - y0) * heightRatio;
    
//     return {
//       position: 'absolute',
//       left: `${x}px`,
//       top: `${y}px`,
//       width: `${width}px`,
//       height: `${height}px`,
//       border: `2px solid ${color}`,
//       backgroundColor: color.replace('1)', '0.3)').replace(')', ', 0.3)'),
//       zIndex: 2000,
//       pointerEvents: 'auto',
//       opacity: 1,
//     };
//   };

//   const boxesData = getFilteredBoxesData();
//   const resultsCount = boxesData ? getResultsCount(boxesData) : 0;
//   const canRenderBoxes =
//     pdfContainerRef?.current &&
//     pageDimensions.width > 0 &&
//     pageDimensions.height > 0 &&
//     boundingBoxesVisible &&
//     boxesData;

//   // --- Render bounding boxes using a portal ---
//   const renderBoundingBoxesPortal = () => {
//     if (!canRenderBoxes) return null;
    
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
//       boxContainer.style.zIndex = '1000';
//       boxContainer.style.border = 'none';
//       boxContainer.style.background = 'transparent';
//       boxContainer.style.boxShadow = 'none';
//       boxContainer.style.overflow = 'hidden';
      
//       // Ensure parent has position: relative
//       pdfContainerRef.current.style.position = 'relative';
//       pdfContainerRef.current.appendChild(boxContainer);
//     }
    
//     return ReactDOM.createPortal(
//       <BoundingBoxes
//         boxesData={boxesData}
//         transformCoord={transformCoordWithContainer}
//         colorMap={colorMap}
//         showTooltips={true}
//       />,
//       boxContainer
//     );
//   };

//   // Check if there are results for the current stage
//   const hasCurrentStageResults = currentStageResults && currentStageResults.length > 0;

//   return (
//     <>
//       {/* Process Controls */}
//       <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
//         <FileMetadataDisplay metadata={metadata} />
//         <Button
//           variant="contained"
//           color="primary"
//           onClick={onProcessPdf}
//           disabled={processingLoading || !fileId}
//           startIcon={processingLoading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
//         >
//           {processingLoading ? "Processing..." : "Process PDF"}
//         </Button>

//         <Tooltip title={!hasClassification ? "File must have a classification to run vision inference" : ""}>
//           <span>
//             <Button
//               variant="contained"
//               color="secondary"
//               onClick={onRunVisionInference}
//               disabled={visionLoading || !fileId || !hasClassification}
//               startIcon={visionLoading ? <CircularProgress size={20} color="inherit" /> : <VisibilityIcon />}
//             >
//               {visionLoading ? "Running..." : "Run Vision Inference"}
//             </Button>
//           </span>
//         </Tooltip>

//         {hasCurrentStageResults && (
//           <Button
//             variant="outlined"
//             color="primary"
//             onClick={toggleResults}
//             startIcon={showResults ? <CancelIcon /> : <SearchIcon />}
//           >
//             {showResults ? "Hide Results" : "Show Results"}
//           </Button>
//         )}

//         {hasCurrentStageResults && (
//           <Typography variant="body2" color="textSecondary">
//             Found {resultsCount} results on page {currentPage} (Stage {selectedStage})
//           </Typography>
//         )}
//       </Box>

//       {/* Query Filters */}
//       {hasCurrentStageResults && (
//         <Paper
//           elevation={1}
//           sx={{
//             p: 2,
//             mb: 2,
//             bgcolor: '#f5f5f5',
//             border: '1px solid #e0e0e0',
//             borderRadius: '4px',
//           }}
//         >
//           <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
//             <Typography variant="subtitle1" fontWeight="bold">
//               Query Types
//             </Typography>
//             <Box>
//               <Button size="small" onClick={() => toggleAllQueryLabels(true)} sx={{ mr: 1 }}>
//                 Show All
//               </Button>
//               <Button size="small" onClick={() => toggleAllQueryLabels(false)}>
//                 Hide All
//               </Button>
//               <IconButton size="small" onClick={() => setShowQueryControls((prev) => !prev)} sx={{ ml: 1 }}>
//                 {showQueryControls ? <ExpandLessIcon /> : <ExpandMoreIcon />}
//               </IconButton>
//             </Box>
//           </Box>
//           <Collapse in={showQueryControls}>
//             <FormGroup sx={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: 1 }}>
//               {currentStageResults.map((result) => {
//                 const label = result.query_label;
//                 const description = result?.description || '';
//                 const count = getQueryLabelResultsCount(label);
//                 const color = getColorForLabel(label);

//                 return (
//                   <Box
//                     key={label}
//                     sx={{
//                       display: 'flex',
//                       flexDirection: 'column',
//                       bgcolor: 'white',
//                       p: 1,
//                       borderRadius: '4px',
//                       border: '1px solid #e0e0e0',
//                       minWidth: '200px',
//                     }}
//                   >
//                     <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
//                       <FormControlLabel
//                         control={
//                           <Switch
//                             checked={queryToggles[label] !== false}
//                             onChange={() => toggleQueryLabel(label)}
//                             size="small"
//                           />
//                         }
//                         label={
//                           <Typography variant="body2" fontWeight="bold">
//                             {label}
//                           </Typography>
//                         }
//                       />
//                       <Chip
//                         label={count}
//                         size="small"
//                         sx={{
//                           bgcolor: getColorForLabel(label) + '20',
//                           border: `1px solid ${getColorForLabel(label)}`,
//                         }}
//                       />
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

//       {/* Error/Success Snackbar */}
//       <Snackbar
//         open={!!localError}
//         autoHideDuration={6000}
//         onClose={() => setLocalError(null)}
//         anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
//       >
//         <Alert severity={localError?.severity || "error"} onClose={() => setLocalError(null)}>
//           {localError?.message || localError}
//         </Alert>
//       </Snackbar>
//     </>
//   );
// };

// export default PdfProcessingResults;


// import React, { useState, useEffect } from 'react';
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
//   IconButton,
//   Tooltip,
// } from '@mui/material';
// import SearchIcon from '@mui/icons-material/Search';
// import CancelIcon from '@mui/icons-material/Cancel';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// import ExpandLessIcon from '@mui/icons-material/ExpandLess';
// import VisibilityIcon from '@mui/icons-material/Visibility';
// import ReactDOM from 'react-dom';

// import BoundingBoxes from './BoundingBoxes';
// import { getResultsCount } from './utils/pdfUtils';

// import { usePdfDataContext } from '../../context/PdfDataContext';
// import FileMetadataDisplay from './FileMetadataDisplay';

// const PdfProcessingResults = ({
//   fileId,
//   stage,
//   pageDimensions,
//   currentPage,
//   pdfContainerRef,
//   scale = 1.5,
//   metadata = {},
// }) => {
//   const {
//     processingResults,
//     processingLoading,
//     visionLoading,
//     error,
//     handleProcessPdf,
//     handleRunVisionInference,
//     clearAllResults,
//     reloadMetadata,
//   } = usePdfDataContext();

//   // Local state
//   const [showResults, setShowResults] = useState(true);
//   const [queryToggles, setQueryToggles] = useState({});
//   const [showQueryControls, setShowQueryControls] = useState(true);
//   const [boundingBoxesVisible, setBoundingBoxesVisible] = useState(false);
//   const [localError, setLocalError] = useState(null);

//   // Clean up the container when component unmounts
//   useEffect(() => {
//     return () => {
//       const boxContainer = document.getElementById('bounding-box-container');
//       if (boxContainer && boxContainer.parentNode) {
//         boxContainer.parentNode.removeChild(boxContainer);
//       }
//     };
//   }, []);

//   // On error from hook, show snackbar
//   useEffect(() => {
//     if (error) setLocalError(error);
//   }, [error]);

//   // Update toggles/results on new process results
//   useEffect(() => {
//     if (!processingResults || processingResults.length === 0) return;
//     const initialToggles = {};
//     processingResults.forEach((result) => {
//       initialToggles[result.query_label] = true;
//     });
//     setQueryToggles(initialToggles);
//     setBoundingBoxesVisible(true);
//     setShowResults(true);
//   }, [processingResults]);

//   // Force bounding box rerender on toggle change
//   useEffect(() => {
//     if (showResults && Object.keys(queryToggles).length > 0) {
//       setBoundingBoxesVisible(false);
//       const timer = setTimeout(() => setBoundingBoxesVisible(true), 50);
//       return () => clearTimeout(timer);
//     }
//   }, [queryToggles, showResults]);

//   // Check if file has classification
//   const hasClassification = !!metadata?.classification;

//   const onProcessPdf = async () => {
//     if (!fileId) {
//       setLocalError("No file selected");
//       return;
//     }
//     clearAllResults();
//     setBoundingBoxesVisible(false);
//     try {
//       await handleProcessPdf({ fileId, stage });
//       await reloadMetadata(); 
//     } catch (err) {
//       setLocalError(err.message);
//     }
//   };
  
//   const onRunVisionInference = async () => {
//     if (!fileId || !hasClassification) {
//       setLocalError("Cannot run vision inference - file must have a classification");
//       return;
//     }
//     try {
//       await handleRunVisionInference({
//         fileId,
//         stage,
//         classificationLabel: metadata.classification,
//       });
//       setLocalError({ message: "Vision inference completed successfully!", severity: "success" });
//       await reloadMetadata(); 
//     } catch (err) {
//       setLocalError(err.message);
//     }
//   };

//   // --- Custom color for query label ---
//   const getColorForLabel = (() => {
//     const cache = {};
//     return (label) => {
//       if (cache[label]) return cache[label];
//       let hash = 0;
//       for (let i = 0; i < label.length; i++) {
//         hash = label.charCodeAt(i) + ((hash << 5) - hash);
//       }
//       const hue = Math.abs(hash) % 360;
//       const saturation = 60 + (Math.abs(hash) % 30);
//       const lightness = 45 + (Math.abs(hash * 7) % 30);
//       const color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
//       cache[label] = color;
//       return color;
//     };
//   })();

//   // Map for label -> color
//   const colorMap = {};
//   (processingResults || []).forEach((r) => {
//     colorMap[r.query_label] = getColorForLabel(r.query_label);
//   });

//   // --- Toggle logic ---
//   const toggleResults = () => {
//     setShowResults((prev) => !prev);
//     setBoundingBoxesVisible((prev) => !prev);
//   };

//   const toggleQueryLabel = (queryLabel) => {
//     setQueryToggles((prev) => ({
//       ...prev,
//       [queryLabel]: !prev[queryLabel],
//     }));
//     setBoundingBoxesVisible((prev) => {
//       if (prev) {
//         setTimeout(() => setBoundingBoxesVisible(true), 10);
//         return false;
//       }
//       return true;
//     });
//   };

//   const toggleAllQueryLabels = (value) => {
//     const updated = {};
//     Object.keys(queryToggles).forEach((key) => {
//       updated[key] = value;
//     });
//     setQueryToggles(updated);
//   };

//   // --- Get PDF dimensions from metadata in the results ---
//   const getPdfDimensions = () => {
//     if (processingResults && processingResults.length > 0) {
//       // Check each result for pdf_metadata or item metadata
//       for (const result of processingResults) {
//         // First, check if there's a top-level pdf_metadata field
//         if (result.pdf_metadata && result.pdf_metadata[currentPage - 1]) {
//           return {
//             width: result.pdf_metadata[currentPage - 1].width,
//             height: result.pdf_metadata[currentPage - 1].height
//           };
//         }
        
//         // Then, check page items for metadata
//         const pageIndex = String(currentPage - 1);
//         const pages = result.results?.pages;
//         if (pages && pages[pageIndex] && pages[pageIndex].length > 0) {
//           // Check if any box has metadata
//           for (const box of pages[pageIndex]) {
//             if (box.meta && box.meta.width && box.meta.height) {
//               return {
//                 width: box.meta.width,
//                 height: box.meta.height
//               };
//             }
//           }
//         }
//       }
//     }
    
//     // Fallback to provided dimensions
//     return pageDimensions;
//   };

//   // --- Filtered boxes data for bounding box display ---
//   const getFilteredBoxesData = () => {
//     if (!processingResults || processingResults.length === 0 || !showResults) return null;
//     try {
//       const formattedData = {};
//       processingResults.forEach((result) => {
//         const queryLabel = result.query_label;
//         if (queryToggles[queryLabel] === false) return;
//         const pages = result.results?.pages;
//         if (!pages) return;
//         const pageIndex = String(currentPage - 1);
//         const pageBoxes = pages[pageIndex];
//         if (!pageBoxes || !pageBoxes.length) return;
//         formattedData[queryLabel] = pageBoxes.map((box) => {
//           const coords = box.bbox || [box.x0, box.y0, box.x1, box.y1];
//           return { ...box, coords, text: box.value || '', queryLabel };
//         });
//       });
//       return Object.keys(formattedData).length > 0 ? formattedData : null;
//     } catch (error) {
//       console.error('Error formatting boxes data:', error);
//       return null;
//     }
//   };

//   // --- Results count helper ---
//   const getQueryLabelResultsCount = (label) => {
//     const queryResult = (processingResults || []).find((r) => r.query_label === label);
//     if (!queryResult || !queryResult.results?.pages) return 0;
//     const pageIndex = String(currentPage - 1);
//     return (queryResult.results.pages[pageIndex] || []).length;
//   };

//   // --- Transform bounding box coordinates ---
//   const transformCoordWithContainer = (coords, color) => {
//     if (!pageDimensions.width || !pageDimensions.height) {
//       return { x: 0, y: 0, width: 0, height: 0, color };
//     }
    
//     let [x0, y0, x1, y1] = coords.map(parseFloat);
//     if (x1 < x0) [x0, x1] = [x1, x0];
//     if (y1 < y0) [y0, y1] = [y1, y0];
    
//     const renderedWidth = pageDimensions.width;
//     const renderedHeight = pageDimensions.height;
    
//     // Get PDF dimensions dynamically
//     const pdfDimensions = getPdfDimensions();
//     const pdfWidth = pdfDimensions.width || 612.0;
//     const pdfHeight = pdfDimensions.height || 792.0;
    
//     const widthRatio = renderedWidth / pdfWidth;
//     const heightRatio = renderedHeight / pdfHeight;
    
//     const x = x0 * widthRatio;
//     const y = y0 * heightRatio;
//     const width = (x1 - x0) * widthRatio;
//     const height = (y1 - y0) * heightRatio;
    
//     return {
//       position: 'absolute',
//       left: `${x}px`,
//       top: `${y}px`,
//       width: `${width}px`,
//       height: `${height}px`,
//       border: `2px solid ${color}`,
//       backgroundColor: color.replace('1)', '0.3)').replace(')', ', 0.3)'),
//       zIndex: 2000,
//       pointerEvents: 'auto',
//       opacity: 1,
//     };
//   };

//   const boxesData = getFilteredBoxesData();
//   const resultsCount = boxesData ? getResultsCount(boxesData) : 0;
//   const canRenderBoxes =
//     pdfContainerRef?.current &&
//     pageDimensions.width > 0 &&
//     pageDimensions.height > 0 &&
//     boundingBoxesVisible &&
//     boxesData;

//   // --- Render bounding boxes using a portal ---
//   const renderBoundingBoxesPortal = () => {
//     if (!canRenderBoxes) return null;
    
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
//       boxContainer.style.zIndex = '1000';
//       boxContainer.style.border = 'none';
//       boxContainer.style.background = 'transparent';
//       boxContainer.style.boxShadow = 'none';
//       boxContainer.style.overflow = 'hidden';
      
//       // Ensure parent has position: relative
//       pdfContainerRef.current.style.position = 'relative';
//       pdfContainerRef.current.appendChild(boxContainer);
//     }
    
//     return ReactDOM.createPortal(
//       <BoundingBoxes
//         boxesData={boxesData}
//         transformCoord={transformCoordWithContainer}
//         colorMap={colorMap}
//         showTooltips={true}
//       />,
//       boxContainer
//     );
//   };

//   return (
//     <>
//       {/* Process Controls */}
//       <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
//       <FileMetadataDisplay metadata={metadata} />
//         <Button
//           variant="contained"
//           color="primary"
//           onClick={onProcessPdf}
//           disabled={processingLoading || !fileId}
//           startIcon={processingLoading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
//         >
//           {processingLoading ? "Processing..." : "Process PDF"}
//         </Button>

//         <Tooltip title={!hasClassification ? "File must have a classification to run vision inference" : ""}>
//           <span>
//             <Button
//               variant="contained"
//               color="secondary"
//               onClick={onRunVisionInference}
//               disabled={visionLoading || !fileId || !hasClassification}
//               startIcon={visionLoading ? <CircularProgress size={20} color="inherit" /> : <VisibilityIcon />}
//             >
//               {visionLoading ? "Running..." : "Run Vision Inference"}
//             </Button>
//           </span>
//         </Tooltip>

//         {processingResults.length > 0 && (
//           <Button
//             variant="outlined"
//             color="primary"
//             onClick={toggleResults}
//             startIcon={showResults ? <CancelIcon /> : <SearchIcon />}
//           >
//             {showResults ? "Hide Results" : "Show Results"}
//           </Button>
//         )}

//         {processingResults.length > 0 && (
//           <Typography variant="body2" color="textSecondary">
//             Found {resultsCount} results on page {currentPage}
//           </Typography>
//         )}

//       </Box>

//       {/* Query Filters */}
//       {processingResults.length > 0 && (
//         <Paper
//           elevation={1}
//           sx={{
//             p: 2,
//             mb: 2,
//             bgcolor: '#f5f5f5',
//             border: '1px solid #e0e0e0',
//             borderRadius: '4px',
//           }}
//         >
//           <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
//             <Typography variant="subtitle1" fontWeight="bold">
//               Query Types
//             </Typography>
//             <Box>
//               <Button size="small" onClick={() => toggleAllQueryLabels(true)} sx={{ mr: 1 }}>
//                 Show All
//               </Button>
//               <Button size="small" onClick={() => toggleAllQueryLabels(false)}>
//                 Hide All
//               </Button>
//               <IconButton size="small" onClick={() => setShowQueryControls((prev) => !prev)} sx={{ ml: 1 }}>
//                 {showQueryControls ? <ExpandLessIcon /> : <ExpandMoreIcon />}
//               </IconButton>
//             </Box>
//           </Box>
//           <Collapse in={showQueryControls}>
//             <FormGroup sx={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: 1 }}>
//               {processingResults.map((result) => {
//                 const label = result.query_label;
//                 const description = result?.description || '';
//                 const count = getQueryLabelResultsCount(label);
//                 const color = getColorForLabel(label);

//                 return (
//                   <Box
//                     key={label}
//                     sx={{
//                       display: 'flex',
//                       flexDirection: 'column',
//                       bgcolor: 'white',
//                       p: 1,
//                       borderRadius: '4px',
//                       border: '1px solid #e0e0e0',
//                       minWidth: '200px',
//                     }}
//                   >
//                     <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
//                       <FormControlLabel
//                         control={
//                           <Switch
//                             checked={queryToggles[label] !== false}
//                             onChange={() => toggleQueryLabel(label)}
//                             size="small"
//                           />
//                         }
//                         label={
//                           <Typography variant="body2" fontWeight="bold">
//                             {label}
//                           </Typography>
//                         }
//                       />
//                       <Chip
//                         label={count}
//                         size="small"
//                         sx={{
//                           bgcolor: getColorForLabel(label) + '20',
//                           border: `1px solid ${getColorForLabel(label)}`,
//                         }}
//                       />
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

//       {/* Error/Success Snackbar */}
//       <Snackbar
//         open={!!localError}
//         autoHideDuration={6000}
//         onClose={() => setLocalError(null)}
//         anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
//       >
//         <Alert severity={localError?.severity || "error"} onClose={() => setLocalError(null)}>
//           {localError?.message || localError}
//         </Alert>
//       </Snackbar>
//     </>
//   );
// };

// export default PdfProcessingResults;

// // import React, { useState, useEffect } from 'react';
// // import {
// //   Button,
// //   CircularProgress,
// //   Snackbar,
// //   Alert,
// //   Box,
// //   Typography,
// //   FormGroup,
// //   FormControlLabel,
// //   Switch,
// //   Chip,
// //   Paper,
// //   Collapse,
// //   IconButton,
// //   Tooltip,
// // } from '@mui/material';
// // import SearchIcon from '@mui/icons-material/Search';
// // import CancelIcon from '@mui/icons-material/Cancel';
// // import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// // import ExpandLessIcon from '@mui/icons-material/ExpandLess';
// // import VisibilityIcon from '@mui/icons-material/Visibility';
// // import ReactDOM from 'react-dom';

// // import BoundingBoxes from './BoundingBoxes';
// // import { getResultsCount } from './utils/pdfUtils';

// // import { usePdfDataContext } from '../../context/PdfDataContext';

// // const PdfProcessingResults = ({
// //   fileId,
// //   stage,
// //   pageDimensions,
// //   currentPage,
// //   pdfContainerRef,
// //   scale = 1.5,
// //   metadata = {},
// // }) => {
// //   const {
// //     processingResults,
// //     processingLoading,      // use this for "Process PDF" button spinner/disable
// //     visionLoading,         // use this for "Vision Inference" button spinner/disable
// //     error,
// //     handleProcessPdf,
// //     handleRunVisionInference,
// //     clearAllResults,
// //     reloadMetadata,
// //   } = usePdfDataContext();

// //   // Local state
// //   const [showResults, setShowResults] = useState(true);
// //   const [queryToggles, setQueryToggles] = useState({});
// //   const [showQueryControls, setShowQueryControls] = useState(true);
// //   const [boundingBoxesVisible, setBoundingBoxesVisible] = useState(false);
// //   const [localError, setLocalError] = useState(null);

// //   // On error from hook, show snackbar
// //   useEffect(() => {
// //     if (error) setLocalError(error);
// //   }, [error]);

// //   // Update toggles/results on new process results
// //   useEffect(() => {
// //     if (!processingResults || processingResults.length === 0) return;
// //     const initialToggles = {};
// //     processingResults.forEach((result) => {
// //       initialToggles[result.query_label] = true;
// //     });
// //     setQueryToggles(initialToggles);
// //     setBoundingBoxesVisible(true);
// //     setShowResults(true);
// //   }, [processingResults]);

// //   // Force bounding box rerender on toggle change
// //   useEffect(() => {
// //     if (showResults && Object.keys(queryToggles).length > 0) {
// //       setBoundingBoxesVisible(false);
// //       const timer = setTimeout(() => setBoundingBoxesVisible(true), 50);
// //       return () => clearTimeout(timer);
// //     }
// //   }, [queryToggles, showResults]);

// //   // Check if file has classification
// //   const hasClassification = !!metadata?.classification;

// //   const onProcessPdf = async () => {
// //     if (!fileId) {
// //       setLocalError("No file selected");
// //       return;
// //     }
// //     clearAllResults();
// //     setBoundingBoxesVisible(false);
// //     try {
// //       await handleProcessPdf({ fileId, stage });   // <-- This manages its own loading state
// //       await reloadMetadata(); 
// //     } catch (err) {
// //       setLocalError(err.message);
// //     }
// //   };
  
// //   const onRunVisionInference = async () => {
// //     if (!fileId || !hasClassification) {
// //       setLocalError("Cannot run vision inference - file must have a classification");
// //       return;
// //     }
// //     try {
// //       await handleRunVisionInference({
// //         fileId,
// //         stage,
// //         classificationLabel: metadata.classification,
// //       }); // <-- This manages its own loading state
// //       setLocalError({ message: "Vision inference completed successfully!", severity: "success" });
// //       await reloadMetadata(); 
// //     } catch (err) {
// //       setLocalError(err.message);
// //     }
// //   };

// //   // --- Custom color for query label ---
// //   const getColorForLabel = (() => {
// //     const cache = {};
// //     return (label) => {
// //       if (cache[label]) return cache[label];
// //       let hash = 0;
// //       for (let i = 0; i < label.length; i++) {
// //         hash = label.charCodeAt(i) + ((hash << 5) - hash);
// //       }
// //       const hue = Math.abs(hash) % 360;
// //       const saturation = 60 + (Math.abs(hash) % 30);
// //       const lightness = 45 + (Math.abs(hash * 7) % 30);
// //       const color = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
// //       cache[label] = color;
// //       return color;
// //     };
// //   })();

// //   // Map for label -> color
// //   const colorMap = {};
// //   (processingResults || []).forEach((r) => {
// //     colorMap[r.query_label] = getColorForLabel(r.query_label);
// //   });

// //   // --- Toggle logic ---
// //   const toggleResults = () => {
// //     setShowResults((prev) => !prev);
// //     setBoundingBoxesVisible((prev) => !prev);
// //   };

// //   const toggleQueryLabel = (queryLabel) => {
// //     setQueryToggles((prev) => ({
// //       ...prev,
// //       [queryLabel]: !prev[queryLabel],
// //     }));
// //     setBoundingBoxesVisible((prev) => {
// //       if (prev) {
// //         setTimeout(() => setBoundingBoxesVisible(true), 10);
// //         return false;
// //       }
// //       return true;
// //     });
// //   };

// //   const toggleAllQueryLabels = (value) => {
// //     const updated = {};
// //     Object.keys(queryToggles).forEach((key) => {
// //       updated[key] = value;
// //     });
// //     setQueryToggles(updated);
// //   };

// //   // --- Filtered boxes data for bounding box display ---
// //   const getFilteredBoxesData = () => {
// //     if (!processingResults || processingResults.length === 0 || !showResults) return null;
// //     try {
// //       const formattedData = {};
// //       processingResults.forEach((result) => {
// //         const queryLabel = result.query_label;
// //         if (queryToggles[queryLabel] === false) return;
// //         const pages = result.results?.pages;
// //         if (!pages) return;
// //         const pageIndex = String(currentPage - 1);
// //         const pageBoxes = pages[pageIndex];
// //         if (!pageBoxes || !pageBoxes.length) return;
// //         formattedData[queryLabel] = pageBoxes.map((box) => {
// //           const coords = box.bbox || [box.x0, box.y0, box.x1, box.y1];
// //           return { ...box, coords, text: box.value || '', queryLabel };
// //         });
// //       });
// //       return Object.keys(formattedData).length > 0 ? formattedData : null;
// //     } catch (error) {
// //       console.error('Error formatting boxes data:', error);
// //       return null;
// //     }
// //   };

// //   // --- Results count helper ---
// //   const getQueryLabelResultsCount = (label) => {
// //     const queryResult = (processingResults || []).find((r) => r.query_label === label);
// //     if (!queryResult || !queryResult.results?.pages) return 0;
// //     const pageIndex = String(currentPage - 1);
// //     return (queryResult.results.pages[pageIndex] || []).length;
// //   };

// //   // --- Transform bounding box coordinates ---
// //   const transformCoordWithContainer = (coords, color) => {
// //     if (!pageDimensions.width || !pageDimensions.height) {
// //       return { x: 0, y: 0, width: 0, height: 0, color };
// //     }
// //     let [x0, y0, x1, y1] = coords.map(parseFloat);
// //     if (x1 < x0) [x0, x1] = [x1, x0];
// //     if (y1 < y0) [y0, y1] = [y1, y0];
// //     const renderedWidth = pageDimensions.width;
// //     const renderedHeight = pageDimensions.height;
// //     const pdfWidth = 612.0;
// //     const pdfHeight = 792.0;
// //     const widthRatio = renderedWidth / pdfWidth;
// //     const heightRatio = renderedHeight / pdfHeight;
// //     const x = x0 * widthRatio;
// //     const y = y0 * heightRatio;
// //     const width = (x1 - x0) * widthRatio;
// //     const height = (y1 - y0) * heightRatio;
// //     return {
// //       position: 'absolute',
// //       left: `${x}px`,
// //       top: `${y}px`,
// //       width: `${width}px`,
// //       height: `${height}px`,
// //       border: `2px solid ${color}`,
// //       backgroundColor: color.replace('1)', '0.3)'),
// //       zIndex: 2000,
// //       pointerEvents: 'auto',
// //       opacity: 1,
// //     };
// //   };

// //   const boxesData = getFilteredBoxesData();
// //   const resultsCount = boxesData ? getResultsCount(boxesData) : 0;
// //   const canRenderBoxes =
// //     pdfContainerRef?.current &&
// //     pageDimensions.width > 0 &&
// //     pageDimensions.height > 0 &&
// //     boundingBoxesVisible &&
// //     boxesData;

// //   // --- Render bounding boxes using a portal ---
// //   const renderBoundingBoxesPortal = () => {
// //     if (!canRenderBoxes) return null;
// //     let boxContainer = document.getElementById('bounding-box-container');
// //     if (!boxContainer) {
// //       boxContainer = document.createElement('div');
// //       boxContainer.id = 'bounding-box-container';
// //       boxContainer.style.position = 'absolute';
// //       boxContainer.style.top = '0';
// //       boxContainer.style.left = '0';
// //       boxContainer.style.width = '100%';
// //       boxContainer.style.height = '100%';
// //       boxContainer.style.pointerEvents = 'none';
// //       boxContainer.style.zIndex = '1000';
// //       pdfContainerRef.current.appendChild(boxContainer);
// //     }
// //     return ReactDOM.createPortal(
// //       <BoundingBoxes
// //         boxesData={boxesData}
// //         transformCoord={transformCoordWithContainer}
// //         colorMap={colorMap}
// //         showTooltips={true}
// //       />,
// //       boxContainer
// //     );
// //   };

// //   return (
// //     <>
// //       {/* Process Controls */}
// //       <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
// //         <Button
// //           variant="contained"
// //           color="primary"
// //           onClick={onProcessPdf}
// //           disabled={processingLoading || !fileId}
// //           startIcon={processingLoading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
// //         >
// //           {processingLoading ? "Processing..." : "Process PDF"}
// //         </Button>

// //         <Tooltip title={!hasClassification ? "File must have a classification to run vision inference" : ""}>
// //           <span>
// //             <Button
// //               variant="contained"
// //               color="secondary"
// //               onClick={onRunVisionInference}
// //               disabled={visionLoading || !fileId || !hasClassification}
// //               startIcon={visionLoading ? <CircularProgress size={20} color="inherit" /> : <VisibilityIcon />}
// //             >
// //               {visionLoading ? "Running..." : "Run Vision Inference"}
// //             </Button>
// //           </span>
// //         </Tooltip>

// //         {processingResults.length > 0 && (
// //           <Button
// //             variant="outlined"
// //             color="primary"
// //             onClick={toggleResults}
// //             startIcon={showResults ? <CancelIcon /> : <SearchIcon />}
// //           >
// //             {showResults ? "Hide Results" : "Show Results"}
// //           </Button>
// //         )}

// //         {processingResults.length > 0 && (
// //           <Typography variant="body2" color="textSecondary">
// //             Found {resultsCount} results on page {currentPage}
// //           </Typography>
// //         )}
// //       </Box>

// //       {/* Query Filters */}
// //       {processingResults.length > 0 && (
// //         <Paper
// //           elevation={1}
// //           sx={{
// //             p: 2,
// //             mb: 2,
// //             bgcolor: '#f5f5f5',
// //             border: '1px solid #e0e0e0',
// //             borderRadius: '4px',
// //           }}
// //         >
// //           <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
// //             <Typography variant="subtitle1" fontWeight="bold">
// //               Query Types
// //             </Typography>
// //             <Box>
// //               <Button size="small" onClick={() => toggleAllQueryLabels(true)} sx={{ mr: 1 }}>
// //                 Show All
// //               </Button>
// //               <Button size="small" onClick={() => toggleAllQueryLabels(false)}>
// //                 Hide All
// //               </Button>
// //               <IconButton size="small" onClick={() => setShowQueryControls((prev) => !prev)} sx={{ ml: 1 }}>
// //                 {showQueryControls ? <ExpandLessIcon /> : <ExpandMoreIcon />}
// //               </IconButton>
// //             </Box>
// //           </Box>
// //           <Collapse in={showQueryControls}>
// //             <FormGroup sx={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: 1 }}>
// //               {processingResults.map((result) => {
// //                 const label = result.query_label;
// //                 const description = result?.description || '';
// //                 const count = getQueryLabelResultsCount(label);
// //                 const color = getColorForLabel(label);

// //                 return (
// //                   <Box
// //                     key={label}
// //                     sx={{
// //                       display: 'flex',
// //                       flexDirection: 'column',
// //                       bgcolor: 'white',
// //                       p: 1,
// //                       borderRadius: '4px',
// //                       border: '1px solid #e0e0e0',
// //                       minWidth: '200px',
// //                     }}
// //                   >
// //                     <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
// //                       <FormControlLabel
// //                         control={
// //                           <Switch
// //                             checked={queryToggles[label] !== false}
// //                             onChange={() => toggleQueryLabel(label)}
// //                             size="small"
// //                           />
// //                         }
// //                         label={
// //                           <Typography variant="body2" fontWeight="bold">
// //                             {label}
// //                           </Typography>
// //                         }
// //                       />
// //                       <Chip
// //                         label={count}
// //                         size="small"
// //                         sx={{
// //                           bgcolor: getColorForLabel(label) + '20',
// //                           border: `1px solid ${getColorForLabel(label)}`,
// //                         }}
// //                       />
// //                     </Box>
// //                     {description && (
// //                       <Typography variant="caption" sx={{ mt: 0.5, color: 'text.secondary' }}>
// //                         {description}
// //                       </Typography>
// //                     )}
// //                   </Box>
// //                 );
// //               })}
// //             </FormGroup>
// //           </Collapse>
// //         </Paper>
// //       )}

// //       {/* Render the bounding boxes via portal */}
// //       {renderBoundingBoxesPortal()}

// //       {/* Error/Success Snackbar */}
// //       <Snackbar
// //         open={!!localError}
// //         autoHideDuration={6000}
// //         onClose={() => setLocalError(null)}
// //         anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
// //       >
// //         <Alert severity={localError?.severity || "error"} onClose={() => setLocalError(null)}>
// //           {localError?.message || localError}
// //         </Alert>
// //       </Snackbar>
// //     </>
// //   );
// // };

// // export default PdfProcessingResults;