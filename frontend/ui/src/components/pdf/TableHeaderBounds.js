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
  Tooltip
} from '@mui/material';
import TableChartIcon from '@mui/icons-material/TableChart';
import CancelIcon from '@mui/icons-material/Cancel';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ReactDOM from 'react-dom';

import BoundingBoxes from './BoundingBoxes';
import { usePdfDataContext } from '../../context/PdfDataContext';

const TableHeaderBounds = ({
  fileId,
  stage,
  pageDimensions,
  currentPage,
  pdfContainerRef,
  scale = 1.5,
  metadata = {}
}) => {
  const {
    tableHeaders,
    tableHeadersLoading,
    handleProcessTableHeaders,
    error,
    reloadMetadata
  } = usePdfDataContext();

  const [showResults, setShowResults] = useState(true);
  const [tableToggles, setTableToggles] = useState({});
  const [showTableControls, setShowTableControls] = useState(true);
  const [boundingBoxesVisible, setBoundingBoxesVisible] = useState(false);
  const [localError, setLocalError] = useState(null);

  const hasClassification = !!metadata?.classification;

  // Clean up the container when component unmounts
  useEffect(() => {
    return () => {
      const boxContainer = document.getElementById('table-header-box-container');
      if (boxContainer && boxContainer.parentNode) {
        boxContainer.parentNode.removeChild(boxContainer);
      }
    };
  }, []);

  // Extract toggles from new table headers result (always an object)
  useEffect(() => {
    if (!tableHeaders || !tableHeaders.results?.pages) return;
    const toggles = {};
    const tables = new Set();
    Object.values(tableHeaders.results.pages).forEach(pageItems => {
      pageItems.forEach(item => {
        if (item.table_index != null) tables.add(item.table_index);
      });
    });
    tables.forEach(tableIndex => {
      toggles[`Table ${tableIndex}`] = true;
    });
    setTableToggles(toggles);
    setBoundingBoxesVisible(true);
    setShowResults(true);
  }, [tableHeaders]);

  useEffect(() => {
    if (showResults && Object.keys(tableToggles).length > 0) {
      setBoundingBoxesVisible(false);
      const timer = setTimeout(() => setBoundingBoxesVisible(true), 50);
      return () => clearTimeout(timer);
    }
  }, [tableToggles, showResults]);

  const onProcessTableHeaders = async () => {
    if (!fileId || !hasClassification) {
      setLocalError("Cannot run table header detection - file must have a classification");
      return;
    }
    setTableToggles({});
    setBoundingBoxesVisible(false);
    try {
      await handleProcessTableHeaders({
        fileId,
        stage,
        classificationLabel: metadata.classification
      });
      await reloadMetadata();
    } catch (err) {
      setLocalError(err.message);
    }
  };

  // --- Color map ---
  const getColorForTable = (() => {
    const cache = {};
    return (tableIndex) => {
      const label = `Table ${tableIndex}`;
      if (cache[label]) return cache[label];
      const tableColors = [
        'hsl(210, 80%, 60%)',
        'hsl(180, 70%, 50%)',
        'hsl(150, 65%, 50%)',
        'hsl(270, 60%, 60%)',
        'hsl(240, 70%, 65%)',
        'hsl(330, 70%, 60%)',
        'hsl(30, 80%, 55%)'
      ];
      const colorIndex = (tableIndex - 1) % tableColors.length;
      const color = tableColors[colorIndex];
      cache[label] = color;
      return color;
    };
  })();

  // Compute color map only for tables on this page
  const colorMap = {};
  if (tableHeaders?.results?.pages) {
    Object.values(tableHeaders.results.pages).forEach(pageItems => {
      pageItems.forEach(item => {
        if (item.table_index != null)
          colorMap[`Table ${item.table_index}`] = getColorForTable(item.table_index);
      });
    });
  }

  const toggleResults = () => {
    setShowResults(!showResults);
    setBoundingBoxesVisible(!showResults);
  };

  const toggleTable = (tableLabel) => {
    setTableToggles(prev => ({
      ...prev,
      [tableLabel]: !prev[tableLabel]
    }));
    setBoundingBoxesVisible(prev => {
      if (prev) {
        setTimeout(() => setBoundingBoxesVisible(true), 10);
        return false;
      }
      return true;
    });
  };

  const toggleAllTables = (value) => {
    const updated = {};
    Object.keys(tableToggles).forEach(key => {
      updated[key] = value;
    });
    setTableToggles(updated);
  };

  const getFilteredBoxesData = () => {
    if (!tableHeaders?.results?.pages || !showResults) return null;
    const formattedData = {};
    const pageIndex = String(currentPage - 1);
    const pageItems = tableHeaders.results.pages[pageIndex];
    if (!pageItems || !pageItems.length) return null;
    pageItems.forEach(item => {
      const tableLabel = `Table ${item.table_index}`;
      if (tableToggles[tableLabel] === false) return;
      if (!formattedData[tableLabel]) formattedData[tableLabel] = [];
      formattedData[tableLabel].push({
        ...item,
        coords: item.bbox,
        text: item.table_title || `Table ${item.table_index}`,
        queryLabel: tableLabel
      });
    });
    return Object.keys(formattedData).length > 0 ? formattedData : null;
  };

  const getTableResultsCount = (tableLabel) => {
    if (!tableHeaders?.results?.pages) return 0;
    const tableIndex = parseInt(tableLabel.replace('Table ', ''));
    const pageIndex = String(currentPage - 1);
    const pageItems = tableHeaders.results.pages[pageIndex] || [];
    return pageItems.filter(item => item.table_index === tableIndex).length;
  };

  // Get PDF dimensions from metadata in the results
  const getPdfDimensions = () => {
    if (tableHeaders?.pdf_metadata && tableHeaders.pdf_metadata[currentPage - 1]) {
      return {
        width: tableHeaders.pdf_metadata[currentPage - 1].width,
        height: tableHeaders.pdf_metadata[currentPage - 1].height
      };
    }
    
    // Or look in the page items meta field
    if (tableHeaders?.results?.pages) {
      const pageIndex = String(currentPage - 1);
      const pageItems = tableHeaders.results.pages[pageIndex];
      if (pageItems && pageItems.length > 0 && pageItems[0].meta) {
        return {
          width: pageItems[0].meta.width,
          height: pageItems[0].meta.height
        };
      }
    }
    
    // Fallback to provided dimensions
    return pageDimensions;
  };

  const transformCoordWithContainer = (coords, color) => {
    // Get dimensions dynamically
    const pdfDimensions = getPdfDimensions();
    
    if (!pageDimensions.width || !pageDimensions.height) 
      return { x: 0, y: 0, width: 0, height: 0, color };
    
    let [x0, y0, x1, y1] = coords.map(Number);
    if (x1 < x0) [x0, x1] = [x1, x0];
    if (y1 < y0) [y0, y1] = [y1, y0];
    
    const renderedWidth = pageDimensions.width;
    const renderedHeight = pageDimensions.height;
    
    // Use dynamically obtained width/height instead of hardcoded values
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
      backgroundColor: color.replace('1)', '0.2)').replace(')', ', 0.2)'),
      zIndex: 2001,
      pointerEvents: 'auto',
      cursor: 'pointer',
      opacity: 1
    };
  };

  const boxesData = getFilteredBoxesData();
  const resultsCount = boxesData ? Object.values(boxesData).reduce((total, boxes) => total + boxes.length, 0) : 0;

  const canRenderBoxes = pdfContainerRef?.current &&
    pageDimensions.width > 0 &&
    pageDimensions.height > 0 &&
    boundingBoxesVisible &&
    boxesData;

  const renderBoundingBoxesPortal = () => {
    if (!canRenderBoxes) return null;
    
    // Find or create container - WITHOUT recreating it each time
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
      
      // Fix: Remove borders/background/shadow
      boxContainer.style.border = 'none';
      boxContainer.style.background = 'transparent';
      boxContainer.style.boxShadow = 'none';
      
      // Ensure the parent container (pdfContainerRef.current) has position: relative!
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
  
  return (
    <>
      {/* Process Controls */}
      <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
        <Tooltip title={!hasClassification ? "File must have a classification to detect table headers" : ""}>
          <span>
            <Button
              variant="contained"
              color="info"
              onClick={onProcessTableHeaders}
              disabled={tableHeadersLoading || !fileId || !hasClassification}
              startIcon={tableHeadersLoading ? <CircularProgress size={20} color="inherit" /> : <TableChartIcon />}
            >
              {tableHeadersLoading ? "Processing..." : "Detect Table Headers"}
            </Button>
          </span>
        </Tooltip>

        {tableHeaders?.results && (
          <Button
            variant="outlined"
            color="info"
            onClick={toggleResults}
            startIcon={showResults ? <CancelIcon /> : <TableChartIcon />}
          >
            {showResults ? "Hide Table Headers" : "Show Table Headers"}
          </Button>
        )}

        {tableHeaders?.results && (
          <Typography variant="body2" color="textSecondary">
            Found {resultsCount} table header{resultsCount !== 1 ? 's' : ''} on page {currentPage}
          </Typography>
        )}
      </Box>

      {/* Table Filters */}
      {tableHeaders?.results && Object.keys(tableToggles).length > 0 && (
        <Paper elevation={1} sx={{
          p: 2, mb: 2,
          bgcolor: '#f5f5f5',
          border: '1px solid #e0e0e0',
          borderRadius: '4px'
        }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="subtitle1" fontWeight="bold">
              Table Headers
            </Typography>
            <Box>
              <Button size="small" onClick={() => toggleAllTables(true)} sx={{ mr: 1 }}>Show All</Button>
              <Button size="small" onClick={() => toggleAllTables(false)}>Hide All</Button>
              <IconButton size="small" onClick={() => setShowTableControls(prev => !prev)} sx={{ ml: 1 }}>
                {showTableControls ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              </IconButton>
            </Box>
          </Box>

          <Collapse in={showTableControls}>
            <FormGroup sx={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: 1 }}>
              {Object.keys(tableToggles).map(tableLabel => {
                const count = getTableResultsCount(tableLabel);
                const color = colorMap[tableLabel] || 'hsl(210, 80%, 60%)';

                return (
                  <Box key={tableLabel} sx={{
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
                            checked={tableToggles[tableLabel] !== false}
                            onChange={() => toggleTable(tableLabel)}
                            size="small"
                          />
                        }
                        label={<Typography variant="body2" fontWeight="bold">{tableLabel}</Typography>}
                      />
                      <Chip
                        label={count}
                        size="small"
                        sx={{
                          bgcolor: color + '20',
                          border: `1px solid ${color}`
                        }}
                      />
                    </Box>
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
        open={!!localError || !!error}
        autoHideDuration={6000}
        onClose={() => setLocalError(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          severity={localError?.severity || error?.severity || "error"} 
          onClose={() => setLocalError(null)}
        >
          {(localError?.message || localError) || (error?.message || error)}
        </Alert>
      </Snackbar>
    </>
  );
};

export default TableHeaderBounds;

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
//   Tooltip
// } from '@mui/material';
// import TableChartIcon from '@mui/icons-material/TableChart';
// import CancelIcon from '@mui/icons-material/Cancel';
// import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
// import ExpandLessIcon from '@mui/icons-material/ExpandLess';
// import ReactDOM from 'react-dom';

// import BoundingBoxes from './BoundingBoxes';
// import { usePdfDataContext } from '../../context/PdfDataContext';

// const TableHeaderBounds = ({
//   fileId,
//   stage,
//   pageDimensions,
//   currentPage,
//   pdfContainerRef,
//   scale = 1.5,
//   metadata = {}
// }) => {
//   const {
//     tableHeaders,          // now always an object, not array!
//     tableHeadersLoading,
//     handleProcessTableHeaders,
//     error,
//     reloadMetadata
//   } = usePdfDataContext();

//   const [showResults, setShowResults] = useState(true);
//   const [tableToggles, setTableToggles] = useState({});
//   const [showTableControls, setShowTableControls] = useState(true);
//   const [boundingBoxesVisible, setBoundingBoxesVisible] = useState(false);
//   const [localError, setLocalError] = useState(null);

//   const hasClassification = !!metadata?.classification;

//   // Clean up the container when component unmounts
//   useEffect(() => {
//     return () => {
//       const boxContainer = document.getElementById('table-header-box-container');
//       if (boxContainer && boxContainer.parentNode) {
//         boxContainer.parentNode.removeChild(boxContainer);
//       }
//     };
//   }, []);

//   // Extract toggles from new table headers result (always an object)
//   useEffect(() => {
//     if (!tableHeaders || !tableHeaders.results?.pages?.pages) return;
//     const toggles = {};
//     const tables = new Set();
//     Object.values(tableHeaders.results.pages.pages).forEach(pageItems => {
//       pageItems.forEach(item => {
//         if (item.table_index != null) tables.add(item.table_index);
//       });
//     });
//     tables.forEach(tableIndex => {
//       toggles[`Table ${tableIndex}`] = true;
//     });
//     setTableToggles(toggles);
//     setBoundingBoxesVisible(true);
//     setShowResults(true);
//   }, [tableHeaders]);

//   useEffect(() => {
//     if (showResults && Object.keys(tableToggles).length > 0) {
//       setBoundingBoxesVisible(false);
//       const timer = setTimeout(() => setBoundingBoxesVisible(true), 50);
//       return () => clearTimeout(timer);
//     }
//   }, [tableToggles, showResults]);

//   const onProcessTableHeaders = async () => {
//     if (!fileId || !hasClassification) {
//       setLocalError("Cannot run table header detection - file must have a classification");
//       return;
//     }
//     setTableToggles({});
//     setBoundingBoxesVisible(false);
//     try {
//       await handleProcessTableHeaders({
//         fileId,
//         stage,
//         classificationLabel: metadata.classification
//       });
//       await reloadMetadata();
//     } catch (err) {
//       setLocalError(err.message);
//     }
//   };

//   // --- Color map ---
//   const getColorForTable = (() => {
//     const cache = {};
//     return (tableIndex) => {
//       const label = `Table ${tableIndex}`;
//       if (cache[label]) return cache[label];
//       const tableColors = [
//         'hsl(210, 80%, 60%)',
//         'hsl(180, 70%, 50%)',
//         'hsl(150, 65%, 50%)',
//         'hsl(270, 60%, 60%)',
//         'hsl(240, 70%, 65%)',
//         'hsl(330, 70%, 60%)',
//         'hsl(30, 80%, 55%)'
//       ];
//       const colorIndex = (tableIndex - 1) % tableColors.length;
//       const color = tableColors[colorIndex];
//       cache[label] = color;
//       return color;
//     };
//   })();

//   // Compute color map only for tables on this page
//   const colorMap = {};
//   if (tableHeaders?.results?.pages?.pages) {
//     Object.values(tableHeaders.results.pages.pages).forEach(pageItems => {
//       pageItems.forEach(item => {
//         if (item.table_index != null)
//           colorMap[`Table ${item.table_index}`] = getColorForTable(item.table_index);
//       });
//     });
//   }

//   const toggleResults = () => {
//     setShowResults(!showResults);
//     setBoundingBoxesVisible(!showResults);
//   };

//   const toggleTable = (tableLabel) => {
//     setTableToggles(prev => ({
//       ...prev,
//       [tableLabel]: !prev[tableLabel]
//     }));
//     setBoundingBoxesVisible(prev => {
//       if (prev) {
//         setTimeout(() => setBoundingBoxesVisible(true), 10);
//         return false;
//       }
//       return true;
//     });
//   };

//   const toggleAllTables = (value) => {
//     const updated = {};
//     Object.keys(tableToggles).forEach(key => {
//       updated[key] = value;
//     });
//     setTableToggles(updated);
//   };

//   const getFilteredBoxesData = () => {
//     if (!tableHeaders?.results?.pages?.pages || !showResults) return null;
//     const formattedData = {};
//     const pageIndex = String(currentPage - 1);
//     const pageItems = tableHeaders.results.pages.pages[pageIndex];
//     if (!pageItems || !pageItems.length) return null;
//     pageItems.forEach(item => {
//       const tableLabel = `Table ${item.table_index}`;
//       if (tableToggles[tableLabel] === false) return;
//       if (!formattedData[tableLabel]) formattedData[tableLabel] = [];
//       formattedData[tableLabel].push({
//         ...item,
//         coords: item.bbox,
//         text: item.table_title || `Table ${item.table_index}`,
//         queryLabel: tableLabel
//       });
//     });
//     return Object.keys(formattedData).length > 0 ? formattedData : null;
//   };

//   const getTableResultsCount = (tableLabel) => {
//     if (!tableHeaders?.results?.pages?.pages) return 0;
//     const tableIndex = parseInt(tableLabel.replace('Table ', ''));
//     const pageIndex = String(currentPage - 1);
//     const pageItems = tableHeaders.results.pages.pages[pageIndex] || [];
//     return pageItems.filter(item => item.table_index === tableIndex).length;
//   };

//   const transformCoordWithContainer = (coords, color) => {
//     if (!pageDimensions.width || !pageDimensions.height) return { x: 0, y: 0, width: 0, height: 0, color };
//     let [x0, y0, x1, y1] = coords.map(Number);
//     if (x1 < x0) [x0, x1] = [x1, x0];
//     if (y1 < y0) [y0, y1] = [y1, y0];
//     const renderedWidth = pageDimensions.width;
//     const renderedHeight = pageDimensions.height;
//     const pdfWidth = 612.0;
//     const pdfHeight = 792.0;
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
//       backgroundColor: color.replace('1)', '0.2)').replace(')', ', 0.2)'),
//       zIndex: 2001,
//       pointerEvents: 'auto',
//       cursor: 'pointer',
//       opacity: 1
//     };
//   };

//   const boxesData = getFilteredBoxesData();
//   const resultsCount = boxesData ? Object.values(boxesData).reduce((total, boxes) => total + boxes.length, 0) : 0;

//   const canRenderBoxes = pdfContainerRef?.current &&
//     pageDimensions.width > 0 &&
//     pageDimensions.height > 0 &&
//     boundingBoxesVisible &&
//     boxesData;

//   const renderBoundingBoxesPortal = () => {
//     if (!canRenderBoxes) return null;
    
//     // Find or create container - WITHOUT recreating it each time
//     let boxContainer = document.getElementById('table-header-box-container');
//     if (!boxContainer) {
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
      
//       // Fix: Remove borders/background/shadow
//       boxContainer.style.border = 'none';
//       boxContainer.style.background = 'transparent';
//       boxContainer.style.boxShadow = 'none';
      
//       // Ensure the parent container (pdfContainerRef.current) has position: relative!
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
//         <Tooltip title={!hasClassification ? "File must have a classification to detect table headers" : ""}>
//           <span>
//             <Button
//               variant="contained"
//               color="info"
//               onClick={onProcessTableHeaders}
//               disabled={tableHeadersLoading || !fileId || !hasClassification}
//               startIcon={tableHeadersLoading ? <CircularProgress size={20} color="inherit" /> : <TableChartIcon />}
//             >
//               {tableHeadersLoading ? "Processing..." : "Detect Table Headers"}
//             </Button>
//           </span>
//         </Tooltip>

//         {tableHeaders?.results && (
//           <Button
//             variant="outlined"
//             color="info"
//             onClick={toggleResults}
//             startIcon={showResults ? <CancelIcon /> : <TableChartIcon />}
//           >
//             {showResults ? "Hide Table Headers" : "Show Table Headers"}
//           </Button>
//         )}

//         {tableHeaders?.results && (
//           <Typography variant="body2" color="textSecondary">
//             Found {resultsCount} table header{resultsCount !== 1 ? 's' : ''} on page {currentPage}
//           </Typography>
//         )}
//       </Box>

//       {/* Table Filters */}
//       {tableHeaders?.results && Object.keys(tableToggles).length > 0 && (
//         <Paper elevation={1} sx={{
//           p: 2, mb: 2,
//           bgcolor: '#f5f5f5',
//           border: '1px solid #e0e0e0',
//           borderRadius: '4px'
//         }}>
//           <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
//             <Typography variant="subtitle1" fontWeight="bold">
//               Table Headers
//             </Typography>
//             <Box>
//               <Button size="small" onClick={() => toggleAllTables(true)} sx={{ mr: 1 }}>Show All</Button>
//               <Button size="small" onClick={() => toggleAllTables(false)}>Hide All</Button>
//               <IconButton size="small" onClick={() => setShowTableControls(prev => !prev)} sx={{ ml: 1 }}>
//                 {showTableControls ? <ExpandLessIcon /> : <ExpandMoreIcon />}
//               </IconButton>
//             </Box>
//           </Box>

//           <Collapse in={showTableControls}>
//             <FormGroup sx={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: 1 }}>
//               {Object.keys(tableToggles).map(tableLabel => {
//                 const count = getTableResultsCount(tableLabel);
//                 const color = colorMap[tableLabel] || 'hsl(210, 80%, 60%)';

//                 return (
//                   <Box key={tableLabel} sx={{
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
//                             checked={tableToggles[tableLabel] !== false}
//                             onChange={() => toggleTable(tableLabel)}
//                             size="small"
//                           />
//                         }
//                         label={<Typography variant="body2" fontWeight="bold">{tableLabel}</Typography>}
//                       />
//                       <Chip
//                         label={count}
//                         size="small"
//                         sx={{
//                           bgcolor: color + '20',
//                           border: `1px solid ${color}`
//                         }}
//                       />
//                     </Box>
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
//         open={!!localError || !!error}
//         autoHideDuration={6000}
//         onClose={() => setLocalError(null)}
//         anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
//       >
//         <Alert 
//           severity={localError?.severity || error?.severity || "error"} 
//           onClose={() => setLocalError(null)}
//         >
//           {(localError?.message || localError) || (error?.message || error)}
//         </Alert>
//       </Snackbar>
//     </>
//   );
// };

// export default TableHeaderBounds;
