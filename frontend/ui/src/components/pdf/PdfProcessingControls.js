// src/components/pdf/PdfProcessingControls.jsx
import React from 'react';
import { Box, Button, CircularProgress, Typography, Tooltip } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import CancelIcon from '@mui/icons-material/Cancel';
import VisibilityIcon from '@mui/icons-material/Visibility';

/**
 * Controls for processing PDFs and displaying results
 */
const PdfProcessingControls = ({
  onProcess,
  onVisionInference,
  onToggleResults,
  showResults,
  processingLoading,
  visionLoading,
  hasResults,
  resultsCount,
  currentPage,
  disabled,
  hasClassification
}) => {
  return (
    <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
      <Button 
        variant="contained" 
        color="primary" 
        onClick={onProcess}
        disabled={processingLoading || disabled}
        startIcon={processingLoading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
      >
        {processingLoading ? "Processing..." : "Process PDF"}
      </Button>
      
      <Tooltip title={!hasClassification ? "File must have a classification to run vision inference" : ""}>
        <span>
          <Button
            variant="contained"
            color="secondary"
            onClick={onVisionInference}
            disabled={visionLoading || disabled || !hasClassification}
            startIcon={visionLoading ? <CircularProgress size={20} color="inherit" /> : <VisibilityIcon />}
          >
            {visionLoading ? "Running..." : "Run Vision Inference"}
          </Button>
        </span>
      </Tooltip>
      
      {hasResults && (
        <Button
          variant="outlined"
          color="primary"
          onClick={onToggleResults}
          startIcon={showResults ? <CancelIcon /> : <SearchIcon />}
        >
          {showResults ? "Hide Results" : "Show Results"}
        </Button>
      )}
      
      {hasResults && (
        <Typography variant="body2" color="textSecondary">
          Found {resultsCount} results on page {currentPage}
        </Typography>
      )}
    </Box>
  );
};

export default PdfProcessingControls;