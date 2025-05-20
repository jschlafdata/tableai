import React from 'react';
import {
  Box,
  Button,
  CircularProgress,
  Tooltip,
  Typography,
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import VisibilityIcon from '@mui/icons-material/Visibility';
import CancelIcon from '@mui/icons-material/Cancel';
import FileMetadataDisplay from './FileMetadataDisplay';

// Optional: keep this a separate boolean or pass it as a prop
const DEBUG = false;

const PdfProcessingControls = ({
  fileId,
  hasClassification,
  processingLoading,
  visionLoading,
  onProcessPdf,
  onRunVisionInference,
  showResults,
  toggleResults,
  resultsCount,
  currentPage,
  effectiveStage,
  hasCurrentStageResults,
  metadata
}) => {
  return (
    <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
      {/* Display file metadata */}
      <FileMetadataDisplay metadata={metadata} />

      {DEBUG && (
        <Typography variant="caption" sx={{ bgcolor: 'yellow' }}>
          Stage: {effectiveStage}, Results: {hasCurrentStageResults ? 'Yes' : 'No'}
        </Typography>
      )}

      {/* Process PDF Button */}
      <Button
        variant="contained"
        color="primary"
        onClick={onProcessPdf}
        disabled={processingLoading || !fileId}
        startIcon={processingLoading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
      >
        {processingLoading ? "Processing..." : "Process PDF"}
      </Button>

      {/* Vision Inference Button */}
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

      {/* Toggle Results Button */}
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

      {/* Results info text */}
      {hasCurrentStageResults && (
        <Typography variant="body2" color="textSecondary">
          Found {resultsCount} results on page {currentPage} (Stage {effectiveStage})
        </Typography>
      )}
    </Box>
  );
};

export default PdfProcessingControls;