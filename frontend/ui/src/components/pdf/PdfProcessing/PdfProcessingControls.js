// 2. UPDATED PdfProcessingControls.jsx
// The intermediate component that handles vision inference options
import React from 'react';
import {
  Box,
  Button,
  CircularProgress,
  Tooltip,
  Typography,
  Divider,
  Paper
} from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import CancelIcon from '@mui/icons-material/Cancel';
import FileMetadataDisplay from './FileMetadataDisplay';
import VisionInferenceOptions from './VisionInferenceOptions';


const TABLE_EXTRACTION_INSTRUCTIONS = `## Instructions

- Count how many complete, visually distinct tables appear in the image.
    - Output this count as \`"number_of_tables"\`.
    - If no tables are identified, output \`"number_of_tables": 0\`.

- For each table, determine whether it has hierarchical headers (multi-level columns).
    - A table has hierarchical headers if there is at least one "parent" column spanning multiple sub-columns.
    - Set \`"headers_hierarchy": true\` if the table has multi-level headers, otherwise \`false\`.

- Return the columns of each table as the lowest-level column names in a flat list.

- Include each table's details under a \`"tables"\` key in the output, using the table index (as a string) as the key.

## Example Output

For a page with two tables:

{
  "number_of_tables": 2,
  "tables": {
    "0": {
      "headers_hierarchy": true,
      "columns": ["Col 1", "Col 2", "Col 3", "Col 4"]
    },
    "1": {
      "headers_hierarchy": false,
      "columns": ["Col X", "Col Y", "Col Z"]
    }
  }
}`;


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
  metadata,
  visionResponse // NEW: Add this prop
}) => {

  const optionsFields = ['temperature', 'top_k', 'top_p', 'top'];

  function buildVisionInferenceRequest(options, fileId, stage, classificationLabel) {
    const optionsDict = {};
    const mainPayload = {};

    Object.entries(options).forEach(([key, value]) => {
      if (optionsFields.includes(key)) {
        optionsDict[key] = value;
      } else {
        mainPayload[key] = value;
      }
    });

    return {
      file_id: fileId,
      stage,
      classification_label: classificationLabel,
      ...mainPayload,
      options: optionsDict,
    };
  }

  // Handle the vision inference with options
  const handleRunVisionInference = (options) => {
    // Build the correct request structure for backend
    const req = buildVisionInferenceRequest(
      options,
      fileId,
      effectiveStage,
      metadata.classification
    );
    
    // Pass the request to the parent handler
    onRunVisionInference(options);
  };

  return (
    <Box sx={{ mb: 2 }}>
      {/* Top row with file metadata and Process PDF button */}
      <Box sx={{ 
          display: 'flex', 
          flexDirection: { xs: 'column', md: 'row' }, 
          alignItems: 'center',
          gap: 2,
          mb: 2
        }}>
          {/* File metadata */}
          <Box>
            <FileMetadataDisplay metadata={metadata} />
            {DEBUG && (
              <Typography variant="caption" sx={{ bgcolor: 'yellow', display: 'block', mt: 1 }}>
                Stage: {effectiveStage}, Results: {hasCurrentStageResults ? 'Yes' : 'No'}
              </Typography>
            )}
          </Box>
          {/* Process PDF button */}
          <Box>
            <Button
              variant="contained"
              color="primary"
              onClick={onProcessPdf}
              disabled={processingLoading || !fileId}
              startIcon={processingLoading ? <CircularProgress size={20} color="inherit" /> : <SearchIcon />}
              sx={{ minWidth: '150px' }}
            >
              {processingLoading ? "Processing..." : "Process PDF"}
            </Button>
          </Box>
        </Box>

      {DEBUG && (
        <Typography variant="caption" sx={{ bgcolor: 'yellow' }}>
          Stage: {effectiveStage}, Results: {hasCurrentStageResults ? 'Yes' : 'No'}
        </Typography>
      )}

      <Box sx={{ display: 'flex', gap: 2, mt: 2, mb: 2 }}>
        {/* Left column - Process PDF */}

        {/* Right column - Vision Inference */}
        <Box sx={{ flex: '1 1 auto' }}>
          <VisionInferenceOptions
            fileId={fileId}
            hasClassification={hasClassification}
            visionLoading={visionLoading}
            onRunVisionInference={handleRunVisionInference}
            visionResponse={visionResponse} // NEW: Pass the response prop
            defaultOptions={{
              model_choice: 'mistralVL',
              temperature: 0,
              top_k: 40,
              top_p: 0.95,
              zoom: 2.0,
              max_attempts: 2,
              timeout: 133,
              prompt: TABLE_EXTRACTION_INSTRUCTIONS
            }}
          />
        </Box>
      </Box>

      {/* Results section - only show if has results */}
      {hasCurrentStageResults && (
        <>
          <Divider sx={{ my: 2 }} />
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
            <Button
              variant="outlined"
              color="primary"
              onClick={toggleResults}
              startIcon={showResults ? <CancelIcon /> : <SearchIcon />}
            >
              {showResults ? "Hide Results" : "Show Results"}
            </Button>

            <Typography variant="body2" color="textSecondary">
              Found {resultsCount} results on page {currentPage} (Stage {effectiveStage})
            </Typography>
          </Box>
        </>
      )}
    </Box>
  );
};

export default PdfProcessingControls;