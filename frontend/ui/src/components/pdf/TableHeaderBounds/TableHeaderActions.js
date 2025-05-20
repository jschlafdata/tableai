import React from 'react';
import { Box, Button, CircularProgress, Tooltip, Typography } from '@mui/material';
import TableChartIcon from '@mui/icons-material/TableChart';
import CancelIcon from '@mui/icons-material/Cancel';

const TableHeaderActions = ({
  hasClassification,
  fileId,
  tableHeaders,
  tableHeadersLoading,
  onProcessTableHeaders,
  showResults,
  toggleResults,
  currentPage
}) => {
  // Count results on the current page
  let resultsCount = 0;
  if (tableHeaders?.results?.pages) {
    const pageIndex = String(currentPage - 1);
    const pageItems = tableHeaders.results.pages[pageIndex] || [];
    resultsCount = pageItems.length;
  }

  return (
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
  );
};

export default TableHeaderActions;