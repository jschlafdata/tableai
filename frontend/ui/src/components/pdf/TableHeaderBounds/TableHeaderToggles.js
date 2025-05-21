import React, { useState } from 'react';
import {
  Paper,
  Box,
  Typography,
  Button,
  IconButton,
  Collapse,
  FormGroup,
  FormControlLabel,
  Switch,
  Chip
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';

const TableHeaderToggles = ({
  tableHeaders,
  tableToggles,
  setTableToggles,
  showResults,
  currentPage,
  colorMap
}) => {
  const [showTableControls, setShowTableControls] = useState(true);

  if (!tableHeaders?.results || Object.keys(tableToggles).length === 0) {
    return null;
  }

  const toggleAllTables = (value) => {
    const updated = {};
    Object.keys(tableToggles).forEach(key => {
      updated[key] = value;
    });
    setTableToggles(updated);
  };

  const toggleTable = (tableLabel) => {
    setTableToggles(prev => ({
      ...prev,
      [tableLabel]: !prev[tableLabel]
    }));
  };

  const getTableResultsCount = (tableLabel) => {
    if (!tableHeaders?.results?.pages) return 0;
    const tableIndex = parseInt(tableLabel.replace('Table ', ''));
    const pageIndex = String(currentPage - 1);
    const pageItems = tableHeaders.results.pages[pageIndex] || [];
    return pageItems.filter(item => item.table_index === tableIndex).length;
  };

  return (
    <Paper
      elevation={1}
      sx={{
        p: 2,
        mb: 2,
        bgcolor: '#f5f5f5',
        border: '1px solid #e0e0e0',
        borderRadius: '4px'
      }}
    >
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 1
        }}
      >
        <Typography variant="subtitle1" fontWeight="bold">
          Table Headers
        </Typography>
        <Box>
          <Button size="small" onClick={() => toggleAllTables(true)} sx={{ mr: 1 }}>
            Show All
          </Button>
          <Button size="small" onClick={() => toggleAllTables(false)}>
            Hide All
          </Button>
          <IconButton
            size="small"
            onClick={() => setShowTableControls(prev => !prev)}
            sx={{ ml: 1 }}
          >
            {showTableControls ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        </Box>
      </Box>

      <Collapse in={showTableControls}>
        <FormGroup sx={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: 1 }}>
          {Object.keys(tableToggles).map(tableLabel => {
            const count = getTableResultsCount(tableLabel);
            const color = colorMap[tableLabel];

            return (
              <Box
                key={tableLabel}
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  bgcolor: 'white',
                  p: 1,
                  borderRadius: '4px',
                  border: '1px solid #e0e0e0',
                  minWidth: '200px'
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}
                >
                  <FormControlLabel
                    control={
                      <Switch
                        checked={tableToggles[tableLabel] !== false}
                        onChange={() => toggleTable(tableLabel)}
                        size="small"
                      />
                    }
                    label={
                      <Typography variant="body2" fontWeight="bold">
                        {tableLabel}
                      </Typography>
                    }
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
  );
};

export default TableHeaderToggles;