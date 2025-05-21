import React, { useState } from 'react';
import {
  Paper,
  Box,
  Button,
  Typography,
  IconButton,
  Collapse,
  FormGroup,
  FormControlLabel,
  Switch,
  Chip,
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';

const QueryFilterPanel = ({
  currentStageResults,
  queryToggles,
  toggleQueryLabel,
  toggleAllQueryLabels,
  getQueryLabelResultsCount,
  getColorForLabel,
}) => {
  const [showQueryControls, setShowQueryControls] = useState(true);

  if (!currentStageResults || currentStageResults.length === 0) {
    return null; // No panel if there's nothing to display
  }

  return (
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
      {/* Header + Show/Hide All */}
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

      {/* Query Toggles */}
      <Collapse in={showQueryControls}>
        <FormGroup sx={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap', gap: 1 }}>
          {currentStageResults.map((result) => {
            const label = result.query_label;
            const description = result?.description || '';
            const count = getQueryLabelResultsCount(label);

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
  );
};

export default QueryFilterPanel;