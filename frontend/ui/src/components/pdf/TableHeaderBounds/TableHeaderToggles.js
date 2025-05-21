import React, { useState, useEffect } from 'react';
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
  Chip,
  List,
  ListItem,
  ListItemText,
  Divider,
  Tooltip,
  Badge
} from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';

const TableHeaderToggles = ({
  tableHeaders,
  tableToggles,
  setTableToggles,
  metadataToggles, // Now receiving metadataToggles from parent
  setMetadataToggles, // Now receiving setMetadataToggles from parent
  showResults,
  currentPage,
  colorMap
}) => {
  const [showTableControls, setShowTableControls] = useState(true);
  const [expandedTables, setExpandedTables] = useState({});

  // Initialize metadata toggles when table headers change and if metadataToggles hasn't been initialized yet
  useEffect(() => {
    if (!tableHeaders?.results?.pages || (metadataToggles && Object.keys(metadataToggles).length > 0)) return;
    
    const newMetadataToggles = {};
    
    Object.values(tableHeaders.results.pages).forEach(pageItems => {
      pageItems.forEach(item => {
        if (item.table_index != null) {
          const tableKey = `Table ${item.table_index}`;
          
          if (!newMetadataToggles[tableKey]) {
            newMetadataToggles[tableKey] = {};
          }
          
          // Add main bbox toggle
          newMetadataToggles[tableKey]['main_bbox'] = true;
          
          // Add metadata fields toggles
          if (item.table_metadata) {
            Object.keys(item.table_metadata).forEach(metaKey => {
              if (item.table_metadata[metaKey]?.bbox) {
                newMetadataToggles[tableKey][metaKey] = true;
              }
            });
          }
          
          // Add any other attributes that have bbox info
          ['bounds_index', 'hierarchy', 'col_hash'].forEach(attr => {
            if (item[attr]?.bbox) {
              newMetadataToggles[tableKey][attr] = true;
            }
          });
        }
      });
    });
    
    // Only update if we have data and setMetadataToggles is available
    if (Object.keys(newMetadataToggles).length > 0 && setMetadataToggles) {
      setMetadataToggles(newMetadataToggles);
    }
  }, [tableHeaders, metadataToggles, setMetadataToggles]);

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

  const toggleTableExpansion = (tableLabel) => {
    setExpandedTables(prev => ({
      ...prev,
      [tableLabel]: !prev[tableLabel]
    }));
  };

  const toggleMetadataField = (tableLabel, field) => {
    // Only update if setMetadataToggles is available
    if (!setMetadataToggles) return;
    
    // Update the metadata toggle for the specific field
    setMetadataToggles(prev => {
      // Create a deep copy to ensure we're not mutating the previous state
      const newToggles = JSON.parse(JSON.stringify(prev || {}));
      
      // Initialize the table entry if it doesn't exist
      if (!newToggles[tableLabel]) {
        newToggles[tableLabel] = {};
      }
      
      // Toggle the field
      newToggles[tableLabel][field] = !newToggles[tableLabel][field];
      
      console.log('TOGGLED METADATA FIELD:', tableLabel, field, newToggles[tableLabel][field]);
      return newToggles;
    });
  };

  const toggleAllMetadataFields = (tableLabel, value) => {
    // Only update if setMetadataToggles is available
    if (!setMetadataToggles || !metadataToggles?.[tableLabel]) return;
    
    setMetadataToggles(prev => {
      // Create a deep copy to ensure we're not mutating the previous state
      const newToggles = JSON.parse(JSON.stringify(prev || {}));
      
      // Set all fields to the specified value
      const fields = Object.keys(newToggles[tableLabel] || {});
      fields.forEach(field => {
        newToggles[tableLabel][field] = value;
      });
      
      console.log('TOGGLED ALL METADATA FIELDS:', tableLabel, value, newToggles[tableLabel]);
      return newToggles;
    });
  };

  const getTableResultsCount = (tableLabel) => {
    if (!tableHeaders?.results?.pages) return 0;
    const tableIndex = parseInt(tableLabel.replace('Table ', ''));
    const pageIndex = String(currentPage - 1);
    const pageItems = tableHeaders.results.pages[pageIndex] || [];
    return pageItems.filter(item => item.table_index === tableIndex).length;
  };

  const getMetadataFieldsForTable = (tableLabel) => {
    if (!tableHeaders?.results?.pages) return [];
    const tableIndex = parseInt(tableLabel.replace('Table ', ''));
    const pageIndex = String(currentPage - 1);
    const pageItems = tableHeaders.results.pages[pageIndex] || [];
    
    // Find the first item for this table to extract metadata fields
    const tableItem = pageItems.find(item => item.table_index === tableIndex);
    
    if (!tableItem) return [];
    
    const fields = [];
    
    // Add main bbox
    fields.push({
      key: 'main_bbox',
      label: 'Main Bounding Box',
      description: 'The primary bounding box for this table element'
    });
    
    // Add metadata fields
    if (tableItem.table_metadata) {
      Object.entries(tableItem.table_metadata).forEach(([key, value]) => {
        if (value?.bbox) {
          fields.push({
            key,
            label: formatFieldName(key),
            description: `Table metadata: ${key}`
          });
        }
      });
    }
    
    // Add other relevant fields
    ['bounds_index', 'hierarchy', 'col_hash', 'columns'].forEach(attr => {
      if (tableItem[attr]?.bbox) {
        fields.push({
          key: attr,
          label: formatFieldName(attr),
          description: `Table attribute: ${attr}`
        });
      }
    });
    
    return fields;
  };

  const formatFieldName = (field) => {
    // Convert snake_case to Title Case
    return field
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const getActiveMetadataCount = (tableLabel) => {
    if (!metadataToggles?.[tableLabel]) return 0;
    return Object.values(metadataToggles[tableLabel]).filter(Boolean).length;
  };

  const getTotalMetadataCount = (tableLabel) => {
    if (!metadataToggles?.[tableLabel]) return 0;
    return Object.keys(metadataToggles[tableLabel]).length;
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
        <FormGroup sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
          {Object.keys(tableToggles).map(tableLabel => {
            const count = getTableResultsCount(tableLabel);
            const color = colorMap[tableLabel];
            const isTableExpanded = expandedTables[tableLabel] || false;
            const activeMetadataCount = getActiveMetadataCount(tableLabel);
            const totalMetadataCount = getTotalMetadataCount(tableLabel);
            
            return (
              <Paper
                key={tableLabel}
                sx={{
                  bgcolor: 'white',
                  borderRadius: '4px',
                  border: `3px solid ${tableToggles[tableLabel] ? color : '#e0e0e0'}`,
                  overflow: 'hidden'
                }}
              >
                <Box
                  sx={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    p: 1,
                    bgcolor: tableToggles[tableLabel] ? `${color}20` : 'transparent',
                    borderBottom: isTableExpanded ? `1px solid ${color}40` : 'none'
                  }}
                >
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
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
                        ml: 1,
                        bgcolor: color + '20',
                        border: `1px solid ${color}`
                      }}
                      title={`${count} items on this page`}
                    />
                    <Tooltip title="Metadata fields visible/total">
                      <Chip
                        label={`${activeMetadataCount}/${totalMetadataCount}`}
                        size="small"
                        sx={{
                          ml: 1,
                          bgcolor: 'transparent',
                          border: '1px solid #aaa'
                        }}
                      />
                    </Tooltip>
                  </Box>
                  <Box>
                    <IconButton
                      size="small"
                      onClick={() => toggleTableExpansion(tableLabel)}
                      disabled={!tableToggles[tableLabel]}
                    >
                      {isTableExpanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </IconButton>
                  </Box>
                </Box>

                <Collapse in={isTableExpanded && tableToggles[tableLabel]}>
                  <Box
                    sx={{
                      p: 1,
                      bgcolor: '#f9f9f9'
                    }}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'flex-end', mb: 1 }}>
                      <Button 
                        size="small" 
                        startIcon={<VisibilityIcon fontSize="small" />}
                        onClick={() => toggleAllMetadataFields(tableLabel, true)}
                        sx={{ mr: 1 }}
                      >
                        Show All Fields
                      </Button>
                      <Button 
                        size="small" 
                        startIcon={<VisibilityOffIcon fontSize="small" />}
                        onClick={() => toggleAllMetadataFields(tableLabel, false)}
                      >
                        Hide All Fields
                      </Button>
                    </Box>
                    
                    <List dense sx={{ width: '100%' }}>
                      {getMetadataFieldsForTable(tableLabel).map((field, index) => (
                        <React.Fragment key={field.key}>
                          {index > 0 && <Divider component="li" />}
                          <ListItem
                            secondaryAction={
                              <FormControlLabel
                                control={
                                  <Switch
                                    checked={metadataToggles?.[tableLabel]?.[field.key] || false}
                                    onChange={() => toggleMetadataField(tableLabel, field.key)}
                                    size="small"
                                  />
                                }
                                label=""
                              />
                            }
                          >
                            <ListItemText 
                              primary={
                                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                  <Typography variant="body2">
                                    {field.label}
                                  </Typography>
                                  <Tooltip title={field.description}>
                                    <IconButton size="small" sx={{ ml: 0.5, p: 0.25 }}>
                                      <InfoOutlinedIcon fontSize="small" sx={{ fontSize: '1rem' }} />
                                    </IconButton>
                                  </Tooltip>
                                </Box>
                              } 
                            />
                          </ListItem>
                        </React.Fragment>
                      ))}
                    </List>
                  </Box>
                </Collapse>
              </Paper>
            );
          })}
        </FormGroup>
      </Collapse>
    </Paper>
  );
};

export default TableHeaderToggles;