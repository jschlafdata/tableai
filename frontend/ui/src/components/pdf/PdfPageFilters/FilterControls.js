// src/components/pdf/FilterControls.jsx
import React from 'react';
import { Box, TextField, Select, MenuItem, FormControl, InputLabel, Grid } from '@mui/material';

const FilterControls = ({
  subDirectories,
  selectedSubDir,
  onSubDirChange,
  searchQuery,
  onSearchChange,
  selectedFileIndex,
  setSelectedFileIndex,
  filteredMetadata,
  categoryFilters,
  onCategoryFilterChange,
  fileIdQuery,
  onFileIdChange,
  classificationFilter,
  onClassificationChange,
  availableClassifications,
  showClassificationFilter,
  selectedStage,
  onStageChange,
  stageOptions
}) => {
  // Get unique category keys from filtered metadata
  const categoryKeys = React.useMemo(() => {
    const keys = new Set();
    filteredMetadata.forEach(meta => {
      if (meta.path_categories) {
        Object.keys(meta.path_categories).forEach(key => keys.add(key));
      }
    });
    return Array.from(keys);
  }, [filteredMetadata]);

  // Get unique category values for each key
  const getCategoryValues = (key) => {
    const values = new Set(['all']);
    filteredMetadata.forEach(meta => {
      if (meta.path_categories && meta.path_categories[key]) {
        const value = meta.path_categories[key];
        if (value !== '' && value !== null) {
          values.add(value);
        }
      }
    });
    return Array.from(values);
  };

  return (
    <Box mb={3} p={2} border="1px solid #ddd" borderRadius="4px" bgcolor="#f9f9f9">
      <Grid container spacing={2}>
        {/* Directory filter */}
        <Grid item xs={12} md={4}>
          <FormControl fullWidth size="small">
            <InputLabel>Directory</InputLabel>
            <Select
              value={selectedSubDir}
              onChange={(e) => onSubDirChange(e.target.value)}
              label="Directory"
            >
              {subDirectories.map((dir) => (
                <MenuItem key={dir} value={dir}>
                  {dir}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>

        {/* File name search */}
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Search Filename"
            size="small"
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
          />
        </Grid>

        {/* File ID filter */}
        <Grid item xs={12} md={4}>
          <TextField
            fullWidth
            label="Filter by File ID"
            size="small"
            value={fileIdQuery}
            onChange={(e) => onFileIdChange(e.target.value)}
          />
        </Grid>

        {/* Stage selection - only show if there are multiple stage options */}
        {stageOptions.length > 0 && (
          <Grid item xs={12} md={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Processing Stage</InputLabel>
              <Select
                value={selectedStage}
                onChange={(e) => onStageChange(e.target.value)}
                label="Processing Stage"
              >
                {stageOptions.map((stage) => (
                  <MenuItem key={stage} value={stage}>
                    {`Stage ${stage}`}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        )}

        {/* Classification filter */}
        {showClassificationFilter && (
          <Grid item xs={12} md={4}>
            <FormControl fullWidth size="small">
              <InputLabel>Classification</InputLabel>
              <Select
                value={classificationFilter}
                onChange={(e) => onClassificationChange(e.target.value)}
                label="Classification"
              >
                {availableClassifications.map((classification) => (
                  <MenuItem key={classification} value={classification}>
                    {classification}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        )}

        {/* Category filters */}
        {categoryKeys.map((key) => (
          <Grid item xs={12} md={4} key={key}>
            <FormControl fullWidth size="small">
              <InputLabel>{key}</InputLabel>
              <Select
                value={categoryFilters[key] || 'all'}
                onChange={(e) => onCategoryFilterChange(key, e.target.value)}
                label={key}
              >
                {getCategoryValues(key).map((value) => (
                  <MenuItem key={value} value={value}>
                    {value}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
        ))}

        {/* File selection dropdown */}
        <Grid item xs={12}>
          <FormControl fullWidth size="small">
            <InputLabel>Selected File</InputLabel>
            <Select
              value={selectedFileIndex}
              onChange={(e) => setSelectedFileIndex(e.target.value)}
              label="Selected File"
            >
              {filteredMetadata.map((meta, index) => (
                <MenuItem key={meta.file_id || index} value={index}>
                  {meta.file_name || meta.name || `File ${index + 1}`}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
      </Grid>
    </Box>
  );
};

export default FilterControls;