import React from 'react';
import { useToggles } from './ToggleContext';
import stageConfig, { getStageToggles } from './stageConfig';

function FilterControls({
  subDirectories,
  selectedSubDir,
  onSubDirChange,
  searchQuery,
  onSearchChange,
  selectedFileIndex,
  setSelectedFileIndex,
  filteredMetadata,
  selectedStage,
  onStageChange,
  categoryFilters,
  onCategoryFilterChange,
  fileIdQuery,
  onFileIdChange,
  classificationFilter,
  onClassificationChange,
  availableClassifications,
  showClassificationFilter
}) {
  // Get toggle state and setter from context
  const { toggles, setToggle, toggleDefinitions } = useToggles();
  
  // Get available toggles for the current stage
  const availableToggles = getStageToggles(selectedStage);

  const categoryOptions = {};

  filteredMetadata.forEach((meta) => {
    const categories = meta.path_categories || {};
    Object.entries(categories).forEach(([key, value]) => {
      if (!categoryOptions[key]) categoryOptions[key] = new Set();
      categoryOptions[key].add(value);
    });
  });
  
  return (
    <div style={styles.controlsContainer}>
      {/* Sub-directory filter */}
      <div style={styles.controlGroup}>
        <label style={styles.label}>Sub Directory:</label>
        <select
          value={selectedSubDir}
          onChange={(e) => onSubDirChange(e.target.value)}
          style={styles.select}
        >
          {subDirectories.map((dir) => (
            <option key={dir} value={dir}>
              {dir}
            </option>
          ))}
        </select>
      </div>

      {/* Search filter */}
      <div style={styles.controlGroup}>
        <label style={styles.label}>Search:</label>
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder="Filter by filename..."
          style={styles.input}
        />
      </div>
      {/* Dynamic category filters (e.g., client, year, month) */}
      {Object.entries(categoryOptions).map(([key, values]) => (
        <div key={key} style={styles.controlGroup}>
          <label style={styles.label}>{key.charAt(0).toUpperCase() + key.slice(1)}:</label>
          <select
            value={categoryFilters?.[key] || 'all'}
            onChange={(e) => onCategoryFilterChange(key, e.target.value)}
            style={styles.select}
          >
            <option value="all">All</option>
            {[...values].sort().map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>
      ))}
      {/* File selector */}
      <div style={styles.controlGroup}>
        <label style={styles.label}>File:</label>
        <select
          value={selectedFileIndex}
          onChange={(e) => setSelectedFileIndex(Number(e.target.value))}
          style={styles.select}
          disabled={filteredMetadata.length === 0}
        >
          {filteredMetadata.map((meta, index) => (
            <option key={meta.dropbox_safe_id || index} value={index}>
              {meta.file_name || meta.dropbox_safe_id || '[unnamed file]'}
              {meta.path_categories
                ? ` (${Object.values(meta.path_categories).filter(Boolean).join(' · ')})`
                : ''}
            </option>
          ))}
        </select>
        <span style={styles.count}>
          {filteredMetadata.length} file{filteredMetadata.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Stage selector */}
      <div style={styles.controlGroup}>
        <label style={styles.label}>Stage:</label>
        <select
          value={selectedStage}
          onChange={(e) => onStageChange(e.target.value)}
          style={styles.select}
        >
          {Object.keys(stageConfig).map((stage) => (
            <option key={stage} value={stage}>
              {stage}
            </option>
          ))}
        </select>
      </div>

        {/* Add classification filter */}
        {showClassificationFilter && availableClassifications && availableClassifications.length > 1 && (
          <div style={styles.controlGroup}>
            <label style={styles.label}>Classification:</label>
            <select
              value={classificationFilter}
              onChange={(e) => onClassificationChange(e.target.value)}
              style={styles.select}
            >
              {availableClassifications.map((label) => (
                <option key={label} value={label}>
                  {label === 'all' ? 'All Classifications' : label}
                </option>
              ))}
            </select>
          </div>
        )}

      {/* New file_id search filter */}
      <div style={styles.controlGroup}>
        <label style={styles.label}>File ID:</label>
        <input
          type="text"
          value={fileIdQuery || ''}
          onChange={(e) => onFileIdChange(e.target.value)}
          placeholder="Filter by file ID..."
          style={styles.input}
        />
      </div>

      {/* Dynamic toggle controls */}
      <div style={styles.togglesContainer}>
        <div style={styles.togglesTitle}>Show/Hide Elements:</div>
        <div style={styles.togglesGrid}>
          {availableToggles.map(({ key, label }) => (
            <div key={key} style={styles.toggleItem}>
              <label>
                <input
                  type="checkbox"
                  checked={toggles[key] || false}
                  onChange={(e) => setToggle(key, e.target.checked)}
                />
                {label}
              </label>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

const styles = {
  controlsContainer: {
    backgroundColor: '#f5f5f5',
    padding: '15px',
    borderRadius: '8px',
    marginBottom: '20px',
    display: 'flex',
    flexWrap: 'wrap',
    justifyContent: 'left',
    gap: '20px'
  },
  controlGroup: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    marginBottom: 0  // horizontal layout = no vertical stacking
  },
  label: {
    fontWeight: 'bold'
  },
  select: {
    padding: '8px',
    borderRadius: '4px',
    border: '1px solid #ccc',
    minWidth: '200px'
  },
  input: {
    padding: '8px',
    borderRadius: '4px',
    border: '1px solid #ccc',
    minWidth: '200px'
  },
  count: {
    marginLeft: '10px',
    color: '#666',
    fontSize: '14px'
  },
  togglesContainer: {
    marginTop: '15px',
    borderTop: '1px solid #ddd',
    paddingTop: '15px',
    width: '100%',
    textAlign: 'left'
  },
  togglesTitle: {
    fontWeight: 'bold',
    marginBottom: '10px'
  },
  togglesGrid: {
    display: 'flex',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: '15px'
  },
  toggleItem: {
    display: 'flex',
    alignItems: 'center'
  }
};

export default FilterControls;
