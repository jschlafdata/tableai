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
  onStageChange
}) {
  // Get toggle state and setter from context
  const { toggles, setToggle, toggleDefinitions } = useToggles();
  
  // Get available toggles for the current stage
  const availableToggles = getStageToggles(selectedStage);
  
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
            <option key={meta.file_name + index} value={index}>
              {meta.file_name}
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
    marginBottom: '20px'
  },
  controlGroup: {
    marginBottom: '10px',
    display: 'flex',
    alignItems: 'center'
  },
  label: {
    minWidth: '120px',
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
    paddingTop: '15px'
  },
  togglesTitle: {
    fontWeight: 'bold',
    marginBottom: '10px'
  },
  togglesGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
    gap: '10px'
  },
  toggleItem: {
    display: 'flex',
    alignItems: 'center'
  }
};

export default FilterControls;
