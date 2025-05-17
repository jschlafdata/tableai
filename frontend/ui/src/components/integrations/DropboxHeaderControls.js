import React from 'react'
import Box from '@mui/material/Box'
import ToggleButton from '@mui/material/ToggleButton'
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup'
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward'
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward'
import SyncButton from './SyncButton'
import ProcessButton from './ProcessButton'
import ForceSwitch from './ForceSwitch'

const DropboxHeaderControls = ({
  selectedPaths,
  fileTypes,
  sortDir,
  ignoreRegex,
  pathCategories,
  autoLabel,
  force,
  onForceChange,    // <-- new prop
  isSyncing,
  onSync,
  onSortChange,
  onFileTypeChange,
  onRegexChange,
  onCategoryChange,
  onLabelChange,
  onProcessComplete,
  pathOrId,
  isFolder,
  filterCounts
}) => {
  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
        <SyncButton
          onSync={onSync}
          loading={isSyncing}
          disabled={!selectedPaths.length}
        />

        {/* Now we control ForceSwitch via the parent’s state */}
        <ForceSwitch force={force} onForceChange={onForceChange} />

        <ProcessButton
          force={force}      // pass force down to ProcessButton
          isFolder={isFolder}
          pathOrId={pathOrId}
          disabled={!pathOrId}
          onComplete={onProcessComplete}
        />

        <ToggleButtonGroup
          value={sortDir}
          exclusive
          onChange={(_, v) => v && onSortChange(v)}
          size="small"
        >
          <ToggleButton value="asc">
            <ArrowUpwardIcon fontSize="small" />
          </ToggleButton>
          <ToggleButton value="desc">
            <ArrowDownwardIcon fontSize="small" />
          </ToggleButton>
        </ToggleButtonGroup>

        <ToggleButtonGroup
          value={fileTypes}
          onChange={onFileTypeChange}
          size="small"
          color="primary"
        >
          <ToggleButton value="pdf">PDF</ToggleButton>
          <ToggleButton value="xlsx">Excel</ToggleButton>
          <ToggleButton value="png">PNG</ToggleButton>
        </ToggleButtonGroup>
      </Box>

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
        <input
          type="text"
          placeholder="Ignore Folder Paths"
          value={ignoreRegex}
          onChange={(e) => onRegexChange(e.target.value)}
          style={{
            height: '32px',
            padding: '4px 8px',
            fontSize: '0.875rem',
            borderRadius: '4px',
            border: '1px solid #ccc',
            width: '200px'
          }}
        />
        <input
          type="text"
          placeholder="Path Categories"
          value={pathCategories}
          onChange={(e) => onCategoryChange(e.target.value)}
          style={{
            height: '32px',
            padding: '4px 8px',
            fontSize: '0.875rem',
            borderRadius: '4px',
            border: '1px solid #ccc',
            width: '200px'
          }}
        />
        <input
          type="text"
          placeholder="Auto Label"
          value={autoLabel}
          onChange={(e) => onLabelChange(e.target.value)}
          style={{
            height: '32px',
            padding: '4px 8px',
            fontSize: '0.875rem',
            borderRadius: '4px',
            border: '1px solid #ccc',
            width: '200px'
          }}
        />
      </Box>
    </Box>
  )
}

export default DropboxHeaderControls