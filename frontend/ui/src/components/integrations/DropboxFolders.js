import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import Box from '@mui/material/Box'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import styles from './DropboxCardListStyles'

import { useBackgroundTask } from '../../hooks/api/tableai/useBackgroundTask'
import { useStage0Summary } from '../../hooks/api/tableai/useSummaries'
import { useDropboxData } from '../../hooks/api/dropbox/useDropboxData'

import DropboxBreadcrumbs from './DropboxBreadcrumbs'
import DropboxHeaderControls from './DropboxHeaderControls'
import DropboxFilterComponent from './DropboxFilterComponent'
import DropboxFileItem from './DropboxFileItem'

export default function DropboxCardListMUI({ path = '' }) {
  const [currentPath, setCurrentPath] = useState(() =>
    sessionStorage.getItem('dropbox_currentPath') || path
  )
  const [sortDir, setSortDir] = useState('asc')
  const [fileTypes, setFileTypes] = useState(['pdf'])
  const [ignoreRegex, setIgnoreRegex] = useState('')
  const [pathCategories, setPathCategories] = useState('')
  const [autoLabel, setAutoLabel] = useState('')
  const [selectedPaths, setSelectedPaths] = useState([])
  const [filteredItems, setFilteredItems] = useState([])
  const [force, setForce] = useState(false)
  const [filterCounts, setFilterCounts] = useState({ complete: 0, recovered: 0, failed: 0 })

  const { summaryMap } = useStage0Summary()
  const navigate = useNavigate()

  useEffect(() => {
    sessionStorage.setItem('dropbox_currentPath', currentPath)
  }, [currentPath])

  const {
    items,
    allFiles,
    history,
    error,
    fetchItems,
    fetchHistory,
    handleSync,
    isSyncing
  } = useDropboxData(currentPath, sortDir, useBackgroundTask({
    pollUrlBase: `/api/tasks`,
    onComplete: () => {
      fetchItems()
      fetchHistory()
    },
    onError: (err) => {
      alert('Sync failed: ' + (err.message || err))
    }
  }).startTask)

  const handleItemClick = (item) => {
    console.time('toggle')
    const isSynced = Boolean(history[item.path_lower])
    const dropboxId = history[item.path_lower]?.dropbox_id
    if (item.type === 'folder') {
      setCurrentPath(item.path_lower.replace(/^\//, ''))
    } else if (isSynced && dropboxId) {
      navigate(`/pdf_viewer?id=${dropboxId}`)
    }
    console.timeEnd('toggle')
  }

  const handleToggleSelect = (e, item) => {
    e.stopPropagation()
    const path = item?.path_lower
    if (!path) return
    console.time('toggle')
    setSelectedPaths(prev => {
      const path = item.path_lower
      return e.target.checked
        ? [...new Set([...prev, path])]
        : prev.filter(p => p !== path)
    })
    console.timeEnd('toggle')
  }

  const selectedItem =
    selectedPaths.length === 1
      ? filteredItems.find(i => i.path_lower === selectedPaths[0])
      : null

  const pathOrId = selectedItem
  ? selectedItem.type === 'folder'
    ? selectedItem.path_lower.replace(/^\//, '')
    : selectedItem.file_id
  : null

  const isFolder = selectedItem?.type === 'folder'

  if (error) {
    return (
      <Box sx={styles.errorBox}>
        <Typography sx={styles.errorText}>Error: {error.message}</Typography>
      </Box>
    )
  }

  return (
    <Box sx={styles.container}>
      <DropboxBreadcrumbs currentPath={currentPath} onCrumbClick={setCurrentPath} />

      <Box sx={styles.header}>
        <Box sx={styles.headerContent}>
          <DropboxHeaderControls
            selectedPaths={selectedPaths}
            fileTypes={fileTypes}
            sortDir={sortDir}
            ignoreRegex={ignoreRegex}
            pathCategories={pathCategories}
            autoLabel={autoLabel}
            isSyncing={isSyncing}
            force={force}
            onForceChange={setForce}
            onSync={() => handleSync(selectedPaths, fileTypes, ignoreRegex, pathCategories, autoLabel, force)}
            onSortChange={setSortDir}
            onFileTypeChange={(e, val) => val.length && setFileTypes(val)}
            onRegexChange={setIgnoreRegex}
            onCategoryChange={setPathCategories}
            onLabelChange={setAutoLabel}
            onProcessComplete={() => {
              fetchItems()
              fetchHistory()
            }}
            pathOrId={pathOrId}
            isFolder={isFolder}
            filterCounts={filterCounts}
          />
          <DropboxFilterComponent
            items={items}
            allFiles={allFiles}
            history={history}
            summaryMap={summaryMap}
            onFiltered={setFilteredItems}
            onCounts={setFilterCounts}
          />
        </Box>
      </Box>

      <Stack spacing={2}>
        {filteredItems.map(item => (
          <DropboxFileItem
            key={item.file_id || item.id || item.path_lower}  // ✅ Use file_id
            item={item}
            isSynced={Boolean(history[item.path_lower])}
            selectedPaths={selectedPaths}
            onSelectToggle={handleToggleSelect}
            onItemClick={handleItemClick}
            summaryMap={summaryMap}
          />
        ))}
      </Stack>
    </Box>
  )
}