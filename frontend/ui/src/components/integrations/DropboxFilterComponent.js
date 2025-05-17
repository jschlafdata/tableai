import React, { useState, useEffect, useMemo } from 'react'
import ToggleButton from '@mui/material/ToggleButton'
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup'
import Box from '@mui/material/Box'
import { useSearchParams } from 'react-router-dom'

const FILTER_TYPES = ['all', 'complete', 'recovered', 'failed']

const DropboxFilterComponent = ({ items, allFiles, history, summaryMap, onFiltered, onCounts }) => {
  const [searchParams, setSearchParams] = useSearchParams()
  const initialFilter = searchParams.get('filter') || 'all'
  const [filter, setFilter] = useState(initialFilter)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setFilter(initialFilter)
  }, [initialFilter])

  const counts = useMemo(() => {
    const result = { complete: 0, recovered: 0, failed: 0 }

    allFiles.forEach(item => {
      if (item.type === 'folder') return

      const dropboxId = history?.[item.path_lower]?.dropbox_safe_id || item.dropbox_safe_id
      const meta = dropboxId && summaryMap?.[dropboxId]
      if (!meta) return

      const isComplete = Boolean(meta.stage0_complete) === false
      const isRecovered = String(meta.recovered_pdf) === '1' || String(meta.recovered_pdf) === 'true'

      if (isRecovered) result.recovered++
      else if (isComplete) result.complete++
      else result.failed++
    })

    return result
  }, [allFiles, summaryMap, history])

  useEffect(() => {
    if (onCounts) onCounts(counts)
  }, [counts, onCounts])

  const handleFilterChange = (_, newFilter) => {
    if (newFilter) {
      console.log('FILTER CHANGED TO:', newFilter)
      console.log("items", items)
      setFilter(newFilter)
      setSearchParams({ filter: newFilter })
    }
  }

  const filteredItems = useMemo(() => {
    if (filter === 'all') return items
  
    return allFiles
    .map(item => ({
      ...item,
      path_lower:
        item.path_lower ||
        history?.[item.file_id]?.path_lower || // ← patch from history by file_id
        history?.[item.path_lower]?.path_lower // fallback
    }))
    .filter(item => {
      if (item.type === 'folder') return false  
      console.time('meta filter')
      const fileId = item.file_id
      const meta = summaryMap?.[fileId]
      if (!meta) return false
  
      const isComplete = Boolean(meta.stage0_complete) === false
      const isRecovered = String(meta.recovered_pdf) === '1' || String(meta.recovered_pdf) === 'true'
      console.timeEnd('meta filter')

      switch (filter) {
        case 'recovered': return isRecovered
        case 'complete': return isComplete
        case 'failed': return !isComplete && !isRecovered
        default: return true
      }
    })
  }, [filter, items, allFiles, summaryMap, history])

  useEffect(() => {
    setLoading(true)
  }, [filter])

  useEffect(() => {
    if (onFiltered) {
      console.log('Firing onFiltered with', filteredItems.length, 'items')
      onFiltered(filteredItems)
      setLoading(false)
    }
  }, [filteredItems, onFiltered])

  return (
    <Box sx={{ mb: 1 }}>
      <ToggleButtonGroup
        value={filter}
        exclusive
        onChange={handleFilterChange}
        size="small"
      >
        <ToggleButton value="all">ALL</ToggleButton>
        <ToggleButton value="complete">COMPLETE</ToggleButton>
        <ToggleButton value="recovered">RECOVERED</ToggleButton>
        <ToggleButton value="failed">FAILED</ToggleButton>
      </ToggleButtonGroup>
    </Box>
  )
}

export default DropboxFilterComponent
