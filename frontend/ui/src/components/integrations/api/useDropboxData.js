import { useCallback, useEffect, useState } from 'react'
import { fetchDropboxFolder, fetchStage0Summary, fetchSyncHistory } from './fetchMetadata'
import { useBackgroundTask } from './useBackgroundTask'
import CONSTANTS from '../../../constants'

export function useDropboxData(currentPath, sortDir) {
  const [items, setItems] = useState([])
  const [allFiles, setAllFiles] = useState([])
  const [history, setHistory] = useState({})
  const [error, setError] = useState(null)

  const { startTask, isRunning } = useBackgroundTask({
    pollUrlBase: `${CONSTANTS.API_BASE_URL}/tasks`,
    onComplete: () => {
      fetchItems()
      fetchHistory()
    },
    onError: (err) => {
      console.error('Sync task failed:', err)
    }
  })

  const fetchItems = useCallback(async () => {
    try {
      setError(null)
      const sortedItems = await fetchDropboxFolder(currentPath, sortDir)
      setItems(sortedItems)
    } catch (err) {
      setError(err)
    }
  }, [currentPath, sortDir])

  const fetchAllFiles = useCallback(async () => {
    const files = await fetchStage0Summary()
    setAllFiles(files)
  }, [])

  const fetchHistory = useCallback(async () => {
    const map = await fetchSyncHistory()
    setHistory(map)
  }, [])

  const handleSync = useCallback(async (selectedPaths, fileTypes, ignoreRegex, pathCategories, autoLabel, force) => {
    if (!selectedPaths.length) return

    const payload = {
      force_refresh: force,
      paths: selectedPaths,
      file_types: fileTypes,
      ignore: ignoreRegex,
      path_categories: pathCategories,
      auto_label: autoLabel
    }

    await startTask(`${CONSTANTS.API_BASE_URL}/${CONSTANTS.DROPBOX_BASE}/sync`, {
      method: 'POST',
      body: payload
    })
  }, [startTask])

  useEffect(() => {
    fetchItems()
    fetchHistory()
  }, [fetchItems, fetchHistory])

  useEffect(() => {
    fetchAllFiles()
  }, [fetchAllFiles])

  return {
    items,
    allFiles,
    history,
    error,
    fetchItems,
    fetchAllFiles,
    fetchHistory,
    handleSync,
    isSyncing: isRunning  // rename for clarity at call site
  }
}