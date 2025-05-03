// src/components/integrations/DropboxCardListMUI.js
import React, { useState, useEffect, useCallback } from 'react'
import Box from '@mui/material/Box'
import Stack from '@mui/material/Stack'
import Breadcrumbs from '@mui/material/Breadcrumbs'
import Link from '@mui/material/Link'
import Typography from '@mui/material/Typography'
import ToggleButton from '@mui/material/ToggleButton'
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup'
import Checkbox from '@mui/material/Checkbox'
import IconButton from '@mui/material/IconButton'
import Avatar from '@mui/material/Avatar'
import AvatarGroup from '@mui/material/AvatarGroup'
import FolderIcon from '@mui/icons-material/Folder'
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile'
import MoreVertIcon from '@mui/icons-material/MoreVert'
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward'
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import CONSTANTS from '../../constants'
import formatDate from '../../utils/dates'
import formatBytes from '../../utils/bytes'
import SyncButton from './SyncButton'
import styles from './DropboxCardListStyles'
import axios from 'axios'

const api = axios.create({
  baseURL: CONSTANTS.API_BASE_URL,
  timeout: 600_000,   // 10 minutes; use 0 for “no timeout”
})


export default function DropboxCardListMUI({ path = '' }) {
  const [currentPath, setCurrentPath] = useState(() => {
        return sessionStorage.getItem('dropbox_currentPath') || path
  })
  const [items, setItems]                 = useState([])
  const [error, setError]                 = useState(null)
  const [sortDir, setSortDir]             = useState('asc')
  const [selectedPaths, setSelectedPaths] = useState([])
  const [history, setHistory]             = useState({})
  const [isSyncing, setIsSyncing]         = useState(false)

  const fetchItems = useCallback(() => {
    setError(null)
    const segments = String(currentPath)
      .split('/')
      .filter(Boolean)
      .map(encodeURIComponent)
      .join('/')
    const url = segments
      ? `${CONSTANTS.API_BASE_URL}/folders/${segments}`
      : `${CONSTANTS.API_BASE_URL}/folders`

    fetch(url)
      .then(res => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
        return res.json()
      })
      .then(data => {
        if (!Array.isArray(data)) throw new Error('Bad format')
        const sorted = data.sort((a, b) =>
          sortDir === 'asc'
            ? a.name.localeCompare(b.name)
            : b.name.localeCompare(a.name)
        )
        setItems(sorted)
        setSelectedPaths([])
      })
      .catch(err => setError(err))
  }, [currentPath, sortDir])

  const fetchHistory = useCallback(() => {
    fetch(`${CONSTANTS.API_BASE_URL}/sync/history`)
      .then(res => res.json())
      .then(records => {
        const map = {}
        records.forEach(r => { map[r.path_lower] = r })
        setHistory(map)
      })
      .catch(console.error)
  }, [])

  useEffect(() => {
    fetchItems()
    fetchHistory()
  }, [fetchItems, fetchHistory])

  const crumbs = [
    { name: 'Home', path: '' },
    ...String(currentPath)
      .split('/')
      .filter(Boolean)
      .map((seg, i, all) => ({
        name: decodeURIComponent(seg),
        path: all.slice(0, i + 1).join('/')
      }))
  ]

  function handleItemClick(item) {
    if (item.type === 'folder') {
      setCurrentPath(item.path_lower.replace(/^\//, ''))
    } else {
      window.open(`/files${item.path_lower}`, '_blank')
    }
  }

  function handleToggleSelect(e, item) {
    e.stopPropagation()
    const p = item.path_lower
    setSelectedPaths(prev =>
      e.target.checked ? [...prev, p] : prev.filter(x => x !== p)
    )
  }

  async function handleSync() {
    if (!selectedPaths.length) return
    setIsSyncing(true)
  
    try {
      const res = await api.post('/sync', { paths: selectedPaths })
      console.log('Synced:', res.data.synced_files)
      fetchItems()
      fetchHistory()
    }
    catch (err) {
      if (err.code === 'ECONNABORTED') {
        alert('Sync timed out – please try again.')
      } else {
        console.error(err)
        alert('Sync failed: ' + (err.message || err))
      }
    }
    finally {
      setIsSyncing(false)
    }
  }

  if (error) {
    return (
      <Box sx={styles.errorBox}>
        <Typography sx={styles.errorText}>
          Error: {error.message}
        </Typography>
      </Box>
    )
  }

  return (
    <Box sx={styles.container}>
      <Breadcrumbs sx={{ mb: 2 }}>
        {crumbs.map(crumb => (
          <Link
            key={crumb.path}
            underline="hover"
            color={crumb.path === currentPath ? 'text.primary' : 'inherit'}
            sx={{ cursor: 'pointer' }}
            onClick={() => setCurrentPath(crumb.path)}
          >
            {crumb.name}
          </Link>
        ))}
      </Breadcrumbs>

      <Box sx={styles.header}>
        <Box sx={styles.headerContent}>
          <SyncButton
            onSync={handleSync}
            loading={isSyncing}
            disabled={!selectedPaths.length}
            sx={{ mr: 2 }}
          />
          <ToggleButtonGroup
            value={sortDir}
            exclusive
            onChange={(_, v) => v && setSortDir(v)}
            size="small"
            sx={styles.toggleButtonGroup}
          >
            <ToggleButton value="asc">
              <ArrowUpwardIcon fontSize="small" />
            </ToggleButton>
            <ToggleButton value="desc">
              <ArrowDownwardIcon fontSize="small" />
            </ToggleButton>
          </ToggleButtonGroup>
        </Box>
      </Box>

      <Stack spacing={2}>
        {items.map(item => {
          const isSynced = Boolean(history[item.path_lower])
          return (
            <Box
              key={item.id}
              sx={{
                ...styles.itemRow,
                cursor: item.type === 'folder' ? 'pointer' : 'default'
              }}
              onClick={() => handleItemClick(item)}
            >
              <Checkbox
                onClick={e => e.stopPropagation()}
                checked={selectedPaths.includes(item.path_lower)}
                onChange={e => handleToggleSelect(e, item)}
              />

              <Box sx={styles.itemContent}>
                {item.type === 'folder' ? (
                  <FolderIcon color="warning" />
                ) : (
                  <InsertDriveFileIcon color="action" />
                )}
                <Typography variant="body2" ml={1}>
                  {item.name}
                </Typography>
                {isSynced && (
                  <CheckCircleIcon
                    color="success"
                    fontSize="small"
                    sx={{ ml: 1 }}
                  />
                )}
              </Box>

              <Typography sx={styles.fileSize}>
                {item.size != null ? formatBytes(item.size) : '—'}
              </Typography>
              <Typography sx={styles.fileType}>{item.type}</Typography>
              <Typography sx={styles.fileDate}>
                {formatDate(item.server_modified)}
              </Typography>

              <AvatarGroup max={3} sx={styles.avatarGroup}>
                {item.shared_with.map((u, i) => (
                  <Avatar
                    key={i}
                    src={u.avatar_url}
                    alt={u.name}
                    sx={styles.avatar}
                  />
                ))}
              </AvatarGroup>

              <Box sx={styles.actionButton}>
                <IconButton
                  size="small"
                  onClick={e => e.stopPropagation()}
                >
                  <MoreVertIcon />
                </IconButton>
              </Box>
            </Box>
          )
        })}
      </Stack>
    </Box>
  )
}