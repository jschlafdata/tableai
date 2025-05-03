// src/components/integrations/SyncButton.js
import React from 'react'
import Button from '@mui/material/Button'
import SyncIcon from '@mui/icons-material/Sync'
import CircularProgress from '@mui/material/CircularProgress'

export default function SyncButton({ onSync, sx, loading = false, ...props }) {
  return (
    <Button
      variant="contained"
      startIcon={
        loading
          ? <CircularProgress size={20} />
          : <SyncIcon />
      }
      onClick={onSync}
      sx={sx}
      disabled={loading || props.disabled}
      {...props}
    >
      {loading ? 'Syncing...' : 'Sync'}
    </Button>
  )
}