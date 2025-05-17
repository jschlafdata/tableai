// src/components/integrations/ProcessButton.js
import React, { useState } from 'react'
import Button from '@mui/material/Button'
import FormControlLabel from '@mui/material/FormControlLabel'
import Switch from '@mui/material/Switch'
import { runTask } from '../../utils/taskRunner'
import CONSTANTS from '../../constants'

const ProcessButton = ({ isFolder = false, pathOrId, disabled = false, onComplete = () => {} }) => {
  const [force, setForce] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)

  const handleClick = async () => {
    if (!pathOrId) return
    setIsProcessing(true)
    try {
      const endpoint = `${CONSTANTS.API_BASE_URL}/tableai/nodes/0`
      const params = new URLSearchParams({
        [isFolder ? 'directory' : 'file_id']: pathOrId,
        force: String(force)
      })
      const url = `${endpoint}?${params.toString()}`
      const result = await runTask(url, { method: 'POST' })
      onComplete(result)
    } catch (err) {
      console.error('Processing failed', err)
      alert('Processing failed: ' + (err.message || err))
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <>
      <FormControlLabel
        control={
          <Switch
            checked={force}
            onChange={(e) => setForce(e.target.checked)}
            color="primary"
          />
        }
        label="Force"
        sx={{ mr: 2 }}
      />
      <Button
        variant="contained"
        color="secondary"
        onClick={handleClick}
        disabled={!pathOrId || disabled || isProcessing}
      >
        {isProcessing ? 'Processing...' : 'Process'}
      </Button>
    </>
  )
}

export default ProcessButton