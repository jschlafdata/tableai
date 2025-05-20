import React, { useState } from 'react'
import Button from '@mui/material/Button'
import { runTask } from './api/taskRunner'
import CONSTANTS from '../../constants'

const ProcessButton = ({
  force = false,
  isFolder = false,
  pathOrId,
  disabled = false,
  onComplete = () => {}
}) => {
  const [isProcessing, setIsProcessing] = useState(false)

  const handleClick = async () => {
    if (!pathOrId) return
    setIsProcessing(true)
    try {
      // New endpoint
      const endpoint = `${CONSTANTS.API_BASE_URL}/dropbox/process`
      
      // Build the request body based on DropboxRegisterRequest
      // If it's a folder, we put pathOrId into `directories`;
      // if it's a file, into `file_ids`.
      const body = {
        file_ids: !isFolder ? [pathOrId] : [],
        directories: isFolder ? [pathOrId] : [],
        force_refresh: Boolean(force),
        stage: 0
      }

      // runTask is presumably a helper that fetches + handles tasks
      const result = await runTask(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
      })

      onComplete(result)
    } catch (err) {
      console.error('Processing failed', err)
      alert('Processing failed: ' + (err.message || err))
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <Button
      variant="contained"
      color="secondary"
      onClick={handleClick}
      disabled={!pathOrId || disabled || isProcessing}
    >
      {isProcessing ? 'Processing...' : 'Process'}
    </Button>
  )
}

export default ProcessButton