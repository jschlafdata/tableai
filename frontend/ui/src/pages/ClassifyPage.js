import React, { useState, useEffect } from 'react'
import {
  Box,
  Button,
  CircularProgress,
  Typography,
  Paper,
  TextField,
  Tooltip,
  IconButton
} from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import ExpandLessIcon from '@mui/icons-material/ExpandLess'
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore'
import NavigateNextIcon from '@mui/icons-material/NavigateNext'

export default function ClassifyPage() {
  const [loading, setLoading] = useState(true)
  const [labels, setLabels] = useState({})
  const [samples, setSamples] = useState({})
  const [expanded, setExpanded] = useState({})
  const [pageIndices, setPageIndices] = useState({})
  const [saving, setSaving] = useState(false)
  const [savedMessage, setSavedMessage] = useState('')

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [labelRes, sampleRes] = await Promise.all([
          fetch('http://localhost:8000/query/classify/labels').then(res => res.json()),
          fetch('http://localhost:8000/query/classify/samples').then(res => res.json())
        ])

        const labelMap = {}
        const expandMap = {}
        const pageIndexMap = {}

        Object.keys(sampleRes).forEach((classification) => {
          labelMap[classification] = labelRes[classification] || ''
          expandMap[classification] = false
          pageIndexMap[classification] = 0
        })

        setLabels(labelMap)
        setSamples(sampleRes)
        setExpanded(expandMap)
        setPageIndices(pageIndexMap)
      } catch (err) {
        console.error('Failed to load initial data', err)
      } finally {
        setLoading(false)
      }
    }

    fetchInitialData()
  }, [])

  const handleLabelChange = (classification, value) => {
    setLabels(prev => ({ ...prev, [classification]: value }))
  }

  const handleToggleExpand = (classification) => {
    setExpanded(prev => ({ ...prev, [classification]: !prev[classification] }))
  }

  const handleNextSample = (classification) => {
    setPageIndices(prev => ({
      ...prev,
      [classification]: Math.min(prev[classification] + 1, samples[classification].length - 1)
    }))
  }

  const handlePrevSample = (classification) => {
    setPageIndices(prev => ({
      ...prev,
      [classification]: Math.max(prev[classification] - 1, 0)
    }))
  }

  const handleSave = () => {
    setSaving(true)
    fetch('http://localhost:8000/ui/classify/labels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(labels)
    })
      .then(res => res.json())
      .then(data => {
        setSavedMessage(`Saved ${data.count} labels`)
      })
      .catch(err => {
        console.error('Failed to save labels', err)
      })
      .finally(() => setSaving(false))
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Classify PDFs and Assign Labels
      </Typography>

      {loading ? (
        <CircularProgress />
      ) : (
        Object.keys(labels).length > 0 && (
          <>
            <Typography variant="h6">Assign Labels:</Typography>
            {Object.entries(labels).map(([classification, label]) => {
              const sampleList = samples[classification] || []
              const currentIndex = pageIndices[classification] || 0
              const currentSample = sampleList[currentIndex]
              const previewUrl = currentSample ? `http://localhost:8000/ui/classify/sample_preview/${currentSample}` : null

              return (
                <Paper key={classification} sx={{ p: 2, my: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center' }}>
                    <Typography variant="body2" sx={{ flex: 1, wordBreak: 'break-word' }}>
                      {classification}
                    </Typography>
                    <IconButton onClick={() => handleToggleExpand(classification)}>
                      {expanded[classification] ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                    </IconButton>
                  </Box>

                  <Tooltip title="Click to edit label" placement="left" arrow enterDelay={300}>
                    <TextField
                      fullWidth
                      label="Label"
                      value={label}
                      onChange={(e) => handleLabelChange(classification, e.target.value)}
                      sx={{ mt: 1 }}
                    />
                  </Tooltip>

                  {expanded[classification] && previewUrl && (
                    <Box sx={{ mt: 2, textAlign: 'left' }}>
                      <img
                        src={previewUrl}
                        alt="PDF Preview"
                        style={{ maxWidth: '100%', maxHeight: 800 }}
                      />
                      <Box sx={{ display: 'flex', justifyContent: 'left', mt: 1 }}>
                        <IconButton onClick={() => handlePrevSample(classification)} disabled={currentIndex === 0}>
                          <NavigateBeforeIcon />
                        </IconButton>
                        <Typography variant="caption" sx={{ px: 2 }}>
                          {currentIndex + 1} / {sampleList.length}
                        </Typography>
                        <IconButton onClick={() => handleNextSample(classification)} disabled={currentIndex === sampleList.length - 1}>
                          <NavigateNextIcon />
                        </IconButton>
                      </Box>
                    </Box>
                  )}
                </Paper>
              )
            })}
            <Button
              variant="outlined"
              onClick={handleSave}
              disabled={saving}
              sx={{ mt: 2 }}
            >
              {saving ? 'Saving...' : 'Save Labels'}
            </Button>
            {savedMessage && (
              <Typography sx={{ mt: 2 }} color="success.main">
                {savedMessage}
              </Typography>
            )}
          </>
        )
      )}
    </Box>
  )
}
