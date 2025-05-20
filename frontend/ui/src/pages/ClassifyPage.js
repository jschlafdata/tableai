import React, { useState, useEffect, useRef } from 'react'
import {
  Box,
  Button,
  CircularProgress,
  Typography,
  Paper,
  TextField,
  Tooltip,
  IconButton,
  Stack,
  Chip
} from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import ExpandLessIcon from '@mui/icons-material/ExpandLess'
import NavigateBeforeIcon from '@mui/icons-material/NavigateBefore'
import NavigateNextIcon from '@mui/icons-material/NavigateNext'
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline'
import SaveIcon from '@mui/icons-material/Save'

export default function ClassifyPage() {
  const [loading, setLoading] = useState(true)
  const [labels, setLabels] = useState({})
  const [originalLabels, setOriginalLabels] = useState({}) // To track which labels were pre-existing
  const [samples, setSamples] = useState({})
  const [expanded, setExpanded] = useState({})
  const [pageIndices, setPageIndices] = useState({})
  const [savingStatus, setSavingStatus] = useState({}) // Track saving status for each classification
  const [savedMessages, setSavedMessages] = useState({}) // Individual success messages
  const [pdfWidths, setPdfWidths] = useState({}) // Track PDF image widths
  
  // Reference to measure container width
  const containerRef = useRef({})
  
  // Function to handle when PDF image loads and get its width
  const handleImageLoad = (classification, event) => {
    const width = event.target.naturalWidth
    setPdfWidths(prev => ({
      ...prev,
      [classification]: width
    }))
  }

  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [existingLabelsRes, sampleRes] = await Promise.all([
          fetch('http://localhost:8000/query/classify/existing_labels').then(res => res.json()),
          fetch('http://localhost:8000/query/classify/samples').then(res => res.json())
        ])

        // Process existing labels data which is an array of {label, classification} objects
        const labelMap = {}
        const originalLabelMap = {} // To track which ones were pre-existing
        
        // Convert the array of objects to a map where classification is the key
        if (Array.isArray(existingLabelsRes)) {
          existingLabelsRes.forEach(item => {
            if (item && item.classification) {
              labelMap[item.classification] = item.label || ''
              // Mark this label as original/pre-existing if it has a non-empty value
              originalLabelMap[item.classification] = !!item.label
            }
          })
        }
        
        const validSampleRes = sampleRes && typeof sampleRes === 'object' ? sampleRes : {}
        const expandMap = {}
        const pageIndexMap = {}
        const initialSavingStatus = {}
        const initialSavedMessages = {}

        // Process samples and ensure all have label entries (even if empty)
        Object.keys(validSampleRes).forEach((classification) => {
          expandMap[classification] = false
          pageIndexMap[classification] = 0
          initialSavingStatus[classification] = false
          initialSavedMessages[classification] = ''
          
          // If no existing label for this classification, initialize with empty string
          if (labelMap[classification] === undefined) {
            labelMap[classification] = ''
            originalLabelMap[classification] = false
          }
        })

        setLabels(labelMap)
        setOriginalLabels(originalLabelMap)
        setSamples(validSampleRes)
        setExpanded(expandMap)
        setPageIndices(pageIndexMap)
        setSavingStatus(initialSavingStatus)
        setSavedMessages(initialSavedMessages)
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

  const handleSaveLabel = (classification) => {
    // Update the saving status for this classification
    setSavingStatus(prev => ({ ...prev, [classification]: true }))
    
    // Create payload with just this single label
    const payload = { [classification]: labels[classification] }
    
    fetch('http://localhost:8000/ui/classify/labels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    })
      .then(res => res.json())
      .then(data => {
        // Mark this classification as having an original label after saving
        setOriginalLabels(prev => ({ ...prev, [classification]: true }))
        
        // Set success message for this classification
        setSavedMessages(prev => ({ 
          ...prev, 
          [classification]: `Saved label successfully` 
        }))
        
        // Clear message after 3 seconds
        setTimeout(() => {
          setSavedMessages(prev => ({ ...prev, [classification]: '' }))
        }, 3000)
      })
      .catch(err => {
        console.error(`Failed to save label for ${classification}`, err)
        setSavedMessages(prev => ({ 
          ...prev, 
          [classification]: `Error saving label` 
        }))
      })
      .finally(() => {
        setSavingStatus(prev => ({ ...prev, [classification]: false }))
      })
  }

  const handleSaveAll = () => {
    // Update all saving statuses
    const allSaving = Object.keys(labels).reduce((acc, classification) => {
      acc[classification] = true
      return acc
    }, {})
    setSavingStatus(allSaving)
    
    fetch('http://localhost:8000/ui/classify/labels', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(labels)
    })
      .then(res => res.json())
      .then(data => {
        // Mark all classifications as having original labels
        const allOriginal = Object.keys(labels).reduce((acc, classification) => {
          acc[classification] = true
          return acc
        }, {})
        setOriginalLabels(allOriginal)
        
        // Set success messages
        const successMessages = Object.keys(labels).reduce((acc, classification) => {
          acc[classification] = `Saved as part of batch update`
          return acc
        }, {})
        setSavedMessages(successMessages)
        
        // Clear messages after 3 seconds
        setTimeout(() => {
          setSavedMessages({})
        }, 3000)
      })
      .catch(err => {
        console.error('Failed to save labels', err)
        
        // Set error messages
        const errorMessages = Object.keys(labels).reduce((acc, classification) => {
          acc[classification] = `Error in batch save`
          return acc
        }, {})
        setSavedMessages(errorMessages)
      })
      .finally(() => {
        // Reset all saving statuses
        const notSaving = Object.keys(labels).reduce((acc, classification) => {
          acc[classification] = false
          return acc
        }, {})
        setSavingStatus(notSaving)
      })
  }

  return (
    <Box sx={styles.container}>
      <Typography variant="h5" gutterBottom>
        Classify PDFs and Assign Labels
      </Typography>

      {loading ? (
        <CircularProgress />
      ) : (
        <>
          {Object.keys(samples).length > 0 ? (
            <>
              <Typography variant="h6" gutterBottom>Assign Labels:</Typography>
              <Box sx={styles.saveAllContainer}>
                <Button
                  variant="contained"
                  onClick={handleSaveAll}
                  disabled={Object.values(savingStatus).some(status => status)}
                  startIcon={<SaveIcon />}
                >
                  Save All Labels
                </Button>
              </Box>
              
              {Object.keys(samples).map((classification) => {
                const label = labels[classification] || ''
                const isOriginal = originalLabels[classification]
                const sampleList = samples[classification] || []
                const currentIndex = pageIndices[classification] || 0
                const currentSample = sampleList[currentIndex]
                const previewUrl = currentSample ? `http://localhost:8000/ui/classify/sample_preview/${currentSample}` : null
                const isSaving = savingStatus[classification]
                const savedMessage = savedMessages[classification]

                return (
                  <Paper 
                    key={classification} 
                    sx={isOriginal ? styles.paperClassified : styles.paper}
                    ref={el => containerRef.current[classification] = el}
                  >
                    {isOriginal && (
                      <Chip 
                        icon={<CheckCircleOutlineIcon />} 
                        label="Classified" 
                        color="success" 
                        size="small"
                        sx={styles.classifiedChip}
                      />
                    )}
                    
                    <Box sx={styles.classificationHeader}>
                      <Typography 
                        variant="body2" 
                        sx={{
                          ...styles.classificationText,
                          fontWeight: isOriginal ? 'bold' : 'normal'
                        }}
                      >
                        {classification}
                      </Typography>
                      <IconButton onClick={() => handleToggleExpand(classification)}>
                        {expanded[classification] ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                      </IconButton>
                    </Box>

                    <Box sx={{
                      ...styles.inputContainer,
                      width: '600px',  // Fixed width for all input fields
                      maxWidth: '100%' // Ensure it doesn't overflow container
                    }}>
                      <TextField
                        fullWidth
                        label="Label"
                        value={label}
                        onChange={(e) => handleLabelChange(classification, e.target.value)}
                        sx={isOriginal ? styles.textFieldClassified : styles.textField}
                      />
                      
                      <Button
                        variant="outlined"
                        color="primary"
                        onClick={() => handleSaveLabel(classification)}
                        disabled={isSaving}
                        startIcon={<SaveIcon />}
                        sx={styles.saveButton}
                      >
                        {isSaving ? 'Saving...' : 'Save'}
                      </Button>
                    </Box>
                    
                    {savedMessage && (
                      <Typography 
                        sx={{
                          ...styles.savedMessage,
                          width: '600px',
                          maxWidth: '100%'
                        }} 
                        color="success.main" 
                        variant="body2"
                      >
                        {savedMessage}
                      </Typography>
                    )}

                    {expanded[classification] && previewUrl && (
                      <Box sx={styles.previewContainer}>
                        <img
                          src={previewUrl}
                          alt="PDF Preview"
                          style={styles.previewImage}
                          onLoad={(e) => handleImageLoad(classification, e)}
                        />
                        <Box sx={styles.paginationContainer}>
                          <IconButton 
                            onClick={() => handlePrevSample(classification)} 
                            disabled={currentIndex === 0}
                          >
                            <NavigateBeforeIcon />
                          </IconButton>
                          <Typography variant="caption" sx={styles.paginationText}>
                            {currentIndex + 1} / {sampleList.length}
                          </Typography>
                          <IconButton 
                            onClick={() => handleNextSample(classification)} 
                            disabled={currentIndex === sampleList.length - 1}
                          >
                            <NavigateNextIcon />
                          </IconButton>
                        </Box>
                      </Box>
                    )}
                  </Paper>
                )
              })}
            </>
          ) : (
            <Typography>No samples found to classify.</Typography>
          )}
        </>
      )}
    </Box>
  )
}

// Styling configuration object for easy adjustment
const styles = {
  // Container styles
  container: { 
    p: 3 
  },
  saveAllContainer: { 
    display: 'flex', 
    justifyContent: 'flex-end', 
    mb: 2 
  },
  inputContainer: { 
    display: 'flex', 
    mt: 1, 
    gap: 1
  },
  previewContainer: { 
    mt: 2, 
    textAlign: 'left'
  },
  paginationContainer: { 
    display: 'flex', 
    justifyContent: 'left', 
    mt: 1 
  },
  
  // Paper styles
  paper: { 
    p: 2, 
    my: 2,
    position: 'relative',
    maxWidth: '750px'  // Constrain width of the entire paper container
  },
  paperClassified: { 
    p: 2, 
    my: 2,
    position: 'relative',
    border: '1px solid #4caf50',
    maxWidth: '750px'  // Constrain width of the entire paper container
  },
  
  // Text and field styles
  classificationHeader: { 
    display: 'flex', 
    alignItems: 'center',
    maxWidth: '600px'  // Match the fixed width for consistency
  },
  classificationText: { 
    flex: 1, 
    wordBreak: 'break-word'
  },
  textField: {
    flex: 1
  },
  textFieldClassified: { 
    flex: 1,
    '& .MuiOutlinedInput-root': {
      bgcolor: 'rgba(76, 175, 80, 0.08)'
    }
  },
  savedMessage: { 
    mt: 1
  },
  paginationText: { 
    px: 2 
  },
  
  // UI element styles
  classifiedChip: { 
    position: 'absolute', 
    top: 8, 
    right: 40 
  },
  previewImage: { 
    maxWidth: '600px',  // Match the fixed width for consistency
    maxHeight: 800 
  },
  saveButton: {
    minWidth: '85px'  // Ensure button has consistent width
  }
}