import React, { useState, useEffect, useMemo } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'

import FilterControls from '../components/pdfs/FilterControls'
import BoundingBoxes from '../components/pdfs/BoundingBoxes'
import PageBreakLines from '../components/pdfs/PageBreakLines'
import { ToggleProvider, useToggles } from '../components/pdfs/ToggleContext'

import {
  getStageNames,
  processBoundingBoxes,
  getColorMap
} from '../components/pdfs/stageConfig'
import { processPdfInfo } from '../components/pdfs/pdfDataProcessor'

pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs'

function ExtractViewerPage() {
  const [metadataList, setMetadataList] = useState([])
  const [selectedFileIndex, setSelectedFileIndex] = useState(0)
  const [numPages, setNumPages] = useState(null)
  const [pageNumber, setPageNumber] = useState(1)
  const [loading, setLoading] = useState(true)

  const [currentMetadata, setCurrentMetadata] = useState(null)
  const [pageDimensions, setPageDimensions] = useState({ width: 0, height: 0 })

  const scale = 2.0
  const stageNames = getStageNames()
  const [selectedStage, setSelectedStage] = useState(stageNames[0] || 'stage1')

  const [searchQuery, setSearchQuery] = useState('')
  const [selectedSubDir, setSelectedSubDir] = useState('all')
  const { toggles } = useToggles()

  // Load file list
  useEffect(() => {
    setLoading(true)
    fetch('http://localhost:8000/extract_metadata')
      .then((res) => res.json())
      .then((data) => {
        const metadataArray = Object.values(data)
        setMetadataList(metadataArray)
      })
      .catch((err) => {
        console.error('Error fetching file list:', err)
      })
      .finally(() => {
        setLoading(false)
      })
  }, [])

  // Compute subdirectories (if present)
  const subDirectories = useMemo(() => {
    const dirs = new Set(
        metadataList.flatMap((meta) => meta?.directories || [])
      )
    return ['all', ...Array.from(dirs)]
  }, [metadataList])

  // Filter logic
  const filteredMetadata = useMemo(() => {
    return metadataList.filter((entry) => {
      const fileId = entry?.file_id || ''
      const label = entry?.label || ''
      const subDir = entry?.sub_dir || ''
      const matchesSearch =
        !searchQuery ||
        fileId.toLowerCase().includes(searchQuery.toLowerCase()) ||
        label.toLowerCase().includes(searchQuery.toLowerCase())
      const matchesSubDir =
        selectedSubDir === 'all' || subDir === selectedSubDir
      return matchesSearch && matchesSubDir
    })
  }, [metadataList, searchQuery, selectedSubDir])

  // Auto reset index if filtering narrows list
  useEffect(() => {
    if (filteredMetadata.length > 0 && selectedFileIndex >= filteredMetadata.length) {
      setSelectedFileIndex(0)
    }
  }, [filteredMetadata])

  // Fetch metadata for selected file
  useEffect(() => {
    const fileEntry = filteredMetadata[selectedFileIndex]
    if (!fileEntry?.file_id) return

    fetch(`http://localhost:8000/extractions/${fileEntry.file_id}`)
      .then((res) => res.json())
      .then((data) => setCurrentMetadata(data))
      .catch((err) => {
        console.error('Error fetching file metadata:', err)
        setCurrentMetadata(null)
      })
  }, [filteredMetadata, selectedFileIndex])

  // Reset pagination on file change
  useEffect(() => {
    setPageNumber(1)
    setNumPages(null)
  }, [selectedFileIndex])

  const pdfInfo = useMemo(() => {
    return processPdfInfo(currentMetadata, selectedStage)
  }, [currentMetadata, selectedStage])

  const boundingBoxesForThisFile = useMemo(() => {
    return processBoundingBoxes(currentMetadata, selectedStage, toggles)
  }, [currentMetadata, selectedStage, toggles])

  const colorMap = useMemo(() => {
    return getColorMap(selectedStage)
  }, [selectedStage])

  const onDocumentLoadSuccess = ({ numPages }) => setNumPages(numPages)
  const onPageRenderSuccess = (page) => {
    setPageDimensions({ width: page.width, height: page.height })
  }

  const goToPrevPage = () => setPageNumber((prev) => (prev > 1 ? prev - 1 : prev))
  const goToNextPage = () => setPageNumber((prev) => (prev < (numPages || 1) ? prev + 1 : prev))

  // Early exits
  if (loading) return <div style={styles.container}>Loading files...</div>
  if (filteredMetadata.length === 0) return <div style={styles.container}>No files match your filter.</div>
  if (!currentMetadata && !loading) return <div style={styles.container}>No metadata loaded.</div>

  const currentEntry = filteredMetadata[selectedFileIndex]
  const fileId = currentEntry?.file_id
  const label = currentEntry?.label
  const pdfPath = `http://localhost:8000/files/.processing/outputs/${selectedStage}/${fileId}.pdf`
  const { combinedPdfWidth, combinedPdfHeight, pageBreaks } = pdfInfo || {}

  return (
    <div style={styles.container}>
      <h1 style={styles.heading}>PDF Viewer</h1>

      <FilterControls
        subDirectories={subDirectories}
        selectedSubDir={selectedSubDir}
        onSubDirChange={setSelectedSubDir}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        selectedFileIndex={selectedFileIndex}
        setSelectedFileIndex={setSelectedFileIndex}
        filteredMetadata={filteredMetadata}
        selectedStage={selectedStage}
        onStageChange={setSelectedStage}
      />

      <div style={styles.fileInfo}>
        <div><strong>File ID:</strong> {fileId}</div>
        <div><strong>Label:</strong> {label || '[unlabeled]'}</div>
        <div><strong>PDF Path:</strong> {pdfPath}</div>
      </div>

      <div style={{ position: 'relative', display: 'inline-block' }}>
        <Document
          file={pdfPath}
          onLoadSuccess={onDocumentLoadSuccess}
          onLoadError={(err) => console.error('PDF load error:', err)}
        >
          <Page
            pageNumber={pageNumber}
            scale={scale}
            onRenderSuccess={onPageRenderSuccess}
            renderAnnotationLayer={false}
            renderTextLayer={false}
          />
        </Document>

        <div style={{ position: 'absolute', top: 0, left: 0 }}>
          {boundingBoxesForThisFile && (
            <BoundingBoxes
              boxesData={boundingBoxesForThisFile}
              transformCoord={(coords, color) => {
                const ratioX = pageDimensions.width / combinedPdfWidth
                const ratioY = pageDimensions.height / combinedPdfHeight
                const [x0, y0, x1, y1] = coords
                return {
                  position: 'absolute',
                  left: x0 * ratioX,
                  top: y0 * ratioY,
                  width: (x1 - x0) * ratioX,
                  height: (y1 - y0) * ratioY,
                  border: `2px solid ${color || 'red'}`,
                  backgroundColor: color || 'rgba(255, 0, 0, 0.2)',
                  pointerEvents: 'none'
                }
              }}
              colorMap={colorMap}
            />
          )}
          {pageBreaks?.length > 0 && (
            <PageBreakLines
              pageBreaks={pageBreaks}
              docWidth={pageDimensions.width}
              docHeight={pageDimensions.height}
              pdfHeight={combinedPdfHeight}
            />
          )}
        </div>
      </div>

      <div style={{ marginTop: '10px' }}>
        <button onClick={goToPrevPage} disabled={pageNumber <= 1} style={styles.button}>Previous Page</button>
        <button onClick={goToNextPage} disabled={pageNumber >= (numPages || 1)} style={styles.button}>Next Page</button>
        <p>Page {pageNumber} of {numPages || 1}</p>
      </div>
    </div>
  )
}

const styles = {
  container: {
    margin: '20px',
    fontFamily: 'Arial, sans-serif'
  },
  heading: {
    color: '#333',
    marginBottom: '20px'
  },
  button: {
    padding: '8px 14px',
    marginRight: '10px',
    border: 'none',
    borderRadius: '4px',
    backgroundColor: '#1976d2',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '14px'
  },
  fileInfo: {
    backgroundColor: '#fff',
    borderRadius: '6px',
    padding: '10px 15px',
    boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
    marginBottom: '20px'
  }
}

const PdfViewerPageWithProviders = () => (
  <ToggleProvider>
    <ExtractViewerPage />
  </ToggleProvider>
)

export default PdfViewerPageWithProviders;