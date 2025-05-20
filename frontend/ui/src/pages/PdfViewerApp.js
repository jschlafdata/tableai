
import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Divider, Box } from '@mui/material';
import { Document, Page, pdfjs } from 'react-pdf';
import FileMetadataDisplay from '../components/pdf/FileMetadataDisplay';
import PdfNavigation from '../components/pdf/PdfNavigation';
import FilterControls from '../components/pdf/FilterControls';
import LoadingState from '../components/pdf/LoadingState';
import PdfProcessingResults from '../components/pdf/PdfProcessingResults';
import TableHeaderBounds from '../components/pdf/TableHeaderBounds/TableHeaderBounds';
import { PdfDataProvider } from '../context/PdfDataContext';
import { usePdfDataContext } from '../context/PdfDataContext';
import { Backdrop, CircularProgress } from '@mui/material';

import '../components/pdf/css/pdf-styles.css';

// Set worker for PDF.js
pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';


function PdfViewerApp() {

  const { 
    metadata, 
    metadataLoading, 
    reloadMetadata,
    // Make sure we clear results when changing files
    clearAllResults
  } = usePdfDataContext();

  // URL and UI states
  const [selectedFileIndex, setSelectedFileIndex] = useState(0);
  // Add a state to track the actual fileId of the selected file
  const [selectedFileId, setSelectedFileId] = useState(null);
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [categoryFilters, setCategoryFilters] = useState({});
  const [fileIdQuery, setFileIdQuery] = useState('');
  const [classificationFilter, setClassificationFilter] = useState('all');
  const [pageDimensions, setPageDimensions] = useState({ width: 0, height: 0 });
  const [selectedStage, setSelectedStage] = useState(0); // Default to stage 0
  const [selectedSubDir, setSelectedSubDir] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const pdfContainerRef = useRef(null);

  const [searchParams] = useSearchParams();
  const scale = 1.5;

  // Debug log to track state updates
  useEffect(() => {
    console.log("State update - selectedFileId:", selectedFileId);
  }, [selectedFileId]);

  // Sync URL param to file selection
  const fileId = searchParams.get('id');
  useEffect(() => {
    if (!fileId || metadata.length === 0) return;
    
    // Log to help diagnose
    console.log("URL fileId param:", fileId);
    console.log("Looking through metadata for matching file");
    
    const index = metadata.findIndex(
      m => (m.file_id === fileId || m.dropbox_id === fileId)
    );
    
    if (index >= 0) {
      console.log("Found matching file at index:", index);
      setSelectedFileIndex(index);
      setSelectedFileId(fileId);
    } else {
      console.log("No matching file found for ID:", fileId);
    }
  }, [fileId, metadata]);

  // Extract subdirectories
  const subDirectories = useMemo(() => {
    const dirs = new Set();
    metadata.forEach(meta => {
      if (meta.type === 'file' && meta.directories && meta.directories.length > 0) {
        dirs.add(meta.directories.join('/'));
      }
    });
    return ['all', ...Array.from(dirs)];
  }, [metadata]);

  // Filter metadata based on all active filters
  const filteredMetadata = useMemo(() => {
    return metadata.filter((meta) => {
      if (meta.type !== 'file') return false;

      const fileName = (meta.file_name || meta.name || '').toLowerCase();
      const matchesSubDir =
        selectedSubDir === 'all' ||
        (meta.directories && meta.directories.join('/').includes(selectedSubDir));

      const matchesSearch = !searchQuery || fileName.includes(searchQuery.toLowerCase());

      const fileId = (meta.file_id || meta.dropbox_id || '').toLowerCase();
      const matchesFileId = !fileIdQuery || fileId.includes(fileIdQuery.toLowerCase());

      const matchesCategories = Object.entries(categoryFilters).every(([key, val]) => {
        if (!val || val === 'all') return true;
        return meta?.path_categories?.[key] === val;
      });

      const matchesClassification =
        classificationFilter === 'all' ||
        meta.classification === classificationFilter;

      return (
        matchesSubDir &&
        matchesSearch &&
        matchesFileId &&
        matchesCategories &&
        matchesClassification
      );
    });
  }, [
    metadata,
    selectedSubDir,
    searchQuery,
    fileIdQuery,
    categoryFilters,
    classificationFilter,
  ]);

  const showEmptyState = filteredMetadata.length === 0;
  
  // Get the current metadata by ID first, then by index if ID not found
  const currentMetadata = useMemo(() => {
    // Debug log
    console.log("Computing currentMetadata", { 
      selectedFileId, 
      filteredMetadataLength: filteredMetadata.length,
      selectedFileIndex
    });
    
    // If we have a selectedFileId, find the file in filteredMetadata that matches that ID
    if (selectedFileId && filteredMetadata.length > 0) {
      const foundFile = filteredMetadata.find(
        m => (m.file_id === selectedFileId || m.dropbox_id === selectedFileId)
      );
      
      // If found, return it
      if (foundFile) {
        console.log("Found file by ID in filtered results:", foundFile.file_name);
        return foundFile;
      } else {
        console.log("File ID not found in filtered results:", selectedFileId);
      }
    }
    
    // Default to index-based selection
    const indexBasedFile = showEmptyState ? null : filteredMetadata[selectedFileIndex] || null;
    console.log("Using index-based file selection:", indexBasedFile?.file_name);
    return indexBasedFile;
  }, [filteredMetadata, selectedFileIndex, selectedFileId, showEmptyState]);

  // Update selectedFileId whenever the current metadata changes (based on index)
  useEffect(() => {
    if (currentMetadata) {
      const newFileId = currentMetadata.file_id || currentMetadata.dropbox_id;
      console.log("Current metadata file ID:", newFileId);
      
      if (newFileId !== selectedFileId) {
        console.log("Updating selectedFileId to:", newFileId);
        setSelectedFileId(newFileId);
        // Clear any previous results when changing files
        clearAllResults();
      }
    }
  }, [currentMetadata, selectedFileId, clearAllResults]);

  // Get available stages for the current file
  const stageOptions = useMemo(() => {
    if (!currentMetadata) return [0];
    const completedStages = currentMetadata.completed_stages || [0];
    return completedStages.sort((a, b) => a - b);
  }, [currentMetadata]);

  useEffect(() => {
    if (!currentMetadata) return;
    // If current selectedStage is not in stageOptions, or stageOptions only has one value, reset
    if (!stageOptions.includes(selectedStage)) {
      // Choose the highest (latest) stage by default, or the first in the list
      setSelectedStage(stageOptions[stageOptions.length - 1] || 0);
    }
  }, [currentMetadata, stageOptions, selectedStage]);

  // Make sure selected stage is valid for the current file
  useEffect(() => {
    if (currentMetadata && stageOptions.length > 0) {
      if (!stageOptions.includes(selectedStage)) {
        setSelectedStage(stageOptions[0]);
      }
    }
  }, [currentMetadata, stageOptions, selectedStage]);

  // Reset selection index if filtered results change
  useEffect(() => {
    if (!showEmptyState && selectedFileIndex >= filteredMetadata.length) {
      console.log("Resetting selectedFileIndex because it's out of bounds");
      setSelectedFileIndex(0);
      
      // Also update the selectedFileId when resetting the index
      if (filteredMetadata.length > 0) {
        const newFile = filteredMetadata[0];
        const newFileId = newFile.file_id || newFile.dropbox_id;
        console.log("Setting selectedFileId to first item:", newFileId);
        setSelectedFileId(newFileId);
      }
    }
  }, [filteredMetadata, selectedFileIndex, showEmptyState]);

  // Reset page when file changes
  useEffect(() => {
    setPageNumber(1);
    setNumPages(null);
  }, [selectedFileId]);  // Changed from selectedFileIndex to selectedFileId

  // Get available classifications
  const availableClassifications = useMemo(() => {
    const classifications = new Set(['all']);
    metadata.forEach(meta => {
      if (meta.classification && meta.classification !== '' && meta.classification !== null) {
        classifications.add(meta.classification);
      }
    });
    return Array.from(classifications);
  }, [metadata]);

  // Navigation handlers
  const goToPrevPage = () => setPageNumber((p) => Math.max(p - 1, 1));
  const goToNextPage = () => setPageNumber((p) => Math.min(p + 1, numPages || 1));

  // Document callbacks
  const onDocumentLoadSuccess = ({ numPages }) => setNumPages(numPages);
  const onPageRenderSuccess = (page) => {
    setPageDimensions({ width: page.width, height: page.height });
  };

  // Generate PDF path
  const getPdfPath = () => {
    if (!currentMetadata) return null;
    const stageStr = String(selectedStage);
    const stagePaths = currentMetadata.stage_paths || {};
    if (stagePaths[stageStr] && stagePaths[stageStr].mount_path) {
      return `http://localhost:8000${stagePaths[stageStr].mount_path}`;
    }
    return `http://localhost:8000/files/stage${stageStr}/${currentMetadata.file_id || currentMetadata.dropbox_id}.pdf`;
  };

  const pdfPath = getPdfPath();
  const currentFileId = currentMetadata?.file_id || currentMetadata?.dropbox_id;

  // Enhanced handler for selecting a file by index 
  const handleFileSelect = useCallback((index) => {
    console.log("handleFileSelect called with index:", index);
    
    if (index >= 0 && index < filteredMetadata.length) {
      const targetFile = filteredMetadata[index];
      const newFileId = targetFile.file_id || targetFile.dropbox_id;
      
      console.log(`Selecting file at index ${index}: ${targetFile.file_name} (${newFileId})`);
      
      setSelectedFileIndex(index);
      setSelectedFileId(newFileId);
      // Clear results when changing files
      clearAllResults();
    }
  }, [filteredMetadata, clearAllResults]);

  return (
    <div style={styles.container}>
      {metadataLoading && (
        <div
          style={{
            position: "fixed",
            top: 0, left: 0,
            width: "100vw", height: "100vh",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            zIndex: 2000,
            pointerEvents: "none", // lets user still interact with page (optional)
            background: "transparent",
          }}
        >
          <CircularProgress size={64} />
        </div>
      )}
      <h1 style={styles.heading}>PDF Viewer</h1>

      <FilterControls
        subDirectories={subDirectories}
        selectedSubDir={selectedSubDir}
        onSubDirChange={setSelectedSubDir}
        searchQuery={searchQuery}
        onSearchChange={setSearchQuery}
        selectedFileIndex={selectedFileIndex}
        setSelectedFileIndex={handleFileSelect}  // Use the new handler
        filteredMetadata={filteredMetadata}
        categoryFilters={categoryFilters}
        onCategoryFilterChange={(key, value) =>
          setCategoryFilters((prev) => ({ ...prev, [key]: value }))
        }
        fileIdQuery={fileIdQuery}
        onFileIdChange={setFileIdQuery}
        classificationFilter={classificationFilter}
        onClassificationChange={setClassificationFilter}
        availableClassifications={availableClassifications}
        showClassificationFilter={true}
        selectedStage={selectedStage}
        onStageChange={setSelectedStage}
        stageOptions={stageOptions}
      />

      {showEmptyState ? (
        <div style={{ padding: '10px', fontStyle: 'italic', color: '#888' }}>
          No files match your search or directory filter.
        </div>
      ) : (
        <>
          {/* Debug information to help track the issue */}
          <div style={{ padding: '10px', border: '1px solid #ddd', margin: '10px 0', backgroundColor: '#f5f5f5' }}>
            <h4 style={{ margin: '0 0 5px 0' }}>Debug Info</h4>
            <p style={{ margin: '0 0 3px 0' }}><strong>Selected File ID:</strong> {selectedFileId || 'None'}</p>
            <p style={{ margin: '0 0 3px 0' }}><strong>Current File Name:</strong> {currentMetadata?.file_name || 'None'}</p>
            <p style={{ margin: '0 0 3px 0' }}><strong>Current File ID:</strong> {currentFileId || 'None'}</p>
          </div>

          {/* Responsive layout for metadata and processing results */}
          <div className="metadata-processing-container" style={styles.metadataProcessingContainer}>
            {/* Left side: FileMetadataDisplay */}
            {/* <div className="metadata-column" style={styles.metadataColumn}>
              <FileMetadataDisplay metadata={currentMetadata} />
            </div> */}

            {/* Right side: Processing Controls */}
            <div className="processing-column" style={styles.processingColumn}>
              <h3 className="column-header">Processing Controls</h3>
              {currentFileId && (
                <>
                  <Box sx={{ mb: 3, pt: 1 }}>
                    <PdfProcessingResults
                      fileId={currentFileId}
                      stage={selectedStage}
                      pageDimensions={pageDimensions}
                      currentPage={pageNumber}
                      pdfContainerRef={pdfContainerRef}
                      scale={scale}
                      metadata={currentMetadata}
                    />
                  </Box>
                  <Divider sx={{ my: 2 }} />
                  <Box sx={{ pt: 1 }}>
                    <TableHeaderBounds
                      fileId={currentFileId}
                      stage={selectedStage}
                      pageDimensions={pageDimensions}
                      currentPage={pageNumber}
                      pdfContainerRef={pdfContainerRef}
                      scale={scale}
                      metadata={currentMetadata}
                    />
                  </Box>
                </>
              )}
            </div>
          </div>

          <Divider sx={{ my: 2 }} />

          {/* PDF Container - Critical for bounding box positioning */}
          <div
            ref={pdfContainerRef}
            className="pdf-container"
            style={{
              position: 'relative',
              display: 'inline-block',
              overflow: 'hidden',
              margin: '0 auto',
            }}
          >
            {pdfPath ? (
              <>
                <Document file={pdfPath} onLoadSuccess={onDocumentLoadSuccess}>
                  <Page
                    pageNumber={pageNumber}
                    scale={scale}
                    onRenderSuccess={onPageRenderSuccess}
                    renderAnnotationLayer={false}
                    renderTextLayer={false}
                  />
                </Document>

                {/* Debug overlay for bounding box testing */}
                <div
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    border: '1px solid red',
                    pointerEvents: 'none',
                    zIndex: 50,
                    opacity: 0.1,
                    backgroundColor: 'rgba(255, 0, 0, 0.1)',
                  }}
                >
                  <div
                    style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      backgroundColor: 'red',
                      color: 'white',
                      padding: '2px 5px',
                      borderRadius: '3px',
                      fontSize: '10px',
                      pointerEvents: 'none',
                    }}
                  >
                    PDF Area
                  </div>
                </div>
                <PdfNavigation
                  pageNumber={pageNumber}
                  numPages={numPages}
                  goToPrevPage={goToPrevPage}
                  goToNextPage={goToNextPage}
                />
              </>
            ) : (
              <div style={{ padding: '20px', border: '1px solid #ddd' }}>
                PDF file not available
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

const styles = {
  container: {
    margin: '20px',
    fontFamily: 'Arial, sans-serif',
  },
  heading: {
    color: '#333',
    marginBottom: '20px',
  },
  metadataProcessingContainer: {
    display: 'flex',
    flexDirection: 'row',
    gap: '20px',
    marginBottom: '20px',
  },
  metadataColumn: {
    flex: '1',
    minWidth: '300px',
  },
  processingColumn: {
    flex: '2',
    border: '1px solid #ddd',
    borderRadius: '4px',
    padding: '15px',
    backgroundColor: '#f9f9f9',
    boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.05)',
  },
};

export default function DisplayPdf() {
  return (
    <PdfDataProvider>
      <PdfViewerApp />
    </PdfDataProvider>
  );
}