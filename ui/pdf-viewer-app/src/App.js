import React, { useState, useEffect, useMemo } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';

// Components
import FilterControls from './FilterControls';
import BoundingBoxes from './BoundingBoxes';
import PageBreakLines from './PageBreakLines';

// Context Provider
import { ToggleProvider, useToggles } from './ToggleContext';

// Configuration and utilities
import { getStageNames, processBoundingBoxes, getColorMap } from './stageConfig';
import { processPdfInfo } from './pdfDataProcessor';

// PDF.js worker
pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';

function App() {
  /********************************************************
   * 1) Define all hooks at the top, unconditionally
   ********************************************************/
  const [metadataList, setMetadataList] = useState([]);
  const [selectedFileIndex, setSelectedFileIndex] = useState(0);
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [loading, setLoading] = useState(true);

  // Filter-related states
  const [selectedSubDir, setSelectedSubDir] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  
  // Get initial stage from configuration
  const stageNames = getStageNames();
  const [selectedStage, setSelectedStage] = useState(stageNames[0] || 'original');
  
  // Get toggles from context
  const { toggles } = useToggles();
  
  // PDF rendering scale & dimensions
  const [pageDimensions, setPageDimensions] = useState({ width: 0, height: 0 });
  const scale = 2.0;

  // ---------- useEffect Hooks ----------
  // Fetch metadata once
  useEffect(() => {
    setLoading(true);
    fetch('/outputs/state/metadata.json')
      .then((res) => res.json())
      .then((data) => {
        const arr = Array.isArray(data) ? data : [data];
        setMetadataList(arr);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error fetching metadata:', err);
        setLoading(false);
      });
  }, []);

  // Reset pageNumber whenever we switch files
  useEffect(() => {
    setPageNumber(1);
    setNumPages(null);
  }, [selectedFileIndex]);

  // ---------- useMemo Hooks ----------
  // Build sub-directory list
  const subDirectories = useMemo(() => {
    const dirs = new Set(metadataList.map((m) => m?.sub_dir).filter(Boolean));
    return ['all', ...Array.from(dirs)];
  }, [metadataList]);

  // Filter the metadata by subdir + search
  const filteredMetadata = useMemo(() => {
    return metadataList.filter((meta) => {
      const subDir = meta?.sub_dir || '';
      const fileName = meta?.file_name || '';
      const matchesSubDir = (selectedSubDir === 'all' || subDir === selectedSubDir);
      const matchesSearch = !searchQuery || fileName.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesSubDir && matchesSearch;
    });
  }, [metadataList, selectedSubDir, searchQuery]);

  // Current metadata
  const currentMetadata = useMemo(() => {
    return filteredMetadata[selectedFileIndex] || null;
  }, [filteredMetadata, selectedFileIndex]);

  // Process PDF info using configuration
  const pdfInfo = useMemo(() => {
    return processPdfInfo(currentMetadata, selectedStage);
  }, [currentMetadata, selectedStage]);

  // Process bounding boxes using configuration - now using toggles object directly
  const boundingBoxesForThisFile = useMemo(() => {
    console.log("Processing bounding boxes with toggle states:", toggles);
    if (!currentMetadata || !selectedStage) {
      console.log("Missing metadata or stage, returning empty boxes");
      return [];
    }
    
    const boxes = processBoundingBoxes(currentMetadata, selectedStage, toggles);
    console.log("Processed boxes result:", boxes);
    
    return boxes;
  }, [
    currentMetadata, 
    selectedStage, 
    toggles // Just depend on the entire toggles object - fewer dependencies to manage
  ]);

  // Get color map for the current stage
  const colorMap = useMemo(() => {
    return getColorMap(selectedStage);
  }, [selectedStage]);

  // ---------- PDF Handlers ----------
  const onDocumentLoadSuccess = ({ numPages }) => setNumPages(numPages);
  const onPageRenderSuccess = (page) => {
    setPageDimensions({ width: page.width, height: page.height });
  };

  // ---------- Navigation Functions ----------
  const goToPrevPage = () => setPageNumber((prev) => (prev > 1 ? prev - 1 : prev));
  const goToNextPage = () => setPageNumber((prev) => (prev < (numPages || 1) ? prev + 1 : prev));

  /********************************************************
   * 2) Early returns happen AFTER the hooks
   ********************************************************/
  if (loading) {
    return <div style={styles.container}>Loading metadata...</div>;
  }
  if (metadataList.length === 0) {
    return <div style={styles.container}>No metadata found</div>;
  }
  if (filteredMetadata.length === 0) {
    return (
      <div style={styles.container}>
        <h1 style={styles.heading}>File Viewer</h1>
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
          // No need to pass individual toggle props - FilterControls will use context
        />
        <div style={{ marginTop: '20px' }}>No files match the current filters</div>
      </div>
    );
  }
  if (!currentMetadata) {
    return <div style={styles.container}>No selected file metadata</div>;
  }

  // Check if the selected stage has valid data
  const { pdfPath } = pdfInfo;
  if (!pdfPath) {
    return (
      <div style={styles.container}>
        <h1 style={styles.heading}>File Viewer</h1>
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
          // No need to pass individual toggle props
        />
        <div style={{ marginTop: '20px' }}>
          Missing data for stage '{selectedStage}' in the current file. Please select a different stage.
        </div>
      </div>
    );
  }

  /********************************************************
   * 3) Destructure values from our processed info
   ********************************************************/
  const { combinedPdfWidth, combinedPdfHeight, pageBreaks } = pdfInfo;

  /********************************************************
   * 4) Render the PDF + Overlays
   ********************************************************/
  return (
    <div style={styles.container}>
      <h1 style={styles.heading}>File Viewer</h1>

      {/* Filter + Stage controls */}
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
        // No need to pass individual toggle props
      />

      {/* Some file info */}
      <div style={styles.fileInfo}>
        <div><strong>File Name:</strong> {currentMetadata.file_name}</div>
        <div><strong>Sub Directory:</strong> {currentMetadata.sub_dir}</div>
        <div><strong>PDF Path:</strong> {pdfPath}</div>
      </div>

      {/* PDF Document */}
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <Document
          file={pdfPath}
          onLoadSuccess={onDocumentLoadSuccess}
          onLoadError={(err) => console.error('Error loading PDF:', err)}
        >
          <Page
            pageNumber={pageNumber}
            scale={scale}
            onRenderSuccess={onPageRenderSuccess}
            renderAnnotationLayer={false}
            renderTextLayer={false}
          />
        </Document>

        {/* Overlays */}
        <div style={{ position: 'absolute', top: 0, left: 0 }}>
          {boundingBoxesForThisFile && (
            <BoundingBoxes
              boxesData={boundingBoxesForThisFile}
              transformCoord={(coords, color) => {
                // scale from PDF coords to rendered size
                const ratioX = pageDimensions.width / combinedPdfWidth;
                const ratioY = pageDimensions.height / combinedPdfHeight;
                const [x0, y0, x1, y1] = coords;

                console.log('Transforming coords:', {
                  original: coords,
                  pageDimensions,
                  combinedPdfDimensions: { width: combinedPdfWidth, height: combinedPdfHeight },
                  ratios: { ratioX, ratioY }
                });

                return {
                  position: 'absolute',
                  left: x0 * ratioX,
                  top: y0 * ratioY,
                  width: (x1 - x0) * ratioX,
                  height: (y1 - y0) * ratioY,
                  border: `2px solid ${color || 'red'}`,
                  backgroundColor: color || 'rgba(255, 0, 0, 0.2)',
                  pointerEvents: 'none'
                };
              }}
              // Pass the color map from our configuration
              colorMap={colorMap}
            />
          )}

          {pageBreaks && pageBreaks.length > 0 && (
            <PageBreakLines
              pageBreaks={pageBreaks}
              docWidth={pageDimensions.width}
              docHeight={pageDimensions.height}
              pdfHeight={combinedPdfHeight}
            />
          )}
        </div>
      </div>

      {/* Navigation Buttons */}
      <div style={{ marginTop: '10px' }}>
        <button
          onClick={goToPrevPage}
          disabled={pageNumber <= 1}
          style={styles.button}
        >
          Previous Page
        </button>
        <button
          onClick={goToNextPage}
          disabled={pageNumber >= (numPages || 1)}
          style={styles.button}
        >
          Next Page
        </button>
      </div>
      <p style={{ marginTop: '8px' }}>
        Page {pageNumber} of {numPages || 1}
      </p>
    </div>
  );
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
};

// Wrap the App with the ToggleProvider
const AppWithProviders = () => (
  <ToggleProvider>
    <App />
  </ToggleProvider>
);

export default AppWithProviders;
