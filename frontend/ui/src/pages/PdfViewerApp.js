// Updated PdfViewerApp.js with Fixed Bounding Box Rendering
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Divider } from '@mui/material';
import { Document, Page, pdfjs } from 'react-pdf';
import FileMetadataDisplay from '../components/pdf/FileMetadataDisplay';
import PdfNavigation from '../components/pdf/PdfNavigation';
import FilterControls from '../components/pdf/FilterControls';
import LoadingState from '../components/pdf/LoadingState';
import PdfProcessingResults from '../components/pdf/PdfProcessingResults';

// Import CSS for PDF styles
import '../components/pdf/css/pdf-styles.css';

// Set worker for PDF.js
pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';

function PdfViewerApp() {
  // ... keep all existing state variables
  const [metadataList, setMetadataList] = useState([]);
  const [selectedFileIndex, setSelectedFileIndex] = useState(0);
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [loading, setLoading] = useState(true);
  const [searchParams] = useSearchParams();
  const [categoryFilters, setCategoryFilters] = useState({});
  const [fileIdQuery, setFileIdQuery] = useState('');
  const [classificationFilter, setClassificationFilter] = useState('all');
  const [pageDimensions, setPageDimensions] = useState({ width: 0, height: 0 });
  const [selectedStage, setSelectedStage] = useState(0); // Default to stage 0
  
  // Reference to the PDF container for proper positioning
  const pdfContainerRef = useRef(null);

  // URL and selection states
  const fileId = searchParams.get('id');
  const [selectedSubDir, setSelectedSubDir] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  
  // Constants
  const scale = 1.5;

  // Keep all the existing useEffect hooks and functions...
  // (fetching metadata, filtering, navigation handlers, etc.)
  
  // Fetch metadata
  useEffect(() => {
    setLoading(true);
    fetch('http://localhost:8000/query/records?merge=true')
      .then(res => res.json())
      .then((data) => {
        setMetadataList(Array.isArray(data) ? data : [data]);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error fetching metadata:', err);
        setLoading(false);
      });
  }, []);

  // Set selected file from URL param
  useEffect(() => {
    if (!fileId || metadataList.length === 0) return;
    
    const index = metadataList.findIndex(
      m => (m.file_id === fileId || m.dropbox_id === fileId)
    );
    if (index >= 0) setSelectedFileIndex(index);
  }, [fileId, metadataList]);

  // Extract subdirectories
  const subDirectories = useMemo(() => {
    const dirs = new Set();
    metadataList.forEach(meta => {
      if (meta.type === 'file' && meta.directories && meta.directories.length > 0) {
        dirs.add(meta.directories.join('/'));
      }
    });
    return ['all', ...Array.from(dirs)];
  }, [metadataList]);

  // Handle file ID filter change
  const handleFileIdChange = (value) => {
    setFileIdQuery(value);
  };

  // Filter metadata based on all active filters
  const filteredMetadata = useMemo(() => {
    return metadataList.filter((meta) => {
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
  
      return matchesSubDir && matchesSearch && matchesFileId && matchesCategories && matchesClassification;
    });
  }, [metadataList, selectedSubDir, searchQuery, fileIdQuery, categoryFilters, classificationFilter]);

  const showEmptyState = filteredMetadata.length === 0;
  const currentMetadata = showEmptyState ? null : filteredMetadata[selectedFileIndex] || null;

  // Get available stages for the current file
  const stageOptions = useMemo(() => {
    if (!currentMetadata) return [0];
    
    const completedStages = currentMetadata.completed_stages || [0];
    return completedStages.sort((a, b) => a - b);
  }, [currentMetadata]);

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
      setSelectedFileIndex(0);
    }
  }, [filteredMetadata, selectedFileIndex, showEmptyState]);

  // Reset page when file changes
  useEffect(() => {
    setPageNumber(1);
    setNumPages(null);
  }, [selectedFileIndex]);

  // Get available classifications
  const availableClassifications = useMemo(() => {
    const classifications = new Set(['all']);
    
    metadataList.forEach(meta => {
      if (meta.classification && meta.classification !== '' && meta.classification !== null) {
        classifications.add(meta.classification);
      }
    });
    
    return Array.from(classifications);
  }, [metadataList]);

  // Navigation handlers
  const goToPrevPage = () => setPageNumber((p) => Math.max(p - 1, 1));
  const goToNextPage = () => setPageNumber((p) => Math.min(p + 1, numPages || 1));

  // Document callbacks
  const onDocumentLoadSuccess = ({ numPages }) => setNumPages(numPages);
  const onPageRenderSuccess = (page) => {
    console.log(`Page rendered. Dimensions: ${page.width}x${page.height}`);
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
    
    // Fallback to a simpler pattern if mount_path is not available
    return `http://localhost:8000/files/stage${stageStr}/${currentMetadata.file_id || currentMetadata.dropbox_id}.pdf`;
  };

  const pdfPath = getPdfPath();
  const currentFileId = currentMetadata?.file_id || currentMetadata?.dropbox_id;

  if (loading) return <LoadingState message="Loading metadata..." />;

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
        categoryFilters={categoryFilters}
        onCategoryFilterChange={(key, value) =>
          setCategoryFilters((prev) => ({ ...prev, [key]: value }))
        }
        fileIdQuery={fileIdQuery}
        onFileIdChange={handleFileIdChange}
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
          {/* Responsive layout for metadata and processing results */}
          <div className="metadata-processing-container">
            {/* Left side: FileMetadataDisplay */}
            <div className="metadata-column">
              <FileMetadataDisplay metadata={currentMetadata} />
            </div>
            
            {/* Right side: PdfProcessingResults in a box */}
            <div className="processing-column">
              <h3 className="column-header">Processing Controls</h3>
              {currentFileId && (
                <PdfProcessingResults
                  fileId={currentFileId}
                  stage={selectedStage}
                  pageDimensions={pageDimensions}
                  currentPage={pageNumber}
                  pdfContainerRef={pdfContainerRef}
                  scale={scale}
                  metadata={currentMetadata}
                />
              )}
            </div>
          </div>

          <Divider sx={{ my: 2 }} />

          {/* PDF Container - Critical for bounding box positioning */}
          <div
            ref={pdfContainerRef}
            className="pdf-container"
            style={{
              position: 'relative',  /* Critical for bounding box positioning */
              display: 'inline-block',
              overflow: 'visible',
              margin: '0 auto'
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

                {/* Adding a debug overlay for bounding box testing */}
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
                    backgroundColor: 'rgba(255, 0, 0, 0.1)'
                  }}
                >
                  <div style={{
                    position: 'absolute',
                    top: '50%',
                    left: '50%',
                    transform: 'translate(-50%, -50%)',
                    backgroundColor: 'red',
                    color: 'white',
                    padding: '2px 5px',
                    borderRadius: '3px',
                    fontSize: '10px',
                    pointerEvents: 'none'
                  }}>
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
    fontFamily: 'Arial, sans-serif'
  },
  heading: {
    color: '#333',
    marginBottom: '20px'
  },
  metadataProcessingContainer: {
    display: 'flex',
    flexDirection: 'row',
    gap: '20px',
    marginBottom: '20px'
  },
  metadataColumn: {
    flex: '1',
    minWidth: '300px'
  },
  processingColumn: {
    flex: '2',
    border: '1px solid #ddd',
    borderRadius: '4px',
    padding: '15px',
    backgroundColor: '#f9f9f9',
    boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.05)'
  }
};

const DisplayPdf = () => (
    <PdfViewerApp />
);

export default DisplayPdf;

// // Updated PdfViewerApp.js with Fixed Bounding Box Rendering
// import React, { useState, useEffect, useMemo, useRef } from 'react';
// import { useSearchParams } from 'react-router-dom';
// import { Divider } from '@mui/material';
// import { Document, Page, pdfjs } from 'react-pdf';
// import { ToggleProvider } from '../components/pdf/ToggleContext';
// import FileMetadataDisplay from '../components/pdf/FileMetadataDisplay';
// import PdfNavigation from '../components/pdf/PdfNavigation';
// import FilterControls from '../components/pdf/FilterControls';
// import LoadingState from '../components/pdf/LoadingState';
// import PdfProcessingResults from '../components/pdf/PdfProcessingResults';

// // Import CSS for PDF styles
// import '../components/pdf/css/pdf-styles.css';

// // Set worker for PDF.js
// pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';

// function PdfViewerApp() {
//   // ... keep all existing state variables
//   const [metadataList, setMetadataList] = useState([]);
//   const [selectedFileIndex, setSelectedFileIndex] = useState(0);
//   const [numPages, setNumPages] = useState(null);
//   const [pageNumber, setPageNumber] = useState(1);
//   const [loading, setLoading] = useState(true);
//   const [searchParams] = useSearchParams();
//   const [categoryFilters, setCategoryFilters] = useState({});
//   const [fileIdQuery, setFileIdQuery] = useState('');
//   const [classificationFilter, setClassificationFilter] = useState('all');
//   const [pageDimensions, setPageDimensions] = useState({ width: 0, height: 0 });
//   const [selectedStage, setSelectedStage] = useState(0); // Default to stage 0
  
//   // Reference to the PDF container for proper positioning
//   const pdfContainerRef = useRef(null);

//   // URL and selection states
//   const fileId = searchParams.get('id');
//   const [selectedSubDir, setSelectedSubDir] = useState('all');
//   const [searchQuery, setSearchQuery] = useState('');
  
//   // Constants
//   const scale = 1.5;

//   // Keep all the existing useEffect hooks and functions...
//   // (fetching metadata, filtering, navigation handlers, etc.)
  
//   // Fetch metadata
//   useEffect(() => {
//     setLoading(true);
//     fetch('http://localhost:8000/query/records?merge=true')
//       .then(res => res.json())
//       .then((data) => {
//         setMetadataList(Array.isArray(data) ? data : [data]);
//         setLoading(false);
//       })
//       .catch((err) => {
//         console.error('Error fetching metadata:', err);
//         setLoading(false);
//       });
//   }, []);

//   // Set selected file from URL param
//   useEffect(() => {
//     if (!fileId || metadataList.length === 0) return;
    
//     const index = metadataList.findIndex(
//       m => (m.file_id === fileId || m.dropbox_id === fileId)
//     );
//     if (index >= 0) setSelectedFileIndex(index);
//   }, [fileId, metadataList]);

//   // Extract subdirectories
//   const subDirectories = useMemo(() => {
//     const dirs = new Set();
//     metadataList.forEach(meta => {
//       if (meta.type === 'file' && meta.directories && meta.directories.length > 0) {
//         dirs.add(meta.directories.join('/'));
//       }
//     });
//     return ['all', ...Array.from(dirs)];
//   }, [metadataList]);

//   // Handle file ID filter change
//   const handleFileIdChange = (value) => {
//     setFileIdQuery(value);
//   };

//   // Filter metadata based on all active filters
//   const filteredMetadata = useMemo(() => {
//     return metadataList.filter((meta) => {
//       if (meta.type !== 'file') return false;
  
//       const fileName = (meta.file_name || meta.name || '').toLowerCase();
//       const matchesSubDir =
//         selectedSubDir === 'all' ||
//         (meta.directories && meta.directories.join('/').includes(selectedSubDir));
  
//       const matchesSearch = !searchQuery || fileName.includes(searchQuery.toLowerCase());
      
//       const fileId = (meta.file_id || meta.dropbox_id || '').toLowerCase();
//       const matchesFileId = !fileIdQuery || fileId.includes(fileIdQuery.toLowerCase());
  
//       const matchesCategories = Object.entries(categoryFilters).every(([key, val]) => {
//         if (!val || val === 'all') return true;
//         return meta?.path_categories?.[key] === val;
//       });
      
//       const matchesClassification = 
//         classificationFilter === 'all' || 
//         meta.classification === classificationFilter;
  
//       return matchesSubDir && matchesSearch && matchesFileId && matchesCategories && matchesClassification;
//     });
//   }, [metadataList, selectedSubDir, searchQuery, fileIdQuery, categoryFilters, classificationFilter]);

//   const showEmptyState = filteredMetadata.length === 0;
//   const currentMetadata = showEmptyState ? null : filteredMetadata[selectedFileIndex] || null;

//   // Get available stages for the current file
//   const stageOptions = useMemo(() => {
//     if (!currentMetadata) return [0];
    
//     const completedStages = currentMetadata.completed_stages || [0];
//     return completedStages.sort((a, b) => a - b);
//   }, [currentMetadata]);

//   // Make sure selected stage is valid for the current file
//   useEffect(() => {
//     if (currentMetadata && stageOptions.length > 0) {
//       if (!stageOptions.includes(selectedStage)) {
//         setSelectedStage(stageOptions[0]);
//       }
//     }
//   }, [currentMetadata, stageOptions, selectedStage]);

//   // Reset selection index if filtered results change
//   useEffect(() => {
//     if (!showEmptyState && selectedFileIndex >= filteredMetadata.length) {
//       setSelectedFileIndex(0);
//     }
//   }, [filteredMetadata, selectedFileIndex, showEmptyState]);

//   // Reset page when file changes
//   useEffect(() => {
//     setPageNumber(1);
//     setNumPages(null);
//   }, [selectedFileIndex]);

//   // Get available classifications
//   const availableClassifications = useMemo(() => {
//     const classifications = new Set(['all']);
    
//     metadataList.forEach(meta => {
//       if (meta.classification && meta.classification !== '' && meta.classification !== null) {
//         classifications.add(meta.classification);
//       }
//     });
    
//     return Array.from(classifications);
//   }, [metadataList]);

//   // Navigation handlers
//   const goToPrevPage = () => setPageNumber((p) => Math.max(p - 1, 1));
//   const goToNextPage = () => setPageNumber((p) => Math.min(p + 1, numPages || 1));

//   // Document callbacks
//   const onDocumentLoadSuccess = ({ numPages }) => setNumPages(numPages);
//   const onPageRenderSuccess = (page) => {
//     console.log(`Page rendered. Dimensions: ${page.width}x${page.height}`);
//     setPageDimensions({ width: page.width, height: page.height });
//   };

//   // Generate PDF path
//   const getPdfPath = () => {
//     if (!currentMetadata) return null;
    
//     const stageStr = String(selectedStage);
//     const stagePaths = currentMetadata.stage_paths || {};
    
//     if (stagePaths[stageStr] && stagePaths[stageStr].mount_path) {
//       return `http://localhost:8000${stagePaths[stageStr].mount_path}`;
//     }
    
//     // Fallback to a simpler pattern if mount_path is not available
//     return `http://localhost:8000/files/stage${stageStr}/${currentMetadata.file_id || currentMetadata.dropbox_id}.pdf`;
//   };

//   const pdfPath = getPdfPath();
//   const currentFileId = currentMetadata?.file_id || currentMetadata?.dropbox_id;

//   if (loading) return <LoadingState message="Loading metadata..." />;

//   return (
//     <div style={styles.container}>
//       <h1 style={styles.heading}>PDF Viewer</h1>

//       <FilterControls
//         subDirectories={subDirectories}
//         selectedSubDir={selectedSubDir}
//         onSubDirChange={setSelectedSubDir}
//         searchQuery={searchQuery}
//         onSearchChange={setSearchQuery}
//         selectedFileIndex={selectedFileIndex}
//         setSelectedFileIndex={setSelectedFileIndex}
//         filteredMetadata={filteredMetadata}
//         categoryFilters={categoryFilters}
//         onCategoryFilterChange={(key, value) =>
//           setCategoryFilters((prev) => ({ ...prev, [key]: value }))
//         }
//         fileIdQuery={fileIdQuery}
//         onFileIdChange={handleFileIdChange}
//         classificationFilter={classificationFilter}
//         onClassificationChange={setClassificationFilter}
//         availableClassifications={availableClassifications}
//         showClassificationFilter={true}
//         selectedStage={selectedStage}
//         onStageChange={setSelectedStage}
//         stageOptions={stageOptions}
//       />

//       {showEmptyState ? (
//         <div style={{ padding: '10px', fontStyle: 'italic', color: '#888' }}>
//           No files match your search or directory filter.
//         </div>
//       ) : (
//         <>
//           {/* Side-by-side layout for metadata and processing results */}
//           <div style={styles.metadataProcessingContainer}>
//             {/* Left side: FileMetadataDisplay */}
//             <div style={styles.metadataColumn}>
//               <FileMetadataDisplay metadata={currentMetadata} />
//             </div>
            
//             {/* Right side: PdfProcessingResults in a box */}
//             <div style={styles.processingColumn}>
//               {currentFileId && (
//                 <PdfProcessingResults
//                   fileId={currentFileId}
//                   stage={selectedStage}
//                   pageDimensions={pageDimensions}
//                   currentPage={pageNumber}
//                   pdfContainerRef={pdfContainerRef}
//                   scale={scale}
//                 />
//               )}
//             </div>
//           </div>

//           <Divider sx={{ my: 2 }} />

//           {/* PDF Container - Critical for bounding box positioning */}
//           <div
//             ref={pdfContainerRef}
//             className="pdf-container"
//             style={{
//               position: 'relative',  /* Critical for bounding box positioning */
//               display: 'inline-block',
//               overflow: 'visible',
//               margin: '0 auto'
//             }}
//           >
//             {pdfPath ? (
//               <>
//                 <Document file={pdfPath} onLoadSuccess={onDocumentLoadSuccess}>
//                   <Page
//                     pageNumber={pageNumber}
//                     scale={scale}
//                     onRenderSuccess={onPageRenderSuccess}
//                     renderAnnotationLayer={false}
//                     renderTextLayer={false}
//                   />
//                 </Document>

//                 {/* Adding a debug overlay for bounding box testing */}
//                 <div 
//                   style={{
//                     position: 'absolute',
//                     top: 0,
//                     left: 0,
//                     width: '100%',
//                     height: '100%',
//                     border: '1px solid red',
//                     pointerEvents: 'none',
//                     zIndex: 50,
//                     opacity: 0.1,
//                     backgroundColor: 'rgba(255, 0, 0, 0.1)'
//                   }}
//                 >
//                   <div style={{
//                     position: 'absolute',
//                     top: '50%',
//                     left: '50%',
//                     transform: 'translate(-50%, -50%)',
//                     backgroundColor: 'red',
//                     color: 'white',
//                     padding: '2px 5px',
//                     borderRadius: '3px',
//                     fontSize: '10px',
//                     pointerEvents: 'none'
//                   }}>
//                     PDF Area
//                   </div>
//                 </div>
//                 <PdfNavigation
//                   pageNumber={pageNumber}
//                   numPages={numPages}
//                   goToPrevPage={goToPrevPage}
//                   goToNextPage={goToNextPage}
//                 />
//               </>
//             ) : (
//               <div style={{ padding: '20px', border: '1px solid #ddd' }}>
//                 PDF file not available
//               </div>
//             )}
//           </div>
//         </>
//       )}
//     </div>
//   );
// }

// const styles = {
//   container: {
//     margin: '20px',
//     fontFamily: 'Arial, sans-serif'
//   },
//   heading: {
//     color: '#333',
//     marginBottom: '20px'
//   },
//   metadataProcessingContainer: {
//     display: 'flex',
//     flexDirection: 'row',
//     gap: '20px',
//     marginBottom: '20px'
//   },
//   metadataColumn: {
//     flex: '1',
//     minWidth: '300px'
//   },
//   processingColumn: {
//     flex: '2',
//     border: '1px solid #ddd',
//     borderRadius: '4px',
//     padding: '15px',
//     backgroundColor: '#f9f9f9',
//     boxShadow: '0px 2px 4px rgba(0, 0, 0, 0.05)'
//   }
// };

// const DisplayPdf = () => (
//   <ToggleProvider>
//     <PdfViewerApp />
//   </ToggleProvider>
// );

// export default DisplayPdf;
