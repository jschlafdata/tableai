import React, { useState, useEffect, useMemo } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Document, Page, pdfjs } from 'react-pdf';
import FilterControls from '../components/pdfs/FilterControls';
import BoundingBoxes from '../components/pdfs/BoundingBoxes';
import PageBreakLines from '../components/pdfs/PageBreakLines';
import { ToggleProvider, useToggles } from '../components/pdfs/ToggleContext';
import { getStageNames, processBoundingBoxes, getColorMap } from '../components/pdfs/stageConfig';
import { processPdfInfo } from '../components/pdfs/pdfDataProcessor';
import { Box } from '@mui/material';

pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';

function PdfViewerApp() {
  const [metadataList, setMetadataList] = useState([]);
  const [extractList, setExtractList] = useState([]);
  const [extractedMetadata, setExtractedMetadata] = useState(null);
  const [extractListLoaded, setExtractListLoaded] = useState(false);
  const [selectedFileIndex, setSelectedFileIndex] = useState(0);
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [loading, setLoading] = useState(true);
  const [metadataLoading, setMetadataLoading] = useState(false);
  const [searchParams] = useSearchParams();
  const { toggles } = useToggles();
  const [categoryFilters, setCategoryFilters] = useState({});
  const [fileIdQuery, setFileIdQuery] = useState('');
  const [classificationFilter, setClassificationFilter] = useState('all');
  const [stage2TablePage, setStage2TablePage] = useState(0);

  const dropboxId = searchParams.get('id');
  const stageNames = getStageNames();
  const [selectedStage, setSelectedStage] = useState(stageNames[0] || 'original');
  const [selectedSubDir, setSelectedSubDir] = useState('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [pageDimensions, setPageDimensions] = useState({ width: 0, height: 0 });
  const scale = 1.5;

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

  useEffect(() => {
    if (!dropboxId || metadataList.length === 0) return;
    const index = metadataList.findIndex(m => m.dropbox_safe_id === dropboxId);
    if (index >= 0) setSelectedFileIndex(index);
  }, [dropboxId, metadataList]);

  const subDirectories = useMemo(() => {
    const dirs = new Set(
      metadataList
        .filter(m => m.type === 'file')
        .map(m => m?.directories?.join('/'))
        .filter(Boolean)
    );
    return ['all', ...Array.from(dirs)];
  }, [metadataList]);

  const handleFileIdChange = (value) => {
    setFileIdQuery(value);
  };
  const filteredMetadata = useMemo(() => {
    return metadataList.filter((meta) => {
      if (meta.type !== 'file') return false;
  
      const fileName = meta?.file_name?.toLowerCase() || '';
      const matchesSubDir =
        selectedSubDir === 'all' ||
        meta.directories?.join('/').includes(selectedSubDir);
  
      const matchesSearch = !searchQuery || fileName.includes(searchQuery.toLowerCase());
      
      // Add file_id filtering
      const fileId = meta?.dropbox_safe_id || '';
      const matchesFileId = !fileIdQuery || fileId.toLowerCase().includes(fileIdQuery.toLowerCase());
  
      const matchesCategories = Object.entries(categoryFilters).every(([key, val]) => {
        if (!val || val === 'all') return true;
        return meta?.path_categories?.[key] === val;
      });
      
      // Add classification filtering
      const matchesClassification = 
        classificationFilter === 'all' || 
        meta.classification_label === classificationFilter;
  
      return matchesSubDir && matchesSearch && matchesFileId && matchesCategories && matchesClassification;
    });
  }, [metadataList, selectedSubDir, searchQuery, fileIdQuery, categoryFilters, classificationFilter]);

  const showEmptyState = filteredMetadata.length === 0;
  const currentMetadata = showEmptyState ? null : filteredMetadata[selectedFileIndex] || null;

  useEffect(() => {
    if (!showEmptyState && selectedFileIndex >= filteredMetadata.length) {
      setSelectedFileIndex(0);
    }
  }, [filteredMetadata, selectedFileIndex, showEmptyState]);

  useEffect(() => {
    setPageNumber(1);
    setNumPages(null);
  }, [selectedFileIndex]);

  // Reset extracted metadata when stage changes
  useEffect(() => {
    // Clear existing metadata when stage changes
    setExtractedMetadata(null);
    
    const fileEntry = filteredMetadata[selectedFileIndex];
    
    // Use dropbox_id as the file_id if file_id doesn't exist
    const fileId = fileEntry?.dropbox_safe_id;
    
    const readyToFetch =
      selectedStage !== 'stage0' && fileId;
    
    if (!readyToFetch) return;
    
    console.log("🧪 Triggered extract fetch", {
      selectedStage,
      selectedFileIndex,
      fileEntry,
      fileId,
    });
    
    setMetadataLoading(true);
    
    fetch(`http://localhost:8000/query/extractions/${fileId}`)
      .then((res) => res.json())
      .then((data) => {
        console.log("✅ Extracted metadata fetched:", data);
        setExtractedMetadata(data);
        console.log("extracted:", data)
        setMetadataLoading(false);
      })
      .catch((err) => {
        console.error('❌ Error fetching extracted metadata:', err);
        setExtractedMetadata(null);
        setMetadataLoading(false);
      });
  }, [selectedStage, filteredMetadata, selectedFileIndex]);

  // Get all available classification labels
  const availableClassifications = useMemo(() => {
      const classifications = new Set(['all']);
      
      metadataList.forEach(meta => {
        if (meta.classification_label) {
          classifications.add(meta.classification_label);
        }
      });
      
      return Array.from(classifications);
  }, [metadataList]);

  // Process PDF info with default values
  const pdfInfo = useMemo(() => {
    if (!extractedMetadata || !selectedStage) return {
      combinedPdfWidth: 612,
      combinedPdfHeight: 792,
      pageBreaks: []
    };

    console.log("using this extracted data:", extractedMetadata)
    
    // Use the first item in the array, just like with bounding boxes
    return processPdfInfo(extractedMetadata[0], selectedStage);
  }, [extractedMetadata, selectedStage]);

  const combinedPdfWidth = pdfInfo?.combinedPdfWidth || 612;
  const combinedPdfHeight = pdfInfo?.combinedPdfHeight || 792;
  const pageBreaks = pdfInfo?.pageBreaks || [];
  // Destructure with default values

  // // Process bounding boxes with index [0]
  const boundingBoxesForThisFile = useMemo(() => {
    if (!extractedMetadata || !selectedStage) return null;
    return processBoundingBoxes(extractedMetadata[0], selectedStage, toggles);
  }, [extractedMetadata, selectedStage, toggles]);

  console.log("📦 All boxes:", boundingBoxesForThisFile);
  console.log("📄 Current pageIndex:", pageNumber - 1);

  const filteredBoxesForCurrentPage = useMemo(() => {
    if (!boundingBoxesForThisFile || !selectedStage) return null;
  
    const result = {};
    for (const [boxType, boxes] of Object.entries(boundingBoxesForThisFile)) {
      result[boxType] = boxes.filter(box => {
        const pageIndex = box?.pageIndex ?? 0;
        return pageIndex === pageNumber - 1; // Page index is 0-based
      });
    }
    return result;
  }, [boundingBoxesForThisFile, pageNumber, selectedStage]);

  const colorMap = useMemo(() => {
    return getColorMap(selectedStage);
  }, [selectedStage]);

  const onDocumentLoadSuccess = ({ numPages }) => setNumPages(numPages);
  const onPageRenderSuccess = (page) => {
    setPageDimensions({ width: page.width, height: page.height });
  };


  const goToPrevPage = () => setPageNumber((p) => Math.max(p - 1, 1));
  const goToNextPage = () => setPageNumber((p) => Math.min(p + 1, numPages || 1));

  if (loading) return <div style={styles.container}>Loading metadata...</div>;

  const pdfPath = currentMetadata
  ? selectedStage === 'stage0'
    ? `http://localhost:8000/files/${selectedStage}/${currentMetadata.dropbox_safe_id}.pdf`
    : `http://localhost:8000/files/extractions/${selectedStage}/${currentMetadata.dropbox_safe_id}.pdf`
  : null;

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
        categoryFilters={categoryFilters}
        onCategoryFilterChange={(key, value) =>
          setCategoryFilters((prev) => ({ ...prev, [key]: value }))
        }
        fileIdQuery={fileIdQuery}
        onFileIdChange={handleFileIdChange}
        classificationFilter={classificationFilter}
        onClassificationChange={setClassificationFilter}
        availableClassifications={availableClassifications}
        showClassificationFilter={true} // Always show since it's from initial data
      />

      {showEmptyState ? (
        <div style={{ padding: '10px', fontStyle: 'italic', color: '#888' }}>
          No files match your search or directory filter.
        </div>
      ) : (
        <>
          <div style={styles.fileInfo}>
            <div><strong>File Name:</strong> {currentMetadata.file_name}</div>
            <div><strong>Directories:</strong> {currentMetadata.directories?.join(' / ')}</div>
            <div><strong>dropbox_id:</strong> {currentMetadata.dropbox_safe_id}</div>
            {currentMetadata.classification_label && (
              <div><strong>Classification:</strong> {currentMetadata.classification_label}</div>
            )}
          </div>

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

            {selectedStage === 'stage2' && extractedMetadata?.[0]?.metadata?.stage3?.clean_stage3 && (
              <Box mt={4} p={2} border="1px solid #ccc" borderRadius="8px" bgcolor="#fafafa">
                <h4>📊 Stage 2 – All Extracted Tables</h4>

                {(() => {
                  const cleanStage3 = extractedMetadata[0].metadata.stage3.clean_stage3;

                  // Extract all pages with table_frame_dict
                  const pagesWithTables = Object.entries(cleanStage3)
                    .filter(([_, data]) => !!data.table_frame_dict)
                    .map(([pageKey, data]) => ({
                      pageKey,
                      tableDict: data.table_frame_dict,
                      sumTests: data.sum_tests || {}
                    }));

                  if (pagesWithTables.length === 0) {
                    return <div>No structured tables available in stage2.</div>;
                  }

                  const { pageKey, tableDict, sumTests } = pagesWithTables[stage2TablePage];
                  const columnNames = Object.keys(tableDict);
                  const rowCount = Object.keys(Object.values(tableDict)[0] || {}).length;

                  return (
                    <>
                      {/* Tab buttons for each page */}
                      <div style={{ display: 'flex', gap: '10px', marginBottom: '10px' }}>
                        {pagesWithTables.map((entry, idx) => (
                          <button
                            key={entry.pageKey}
                            onClick={() => setStage2TablePage(idx)}
                            style={{
                              padding: '6px 10px',
                              backgroundColor: idx === stage2TablePage ? '#1976d2' : '#eee',
                              color: idx === stage2TablePage ? '#fff' : '#333',
                              border: '1px solid #ccc',
                              borderRadius: '4px',
                              cursor: 'pointer'
                            }}
                          >
                            Page {parseInt(entry.pageKey, 10) + 1}
                          </button>
                        ))}
                      </div>

                      {/* Table Display */}
                      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                        <thead>
                          <tr>
                            {columnNames.map((colName, colIdx) => {
                              const testResult = sumTests?.[String(colIdx)];
                              const bgColor = testResult
                                ? testResult.result
                                  ? 'rgba(0, 200, 0, 0.2)' // green
                                  : 'rgba(255, 0, 0, 0.2)' // red
                                : 'transparent';

                              return (
                                <th
                                  key={colName}
                                  style={{
                                    borderBottom: '2px solid #999',
                                    textAlign: 'left',
                                    padding: '6px',
                                    fontWeight: 'bold',
                                    backgroundColor: bgColor
                                  }}
                                  title={
                                    testResult
                                      ? `Extracted: ${testResult.extracted}, Calculated: ${testResult.calculated}`
                                      : ''
                                  }
                                >
                                  {colName}
                                </th>
                              );
                            })}
                          </tr>
                        </thead>
                        <tbody>
                          {Array.from({ length: rowCount }).map((_, rowIndex) => (
                            <tr key={rowIndex}>
                              {columnNames.map((colName, colIdx) => {
                                const testResult = sumTests?.[String(colIdx)];
                                const cellBg = testResult
                                  ? testResult.result
                                    ? 'rgba(0, 200, 0, 0.08)'
                                    : 'rgba(255, 0, 0, 0.08)'
                                  : 'transparent';

                                return (
                                  <td
                                    key={`${colName}-${rowIndex}`}
                                    style={{
                                      borderBottom: '1px solid #ddd',
                                      padding: '4px',
                                      fontFamily: 'monospace',
                                      backgroundColor: cellBg
                                    }}
                                  >
                                    {tableDict[colName][String(rowIndex)] ?? ''}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </>
                  );
                })()}
              </Box>
            )}

            <div style={{ position: 'absolute', top: 0, left: 0 }}>
              {boundingBoxesForThisFile && (
                <BoundingBoxes
                  boxesData={filteredBoxesForCurrentPage}
                  transformCoord={(coords, color, pageIndex = 0) => {
                    // If stage3 with per-page bboxes:
                    if (selectedStage === 'stage3') {
                      const stage3Pages = extractedMetadata?.[0]?.metadata?.stage3?.pages;
                      const pageData = stage3Pages?.[String(pageIndex)];
                      console.log("📄 Page", pageNumber, "→ displaying", Object.values(filteredBoxesForCurrentPage || {}).flat().length, "boxes", "stage3Pages -->", stage3Pages);
                      if (!pageData) return {};
                  
                      const ratioX = pageDimensions.width / pageData.page_width;
                      const ratioY = pageDimensions.height / pageData.page_height;
                      
                      console.log("📄 ratioX", ratioX, "→ ratioY", ratioY, "boxes");

                      const [x0, y0, x1, y1] = coords;
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
                    }
                  
                    // Default: use global combined PDF dimensions
                    const ratioX = pageDimensions.width / combinedPdfWidth;
                    const ratioY = pageDimensions.height / combinedPdfHeight;
                  
                    const [x0, y0, x1, y1] = coords;
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
              
            {selectedStage !== 'stage2' &&
              extractedMetadata?.[0]?.metadata?.stage3?.clean_stage3?.[String(pageNumber - 1)]?.table_frame_dict && (
              <Box mt={4} p={2} border="1px solid #ccc" borderRadius="8px" bgcolor="#fafafa">
                <h4>📊 Page {pageNumber} - Structured Table</h4>

                {(() => {
                  const pageKey = String(pageNumber - 1);
                  const cleanPageData = extractedMetadata[0].metadata.stage3.clean_stage3[pageKey];
                  const tableDict = cleanPageData.table_frame_dict;
                  const sumTests = cleanPageData.sum_tests || {};

                  const columnNames = Object.keys(tableDict);
                  const rowCount = Object.keys(Object.values(tableDict)[0] || {}).length;

                  return (
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          {columnNames.map((colName, colIdx) => {
                            const testResult = sumTests?.[String(colIdx)];
                            const bgColor = testResult
                              ? testResult.result
                                ? 'rgba(0, 200, 0, 0.2)' // green
                                : 'rgba(255, 0, 0, 0.2)' // red
                              : 'transparent';

                            return (
                              <th
                                key={colName}
                                style={{
                                  borderBottom: '2px solid #999',
                                  textAlign: 'left',
                                  padding: '6px',
                                  fontWeight: 'bold',
                                  backgroundColor: bgColor
                                }}
                                title={
                                  testResult
                                    ? `Extracted: ${testResult.extracted}, Calculated: ${testResult.calculated}`
                                    : ''
                                }
                              >
                                {colName}
                              </th>
                            );
                          })}
                        </tr>
                      </thead>
                      <tbody>
                        {Array.from({ length: rowCount }).map((_, rowIndex) => (
                          <tr key={rowIndex}>
                            {columnNames.map((colName, colIdx) => {
                              const testResult = sumTests?.[String(colIdx)];
                              const cellBg = testResult
                                ? testResult.result
                                  ? 'rgba(0, 200, 0, 0.08)'
                                  : 'rgba(255, 0, 0, 0.08)'
                                : 'transparent';

                              return (
                                <td
                                  key={`${colName}-${rowIndex}`}
                                  style={{
                                    borderBottom: '1px solid #ddd',
                                    padding: '4px',
                                    fontFamily: 'monospace',
                                    backgroundColor: cellBg
                                  }}
                                >
                                  {tableDict[colName][String(rowIndex)] ?? ''}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  );
                })()}
              </Box>
            )}
            {metadataLoading && (
              <div style={{ 
                position: 'absolute', 
                top: '50%', 
                left: '50%', 
                transform: 'translate(-50%, -50%)',
                background: 'rgba(255, 255, 255, 0.8)',
                padding: '10px',
                borderRadius: '5px',
                boxShadow: '0 2px 5px rgba(0,0,0,0.2)'
              }}>
                Loading metadata...
              </div>
            )}
          </div>

          <div style={{ marginTop: '10px' }}>
            <button onClick={goToPrevPage} disabled={pageNumber <= 1} style={styles.button}>
              Previous Page
            </button>
            <button onClick={goToNextPage} disabled={pageNumber >= (numPages || 1)} style={styles.button}>
              Next Page
            </button>
          </div>
          <p style={{ marginTop: '8px' }}>
            Page {pageNumber} of {numPages || 1}
          </p>
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

const DisplayPdf = () => (
  <ToggleProvider>
    <PdfViewerApp />
  </ToggleProvider>
);

export default DisplayPdf;