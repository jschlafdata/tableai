// pages/BasicPdfViewerPage.jsx
import React, { useState, useMemo, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { TABLEAI_SERVICE_ENDPOINTS } from '../../schemas/tableai/tableaiServiceEndpoints';
import { TABLEAI_SERVICES } from '../../schemas/tableai/serviceBaseUrls';
import { useFilteredMetadata } from '../../hooks/api/tableai/useFilteredMetadata';
import FilterControls from '../../components/filters/FilterControls';

pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';

export default function BasicPdfViewerPage() {
  const {
    filters,
    setFilters,
    filteredMetadata,
    metadataList, // For debugging 
    uiOptions,
    isReady,
    loadingFilters,
    filterError
  } = useFilteredMetadata();

  const [numPages, setNumPages] = useState(null);
  const [selectedFileIndex, setSelectedFileIndex] = useState(0);
  const onLoadSuccess = ({ numPages }) => setNumPages(numPages);

  // Get available subdirectories from all files
  const subDirectories = useMemo(() => {
    const dirs = new Set(['all']);
    
    if (Array.isArray(metadataList)) {
      metadataList.forEach((record) => {
        if (Array.isArray(record.directories)) {
          record.directories.forEach((dir) => {
            if (dir) dirs.add(dir);
          });
        }
      });
    }
    
    return Array.from(dirs);
  }, [metadataList]);

  // Update selected file index when filtered list changes
  useEffect(() => {
    if (filteredMetadata.length > 0 && selectedFileIndex >= filteredMetadata.length) {
      setSelectedFileIndex(0);
    }
  }, [filteredMetadata, selectedFileIndex]);

  // Loading state
  if (loadingFilters && !isReady) {
    return <div>Loading PDF viewer...</div>;
  }

  // Error state
  if (filterError) {
    return (
      <div>
        <h3>Error loading filters</h3>
        <p>{filterError.message || 'Unknown error'}</p>
        <button onClick={() => window.location.reload()}>Reload page</button>
      </div>
    );
  }

  // No files available
  if (!Array.isArray(filteredMetadata) || filteredMetadata.length === 0) {
    return (
      <div>
        <FilterControls
          subDirectories={subDirectories}
          selectedSubDir={filters?.subDirectory || 'all'}
          onSubDirChange={(val) => setFilters((prev) => ({ ...prev, subDirectory: val }))}
          searchQuery={filters?.searchQuery || ''}
          onSearchChange={(val) => setFilters((prev) => ({ ...prev, searchQuery: val }))}
          selectedFileIndex={selectedFileIndex}
          setSelectedFileIndex={setSelectedFileIndex}
          filteredMetadata={[]}
          selectedStage={filters?.selectedStage || 'stage0'}
          onStageChange={(val) => setFilters((prev) => ({ ...prev, selectedStage: val }))}
          categoryFilters={filters?.categoryFilters || {}}
          onCategoryFilterChange={(key, val) =>
            setFilters((prev) => ({
              ...prev,
              categoryFilters: { ...prev.categoryFilters, [key]: val },
            }))
          }
          fileIdQuery={filters?.fileIdQuery || ''}
          onFileIdChange={(val) => setFilters((prev) => ({ ...prev, fileIdQuery: val }))}
          classificationFilter={filters?.classificationFilter || 'all'}
          onClassificationChange={(val) => setFilters((prev) => ({ ...prev, classificationFilter: val }))}
          availableClassifications={uiOptions?.availableClassifications || ['all']}
          showClassificationFilter={uiOptions?.showClassificationFilter || false}
        />
        
        <div style={styles.noFilesMessage}>
          <h3>No files available</h3>
          <p>No files match your filter criteria. Try adjusting your filters or check if files are loaded correctly.</p>
          {metadataList.length > 0 && (
            <p>There are {metadataList.length} files in total, but none match the current filters.</p>
          )}
          <button 
            onClick={() => setFilters(prev => ({ 
              ...prev, 
              subDirectory: 'all',
              searchQuery: '',
              fileIdQuery: '',
              categoryFilters: {},
              classificationFilter: 'all'
            }))}
            style={styles.resetButton}
          >
            Reset Filters
          </button>
        </div>
      </div>
    );
  }

  // Select file from filtered list
  const selectedFile = filteredMetadata[selectedFileIndex] || filteredMetadata[0];
  const pdfUrl = selectedFile?.fastapi_url
    ? `${TABLEAI_SERVICES.base}${selectedFile.fastapi_url}`
    : null;

  return (
    <div>
      <FilterControls
        subDirectories={subDirectories}
        selectedSubDir={filters.subDirectory}
        onSubDirChange={(val) => setFilters((prev) => ({ ...prev, subDirectory: val }))}
        searchQuery={filters.searchQuery}
        onSearchChange={(val) => setFilters((prev) => ({ ...prev, searchQuery: val }))}
        selectedFileIndex={selectedFileIndex}
        setSelectedFileIndex={setSelectedFileIndex}
        filteredMetadata={filteredMetadata}
        selectedStage={filters.selectedStage}
        onStageChange={(val) => setFilters((prev) => ({ ...prev, selectedStage: val }))}
        categoryFilters={filters.categoryFilters}
        onCategoryFilterChange={(key, val) =>
          setFilters((prev) => ({
            ...prev,
            categoryFilters: { ...prev.categoryFilters, [key]: val },
          }))
        }
        fileIdQuery={filters.fileIdQuery}
        onFileIdChange={(val) => setFilters((prev) => ({ ...prev, fileIdQuery: val }))}
        classificationFilter={filters.classificationFilter}
        onClassificationChange={(val) => setFilters((prev) => ({ ...prev, classificationFilter: val }))}
        availableClassifications={uiOptions.availableClassifications || ['all']}
        showClassificationFilter={uiOptions.showClassificationFilter || false}
      />

      {pdfUrl ? (
        <div className="pdf-container" style={styles.pdfContainer}>
          <Document 
            file={pdfUrl} 
            onLoadSuccess={onLoadSuccess}
            error={
              <div style={styles.errorMessage}>
                <p>Failed to load PDF. The URL might be invalid or the file might not exist.</p>
                <p>URL: {pdfUrl}</p>
              </div>
            }
            loading={<div>Loading PDF...</div>}
          >
            {Array.from(new Array(numPages || 0), (_, index) => (
              <Page 
                key={`page_${index + 1}`} 
                pageNumber={index + 1} 
                renderTextLayer={false}
                renderAnnotationLayer={false}
              />
            ))}
          </Document>
        </div>
      ) : (
        <div style={styles.errorMessage}>
          <p>No PDF URL available for the selected file.</p>
          <p>Selected file: {selectedFile?.file_name || 'Unknown'}</p>
          <p>This may happen if the fastapi_url property is missing in the file metadata.</p>
        </div>
      )}
    </div>
  );
}

const styles = {
  pdfContainer: {
    marginTop: '20px',
    border: '1px solid #e0e0e0',
    borderRadius: '4px',
    padding: '20px',
    backgroundColor: '#f9f9f9'
  },
  errorMessage: {
    padding: '20px',
    backgroundColor: '#fff3f3',
    border: '1px solid #ffcdd2',
    borderRadius: '4px',
    color: '#c62828',
    marginTop: '20px'
  },
  noFilesMessage: {
    padding: '20px',
    backgroundColor: '#f5f5f5',
    border: '1px solid #e0e0e0',
    borderRadius: '4px',
    marginTop: '20px',
    textAlign: 'center'
  },
  resetButton: {
    marginTop: '15px',
    padding: '8px 16px',
    backgroundColor: '#2196f3',
    color: 'white',
    border: 'none',
    borderRadius: '4px',
    cursor: 'pointer',
    fontSize: '14px'
  }
};