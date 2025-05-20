import React, { useState, useEffect, useMemo } from 'react';
import { Box, Typography, Snackbar, Alert } from '@mui/material';
import ReactDOM from 'react-dom';

import { usePdfDataContext } from '../../../context/PdfDataContext';

import TableHeaderActions from './TableHeaderActions';
import TableHeaderToggles from './TableHeaderToggles';
import BoundingBoxesPortal from './BoundingBoxesPortal';
import ErrorSnackbar from './ErrorSnackbar';
import { getColorForTable } from '../utils/getColorForTable';

const TableHeaderBounds = ({
  fileId,
  stage,
  pageDimensions,
  currentPage,
  pdfContainerRef,
  scale = 1.5,
  metadata = {}
}) => {
  const {
    tableHeaders,
    tableHeadersLoading,
    handleProcessTableHeaders,
    error,
    reloadMetadata
  } = usePdfDataContext();

  const [showResults, setShowResults] = useState(true);
  const [tableToggles, setTableToggles] = useState({});
  const [boundingBoxesVisible, setBoundingBoxesVisible] = useState(false);
  const [localError, setLocalError] = useState(null);

  const hasClassification = !!metadata?.classification;

  /** Cleanup the container when unmounting */
  useEffect(() => {
    return () => {
      const boxContainer = document.getElementById('table-header-box-container');
      if (boxContainer && boxContainer.parentNode) {
        boxContainer.parentNode.removeChild(boxContainer);
      }
    };
  }, []);

  /**
   * Whenever `tableHeaders` changes, extract new toggle states.
   */
  useEffect(() => {
    if (!tableHeaders || !tableHeaders.results?.pages) return;
    
    const toggles = {};
    const tables = new Set();
    Object.values(tableHeaders.results.pages).forEach(pageItems => {
      pageItems.forEach(item => {
        if (item.table_index != null) {
          tables.add(item.table_index);
        }
      });
    });

    tables.forEach(tableIndex => {
      toggles[`Table ${tableIndex}`] = true;
    });

    setTableToggles(toggles);
    setBoundingBoxesVisible(true);
    setShowResults(true);
  }, [tableHeaders]);

  /**
   * If we toggle the results or toggles, re-show bounding boxes after a short delay
   * to re-inject them properly into the portal.
   */
  useEffect(() => {
    if (showResults && Object.keys(tableToggles).length > 0) {
      setBoundingBoxesVisible(false);
      const timer = setTimeout(() => setBoundingBoxesVisible(true), 50);
      return () => clearTimeout(timer);
    }
  }, [tableToggles, showResults]);

  /**
   * Start table header detection & handle errors
   */
  const onProcessTableHeaders = async () => {
    if (!fileId || !hasClassification) {
      setLocalError("Cannot run table header detection — file must have a classification.");
      return;
    }
    setTableToggles({});
    setBoundingBoxesVisible(false);
    try {
      await handleProcessTableHeaders({
        fileId,
        stage,
        classificationLabel: metadata.classification
      });
      await reloadMetadata();
    } catch (err) {
      setLocalError(err.message);
    }
  };

  /**
   * Show/hide the entire result set
   */
  const toggleResults = () => {
    setShowResults(!showResults);
    setBoundingBoxesVisible(!showResults);
  };

  const colorMap = useMemo(() => {
    if (!tableHeaders?.results?.pages) return {};
    const newMap = {};
    // Gather all tableIndexes
    const indexes = new Set();
    Object.values(tableHeaders.results.pages).forEach(pageItems => {
      pageItems.forEach(item => {
        if (item.table_index != null) {
          indexes.add(item.table_index);
        }
      });
    });
    // Assign color to each
    indexes.forEach(index => {
      newMap[`Table ${index}`] = getColorForTable(index);
    });
    return newMap;
  }, [tableHeaders]);

  return (
    <>
      {/* Actions (Detect, Show/Hide, etc.) */}
      <TableHeaderActions
        hasClassification={hasClassification}
        fileId={fileId}
        tableHeaders={tableHeaders}
        tableHeadersLoading={tableHeadersLoading}
        onProcessTableHeaders={onProcessTableHeaders}
        showResults={showResults}
        toggleResults={toggleResults}
        currentPage={currentPage}
      />

      {/* Toggles panel (only show if we have results) */}
      {tableHeaders?.results && Object.keys(tableToggles).length > 0 && (
        <TableHeaderToggles
          tableHeaders={tableHeaders}
          tableToggles={tableToggles}
          setTableToggles={setTableToggles}
          showResults={showResults}
          colorMap={colorMap}
          currentPage={currentPage}
        />
      )}

      {/* The bounding boxes overlay (portal) */}
      {showResults && (
        <BoundingBoxesPortal
          pdfContainerRef={pdfContainerRef}
          tableHeaders={tableHeaders}
          tableToggles={tableToggles}
          boundingBoxesVisible={boundingBoxesVisible}
          pageDimensions={pageDimensions}
          currentPage={currentPage}
          colorMap={colorMap}
        />
      )}

      {/* Error/Success Snackbar */}
      <ErrorSnackbar
        error={error}
        localError={localError}
        onClose={() => setLocalError(null)}
      />
    </>
  );
};

export default TableHeaderBounds;