import React, { useState, useEffect, useMemo } from 'react';
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
  metadata = {},
  headerResponse, 
}) => {
  const [showResults, setShowResults] = useState(true);
  const [tableToggles, setTableToggles] = useState({});
  const [metadataToggles, setMetadataToggles] = useState({});
  const [boundingBoxesVisible, setBoundingBoxesVisible] = useState(false);
  const [localError, setLocalError] = useState(null);

  console.log("LOGGING HEADERS ACTIVITY:", headerResponse);

  // Example: if you still need classification info
  const hasClassification = !!metadata?.classification;

  // We'll just refer to the passed-in data as "tableHeaders" for convenience
  const tableHeaders = headerResponse || {};

  /**
   * Whenever the tableHeaders update, extract new toggle states.
   */
  useEffect(() => {
    if (!tableHeaders?.results?.pages) return;
    
    const toggles = {};
    const metaToggles = {};
    const tables = new Set();

    // Collect all table_index values across pages
    Object.values(tableHeaders.results.pages).forEach(pageItems => {
      pageItems.forEach(item => {
        if (item.table_index != null) {
          tables.add(item.table_index);
          
          // Initialize metadata toggles for this table
          const tableKey = `Table ${item.table_index}`;
          if (!metaToggles[tableKey]) {
            metaToggles[tableKey] = {
              main_bbox: true // By default, show the main bounding box
            };
          }
          
          // Enable toggles for all metadata fields by default
          if (item.table_metadata) {
            Object.entries(item.table_metadata).forEach(([metaKey, metaObj]) => {
              if (metaObj?.bbox) {
                metaToggles[tableKey][metaKey] = true;
              }
            });
          }
          
          // Enable toggles for other attributes with bbox info
          ['bounds_index', 'hierarchy', 'col_hash'].forEach(attr => {
            if (item[attr]?.bbox) {
              metaToggles[tableKey][attr] = true;
            }
          });
        }
      });
    });

    // By default, enable toggles for each table found
    tables.forEach(tableIndex => {
      toggles[`Table ${tableIndex}`] = true;
    });

    setTableToggles(toggles);
    setMetadataToggles(metaToggles);
    setBoundingBoxesVisible(true);
    setShowResults(true);
  }, [tableHeaders]);

  /**
   * If we toggle the results or toggles, re-inject bounding boxes after a short delay.
   */
  useEffect(() => {
    if (showResults && Object.keys(tableToggles).length > 0) {
      setBoundingBoxesVisible(false);
      const timer = setTimeout(() => setBoundingBoxesVisible(true), 50);
      return () => clearTimeout(timer);
    }
  }, [tableToggles, metadataToggles, showResults]);

  /**
   * Show/hide the entire result set
   */
  const toggleResults = () => {
    setShowResults(prev => !prev);
    setBoundingBoxesVisible(prev => !prev);
  };

  /**
   * Create a color map for each distinct table index
   */
  const colorMap = useMemo(() => {
    if (!tableHeaders?.results?.pages) return {};
    const newMap = {};
    const indexes = new Set();

    Object.values(tableHeaders.results.pages).forEach(pageItems => {
      pageItems.forEach(item => {
        if (item.table_index != null) {
          indexes.add(item.table_index);
        }
      });
    });
    indexes.forEach(index => {
      newMap[`Table ${index}`] = getColorForTable(index);
    });
    return newMap;
  }, [tableHeaders]);

  return (
    <>
      {/* 
          TableHeaderActions can still render a "Detect Headers" button if you like,
          but if you no longer need that (since the data is already cached),
          you can remove it or simplify it.
       */}
      <TableHeaderActions
        hasClassification={hasClassification}
        fileId={fileId}
        // Pass in the data if needed, or remove these props if not used
        tableHeaders={tableHeaders}
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
          metadataToggles={metadataToggles}
          setMetadataToggles={setMetadataToggles}
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
          metadataToggles={metadataToggles}
          boundingBoxesVisible={boundingBoxesVisible}
          pageDimensions={pageDimensions}
          currentPage={currentPage}
          colorMap={colorMap}
        />
      )}

      {/* Error/Success Snackbar */}
      <ErrorSnackbar
        error={localError}
        onClose={() => setLocalError(null)}
      />
    </>
  );
};

export default TableHeaderBounds;