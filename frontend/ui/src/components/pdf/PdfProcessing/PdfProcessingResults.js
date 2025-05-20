import React, { useState, useEffect } from 'react';
import {
  Snackbar,
  Alert,
} from '@mui/material';
import ReactDOM from 'react-dom';

import PdfProcessingControls from './PdfProcessingControls';
import QueryFilterPanel from './QueryFilterPanel';
import BoundingBoxes from '../utils/BoundingBoxes';

import { getResultsCount } from '../utils/pdfUtils';
import { usePdfDataContext } from '../services/PdfDataContext';
import transformCoordWithContainer from '../utils/transformCoordWithContainer';
import getFilteredBoxesData from '../utils/getFilteredBoxesData';
import { createLabelColorGenerator } from '../utils/getColorForLabel';

// For debugging
const DEBUG = false;

const PdfProcessingResults = ({
  fileId,
  stage, // For backward compatibility
  selectedStage, // New prop to determine which stage results to display
  pageDimensions,
  currentPage,
  pdfContainerRef,
  scale = 1.5,
  metadata = {},
}) => {
  // Use selectedStage if provided, otherwise fall back to stage for backward compatibility
  const effectiveStage = selectedStage !== undefined ? selectedStage : stage;
  
  const {
    processingResults,
    processingLoading,
    visionLoading,
    error,
    handleProcessPdf,
    handleRunVisionInference,
    clearAllResults,
    reloadMetadata,
  } = usePdfDataContext();

  // ---- Local State ----
  const [showResults, setShowResults] = useState(true);
  const [queryToggles, setQueryToggles] = useState({});
  const [boundingBoxesVisible, setBoundingBoxesVisible] = useState(false);
  const [localError, setLocalError] = useState(null);

  // ---- Clean up bounding box container on unmount ----
  useEffect(() => {
    return () => {
      const boxContainer = document.getElementById('bounding-box-container');
      if (boxContainer && boxContainer.parentNode) {
        boxContainer.parentNode.removeChild(boxContainer);
      }
    };
  }, []);

  // ---- Sync error from hook to local error ----
  useEffect(() => {
    if (error) setLocalError(error);
  }, [error]);

  // ---- Determine current stage results ----
  const getCurrentStageResults = () => {
    if (!processingResults) return [];
    if (DEBUG) {
      console.log("Processing Results:", processingResults);
      console.log("Effective Stage:", effectiveStage);
    }
    
    // If new dictionary format
    if (processingResults[`stage${effectiveStage}`]) {
      return processingResults[`stage${effectiveStage}`] || [];
    }
    
    // Fallback (backward-compat array)
    return Array.isArray(processingResults) ? processingResults : [];
  };

  const currentStageResults = getCurrentStageResults();
  
  if (DEBUG) {
    console.log("Current Stage Results:", currentStageResults);
  }

  // ---- Setup toggles whenever new results come in ----
  useEffect(() => {
    if (!currentStageResults || currentStageResults.length === 0) return;
    if (DEBUG) {
      console.log("Setting up toggles for results:", currentStageResults);
    }
    
    const initialToggles = {};
    currentStageResults.forEach((result) => {
      initialToggles[result.query_label] = true;
    });
    setQueryToggles(initialToggles);
    setBoundingBoxesVisible(true);
    setShowResults(true);
  }, [processingResults, effectiveStage]);

  // ---- Re-render bounding boxes on toggles change ----
  useEffect(() => {
    if (showResults && Object.keys(queryToggles).length > 0) {
      setBoundingBoxesVisible(false);
      const timer = setTimeout(() => setBoundingBoxesVisible(true), 50);
      return () => clearTimeout(timer);
    }
  }, [queryToggles, showResults]);

  // ---- Helpers ----
  const hasClassification = !!metadata?.classification;
  const hasCurrentStageResults = currentStageResults && currentStageResults.length > 0;

  const onProcessPdf = async () => {
    if (!fileId) {
      setLocalError("No file selected");
      return;
    }
    clearAllResults();
    setBoundingBoxesVisible(false);
    try {
      await handleProcessPdf({ fileId });
      await reloadMetadata();
    } catch (err) {
      setLocalError(err.message);
    }
  };
  
  const onRunVisionInference = async (visionOptions) => {
    if (!fileId || !hasClassification) {
      setLocalError("Cannot run vision inference - file must have a classification");
      return;
    }
    try {
      await handleRunVisionInference({
        fileId,
        stage: effectiveStage,                   // from props or context
        classificationLabel: metadata.classification, // from your file metadata
        ...visionOptions                         // spread all the options from VisionInferenceOptions
      });
      setLocalError({ message: "Vision inference completed successfully!", severity: "success" });
      await reloadMetadata();
    } catch (err) {
      setLocalError(err.message);
    }
  };

  const getColorForLabel = createLabelColorGenerator();
  
  // Build color map
  const colorMap = {};
  (currentStageResults || []).forEach((r) => {
    colorMap[r.query_label] = getColorForLabel(r.query_label);
  });

  // --- Toggling logic ---
  const toggleResults = () => {
    setShowResults((prev) => !prev);
    setBoundingBoxesVisible((prev) => !prev);
  };

  const toggleQueryLabel = (queryLabel) => {
    setQueryToggles((prev) => ({
      ...prev,
      [queryLabel]: !prev[queryLabel],
    }));
    // Force bounding box re-render
    setBoundingBoxesVisible((prev) => {
      if (prev) {
        setTimeout(() => setBoundingBoxesVisible(true), 10);
        return false;
      }
      return true;
    });
  };

  const toggleAllQueryLabels = (value) => {
    const updated = {};
    Object.keys(queryToggles).forEach((key) => {
      updated[key] = value;
    });
    setQueryToggles(updated);
  };


  const boxesData = getFilteredBoxesData({
    data: currentStageResults,
    pageNumber: currentPage,
    showResults,
    toggles: queryToggles
  });

  const getQueryLabelResultsCount = (label) => {
    const queryResult = (currentStageResults || []).find((r) => r.query_label === label);
    if (!queryResult || !queryResult.results?.pages) return 0;
    const pageIndex = String(currentPage - 1);
    return (queryResult.results.pages[pageIndex] || []).length;
  };

  const transformCoord = (coords, color) =>
    transformCoordWithContainer(coords, {
      color,
      containerDimensions: pageDimensions,
      pdfData: currentStageResults,
      pageNumber: currentPage,
      backgroundOpacity: 0.3
  });

  // --- Render bounding boxes via portal ---
  const resultsCount = boxesData ? getResultsCount(boxesData) : 0;
  const canRenderBoxes =
    pdfContainerRef?.current &&
    pageDimensions.width > 0 &&
    pageDimensions.height > 0 &&
    boundingBoxesVisible &&
    boxesData;

  const renderBoundingBoxesPortal = () => {
    if (!canRenderBoxes) {
      if (DEBUG) {
        console.log("Cannot render boxes:", {
          containerExists: !!pdfContainerRef?.current,
          dimensions: pageDimensions,
          boundingBoxesVisible,
          hasBoxesData: !!boxesData,
        });
      }
      return null;
    }
    let boxContainer = document.getElementById('bounding-box-container');
    if (!boxContainer) {
      boxContainer = document.createElement('div');
      boxContainer.id = 'bounding-box-container';
      boxContainer.style.position = 'absolute';
      boxContainer.style.top = '0';
      boxContainer.style.left = '0';
      boxContainer.style.width = '100%';
      boxContainer.style.height = '100%';
      boxContainer.style.pointerEvents = 'none';
      boxContainer.style.zIndex = '1000';
      boxContainer.style.border = 'none';
      boxContainer.style.background = 'transparent';
      boxContainer.style.boxShadow = 'none';
      boxContainer.style.overflow = 'hidden';

      pdfContainerRef.current.style.position = 'relative';
      pdfContainerRef.current.appendChild(boxContainer);
    }

    return ReactDOM.createPortal(
      <BoundingBoxes
        boxesData={boxesData}
        transformCoord={transformCoord}
        colorMap={colorMap}
        showTooltips={true}
      />,
      boxContainer
    );
  };

  // ---- Render ----
  return (
    <>
      {/* Top bar / process controls */}
      <PdfProcessingControls
        fileId={fileId}
        hasClassification={hasClassification}
        processingLoading={processingLoading}
        visionLoading={visionLoading}
        onProcessPdf={onProcessPdf}
        onRunVisionInference={onRunVisionInference}
        showResults={showResults}
        toggleResults={toggleResults}
        resultsCount={resultsCount}
        currentPage={currentPage}
        effectiveStage={effectiveStage}
        hasCurrentStageResults={hasCurrentStageResults}
        metadata={metadata}
      />

      {/* Query filters panel */}
      {hasCurrentStageResults && (
        <QueryFilterPanel
          currentStageResults={currentStageResults}
          queryToggles={queryToggles}
          toggleQueryLabel={toggleQueryLabel}
          toggleAllQueryLabels={toggleAllQueryLabels}
          getQueryLabelResultsCount={getQueryLabelResultsCount}
          getColorForLabel={getColorForLabel}
        />
      )}

      {/* Render bounding boxes via portal */}
      {renderBoundingBoxesPortal()}

      {/* Error/Success Snackbar */}
      <Snackbar
        open={!!localError}
        autoHideDuration={6000}
        onClose={() => setLocalError(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert severity={localError?.severity || "error"} onClose={() => setLocalError(null)}>
          {localError?.message || localError}
        </Alert>
      </Snackbar>
    </>
  );
};

export default PdfProcessingResults;