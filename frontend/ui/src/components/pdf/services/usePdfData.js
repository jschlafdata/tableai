// components/pdf/services/usePdfData.js

import { useState, useCallback } from 'react';
import {
  fetchMetadata,
  processPdf,
  runVisionInference,
  fetchTableHeaders
} from './pdfApi';

export function usePdfData() {
  // State for various API data
    const [metadata, setMetadata] = useState([]);
    const [processingResults, setProcessingResults] = useState([]);
    const [visionResults, setVisionResults] = useState([]);
    const [tableHeaders, setTableHeaders] = useState([]);
  
    // Split loading state
    const [metadataLoading, setMetadataLoading] = useState(false);
    const [processingLoading, setProcessingLoading] = useState(false);
    const [visionLoading, setVisionLoading] = useState(false);
    const [error, setError] = useState(null);
    const [tableHeadersLoading, setTableHeadersLoading] = useState(false);
  
    const reloadMetadata = useCallback(async () => {
      setMetadataLoading(true);
      setError(null);
      try {
        const data = await fetchMetadata();
        setMetadata(Array.isArray(data) ? data : [data]);
      } catch (err) {
        setError(err.message);
      } finally {
        setMetadataLoading(false);
      }
    }, []);
  
    const handleProcessPdf = useCallback(async ({ fileId }) => {
      setProcessingLoading(true);
      setError(null);
      try {
        const results = await processPdf({ fileId });
        setProcessingResults(results);
        return results;
      } catch (err) {
        setError(err.message);
        throw err;
      } finally {
        setProcessingLoading(false);
      }
    }, []);
  
    const handleRunVisionInference = useCallback(async ({ fileId, stage, classificationLabel }) => {
      setVisionLoading(true);
      setError(null);
      try {
        const results = await runVisionInference({ fileId, stage, classificationLabel });
        setVisionResults(results);
        return results;
      } catch (err) {
        setError(err.message);
        throw err;
      } finally {
        setVisionLoading(false);
      }
    }, []);

  // Table header bounds detection (example)
  const handleProcessTableHeaders = useCallback(async ({ fileId, stage, classificationLabel }) => {
    setTableHeadersLoading(true);
    setError(null);
    try {
      const data = await fetchTableHeaders({ fileId, stage, classificationLabel });
      setTableHeaders(data); // or format as needed
      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setTableHeadersLoading(false);
    }
  }, []);

  const clearAllResults = useCallback(() => {
    setProcessingResults([]);
    setVisionResults([]);
    setTableHeaders([]);
    setError(null);
  }, []);

  return {
    metadata,
    processingResults,
    visionResults,
    tableHeaders,
    metadataLoading,
    processingLoading,
    visionLoading,
    error,
    reloadMetadata,
    handleProcessPdf,
    handleRunVisionInference,
    handleProcessTableHeaders,
    clearAllResults,
  };
}