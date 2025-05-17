import { useEffect, useMemo, useState, useRef } from 'react';
import { useTableAIRequest } from './useTableAIRequest';
import { TABLEAI_SERVICE_ENDPOINTS } from '../../../schemas/tableai/tableaiServiceEndpoints';

export function useFilteredMetadata() {
  const {
    data: filterData,
    loading: loadingFilters,
    error: filterError,
  } = useTableAIRequest(TABLEAI_SERVICE_ENDPOINTS.query.filterMetadata);

  // Initialize with default values
  const [filters, setFilters] = useState({
    subDirectory: 'all',
    searchQuery: '',
    fileIdQuery: '',
    selectedStage: 'stage0',
    selectedFileIndex: 0,
    categoryFilters: {},
    classificationFilter: 'all'
  });
  
  const [uiOptions, setUiOptions] = useState({
    availableClassifications: ['all'],
    showClassificationFilter: false
  });

  // Track if this is the first render
  const isInitialized = useRef(false);
  
  // Get the metadata list directly from the filter data
  const metadataList = useMemo(() => {
    return Array.isArray(filterData?.files) ? filterData.files : [];
  }, [filterData]);

  // Initialize filters and UI options from API
  useEffect(() => {
    if (filterData) {
      console.log('Filter data received:', filterData);
      
      // Update filters from API data if available
      if (filterData.filters) {
        setFilters(prev => ({
          ...prev,
          subDirectory: filterData.filters.subDirectory || prev.subDirectory,
          searchQuery: filterData.filters.searchQuery || prev.searchQuery,
          fileIdQuery: filterData.filters.fileIdQuery || prev.fileIdQuery,
          selectedStage: filterData.filters.selectedStage || prev.selectedStage,
          categoryFilters: filterData.filters.categoryFilters || prev.categoryFilters,
          classificationFilter: filterData.filters.classificationFilter || prev.classificationFilter
        }));
      }

      // Update UI options from API data if available
      if (filterData.uiOptions) {
        setUiOptions(prev => ({
          ...prev,
          ...filterData.uiOptions
        }));
      }
      
      isInitialized.current = true;
    }
  }, [filterData]);

  // Apply filters to metadata
  const filteredMetadata = useMemo(() => {
    if (!Array.isArray(metadataList) || metadataList.length === 0) {
      console.log('No metadata list to filter');
      return [];
    }

    console.log(`Filtering ${metadataList.length} metadata items with filters:`, filters);
    
    return metadataList.filter((item) => {
      if (!item) return false;

      // File ID filter
      const matchesFileId =
        !filters.fileIdQuery ||
        (item.uuid && item.uuid.toLowerCase().includes(filters.fileIdQuery.toLowerCase())) ||
        (item.dropbox_safe_id && item.dropbox_safe_id.toLowerCase().includes(filters.fileIdQuery.toLowerCase()));

      // Search query filter (case insensitive)
      const matchesSearch =
        !filters.searchQuery ||
        (item.file_name && item.file_name.toLowerCase().includes(filters.searchQuery.toLowerCase()));

      // Subdirectory filter
      const directories = Array.isArray(item.directories)
        ? item.directories
        : [];

      const joinedDir = directories.join('/');
      const matchesSubDir =
          filters.subDirectory === 'all' || joinedDir.includes(filters.subDirectory);
      
      // Classification filter
      const matchesClassification =
        filters.classificationFilter === 'all' || item.classification === filters.classificationFilter;

      // Path categories filter
      let pathCats = {};
      if (typeof item.path_categories === 'string') {
        try {
          pathCats = JSON.parse(item.path_categories || '{}');
        } catch (err) {
          console.warn('Invalid path_categories JSON:', err);
        }
      } else if (typeof item.path_categories === 'object' && item.path_categories !== null) {
        pathCats = item.path_categories;
      }

      const matchesCategoryFilters = Object.entries(filters.categoryFilters || {}).every(
        ([key, value]) => value === 'all' || pathCats?.[key] === value
      );

      return (
        matchesFileId &&
        matchesSearch &&
        matchesSubDir &&
        matchesClassification &&
        matchesCategoryFilters
      );
    });
  }, [metadataList, filters]);

  // Consider ready when either:
  // 1. We have filter data from the API
  // 2. We've encountered an error but still want to proceed
  // 3. Loading is complete (even if no data was returned)
  const isReady = isInitialized.current || !!filterError || !loadingFilters;

  // For debugging
  useEffect(() => {
    console.log('FilteredMetadata status:', {
      isReady,
      loadingFilters,
      filterData, 
      hasFilterData: !!filterData,
      filterError: filterError?.message,
      metadataCount: metadataList.length,
      filteredCount: filteredMetadata.length
    });
  }, [isReady, loadingFilters, filterData, filterError, metadataList.length, filteredMetadata.length]);

  return {
    filters,
    setFilters,
    filteredMetadata,
    metadataList, // Expose the raw metadata list for debugging
    uiOptions,
    loadingFilters,
    filterError,
    isReady
  };
}

