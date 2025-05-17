import React from 'react';
import { useTableAIRequest } from '../../hooks/api/tableai/useTableAIRequest';
import { TABLEAI_SERVICE_ENDPOINTS } from '../../schemas/tableai/tableaiServiceEndpoints';
import { useFilteredMetadata } from './useFilteredMetadata';
import FilterControls from './FilterControls';

export default function PdfDashboardPage() {
  const { data: metadataList = [], loading: loadingRecords } = useTableAIRequest(
    TABLEAI_SERVICE_ENDPOINTS.query.records
  );

  const {
    filters,
    setFilters,
    filteredMetadata,
    uiOptions,
    loadingFilters
  } = useFilteredMetadata(metadataList);

  if (loadingRecords || loadingFilters) return <div>Loading...</div>;

  return (
    <FilterControls
      subDirectories={[]}
      selectedSubDir={filters.subDirectory}
      onSubDirChange={(val) => setFilters((prev) => ({ ...prev, subDirectory: val }))}
      searchQuery={filters.searchQuery}
      onSearchChange={(val) => setFilters((prev) => ({ ...prev, searchQuery: val }))}
      selectedFileIndex={filters.selectedFileIndex}
      setSelectedFileIndex={(val) => setFilters((prev) => ({ ...prev, selectedFileIndex: val }))}
      filteredMetadata={filteredMetadata}
      selectedStage={filters.selectedStage}
      onStageChange={(val) => setFilters((prev) => ({ ...prev, selectedStage: val }))}
      categoryFilters={filters.categoryFilters}
      onCategoryFilterChange={(key, val) =>
        setFilters((prev) => ({
          ...prev,
          categoryFilters: {
            ...prev.categoryFilters,
            [key]: val
          }
        }))
      }
      fileIdQuery={filters.fileIdQuery}
      onFileIdChange={(val) => setFilters((prev) => ({ ...prev, fileIdQuery: val }))}
      classificationFilter={filters.classificationFilter}
      onClassificationChange={(val) => setFilters((prev) => ({ ...prev, classificationFilter: val }))}
      availableClassifications={uiOptions.availableClassifications}
      showClassificationFilter={uiOptions.showClassificationFilter}
    />
  );
}