// pdfDataProcessor.js
// Utility functions for processing PDF data based on stage configuration

import stageConfig, { getValue } from './stageConfig';

/**
 * Process PDF information based on stage configuration
 * @param {Object} metadata - The current file metadata
 * @param {String} selectedStage - The currently selected stage
 * @returns {Object} PDF information with path and dimensions
 */
export const processPdfInfo = (metadata, selectedStage) => {
  console.log("PROCESSOR META", metadata)
  if (!metadata) {
    return { 
      pdfPath: '', 
      combinedPdfWidth: 612, 
      combinedPdfHeight: 792, 
      pageBreaks: [] 
    };
  }

  // Get configuration for selected stage
  const config = stageConfig[selectedStage];
  if (!config) {
    console.error(`No configuration found for stage: ${selectedStage}`);
    return { 
      pdfPath: '', 
      combinedPdfWidth: 612, 
      combinedPdfHeight: 792, 
      pageBreaks: [] 
    };
  }

  // Get PDF path and dimensions from metadata using config
  const pdfPath = getValue(metadata, config.pathKey, '');
  const combinedPdfWidth = getValue(metadata, config.dimensionKeys.width, 612);
  const combinedPdfHeight = getValue(metadata, config.dimensionKeys.height, 792);
  
  // Get page breaks if configured
  const pageBreaks = config.pageBreaksKey 
    ? getValue(metadata, config.pageBreaksKey, [])
    : [];

  return { pdfPath, combinedPdfWidth, combinedPdfHeight, pageBreaks };
};

/**
 * Process bounding boxes based on stage configuration and toggle states
 * @param {Object} metadata - The current file metadata
 * @param {String} selectedStage - The currently selected stage
 * @param {Object} toggleStates - Object with boolean toggle states
 * @returns {Object|null} Bounding box data or null if none available
 */
export const processBoundingBoxes = (metadata, selectedStage, toggleStates) => {
  console.log("✅ Extracted metadata processing:", metadata)
  if (!metadata) return null;

  // Get configuration for selected stage
  const config = stageConfig[selectedStage];
  if (!config) return null;

  const result = {};
  
  // Process each overlay config
  for (const overlayConfig of config.overlayConfigs) {
    // Log for debugging
    console.log(`Processing overlay: ${overlayConfig.type}`);
    console.log(`Toggle key: ${overlayConfig.toggleKey}, value: ${toggleStates[overlayConfig.toggleKey]}`);
    
    // Check if this overlay is enabled via its toggle
    if (overlayConfig.toggleKey && !toggleStates[overlayConfig.toggleKey]) {
      console.log(`Skipping overlay ${overlayConfig.type} because toggle is off`);
      // Skip this overlay if its toggle is disabled
      continue;
    }
    
    // Get the data from metadata
    let data = getValue(metadata, overlayConfig.dataKey);
    console.log(`Data for ${overlayConfig.type}:`, data ? (Array.isArray(data) ? `Array of length ${data.length}` : data) : 'null');
    
    // Apply transform function if provided
    if (data && overlayConfig.dataTransform) {
      data = overlayConfig.dataTransform(data);
    }
    
    // Add to result if we have data
    if (data && (Array.isArray(data) ? data.length > 0 : true)) {
      result[overlayConfig.type] = data;
      console.log(`Added ${overlayConfig.type} to result`);
    }
  }
  
  return Object.keys(result).length > 0 ? result : null;
};

/**
 * Get color map for bounding boxes based on stage configuration
 * @param {String} selectedStage - The currently selected stage
 * @returns {Object} Color map for different box types
 */
export const getColorMap = (selectedStage) => {
  const config = stageConfig[selectedStage];
  if (!config) return {};
  
  const colorMap = {};
  for (const overlayConfig of config.overlayConfigs) {
    colorMap[overlayConfig.type] = overlayConfig.color;
  }
  
  return colorMap;
};