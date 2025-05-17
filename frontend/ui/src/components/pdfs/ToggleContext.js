// ToggleContext.js
import React, { createContext, useContext, useState, useMemo } from 'react';

// Define all available toggles with their default values and labels
const TOGGLE_DEFINITIONS = {
  showTableBounds: { default: true, label: 'Table Bounds' },
  showTableNames: { default: true, label: 'Table Names' },
  showRecurringBlocks: { default: true, label: 'Recurring Blocks' },
  showDataBlocks: { default: true, label: 'Data Blocks' },
  showWhitespaceBlocks: { default: true, label: 'Whitespace Blocks' },
  showTableContinued: { default: true, label: 'Table Continued' },
  showInternalHeaders: { default: true, label: 'Internal Headers' },
  showInverseTables: { default: true, label: 'Inverse Tables' },
  showMultiTables: { default: true, label: 'Multi Tables' },
  showTableHeaders: { default: true, label: 'Table Headers' },
  showTableTotals: { default: true, label: 'Table Totals' },
  showColumnAlignments: { default: true, label: 'Column Alignments' },
  showSpanningText: { default: true, label: 'Spanning Text' },
  showTableCells: { default: true, label: 'Table Cells' },
  showRowBounds: { default: true, label: 'Table Rows' },
  showCellBounds: { default: true, label: 'Table Cells' },
  showTotalsBox: { default: true, label: 'Totals Box' },
  showTotalsCells: { default: true, label: 'Totals Cells' },
  // Add new toggles here - they'll automatically be integrated
};

// Create the context
const ToggleContext = createContext();

// Create provider component
export function ToggleProvider({ children }) {
  // Initialize all toggle states using the definitions
  const [toggles, setToggles] = useState(() => {
    const initialStates = {};
    Object.entries(TOGGLE_DEFINITIONS).forEach(([key, config]) => {
      initialStates[key] = config.default;
    });
    return initialStates;
  });

  // Toggle setter function
  const setToggle = (key, value) => {
    if (key in toggles) {
      setToggles(prev => ({ ...prev, [key]: value }));
    } else {
      console.warn(`Attempted to set undefined toggle: ${key}`);
    }
  };

  // Get all toggle definitions with current values
  const toggleDefinitions = useMemo(() => {
    return Object.entries(TOGGLE_DEFINITIONS).map(([key, config]) => ({
      key,
      label: config.label,
      value: toggles[key]
    }));
  }, [toggles]);

  // Context value
  const value = {
    toggles,
    setToggle,
    toggleDefinitions
  };

  return (
    <ToggleContext.Provider value={value}>
      {children}
    </ToggleContext.Provider>
  );
}

// Hook for accessing the toggle context
export function useToggles() {
  const context = useContext(ToggleContext);
  if (context === undefined) {
    throw new Error('useToggles must be used within a ToggleProvider');
  }
  return context;
}

// Helper to get all toggle keys
export function getAllToggleKeys() {
  return Object.keys(TOGGLE_DEFINITIONS);
}

// Helper to get toggle details
export function getToggleDefinitions() {
  return TOGGLE_DEFINITIONS;
}
