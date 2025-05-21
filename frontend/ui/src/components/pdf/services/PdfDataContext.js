// components/pdf/services/PdfDataContext.js
import React, { createContext, useContext, useEffect } from 'react';
import { usePdfData } from './usePdfData';

const PdfDataContext = createContext();

export function PdfDataProvider({ children }) {
  const pdfData = usePdfData();

  // Only call reloadMetadata ONCE on mount here!
  useEffect(() => {
    pdfData.reloadMetadata();
    // eslint-disable-next-line
  }, []); // <--- ONLY empty array!

  return (
    <PdfDataContext.Provider value={pdfData}>
      {children}
    </PdfDataContext.Provider>
  );
}

export function usePdfDataContext() {
  return useContext(PdfDataContext);
}