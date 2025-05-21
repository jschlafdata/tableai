// src/components/pdf/PdfNavigation.jsx
import React from 'react';

const PdfNavigation = ({ pageNumber, numPages, goToPrevPage, goToNextPage }) => {
  return (
    <div>
      <div style={{ marginTop: '10px' }}>
        <button 
          onClick={goToPrevPage} 
          disabled={pageNumber <= 1} 
          style={styles.button}
        >
          Previous Page
        </button>
        <button 
          onClick={goToNextPage} 
          disabled={pageNumber >= (numPages || 1)} 
          style={styles.button}
        >
          Next Page
        </button>
      </div>
      <p style={{ marginTop: '8px' }}>
        Page {pageNumber} of {numPages || 1}
      </p>
    </div>
  );
};

const styles = {
  button: {
    padding: '8px 14px',
    marginRight: '10px',
    border: 'none',
    borderRadius: '4px',
    backgroundColor: '#1976d2',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '14px'
  }
};

export default PdfNavigation;