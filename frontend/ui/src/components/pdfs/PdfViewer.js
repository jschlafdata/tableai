import React, { useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// Load the PDF.js worker from your public folder or CDN
pdfjs.GlobalWorkerOptions.workerSrc = '/pdf.worker.min.mjs';

const PdfViewer = ({ fileUrl }) => {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);

  const onDocumentLoadSuccess = ({ numPages }) => {
    setNumPages(numPages);
    setPageNumber(1);
  };

  const goToPrevPage = () => setPageNumber(p => Math.max(p - 1, 1));
  const goToNextPage = () => setPageNumber(p => Math.min(p + 1, numPages));

  return (
    <div style={{ textAlign: 'left' }}>
      <Document
        file={fileUrl}
        onLoadSuccess={onDocumentLoadSuccess}
        onLoadError={(err) => console.error('PDF load error:', err)}
      >
        <Page
          pageNumber={pageNumber}
          renderAnnotationLayer={false}
          renderTextLayer={false}
          scale={2}
        />
      </Document>

      <div style={{ marginTop: 10 }}>
        <button onClick={goToPrevPage} disabled={pageNumber <= 1}>Previous</button>
        <span style={{ margin: '0 10px' }}>
          Page {pageNumber} of {numPages || 1}
        </span>
        <button onClick={goToNextPage} disabled={pageNumber >= numPages}>Next</button>
      </div>
    </div>
  );
};

export default PdfViewer;
