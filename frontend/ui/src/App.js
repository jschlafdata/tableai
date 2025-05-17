// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import FolderIcon from '@mui/icons-material/Folder';
import Layout from './components/Layout';
import CONSTANTS from './constants'
import DropboxPage from './pages/DropboxPage';
import DisplayPdf from './pages/PdfViewerApp';
import ClassifyPage from './pages/ClassifyPage'
import ExtractViewerPage from './pages/ExtractViewerPage'
import BasicPdfViewerPage from './components/pdfs/basicPdfViewer'

// import ApiTestPage from './pages/ApiTestPage'

// export default function App() {
//   return (
//       <Routes>
//         <Route path="/test" element={<ApiTestPage />} />
//         <Route path="integrations/dropbox" element={<DropboxPage />} />
//         {/* your other routes */}
//       </Routes>
//   )
// }


export default function App() {
  const menuConfig = {
    mainTitle: 'INTEGRATIONS',
    main: [
      { name: 'Dropbox', icon: <FolderIcon />, path: '/integrations/dropbox' }
    ],
    tools: [
      { name: 'PDF Viewer', icon: <FolderIcon />, path: '/pdf_viewer' },
      { name: 'Classifier', icon: <FolderIcon />, path: '/classify' },
      { name: 'Extractor', icon: <FolderIcon />, path: '/viewer' }
    ],
    misc: [
      { name: 'About', icon: <FolderIcon />, path: '/about' }
    ]
  };

  return (
    <Routes>
      <Route path="/" element={<Layout menuConfig={menuConfig} />}>
        <Route index element={<Navigate to="/pdf_viewer" replace />} />
        <Route path="pdf_viewer" element={<DisplayPdf />} />
        <Route path="integrations/dropbox" element={<DropboxPage />} />
        <Route path="/viewer" element={<ExtractViewerPage />} />
        <Route path="/basic-pdf-viewer" element={<BasicPdfViewerPage />} />
        <Route path="/classify" element={<ClassifyPage />} />
        <Route path="*" element={<div>404 Not Found</div>} />
      </Route>
    </Routes>
  );
}
