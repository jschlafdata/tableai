// src/App.js
import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom'
import FolderIcon from '@mui/icons-material/Folder';
import Layout from './components/Layout';
import DropboxPage from './pages/DropboxPage';
import DisplayPdf from './pages/PdfViewerApp';
import ClassifyPage from './pages/ClassifyPage'
import LocalIcon from './components/ui/LocalIcon';
import dataClassificationIcon from './assets/icons/dataClassificationIcon.png'
import pdfViewer from './assets/icons/pdfViewer.png'


export default function App() {
  const menuConfig = {
    mainTitle: 'INTEGRATIONS',
    main: [
      { name: 'Dropbox', icon: <FolderIcon />, path: '/integrations/dropbox' }
    ],
    tools: [
      { name: 'PDF Viewer', 
        icon: <LocalIcon src={pdfViewer} alt="PdfViewer" />,
        path: '/pdf_viewer' },
      { name: 'Classifier', 
        icon: <LocalIcon src={dataClassificationIcon} alt="Classification" />,  
        path: '/classify' },
    ],
  };

  return (
    <Routes>
      <Route path="/" element={<Layout menuConfig={menuConfig} />}>
        <Route index element={<Navigate to="/pdf_viewer" replace />} />
        <Route path="pdf_viewer" element={<DisplayPdf />} />
        <Route path="integrations/dropbox" element={<DropboxPage />} />
        <Route path="/classify" element={<ClassifyPage />} />
        <Route path="*" element={<div>404 Not Found</div>} />
      </Route>
    </Routes>
  );
}
