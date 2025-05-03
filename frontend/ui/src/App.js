// src/App.js
import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import FolderIcon from '@mui/icons-material/Folder';
import Layout from './components/Layout';
import DropboxPage from './pages/DropboxPage';

export default function App() {
  const menuConfig = {
    mainTitle: 'INTEGRATIONS',
    main: [
      {
        name: 'Dropbox',
        icon: <FolderIcon />,
        path: '/integrations/dropbox'
      }
    ],
    misc: []
  };

  return (
    <Routes>
      {/* Layout wraps all of these */}
      <Route path="/" element={<Layout menuConfig={menuConfig} />}>

        {/* When user hits “/”, send them to the Dropbox route */}
        <Route index element={<Navigate to="/integrations/dropbox" replace />} />

        {/* Only this route needs to exist */}
        <Route
          path="integrations/dropbox"
          element={<DropboxPage />}
        />

        {/* Anything else is a 404 */}
        <Route path="*" element={<div>404 Not Found</div>} />
      </Route>
    </Routes>
  );
}
