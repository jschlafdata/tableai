import React from 'react';
import Box from '@mui/material/Box';
import DropboxFolders from '../components/integrations/DropboxFolders';

export default function DropboxPage() {
  return (
    <Box
      sx={{
        bgcolor: '#f9fafb',
        minHeight: '100vh',
        py: 4,
        px: 2
      }}
    >
      <Box
        sx={{
          bgcolor: '#fff',
          border: '1px solid #e0e0e0',
          boxShadow: '0 2px 8px rgba(0, 0, 0, 0.05)',
          borderRadius: 2,
          maxWidth: 1200,
          p: 4,
          ml: 2 // 👈 pushes away from sidebar while keeping it left-aligned
        }}
      >
        <Box
          component="img"
          src="/Dropbox_logo_2017.png"
          alt="Logo"
          sx={{ height: 35, width: 'auto', mb: 3 }}
        />
        <DropboxFolders />
      </Box>
    </Box>
  );
}
