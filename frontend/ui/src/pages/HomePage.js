// src/pages/HomePage.js
import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';

export default function HomePage() {
  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Welcome to TableAI
      </Typography>
      <Typography>
        Use the sidebar to navigate to the Dropbox Explorer.
      </Typography>
    </Box>
  );
}
