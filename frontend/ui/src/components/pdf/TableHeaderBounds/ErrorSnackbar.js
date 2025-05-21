import React from 'react';
import { Snackbar, Alert } from '@mui/material';

const ErrorSnackbar = ({ error, localError, onClose }) => {
  const open = Boolean(error) || Boolean(localError);

  if (!open) {
    return null;
  }

  // Figure out the best message to show
  const message = localError?.message || localError || error?.message || error || '';

  return (
    <Snackbar
      open={open}
      autoHideDuration={6000}
      onClose={onClose}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
    >
      <Alert severity="error" onClose={onClose}>
        {message}
      </Alert>
    </Snackbar>
  );
};

export default ErrorSnackbar;