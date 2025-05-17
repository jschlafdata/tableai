// src/components/pdfs/LoadingState.js
import React from 'react';

const LoadingState = ({ message = 'Loading...' }) => {
  return (
    <div style={styles.container}>
      {message}
    </div>
  );
};

const styles = {
  container: {
    margin: '20px',
    fontFamily: 'Arial, sans-serif',
    padding: '20px',
    backgroundColor: '#f5f5f5',
    borderRadius: '8px',
    textAlign: 'center',
    color: '#666'
  }
};

export default LoadingState;