// src/theme.js
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Public Sans Variable"',
    body2: {
      fontSize: '1.25',
    },
  },
  components: {
    MuiToggleButton: {
      styleOverrides: {
        root: {
          padding: '4px 8px',
        },
      },
    },
  },
});

export default theme;