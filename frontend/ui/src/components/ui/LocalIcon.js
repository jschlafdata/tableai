// src/components/icons/LocalIcon.jsx
import React from 'react';
import Box from '@mui/material/Box';
import sidebarStyles from '../../config/sidebarStyles';

/**
 * A component for displaying local image icons that respect the sidebar styling
 * 
 * @param {Object} props - Component props
 * @param {string} props.src - Source URL for the icon image
 * @param {string} props.alt - Alt text for the icon
 * @param {boolean} props.active - Whether this icon is in active state
 * @param {Object} props.style - Optional style override
 */
const LocalIcon = ({ src, alt, active = false, style = null, ...props }) => {
  // Get default styles or use provided style object
  const styles = style || sidebarStyles.default;
  
  return (
    <Box
      component="img"
      src={src}
      alt={alt}
      sx={styles.localIcon({ active })}
      {...props}
    />
  );
};

export default LocalIcon;