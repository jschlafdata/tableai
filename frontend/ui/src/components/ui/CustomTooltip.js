import React from 'react'
import Tooltip from '@mui/material/Tooltip'
import Box from '@mui/material/Box'
import Typography from '@mui/material/Typography'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import LightbulbOutlinedIcon from '@mui/icons-material/LightbulbOutlined'
import WarningAmberOutlinedIcon from '@mui/icons-material/WarningAmberOutlined'
import ErrorOutlineOutlinedIcon from '@mui/icons-material/ErrorOutlineOutlined'
import NotesOutlinedIcon from '@mui/icons-material/NotesOutlined'

const VARIANT_STYLES = {
  note: {
    icon: <NotesOutlinedIcon fontSize="small" sx={{ mr: 1 }} />,
    color: '#444',
    background: '#f9f9f9',
    border: '1px solid #ddd',
  },
  tip: {
    icon: <LightbulbOutlinedIcon fontSize="small" sx={{ mr: 1 }} />,
    color: '#1a4731',
    background: '#e6f4ea',
    border: '1px solid #c8e6c9',
  },
  info: {
    icon: <InfoOutlinedIcon fontSize="small" sx={{ mr: 1 }} />,
    color: '#0d3b66',
    background: '#e1f5fe',
    border: '1px solid #b3e5fc',
  },
  warning: {
    icon: <WarningAmberOutlinedIcon fontSize="small" sx={{ mr: 1 }} />,
    color: '#7a4f01',
    background: '#fff8e1',
    border: '1px solid #ffe082',
  },
  danger: {
    icon: <ErrorOutlineOutlinedIcon fontSize="small" sx={{ mr: 1 }} />,
    color: '#761b18',
    background: '#fdecea',
    border: '1px solid #f5c6cb',
  },
}

const CustomTooltip = ({ variant = 'info', title = '', children }) => {
  const style = VARIANT_STYLES[variant] || VARIANT_STYLES.info

  return (
    <Tooltip
      title={
        <Box
          sx={{
            bgcolor: style.background,
            color: style.color,
            border: style.border,
            borderRadius: 1,
            padding: 1.2,
            maxWidth: 280,
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
            {style.icon}
            <Typography variant="subtitle2" sx={{ fontWeight: 700 }}>
              {variant.toUpperCase()}
            </Typography>
          </Box>
          <Typography variant="body2">{title}</Typography>
        </Box>
      }
      arrow
      placement="top"
    >
      {children}
    </Tooltip>
  )
}

export default CustomTooltip