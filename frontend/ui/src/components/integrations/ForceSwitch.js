
import React from 'react'
import FormControlLabel from '@mui/material/FormControlLabel'
import Switch from '@mui/material/Switch'

const ForceSwitch = ({ force, onForceChange }) => {
  return (
    <FormControlLabel
      control={
        <Switch
          checked={force}
          onChange={(e) => onForceChange(e.target.checked)}
          color="primary"
        />
      }
      label="Force"
      sx={{ mr: 2 }}
    />
  )
}

export default ForceSwitch