import React from 'react'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import WarningAmberIcon from '@mui/icons-material/WarningAmber'
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty'
import Tooltip from '@mui/material/Tooltip'
import Filter1OutlinedIcon from '@mui/icons-material/Filter1Outlined';
import CustomTooltip from '../ui/CustomTooltip'

const Stage0StatusIcon = ({ fileId, summaryMap }) => {
  if (!fileId || !summaryMap) return null

  const meta = summaryMap[fileId]
//   console.log('Rendering icon for:', fileId, meta)

  if (!meta) return null

  const isComplete = Boolean(meta.stage0_complete) === false
//   console.log("is complete:", isComplete)
  const isRecovered = String(meta.recovered_pdf) === "1" || String(meta.recovered_pdf) === "true"
//   console.log("is isRecovered:", isRecovered)

  if (isRecovered) {
    return (
      <CustomTooltip variant="warning" title="This file was recovered using OCR scanning.">
        <Filter1OutlinedIcon color="warning" fontSize="medium" sx={{ ml: 2, color: '#F28C28' }} />
      </CustomTooltip>
    )
  }

  if (isComplete) {
    return ( 
    <CustomTooltip variant="tip" title="This file passed stage 0 validation.">
        <Filter1OutlinedIcon sx={{ ml: 2, color: '#7e62fc' }} />
    </CustomTooltip>
    )
  }

  return (
    <CustomTooltip variant="danger" title="Stage 0 validation pending or failed">
      <Filter1OutlinedIcon color="disabled" fontSize="medium" sx={{ ml: 2, color: '#EE4B2B' }} />
    </CustomTooltip>
  )
}

export default Stage0StatusIcon