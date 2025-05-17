import React from 'react'
import Box from '@mui/material/Box'
import Typography from '@mui/material/Typography'
import Checkbox from '@mui/material/Checkbox'
import IconButton from '@mui/material/IconButton'
import AvatarGroup from '@mui/material/AvatarGroup'
import Avatar from '@mui/material/Avatar'
import FolderIcon from '@mui/icons-material/Folder'
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile'
import MoreVertIcon from '@mui/icons-material/MoreVert'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import Stage0StatusIcon from '../pdfs/Stage0StatusIcon'
import formatBytes from '../../utils/bytes'
import formatDate from '../../utils/dates'
import styles from './DropboxCardListStyles'

export default function DropboxFileItem({
  item,
  isSynced,
  selectedPaths,
  onSelectToggle,
  onItemClick,
  summaryMap
}) {
  return (
    <Box
      sx={{
        ...styles.itemRow,
        cursor: item.type === 'folder' || isSynced ? 'pointer' : 'not-allowed',
        opacity: item.type === 'file' && !isSynced ? 0.5 : 1
      }}
      onClick={() => onItemClick(item)}
    >
    <Checkbox
        checked={selectedPaths.includes(item.path_lower)}
        onChange={(e) => onSelectToggle(e, item)}
        onClick={(e) => e.stopPropagation()}  // Prevent bubbling up
        />
      <Box sx={styles.itemContent}>
        {item.type === 'folder' ? <FolderIcon color="warning" /> : <InsertDriveFileIcon color="action" />}
        <Typography variant="body2" ml={1}>{item.name}</Typography>
        {isSynced && <CheckCircleIcon color="success" fontSize="small" sx={{ ml: 1 }} />}
        <Stage0StatusIcon fileId={item.file_id || item.id} summaryMap={summaryMap} />
      </Box>
      <Typography sx={styles.fileSize}>{item.size != null ? formatBytes(item.size) : '—'}</Typography>
      <Typography sx={styles.fileType}>{item.type}</Typography>
      <Typography sx={styles.fileDate}>{formatDate(item.server_modified)}</Typography>
      <AvatarGroup max={3} sx={styles.avatarGroup}>
        {(item.shared_with ?? []).map((u, i) => (
          <Avatar key={i} src={u.avatar_url} alt={u.name} sx={styles.avatar} />
        ))}
      </AvatarGroup>
      <Box sx={styles.actionButton}>
        <IconButton size="large" onClick={e => e.stopPropagation()}>
          <MoreVertIcon />
        </IconButton>
      </Box>
    </Box>
  )
}
