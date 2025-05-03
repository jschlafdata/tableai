// src/components/integrations/DropboxCardListStyles.js
import { alpha } from '@mui/material/styles';

const DropboxCardListStyles = {
  container: {
    width: '100%',
    maxWidth: {
      xs: '100%',
      sm: 600,
      md: 800,
      lg: 1000
    },
    mx: 'auto',
    mt: 3,
    px: { xs: 1, sm: 2 }
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    bgcolor: 'grey.100',
    px: { xs: 1.5, sm: 2 },
    py: 1.5,
    borderRadius: 2,
    mb: 2
  },
  headerContent: {
    display: 'flex',
    alignItems: 'center',
    flexGrow: 1
  },
  toggleButtonGroup: {
    ml: { xs: 0, sm: 1 }
  },
  itemStack: {
  },
  itemRow: {
    display: 'flex',
    flexDirection: { xs: 'column', sm: 'row' },
    alignItems: { xs: 'flex-start', sm: 'center' },
    bgcolor: 'background.paper',
    p: { xs: 1.5, sm: 2 },
    borderRadius: 2,
    boxShadow: 1,
    mb: 2,                             // ← add 16px (2*8px) of space below each row
    transition: 'box-shadow 0.2s ease-in-out, background-color 0.2s ease-in-out',
    '&:hover': {
      boxShadow: 3,
      bgcolor: theme => alpha(      // ← lighter tint of your primary color
        theme.palette.primary.main,
        0.1
      ),
    }
  },
  itemMainSection: {
    display: 'flex',
    alignItems: 'center',
    width: '100%',
    mb: { xs: 1, sm: 0 }
  },
  itemContent: {
    display: 'flex',
    alignItems: 'center',
    ml: 1,
    flexGrow: 1,
    overflow: 'hidden'
  },
  itemName: {
    variant: 'body2',
    ml: 1,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap'
  },
  metadataContainer: {
    display: 'flex',
    width: '100%',
    justifyContent: { xs: 'space-between', sm: 'flex-end' },
    alignItems: 'center',
    mt: { xs: 1, sm: 0 },
    ml: { xs: 0, sm: 2 }
  },
  fileSize: {
    variant: 'body2',
    color: 'textSecondary',
    width: { xs: 'auto', sm: 80 },
    textAlign: { xs: 'left', sm: 'right' },
    mr: { xs: 0, sm: 2 }
  },
  fileType: {
    variant: 'body2',
    color: 'textSecondary',
    textTransform: 'capitalize',
    width: { xs: 'auto', sm: 60 },
    mr: { xs: 0, sm: 2 }
  },
  fileDate: {
    variant: 'body2',
    color: 'textSecondary',
    width: { xs: 'auto', sm: 120 },
    mr: { xs: 0, sm: 2 },
    display: { xs: 'none', sm: 'block' }
  },
  avatarGroup: {
    width: { xs: 'auto', sm: 100 },
    justifyContent: 'flex-start',
    mr: { xs: 0, sm: 2 }
  },
  avatar: {
    width: 24,
    height: 24
  },
  actionButton: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center'
  },
  errorBox: {
    p: 2,
    bgcolor: 'error.light',
    borderRadius: 1
  },
  errorText: {
    color: 'error.dark'
  }
};

export default DropboxCardListStyles;