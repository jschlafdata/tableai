// src/config/menuConfig.js
import FolderIcon from '@mui/icons-material/Folder';
import DescriptionIcon from '@mui/icons-material/Description';
import WorkIcon from '@mui/icons-material/Work';
import ExploreIcon from '@mui/icons-material/Explore';
import EmailIcon from '@mui/icons-material/Email';
import ChatIcon from '@mui/icons-material/Chat';
import CalendarMonthIcon from '@mui/icons-material/CalendarMonth';
import ViewKanbanIcon from '@mui/icons-material/ViewKanban';
import PermIdentityIcon from '@mui/icons-material/PermIdentity';
import LayersIcon from '@mui/icons-material/Layers';
import BlockIcon from '@mui/icons-material/Block';
import LabelIcon from '@mui/icons-material/Label';
import StoreIcon from '@mui/icons-material/Store';
import CloudIcon from '@mui/icons-material/Cloud';
import DriveFileRenameOutlineIcon from '@mui/icons-material/DriveFileRenameOutline';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import PeopleIcon from '@mui/icons-material/People';
import CodeIcon from '@mui/icons-material/Code';
import TableChartIcon from '@mui/icons-material/TableChart';
import DatabaseIcon from '@mui/icons-material/Storage';

// Define your menu structure with nested items
const menuConfig = {
  // Main menu items
  main: [
    { 
      name: 'Blog', 
      icon: <DescriptionIcon />, 
      path: '/blog'
    },
    { 
      name: 'Integrations', 
      icon: <FolderIcon />, 
      path: '/integrations',
      children: [
        {
          name: 'Document Stores',
          icon: <StoreIcon />,
          path: '/integrations/document-stores',
          children: [
            {
              name: 'Google Drive',
              icon: <DriveFileRenameOutlineIcon />,
              path: '/integrations/document-stores/gdrive'
            },
            {
              name: 'Dropbox',
              icon: <CloudIcon />,
              path: '/integrations/document-stores/dropbox'
            },
            {
              name: 'OneDrive',
              icon: <InsertDriveFileIcon />,
              path: '/integrations/document-stores/onedrive'
            }
          ]
        },
        {
          name: 'File Attachments',
          icon: <AttachFileIcon />,
          path: '/integrations/file-attachments'
        },
        {
          name: 'Data Sources',
          icon: <DatabaseIcon />,
          path: '/integrations/data-sources',
          children: [
            {
              name: 'SQL Databases',
              icon: <TableChartIcon />,
              path: '/integrations/data-sources/sql'
            },
            {
              name: 'APIs',
              icon: <CodeIcon />,
              path: '/integrations/data-sources/apis'
            }
          ]
        }
      ]
    },
    { 
      name: 'Teams', 
      icon: <PeopleIcon />, 
      path: '/teams',
      children: [
        {
          name: 'Management',
          icon: <WorkIcon />,
          path: '/teams/management'
        },
        {
          name: 'Members',
          icon: <PermIdentityIcon />,
          path: '/teams/members'
        }
      ]
    },
    { 
      name: 'Mail', 
      icon: <EmailIcon />, 
      path: '/mail',
      hasBadge: true,
      badgeContent: '+32',
      badgeColor: '#ff9966'
    },
    { 
      name: 'Calendar', 
      icon: <CalendarMonthIcon />, 
      path: '/calendar'
    }
  ],
  
  // Misc menu items
  misc: [
    { 
      name: 'Settings', 
      icon: <LayersIcon />, 
      path: '/settings',
      children: [
        {
          name: 'Permissions',
          icon: <PermIdentityIcon />,
          path: '/settings/permissions',
          description: 'Only admin can access.'
        },
        {
          name: 'Preferences',
          icon: <ChatIcon />,
          path: '/settings/preferences'
        }
      ]
    },
    { 
      name: 'Disabled', 
      icon: <BlockIcon />, 
      path: '/disabled',
      disabled: true
    },
    { 
      name: 'Label', 
      icon: <LabelIcon />, 
      path: '/label',
      hasBadge: true,
      badgeContent: 'NEW',
      badgeColor: '#88e2e2',
      badgeTextColor: '#333'
    }
  ]
};

export default menuConfig;