import React from 'react'
import Breadcrumbs from '@mui/material/Breadcrumbs'
import Link from '@mui/material/Link'

export default function DropboxBreadcrumbs({ currentPath, onCrumbClick }) {
  const crumbs = [
    { name: 'Home', path: '' },
    ...String(currentPath)
      .split('/')
      .filter(Boolean)
      .map((seg, i, all) => ({
        name: decodeURIComponent(seg),
        path: all.slice(0, i + 1).join('/')
      }))
  ]

  return (
    <Breadcrumbs sx={{ mb: 2 }}>
      {crumbs.map(crumb => (
        <Link
          key={crumb.path}
          underline="hover"
          color={crumb.path === currentPath ? 'text.primary' : 'inherit'}
          sx={{ cursor: 'pointer' }}
          onClick={() => onCrumbClick(crumb.path)}
        >
          {crumb.name}
        </Link>
      ))}
    </Breadcrumbs>
  )
}