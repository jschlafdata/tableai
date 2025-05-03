// src/components/Layout.js
import React, { useState } from 'react';
import { Link, useLocation, Outlet } from 'react-router-dom';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Badge from '@mui/material/Badge';
import {
  Sidebar,
  Menu,
  MenuItem,
  SubMenu,
  sidebarClasses
} from 'react-pro-sidebar';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import ChevronRightIcon from '@mui/icons-material/ChevronRight';
import KeyboardArrowDownIcon from '@mui/icons-material/KeyboardArrowDown';

const SIDEBAR_WIDTH = 280;
const COLLAPSED_WIDTH = 95;

const NestedMenuItem = ({ item, collapsed, location, level = 0 }) => {
  const active = item.path === location.pathname;
  const iconSized = React.cloneElement(item.icon, {
    sx: { fontSize: 30, color: active ? '#312E31' : '#767676' }
  });

  if (item.children?.length) {
    return (
      <SubMenu
        label={!collapsed && item.name}
        icon={iconSized}
        routerLink={<Link to={item.path} />}
        active={active}
        defaultOpen={active}
        renderExpandIcon={({ open }) =>
          open ? <KeyboardArrowDownIcon /> : <ChevronRightIcon />
        }
        rootStyles={{
          [`.${sidebarClasses.subMenuContent}`]: { backgroundColor: 'transparent' }
        }}
        level={level}
      >
        {item.children.map((child, i) => (
          <NestedMenuItem
            key={i}
            item={child}
            collapsed={collapsed}
            location={location}
            level={level + 1}
          />
        ))}
      </SubMenu>
    );
  }

  return (
    <MenuItem
      icon={iconSized}
      routerLink={<Link to={item.path} />}
      active={active}
      rootStyles={{ paddingLeft: level * 16 }}
    >
      {!collapsed && item.name}
      {!collapsed && item.hasBadge && (
        <Badge
          badgeContent={item.badgeContent}
          sx={{
            ml: 1,
            '& .MuiBadge-badge': {
              backgroundColor: item.badgeColor || '#88e2e2',
              color: item.badgeTextColor || '#333',
              fontSize: 10,
              height: 20,
              minWidth: 20,
            }
          }}
        />
      )}
    </MenuItem>
  );
};

const MenuSection = ({ title, items, collapsed, location }) => (
  <>
    {title && !collapsed && (
      <Box sx={{ px: 2, pt: 2 }}>
        <Typography variant="caption" color="textSecondary">
          {title}
        </Typography>
      </Box>
    )}
    {items.map((it, i) => (
      <NestedMenuItem
        key={i}
        item={it}
        collapsed={collapsed}
        location={location}
      />
    ))}
  </>
);

export default function Layout({ menuConfig }) {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);
  const { mainTitle = null, main = [], misc = [] } = menuConfig || {};

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      {/* Sidebar Container */}
      <Box
  onClick={() => collapsed && setCollapsed(false)}
  sx={{
    width: collapsed ? COLLAPSED_WIDTH : SIDEBAR_WIDTH,
    transition: 'width 0.3s ease',
    bgcolor: '#ffffff'
  }}
>
        {/* Header */}
        <Box
  sx={{
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    px: 2,
    height: 64,
    minWidth: COLLAPSED_WIDTH, // ensure it never shrinks smaller than the icon area
    bgcolor: 'rgba(255,255,255,0.22)',
    color: '#660dfc',
  }}
>
  {/* Logo always visible */}
  <Box sx={{ display: 'flex', alignItems: 'center' }}>
    <Box
      component="img"
      src="/1x/table_ai_icon.png"
      alt="Logo"
      sx={{ height: 32, width: 'auto', mr: collapsed ? 0 : 1 }}
    />
    {!collapsed && (
      <Typography variant="h6" sx={{ fontWeight: 600 }}>
        TableAI
      </Typography>
    )}
  </Box>

  {/* Arrow always visible */}
  <Box
    onClick={e => {
      e.stopPropagation();
      setCollapsed(c => !c);
    }}
    sx={{
      width: 32,
      height: 32,
      borderRadius: '50%',
      border: '1px solid rgba(255,255,255,0.2)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      cursor: 'pointer',
      backgroundColor: '#fff',
      ml: 1,
      mr: 3
    }}
  >
    {collapsed ? <ChevronRightIcon /> : <ArrowBackIcon />}
  </Box>
</Box>
        {/* Sidebar Menu */}
        <Sidebar
          width={SIDEBAR_WIDTH}
          collapsedWidth={COLLAPSED_WIDTH}
          collapsed={collapsed}
          rootStyles={{
            [`.${sidebarClasses.container}`]: {
              height: 'calc(100% - 64px)',
              boxShadow: 'rgba(147, 93, 249, 0.48) 2px 2px 2px',
              borderRadius: 8
            }
          }}
        >
            <Menu
            menuItemStyles={{
                root: {
                px: 2,
                py: 1
                },
                button: ({ active }) => ({
                borderRadius: 8, // 👈 rounded corners
                padding: '6px 12px', // 👈 vertical/horizontal padding
                margin: '8px 8px', // 👈 spacing between items
                backgroundColor: active ? 'rgba(134, 81, 214, 0)' : 'transparent',
                '&:hover': {
                    backgroundColor: 'rgba(134,81,214,0.09)'
                },
                transition: 'background-color 0.2s ease'
                })
            }}
            >
            <MenuSection
              title={mainTitle}
              items={main}
              collapsed={collapsed}
              location={location}
            />
            <MenuSection
              title="MISC"
              items={misc}
              collapsed={collapsed}
              location={location}
            />
          </Menu>
        </Sidebar>
      </Box>

      {/* Main Content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          bgcolor: '#fffff',
          overflowY: 'auto',
          borderRadius: '5%',
          border: '1px solid rgba(255, 255, 255, 0.04)',
        }}
      >
        <Outlet />
      </Box>
    </Box>
  );
}
