import React, { useState } from 'react';
import { Link, useLocation, Outlet } from 'react-router-dom'; // ✅ Add these
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
import sidebarStyles from '../config/sidebarStyles';

const NestedMenuItem = ({ item, collapsed, location, level = 0, style }) => {
  const active = item.path === location.pathname;
  const iconSized = React.cloneElement(item.icon, {
    sx: style.icon({ active })
  });

  if (item.children?.length) {
    return (
      <SubMenu
        label={!collapsed && item.name}
        icon={iconSized}
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
            style={style}
          />
        ))}
      </SubMenu>
    );
  }

  return (
    <MenuItem icon={iconSized} active={active} rootStyles={{ paddingLeft: level * 16 }}>
      <Link to={item.path} style={{ textDecoration: 'none', color: 'inherit', display: 'flex', alignItems: 'center', width: '100%' }}>
        {!collapsed && item.name}
        {!collapsed && item.hasBadge && (
          <Badge badgeContent={item.badgeContent} sx={style.badge(item)} />
        )}
      </Link>
    </MenuItem>
  );
};

const MenuSection = ({ title, items, collapsed, location, style }) => (
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
        style={style}
      />
    ))}
  </>
);

export default function Layout({ menuConfig }) {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(false);

  const theme = menuConfig?.theme || 'default';
  const style = sidebarStyles[theme] || sidebarStyles.default;
  const sections = Object.entries(menuConfig || {});

  return (
    <Box sx={{ display: 'flex', height: '100vh' }}>
      {/* Sidebar Container */}
      <Box
        onClick={() => collapsed && setCollapsed(false)}
        sx={{
          width: collapsed ? style.sidebar.collapsedWidth : style.sidebar.width,
          transition: 'width 0.3s ease',
          bgcolor: '#ffffff'
        }}
      >
        {/* Header */}
        <Box sx={style.content.header}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Box
              component="img"
              src="/1x/table_ai_icon.png"
              alt="Logo"
              sx={style.logo(collapsed)}
            />
            {!collapsed && (
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                TableAI
              </Typography>
            )}
          </Box>
          <Box
            onClick={e => {
              e.stopPropagation();
              setCollapsed(c => !c);
            }}
            sx={style.toggleButton}
          >
            {collapsed ? <ChevronRightIcon /> : <ArrowBackIcon />}
          </Box>
        </Box>

        {/* Sidebar Menu */}
        <Sidebar
          width={style.sidebar.width}
          collapsedWidth={style.sidebar.collapsedWidth}
          collapsed={collapsed}
          rootStyles={{
            [`.${sidebarClasses.container}`]: style.sidebar.container
          }}
        >
          <Menu menuItemStyles={style.menuItem}>
            {sections.map(([key, value]) => {
              if (!Array.isArray(value)) return null;
              const title = key === 'main' ? menuConfig.mainTitle : key.toUpperCase();
              return (
                <MenuSection
                  key={key}
                  title={title}
                  items={value}
                  collapsed={collapsed}
                  location={location}
                  style={style}
                />
              );
            })}
          </Menu>
        </Sidebar>
      </Box>

      {/* Main Content */}
      <Box component="main" sx={style.content.main}>
        <Outlet />
      </Box>
    </Box>
  );
}
