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
import LocalIcon from './ui/LocalIcon'

const NestedMenuItem = ({ item, collapsed, location, level = 0, style }) => {
  const active = item.path === location.pathname;
  
  // Handle different icon types based on their properties
  let iconElement;
  
  if (React.isValidElement(item.icon)) {
    // Check if it's our LocalIcon component or has a src property (img tag)
    if (item.icon.type === LocalIcon || (item.icon.props && item.icon.props.src)) {
      // For our custom LocalIcon, pass the active prop
      iconElement = React.cloneElement(item.icon, {
        active,
        style
      });
    } else {
      // For Material-UI icons
      iconElement = React.cloneElement(item.icon, {
        sx: style.icon({ active })
      });
    }
  } else {
    // Fallback for no icon
    iconElement = null;
  }

  if (item.children?.length) {
    return (
      <SubMenu
        label={!collapsed && (
          <Typography sx={{
            ...style.menuItem.text,
            ...(level > 0 && style.menuItem.subMenuText)
          }}>
            {item.name}
          </Typography>
        )}
        icon={iconElement}
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
    <MenuItem icon={iconElement} active={active} rootStyles={{ paddingLeft: level * 16 }}>
      <Link 
        to={item.path} 
        style={{ 
          textDecoration: 'none', 
          color: 'inherit', 
          display: 'flex', 
          alignItems: 'center', 
          width: '100%' 
        }}
      >
        {!collapsed && (
          <>
            <Typography sx={{
              ...style.menuItem.text,
              ...(active ? style.menuItem.activeText : {}),
              ...(level > 0 && style.menuItem.subMenuText)
            }}>
              {item.name}
            </Typography>
            
            {item.hasBadge && (
              <Badge badgeContent={item.badgeContent} sx={style.badge(item)} />
            )}
          </>
        )}
      </Link>
    </MenuItem>
  );
};

const MenuSection = ({ title, items, collapsed, location, style }) => (
  <>
    {title && !collapsed && (
      <Box sx={{ ...style.sectionTitle }}>
        <Typography variant="caption" sx={{ 
          fontFamily: style.sectionTitle.fontFamily,
          fontSize: style.sectionTitle.fontSize,
          fontWeight: style.sectionTitle.fontWeight,
          textTransform: style.sectionTitle.textTransform,
          letterSpacing: style.sectionTitle.letterSpacing,
          color: style.sectionTitle.color
        }}>
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