// src/components/layouts/PageLayout.tsx - Simplified version
import { useState } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import type { CSSProperties, ReactNode } from "react";
import type { User } from "../../types/user";

interface MenuItem {
  name: string;
  path: string;
  icon?: string;
  children?: MenuItem[];
}

interface MenuConfig {
  mainTitle?: string;
  main: MenuItem[];
}

interface PageLayoutProps {
  children: ReactNode;
  title?: string;
  showSidebar?: boolean;
  menuConfig?: MenuConfig;
  headerRight?: ReactNode; // Pass the UserDropdown as a prop instead
  user?: User | null; // Add user prop for compatibility
}

const PageLayout: React.FC<PageLayoutProps> = ({
  children,
  title = "Dashboard",
  showSidebar = true,
  menuConfig,
  headerRight,
  user, // Accept user prop but it's optional
}) => {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const currentPath = location.pathname;

  const defaultMenuConfig: MenuConfig = {
    mainTitle: "Main Menu",
    main: [
      { name: "Dashboard", path: "/", icon: "🏠" },
      { name: "Document Processing", path: "/processing", icon: "📄" },
      { name: "Cloud Storage", path: "/cloud_storage/documents", icon: "☁️" },
      {
        name: "Document Viewer",
        path: "/processing/pdf_viewer",
        icon: "📄",
        children: [
          { name: "Classifications", path: "/processing/classifications", icon: "🔗" },
        ],
      },
      { name: "S3 PDF Viewer", path: "/processing/pdf/s3_viewer", icon: "📄" },
      {
        name: "Profile",
        path: "/profile",
        icon: "👤",
        children: [
          { name: "Integrations", path: "/profile/integrations", icon: "🔗" },
        ],
      },
    ],
  };

  const activeMenuConfig = menuConfig || defaultMenuConfig;

  const styles: Record<string, CSSProperties> = {
    container: {
      display: "flex",
      height: "100vh",
      fontFamily: "system-ui, -apple-system, sans-serif",
    },
    sidebar: {
      width: collapsed ? "60px" : "250px",
      backgroundColor: "#ffffff",
      borderRight: "1px solid #e5e7eb",
      transition: "width 0.3s ease",
      overflow: "hidden",
      display: showSidebar ? "block" : "none",
    },
    header: {
      padding: "16px",
      borderBottom: "1px solid #e5e7eb",
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
    },
    logo: {
      display: "flex",
      alignItems: "center",
      gap: "12px",
    },
    logoIcon: {
      width: "32px",
      height: "32px",
      borderRadius: "8px",
      backgroundColor: "#2563eb",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      color: "white",
      fontSize: "18px",
      fontWeight: "bold",
    },
    logoText: {
      fontSize: "18px",
      fontWeight: 600,
      color: "#111827",
      display: collapsed ? "none" : "block",
    },
    toggleButton: {
      padding: "8px",
      border: "none",
      backgroundColor: "transparent",
      cursor: "pointer",
      borderRadius: "4px",
      color: "#6b7280",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
    },
    menu: {
      padding: "16px 0",
    },
    menuSection: {
      marginBottom: "24px",
    },
    sectionTitle: {
      padding: "0 16px 8px 16px",
      fontSize: "12px",
      fontWeight: 600,
      color: "#6b7280",
      textTransform: "uppercase",
      letterSpacing: "0.05em",
      display: collapsed ? "none" : "block",
    },
    menuItem: {
      display: "block",
      padding: "8px 16px",
      textDecoration: "none",
      color: "#374151",
      fontSize: "14px",
      borderRadius: "0px",
      margin: "0 8px",
      transition: "all 0.2s",
      position: "relative",
      cursor: "pointer",
    },
    menuItemActive: {
      backgroundColor: "#eff6ff",
      color: "#2563eb",
      fontWeight: 500,
    },
    menuItemIcon: {
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      width: "20px",
      height: "20px",
      marginRight: collapsed ? "0" : "12px",
      fontSize: "16px",
    },
    menuItemText: {
      display: collapsed ? "none" : "inline",
    },
    subMenu: {
      paddingLeft: "32px",
    },
    content: {
      flex: 1,
      backgroundColor: "#f9fafb",
      overflow: "auto",
      display: "flex",
      flexDirection: "column",
    },
    contentHeader: {
      backgroundColor: "white",
      borderBottom: "1px solid #e5e7eb",
      padding: "16px 24px",
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      minHeight: "60px",
    },
    contentTitle: {
      fontSize: "1.5rem",
      fontWeight: 600,
      color: "#111827",
      margin: 0,
    },
    contentBody: {
      flex: 1,
      padding: "24px",
      overflow: "auto",
    },
  };

  const renderMenuItem = (item: MenuItem, isSubItem = false) => {
    const isActive = currentPath === item.path;
    const [hover, setHover] = useState(false);

    const menuItemStyle: CSSProperties = {
      ...styles.menuItem,
      ...(isActive ? styles.menuItemActive : {}),
      ...(isSubItem ? styles.subMenu : {}),
      backgroundColor: hover && !isActive ? "#f3f4f6" : (isActive ? "#eff6ff" : "transparent"),
    };

    return (
      <div key={item.path}>
        <div
          style={menuItemStyle}
          onClick={() => navigate(item.path)}
          onMouseEnter={() => setHover(true)}
          onMouseLeave={() => setHover(false)}
        >
          <span style={styles.menuItemIcon}>{item.icon || "•"}</span>
          <span style={styles.menuItemText}>{item.name}</span>
        </div>
        {item.children && !collapsed && (
          <div>{item.children.map((child) => renderMenuItem(child, true))}</div>
        )}
      </div>
    );
  };

  return (
    <div style={styles.container}>
      {showSidebar && (
        <div style={styles.sidebar}>
          <div style={styles.header}>
            <div style={styles.logo}>
              <div style={styles.logoIcon}>T</div>
              <span style={styles.logoText}>TableAI</span>
            </div>
            <button
              style={styles.toggleButton}
              onClick={() => setCollapsed(!collapsed)}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#f3f4f6")}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
            >
              {collapsed ? "→" : "←"}
            </button>
          </div>

          <div style={styles.menu}>
            <div style={styles.menuSection}>
              <div style={styles.sectionTitle}>
                {activeMenuConfig?.mainTitle || "Menu"}
              </div>
              {activeMenuConfig.main?.map((item) => renderMenuItem(item))}
            </div>
          </div>
        </div>
      )}

      <div style={styles.content}>
        <div style={styles.contentHeader}>
          <h1 style={styles.contentTitle}>{title}</h1>
          {headerRight}
        </div>
        <div style={styles.contentBody}>{children}</div>
      </div>
    </div>
  );
};

export default PageLayout;