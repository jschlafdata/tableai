// src/components/auth/UserDropdown.tsx
import React, { useState, useEffect, useRef } from "react";
import { LogOut, User as UserIcon, Settings, Shield } from "lucide-react";
import { useNavigate } from "react-router-dom";
import type { CSSProperties } from "react";
import type { User } from "../../types/user";
import { setToken } from "../../api";

interface UserDropdownProps {
  user: User | null;
  style?: "dropdown" | "simple";
  showUserInfo?: boolean;
  className?: string;
  onLogout?: () => void;
}

const UserDropdown: React.FC<UserDropdownProps> = ({
  user,
  style = "dropdown",
  showUserInfo = true,
  className = "",
  onLogout,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isLoggingOut, setIsLoggingOut] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener("click", handleClickOutside);
    }

    return () => {
      document.removeEventListener("click", handleClickOutside);
    };
  }, [isOpen]);

  const handleLogout = async () => {
    setIsLoggingOut(true);
    setToken(null); // Clear the token
    onLogout?.(); // Call parent's logout handler if provided
    navigate("/"); // Navigate to home/login
  };

  const styles: Record<string, CSSProperties> = {
    container: {
      position: "relative",
      display: "inline-block",
    },
    button: {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      padding: "8px 12px",
      backgroundColor: "transparent",
      border: "none",
      borderRadius: "8px",
      cursor: "pointer",
      transition: "background-color 0.2s ease",
    },
    simpleButton: {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      padding: "8px 16px",
      backgroundColor: "transparent",
      color: "#6b7280",
      border: "1px solid #d1d5db",
      borderRadius: "8px",
      cursor: "pointer",
      fontSize: "14px",
      transition: "all 0.2s ease",
    },
    avatar: {
      width: "32px",
      height: "32px",
      borderRadius: "50%",
      backgroundColor: "#2563eb",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      color: "white",
      fontSize: "14px",
      fontWeight: "bold",
    },
    userInfo: {
      textAlign: "left" as const,
    },
    userName: {
      fontSize: "14px",
      fontWeight: 500,
      color: "#111827",
      lineHeight: 1.2,
    },
    userEmail: {
      fontSize: "12px",
      color: "#6b7280",
      lineHeight: 1.2,
    },
    arrow: {
      marginLeft: "4px",
      transform: isOpen ? "rotate(180deg)" : "rotate(0deg)",
      transition: "transform 0.2s ease",
      fontSize: "12px",
    },
    dropdown: {
      position: "absolute" as const,
      top: "100%",
      right: "0",
      marginTop: "4px",
      backgroundColor: "white",
      borderRadius: "8px",
      boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
      border: "1px solid #e5e7eb",
      minWidth: "200px",
      zIndex: 1000,
      overflow: "hidden",
    },
    dropdownHeader: {
      padding: "12px 16px",
      borderBottom: "1px solid #e5e7eb",
      backgroundColor: "#f9fafb",
    },
    dropdownMenu: {
      padding: "8px 0",
    },
    menuItem: {
      width: "100%",
      display: "flex",
      alignItems: "center",
      gap: "12px",
      padding: "8px 16px",
      backgroundColor: "transparent",
      border: "none",
      cursor: "pointer",
      fontSize: "14px",
      color: "#374151",
      textAlign: "left" as const,
      transition: "background-color 0.2s",
    },
    menuItemDanger: {
      color: "#dc2626",
    },
    divider: {
      margin: "8px 0",
      border: "none",
      borderTop: "1px solid #e5e7eb",
    },
  };

  if (!user) {
    return null;
  }

  const getUserInitial = () => {
    return user.full_name?.charAt(0).toUpperCase() || user.email.charAt(0).toUpperCase();
  };

  if (style === "simple") {
    return (
      <button
        onClick={handleLogout}
        disabled={isLoggingOut}
        className={className}
        style={{
          ...styles.simpleButton,
          opacity: isLoggingOut ? 0.5 : 1,
          cursor: isLoggingOut ? "not-allowed" : "pointer",
        }}
        onMouseEnter={(e) => {
          if (!isLoggingOut) {
            e.currentTarget.style.backgroundColor = "#fee2e2";
            e.currentTarget.style.borderColor = "#fecaca";
            e.currentTarget.style.color = "#dc2626";
          }
        }}
        onMouseLeave={(e) => {
          if (!isLoggingOut) {
            e.currentTarget.style.backgroundColor = "transparent";
            e.currentTarget.style.borderColor = "#d1d5db";
            e.currentTarget.style.color = "#6b7280";
          }
        }}
      >
        <LogOut style={{ width: "16px", height: "16px" }} />
        {isLoggingOut ? "Signing out..." : "Sign out"}
      </button>
    );
  }

  return (
    <div ref={dropdownRef} style={styles.container} className="user-dropdown">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={className}
        style={styles.button}
        onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#f3f4f6")}
        onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
      >
        <div style={styles.avatar}>{getUserInitial()}</div>
        
        {showUserInfo && (
          <div style={styles.userInfo}>
            <div style={styles.userName}>{user.full_name || "User"}</div>
            <div style={styles.userEmail}>{user.email}</div>
          </div>
        )}
        
        <div style={styles.arrow}>▼</div>
      </button>

      {isOpen && (
        <div style={styles.dropdown}>
          <div style={styles.dropdownHeader}>
            <div style={{ ...styles.userName, marginBottom: "2px" }}>
              {user.full_name || "User"}
            </div>
            <div style={styles.userEmail}>{user.email}</div>
          </div>

          <div style={styles.dropdownMenu}>
            <button
              style={styles.menuItem}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#f3f4f6")}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
              onClick={() => {
                setIsOpen(false);
                navigate("/profile/info");
              }}
            >
              <UserIcon style={{ width: "16px", height: "16px" }} />
              View Profile
            </button>
            
            <button
              style={styles.menuItem}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#f3f4f6")}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
              onClick={() => {
                setIsOpen(false);
                navigate("/profile/settings");
              }}
            >
              <Settings style={{ width: "16px", height: "16px" }} />
              Settings
            </button>
            
            <button
              style={styles.menuItem}
              onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = "#f3f4f6")}
              onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = "transparent")}
              onClick={() => {
                setIsOpen(false);
                navigate("/profile/security");
              }}
            >
              <Shield style={{ width: "16px", height: "16px" }} />
              Security
            </button>

            <hr style={styles.divider} />

            <button
              onClick={handleLogout}
              disabled={isLoggingOut}
              style={{
                ...styles.menuItem,
                ...styles.menuItemDanger,
                opacity: isLoggingOut ? 0.5 : 1,
                cursor: isLoggingOut ? "not-allowed" : "pointer",
              }}
              onMouseEnter={(e) => {
                if (!isLoggingOut) {
                  e.currentTarget.style.backgroundColor = "#fee2e2";
                }
              }}
              onMouseLeave={(e) => {
                if (!isLoggingOut) {
                  e.currentTarget.style.backgroundColor = "transparent";
                }
              }}
            >
              <LogOut style={{ width: "16px", height: "16px" }} />
              {isLoggingOut ? "Signing out..." : "Sign out"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default UserDropdown;