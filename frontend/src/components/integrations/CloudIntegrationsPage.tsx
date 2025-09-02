// src/components/integrations/CloudIntegrationsPage.tsx
import { useState, useEffect } from "react";
import type { CSSProperties } from "react";
import { 
  CheckCircle, 
  AlertCircle, 
  Settings, 
  ExternalLink, 
  Trash2, 
  RefreshCw, 
  AlertTriangle, 
  Download, 
  Loader 
} from "lucide-react";
import PageLayout from "../layouts/PageLayout";
import UserDropdown from "../auth/UserDropdown";
import type { User } from "../../types/user";
import { setToken } from "../../api";

// Types
interface AccountInfo {
  email?: string;
  connected_at?: string;
}

interface Integration {
  connected: boolean;
  email: string | null;
  connectedAt: string | null;
  status: "connected" | "disconnected" | "needs_reauth";
  account_info: AccountInfo | null;
  needs_reauth: boolean;
  expires_at: string | null;
  last_sync: string | null;
  sync_status: "success" | "error" | null;
}

interface IntegrationConfig {
  name: string;
  icon: string;
  description: string;
  color: string;
  features: string[];
}

interface CloudIntegrationsPageProps {
  user: User | null;
}

type IntegrationKey = "dropbox" | "googleDrive";
type LoadingState = Record<string, boolean>;
type MessageState = { type: "success" | "error" | ""; text: string };

// Mock data - replace with actual imports when available
const DROPBOX_ICON = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 48'%3E%3Cpath fill='%230061FF' d='M12 8l12 8-12 8L0 16zm24 0l12 8-12 8-12-8zm-24 18l12 8 12-8 12 8-12 8-12-8z'/%3E%3C/svg%3E";
const GOOGLE_DRIVE_ICON = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 48 48'%3E%3Cpath fill='%234285F4' d='M16 0l16 28h16L32 0z'/%3E%3Cpath fill='%2334A853' d='M0 28l8 14h32l-8-14z'/%3E%3Cpath fill='%23FBBC04' d='M16 0L0 28l16 14V14z'/%3E%3C/svg%3E";

const CloudIntegrationsPage: React.FC<CloudIntegrationsPageProps> = ({ user }) => {
  // State
  const [integrations, setIntegrations] = useState<Record<IntegrationKey, Integration>>({
    dropbox: {
      connected: false,
      email: null,
      connectedAt: null,
      status: "disconnected",
      account_info: null,
      needs_reauth: false,
      expires_at: null,
      last_sync: null,
      sync_status: null
    },
    googleDrive: {
      connected: false,
      email: null,
      connectedAt: null,
      status: "disconnected",
      account_info: null,
      needs_reauth: false,
      expires_at: null,
      last_sync: null,
      sync_status: null
    }
  });

  const [loading, setLoading] = useState<LoadingState>({});
  const [message, setMessage] = useState<MessageState>({ type: "", text: "" });
  const [currentUserId, setCurrentUserId] = useState<number | null>(null);

  const integrationConfigs: Record<IntegrationKey, IntegrationConfig> = {
    dropbox: {
      name: "Dropbox",
      icon: DROPBOX_ICON,
      description: "Sync files with your Dropbox account",
      color: "#0061FF",
      features: ["File sync", "Backup storage", "Shared folders"]
    },
    googleDrive: {
      name: "Google Drive",
      icon: GOOGLE_DRIVE_ICON,
      description: "Connect to Google Drive for file management",
      color: "#4285F4",
      features: ["Document storage", "Real-time collaboration", "Google Workspace integration"]
    }
  };

  // Handlers (mock implementations)
  const handleSyncMetadata = async (integrationKey: IntegrationKey) => {
    setLoading(prev => ({ ...prev, [`${integrationKey}_sync`]: true }));
    setMessage({ type: "", text: "" });

    try {
      // Simulate sync operation
      await new Promise(resolve => setTimeout(resolve, 2000));

      setIntegrations(prev => ({
        ...prev,
        [integrationKey]: {
          ...prev[integrationKey],
          last_sync: new Date().toISOString(),
          sync_status: "success"
        }
      }));

      setMessage({
        type: "success",
        text: `Synced files from ${integrationConfigs[integrationKey].name}`
      });

      setTimeout(() => setMessage({ type: "", text: "" }), 5000);
    } catch (error) {
      setIntegrations(prev => ({
        ...prev,
        [integrationKey]: {
          ...prev[integrationKey],
          sync_status: "error"
        }
      }));

      setMessage({
        type: "error",
        text: `Sync failed for ${integrationConfigs[integrationKey].name}`
      });
    } finally {
      setLoading(prev => ({ ...prev, [`${integrationKey}_sync`]: false }));
    }
  };

  const loadIntegrationStatus = async () => {
    // Mock loading integration status
    console.log("Loading integration statuses...");
    
    // Simulate loading with mock data
    setTimeout(() => {
      // Example: Set Dropbox as connected
      setIntegrations(prev => ({
        ...prev,
        dropbox: {
          connected: true,
          email: "user@example.com",
          connectedAt: new Date().toISOString(),
          status: "connected",
          account_info: {
            email: "user@example.com",
            connected_at: new Date().toISOString()
          },
          needs_reauth: false,
          expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
          last_sync: new Date().toISOString(),
          sync_status: "success"
        }
      }));
    }, 1000);
  };

  const handleConnect = async (integrationKey: IntegrationKey) => {
    setLoading(prev => ({ ...prev, [integrationKey]: true }));
    setMessage({ type: "", text: "" });

    try {
      // Simulate OAuth flow
      console.log(`Starting ${integrationKey} authentication...`);
      
      // In real app, this would redirect to OAuth URL
      alert(`Would redirect to ${integrationConfigs[integrationKey].name} OAuth login`);
      
      // Simulate successful connection
      setTimeout(() => {
        setIntegrations(prev => ({
          ...prev,
          [integrationKey]: {
            connected: true,
            email: "user@example.com",
            connectedAt: new Date().toISOString(),
            status: "connected",
            account_info: {
              email: "user@example.com",
              connected_at: new Date().toISOString()
            },
            needs_reauth: false,
            expires_at: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString(),
            last_sync: null,
            sync_status: null
          }
        }));
        
        setMessage({
          type: "success",
          text: `Successfully connected to ${integrationConfigs[integrationKey].name}`
        });
      }, 2000);
    } catch (error) {
      console.error("Connection error:", error);
      setMessage({
        type: "error",
        text: `Failed to connect to ${integrationConfigs[integrationKey].name}`
      });
    } finally {
      setLoading(prev => ({ ...prev, [integrationKey]: false }));
    }
  };

  const handleReauth = async (integrationKey: IntegrationKey) => {
    setLoading(prev => ({ ...prev, [`${integrationKey}_reauth`]: true }));
    setMessage({ type: "", text: "" });

    try {
      console.log(`Re-authenticating ${integrationKey}...`);
      
      // Simulate re-auth
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setIntegrations(prev => ({
        ...prev,
        [integrationKey]: {
          ...prev[integrationKey],
          needs_reauth: false,
          status: "connected"
        }
      }));
      
      setMessage({
        type: "success",
        text: `Re-authenticated with ${integrationConfigs[integrationKey].name}`
      });
    } catch (error) {
      console.error("Re-auth error:", error);
      setMessage({
        type: "error",
        text: `Failed to re-authenticate with ${integrationConfigs[integrationKey].name}`
      });
    } finally {
      setLoading(prev => ({ ...prev, [`${integrationKey}_reauth`]: false }));
    }
  };

  const handleDisconnect = async (integrationKey: IntegrationKey) => {
    const config = integrationConfigs[integrationKey];
    if (!window.confirm(`Are you sure you want to disconnect from ${config.name}?`)) {
      return;
    }

    setLoading(prev => ({ ...prev, [integrationKey]: true }));

    try {
      // Simulate disconnection
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setIntegrations(prev => ({
        ...prev,
        [integrationKey]: {
          connected: false,
          email: null,
          connectedAt: null,
          status: "disconnected",
          account_info: null,
          needs_reauth: false,
          expires_at: null,
          last_sync: null,
          sync_status: null
        }
      }));

      setMessage({
        type: "success",
        text: `Successfully disconnected from ${config.name}`
      });
    } catch (error) {
      console.error("Disconnect error:", error);
      setMessage({
        type: "error",
        text: `Failed to disconnect from ${config.name}`
      });
    } finally {
      setLoading(prev => ({ ...prev, [integrationKey]: false }));
    }
  };

  useEffect(() => {
    if (user) {
      setCurrentUserId(user.id);
    }
    loadIntegrationStatus();
  }, [user]);

  const getStatusDisplay = (integration: Integration) => {
    if (integration.needs_reauth) {
      return {
        color: "#f59e0b",
        text: "Needs Re-authorization",
        icon: <AlertTriangle style={{ width: "10px", height: "10px" }} />
      };
    } else if (integration.connected) {
      return {
        color: "#16a34a",
        text: "Connected",
        icon: null
      };
    } else {
      return {
        color: "#6b7280",
        text: "Not connected",
        icon: null
      };
    }
  };

  const isTokenExpiringSoon = (expiresAt: string | null) => {
    if (!expiresAt) return false;
    const expirationTime = new Date(expiresAt);
    const now = new Date();
    const hoursUntilExpiry = (expirationTime.getTime() - now.getTime()) / (1000 * 60 * 60);
    return hoursUntilExpiry < 24;
  };

  const styles: Record<string, CSSProperties> = {
    container: {
      padding: "24px",
      backgroundColor: "#f9fafb",
      minHeight: "100vh"
    },
    messageBox: {
      padding: "12px 16px",
      borderRadius: "8px",
      marginBottom: "24px",
      display: "flex",
      alignItems: "center",
      gap: "8px"
    },
    gridContainer: {
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(450px, 1fr))",
      gap: "24px",
      maxWidth: "1200px"
    }
  };

  return (
    <PageLayout 
      title="Cloud Integrations"
      headerRight={
        <UserDropdown 
          user={user} 
          onLogout={() => {
            setToken(null);
            window.location.href = '/';
          }} 
        />
      }
    >
      <div style={styles.container}>
        {message.text && (
          <div style={{
            ...styles.messageBox,
            backgroundColor: message.type === "success" ? "#dcfce7" : "#fee2e2",
            border: `1px solid ${message.type === "success" ? "#bbf7d0" : "#fecaca"}`,
            color: message.type === "success" ? "#16a34a" : "#dc2626"
          }}>
            {message.type === "success" ? (
              <CheckCircle style={{ width: "16px", height: "16px" }} />
            ) : (
              <AlertCircle style={{ width: "16px", height: "16px" }} />
            )}
            {message.text}
          </div>
        )}

        <div style={styles.gridContainer}>
          {(Object.entries(integrationConfigs) as [IntegrationKey, IntegrationConfig][]).map(([key, config]) => (
            <IntegrationCard
              key={key}
              integrationKey={key}
              config={config}
              integration={integrations[key]}
              loading={loading}
              onConnect={handleConnect}
              onDisconnect={handleDisconnect}
              onReauth={handleReauth}
              onSync={handleSyncMetadata}
              getStatusDisplay={getStatusDisplay}
              isTokenExpiringSoon={isTokenExpiringSoon}
            />
          ))}
        </div>
      </div>
    </PageLayout>
  );
};

// Integration Card Component
interface IntegrationCardProps {
  integrationKey: IntegrationKey;
  config: IntegrationConfig;
  integration: Integration;
  loading: LoadingState;
  onConnect: (key: IntegrationKey) => void;
  onDisconnect: (key: IntegrationKey) => void;
  onReauth: (key: IntegrationKey) => void;
  onSync: (key: IntegrationKey) => void;
  getStatusDisplay: (integration: Integration) => any;
  isTokenExpiringSoon: (expiresAt: string | null) => boolean;
}

const IntegrationCard: React.FC<IntegrationCardProps> = ({
  integrationKey,
  config,
  integration,
  loading,
  onConnect,
  onDisconnect,
  onReauth,
  onSync,
  getStatusDisplay,
  isTokenExpiringSoon
}) => {
  const statusDisplay = getStatusDisplay(integration);
  const tokenExpiringSoon = isTokenExpiringSoon(integration.expires_at);

  const styles: Record<string, CSSProperties> = {
    card: {
      backgroundColor: "white",
      borderRadius: "12px",
      padding: "24px",
      boxShadow: "0 1px 3px 0 rgba(0, 0, 0, 0.1)",
      border: integration.connected && !integration.needs_reauth
        ? `2px solid ${config.color}20`
        : integration.needs_reauth
        ? "2px solid #fbbf2420"
        : "1px solid #e5e7eb"
    },
    header: {
      display: "flex",
      alignItems: "center",
      marginBottom: "12px"
    },
    icon: {
      width: "48px",
      height: "48px",
      marginRight: "16px",
      objectFit: "contain" as const
    },
    title: {
      fontSize: "1.5rem",
      fontWeight: "bold",
      color: "#111827",
      margin: 0
    },
    statusContainer: {
      display: "flex",
      alignItems: "center",
      marginTop: "8px"
    },
    statusDot: {
      width: "10px",
      height: "10px",
      borderRadius: "50%",
      backgroundColor: statusDisplay.color,
      marginRight: "8px"
    },
    statusText: {
      fontSize: "16px",
      color: statusDisplay.color,
      fontWeight: "500",
      display: "flex",
      alignItems: "center",
      gap: "4px"
    },
    description: {
      color: "#6b7280",
      fontSize: "16px",
      marginBottom: "20px",
      lineHeight: 1.6
    },
    warningBox: {
      backgroundColor: "#fef3c7",
      border: "1px solid #fbbf24",
      padding: "12px 16px",
      borderRadius: "8px",
      marginBottom: "16px",
      display: "flex",
      alignItems: "center",
      gap: "8px"
    },
    infoBox: {
      backgroundColor: "#f9fafb",
      padding: "16px",
      borderRadius: "8px",
      marginBottom: "20px"
    },
    featuresList: {
      margin: 0,
      paddingLeft: "20px",
      fontSize: "14px",
      color: "#6b7280",
      lineHeight: 1.6
    },
    buttonContainer: {
      display: "flex",
      gap: "12px",
      flexWrap: "wrap" as const
    },
    primaryButton: {
      padding: "12px 24px",
      fontSize: "16px",
      fontWeight: "600",
      border: "none",
      color: "white",
      borderRadius: "8px",
      cursor: "pointer",
      display: "flex",
      alignItems: "center",
      gap: "8px",
      transition: "background-color 0.2s"
    },
    secondaryButton: {
      padding: "10px 20px",
      fontSize: "14px",
      borderRadius: "8px",
      cursor: "pointer",
      display: "flex",
      alignItems: "center",
      gap: "8px",
      fontWeight: "500",
      transition: "background-color 0.2s"
    }
  };

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <img src={config.icon} alt={`${config.name} logo`} style={styles.icon} />
        <div>
          <h3 style={styles.title}>{config.name}</h3>
          <div style={styles.statusContainer}>
            <div style={styles.statusDot} />
            <span style={styles.statusText}>
              {statusDisplay.icon}
              {statusDisplay.text}
            </span>
          </div>
        </div>
      </div>

      <p style={styles.description}>{config.description}</p>

      {integration.needs_reauth && (
        <div style={styles.warningBox}>
          <AlertTriangle style={{ width: "16px", height: "16px", color: "#f59e0b" }} />
          <span style={{ fontSize: "14px", color: "#92400e" }}>
            Your connection has expired. Please re-authorize to continue using this integration.
          </span>
        </div>
      )}

      {integration.connected && !integration.needs_reauth && (
        <div style={styles.infoBox}>
          <div style={{ fontSize: "14px", color: "#6b7280", marginBottom: "6px" }}>
            Connected Account
          </div>
          <div style={{ fontSize: "16px", fontWeight: "500", color: "#111827" }}>
            {integration.email}
          </div>
          {integration.connectedAt && (
            <div style={{ fontSize: "14px", color: "#6b7280", marginTop: "6px" }}>
              Connected on {new Date(integration.connectedAt).toLocaleDateString()}
            </div>
          )}
          {integration.last_sync && (
            <div style={{ 
              fontSize: "12px", 
              color: "#6b7280", 
              marginTop: "4px",
              display: "flex",
              alignItems: "center",
              gap: "4px"
            }}>
              Last sync: {new Date(integration.last_sync).toLocaleString()}
              {integration.sync_status === "success" && <CheckCircle style={{ width: "12px", height: "12px", color: "#16a34a" }} />}
              {integration.sync_status === "error" && <AlertCircle style={{ width: "12px", height: "12px", color: "#dc2626" }} />}
            </div>
          )}
        </div>
      )}

      <div style={{ marginBottom: "24px" }}>
        <div style={{
          fontSize: "14px",
          fontWeight: "600",
          color: "#374151",
          marginBottom: "10px"
        }}>
          Features:
        </div>
        <ul style={styles.featuresList}>
          {config.features.map((feature, index) => (
            <li key={index} style={{ marginBottom: "4px" }}>{feature}</li>
          ))}
        </ul>
      </div>

      <div style={styles.buttonContainer}>
        {integration.needs_reauth ? (
          <>
            <button
              onClick={() => onReauth(integrationKey)}
              disabled={loading[`${integrationKey}_reauth`]}
              style={{
                ...styles.primaryButton,
                backgroundColor: loading[`${integrationKey}_reauth`] ? "#9ca3af" : "#f59e0b",
                opacity: loading[`${integrationKey}_reauth`] ? 0.5 : 1,
                cursor: loading[`${integrationKey}_reauth`] ? "not-allowed" : "pointer"
              }}
            >
              <RefreshCw style={{ width: "18px", height: "18px" }} />
              {loading[`${integrationKey}_reauth`] ? "Re-authorizing..." : "Re-authorize"}
            </button>
            <button
              onClick={() => onDisconnect(integrationKey)}
              disabled={loading[integrationKey]}
              style={{
                ...styles.secondaryButton,
                border: "1px solid #dc2626",
                color: "#dc2626",
                backgroundColor: "transparent",
                opacity: loading[integrationKey] ? 0.5 : 1,
                cursor: loading[integrationKey] ? "not-allowed" : "pointer"
              }}
            >
              <Trash2 style={{ width: "16px", height: "16px" }} />
              {loading[integrationKey] ? "Disconnecting..." : "Disconnect"}
            </button>
          </>
        ) : integration.connected ? (
          <>
            <button
              onClick={() => onSync(integrationKey)}
              disabled={loading[`${integrationKey}_sync`]}
              style={{
                ...styles.primaryButton,
                backgroundColor: loading[`${integrationKey}_sync`] ? "#9ca3af" : config.color,
                opacity: loading[`${integrationKey}_sync`] ? 0.5 : 1,
                cursor: loading[`${integrationKey}_sync`] ? "not-allowed" : "pointer"
              }}
            >
              {loading[`${integrationKey}_sync`] ? (
                <>
                  <Loader style={{ width: "18px", height: "18px", animation: "spin 1s linear infinite" }} />
                  Syncing...
                </>
              ) : (
                <>
                  <Download style={{ width: "18px", height: "18px" }} />
                  Sync Metadata
                </>
              )}
            </button>
            <button
              onClick={() => onDisconnect(integrationKey)}
              disabled={loading[integrationKey]}
              style={{
                ...styles.secondaryButton,
                border: "1px solid #dc2626",
                color: "#dc2626",
                backgroundColor: "transparent",
                opacity: loading[integrationKey] ? 0.5 : 1,
                cursor: loading[integrationKey] ? "not-allowed" : "pointer"
              }}
            >
              <Trash2 style={{ width: "16px", height: "16px" }} />
              {loading[integrationKey] ? "Disconnecting..." : "Disconnect"}
            </button>
            <button
              style={{
                ...styles.secondaryButton,
                border: "1px solid #6b7280",
                color: "#6b7280",
                backgroundColor: "transparent"
              }}
            >
              <Settings style={{ width: "16px", height: "16px" }} />
              Settings
            </button>
          </>
        ) : (
          <button
            onClick={() => onConnect(integrationKey)}
            disabled={loading[integrationKey]}
            style={{
              ...styles.primaryButton,
              backgroundColor: loading[integrationKey] ? "#9ca3af" : config.color,
              opacity: loading[integrationKey] ? 0.5 : 1,
              cursor: loading[integrationKey] ? "not-allowed" : "pointer"
            }}
          >
            <ExternalLink style={{ width: "18px", height: "18px" }} />
            {loading[integrationKey] ? "Connecting..." : "Connect"}
          </button>
        )}
      </div>
    </div>
  );
};

export default CloudIntegrationsPage;