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
import axios from "axios";
import PageLayout from "../layouts/PageLayout";
import UserDropdown from "../auth/UserDropdown";
import type { User } from "../../types/user";
import { setToken } from "../../api";
import { 
  getTestAttributes, 
  generateTestId,
  type TestAction,
  type StatusType
} from "../../utils/testUtils";

// Types
interface OAuthStatus {
  connected: boolean;
  provider: string;
  expires_at: string | null;
  has_refresh_token: boolean;
  needs_reauth?: boolean;
  account_info?: {
    email?: string;
    connected_at?: string;
  };
}

interface Integration {
  connected: boolean;
  email: string | null;
  connectedAt: string | null;
  status: "connected" | "disconnected" | "needs_reauth";
  account_info: { email?: string; connected_at?: string } | null;
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
  apiPath: string; // API endpoint path for this integration
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

// API configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Create axios instance with auth header
const api = axios.create({
  baseURL: API_BASE_URL,
});

// Add auth token from localStorage
const token = localStorage.getItem("token");
if (token) {
  api.defaults.headers.common.Authorization = `Bearer ${token}`;
}

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
      features: ["File sync", "Backup storage", "Shared folders"],
      apiPath: "/oauth/dropbox"
    },
    googleDrive: {
      name: "Google Drive",
      icon: GOOGLE_DRIVE_ICON,
      description: "Connect to Google Drive for file management",
      color: "#4285F4",
      features: ["Document storage", "Real-time collaboration", "Google Workspace integration"],
      apiPath: "/oauth/googledrive" // Update when endpoint is available
    }
  };

  // Helper function to map OAuth status to Integration
  const mapOAuthStatusToIntegration = (status: OAuthStatus | null): Partial<Integration> => {
    if (!status) {
      return {
        connected: false,
        email: null,
        connectedAt: null,
        status: "disconnected",
        account_info: null,
        needs_reauth: false,
        expires_at: null,
      };
    }

    // Check if token is expired or about to expire
    const needsReauth = status.needs_reauth || 
      (status.expires_at ? new Date(status.expires_at) < new Date() : false);

    return {
      connected: status.connected && !needsReauth,
      email: status.account_info?.email || null,
      connectedAt: status.account_info?.connected_at || null,
      status: needsReauth ? "needs_reauth" : (status.connected ? "connected" : "disconnected"),
      account_info: status.account_info || null,
      needs_reauth: needsReauth,
      expires_at: status.expires_at,
    };
  };

  // Load integration status from API
  const loadIntegrationStatus = async () => {
    // Load Dropbox status
    try {
      const dropboxResponse = await api.get<OAuthStatus>("/oauth/dropbox/status");
      setIntegrations(prev => ({
        ...prev,
        dropbox: {
          ...prev.dropbox,
          ...mapOAuthStatusToIntegration(dropboxResponse.data)
        }
      }));
    } catch (error) {
      console.error("Error loading Dropbox status:", error);
      // If 404 or auth error, assume not connected
      if (axios.isAxiosError(error) && (error.response?.status === 404 || error.response?.status === 401)) {
        setIntegrations(prev => ({
          ...prev,
          dropbox: {
            ...prev.dropbox,
            ...mapOAuthStatusToIntegration(null)
          }
        }));
      }
    }

    // Load Google Drive status when endpoint is available
    // For now, keep it disconnected
    // try {
    //   const gdriveResponse = await api.get<OAuthStatus>("/oauth/googledrive/status");
    //   setIntegrations(prev => ({
    //     ...prev,
    //     googleDrive: {
    //       ...prev.googleDrive,
    //       ...mapOAuthStatusToIntegration(gdriveResponse.data)
    //     }
    //   }));
    // } catch (error) {
    //   console.error("Error loading Google Drive status:", error);
    // }
  };

  // Handle OAuth connection
  const handleConnect = async (integrationKey: IntegrationKey) => {
    setLoading(prev => ({ ...prev, [integrationKey]: true }));
    setMessage({ type: "", text: "" });

    try {
      const config = integrationConfigs[integrationKey];
      
      // Special handling for Google Drive (not implemented yet)
      if (integrationKey === "googleDrive") {
        setMessage({
          type: "error",
          text: "Google Drive integration is coming soon!"
        });
        setLoading(prev => ({ ...prev, [integrationKey]: false }));
        return;
      }

      // Get OAuth authorization URL from backend
      const response = await api.get<{ authorize_url: string }>(`${config.apiPath}/start`);
      
      if (response.data.authorize_url) {
        // Save current page to return after OAuth
        sessionStorage.setItem('oauth_return_url', window.location.href);
        
        // Redirect to OAuth provider
        window.location.href = response.data.authorize_url;
      } else {
        throw new Error("No authorization URL received");
      }
    } catch (error) {
      console.error("Connection error:", error);
      let errorMessage = `Failed to connect to ${integrationConfigs[integrationKey].name}`;
      
      if (axios.isAxiosError(error)) {
        if (error.response?.status === 401) {
          errorMessage = "Please log in to connect integrations";
        } else if (error.response?.data?.detail) {
          errorMessage = error.response.data.detail;
        }
      }
      
      setMessage({
        type: "error",
        text: errorMessage
      });
      setLoading(prev => ({ ...prev, [integrationKey]: false }));
    }
  };

  // Handle re-authentication (similar to connect)
  const handleReauth = async (integrationKey: IntegrationKey) => {
    // Re-auth is essentially the same as connect for OAuth
    await handleConnect(integrationKey);
  };

  // Handle disconnect
  const handleDisconnect = async (integrationKey: IntegrationKey) => {
    const config = integrationConfigs[integrationKey];
    if (!window.confirm(`Are you sure you want to disconnect from ${config.name}?`)) {
      return;
    }

    setLoading(prev => ({ ...prev, [integrationKey]: true }));

    try {
      await api.post(`${config.apiPath}/disconnect`);
      
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
      let errorMessage = `Failed to disconnect from ${config.name}`;
      
      if (axios.isAxiosError(error) && error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      }
      
      setMessage({
        type: "error",
        text: errorMessage
      });
    } finally {
      setLoading(prev => ({ ...prev, [integrationKey]: false }));
    }
  };

  // Handle sync metadata (using the startDropboxSync from api.ts)
  const handleSyncMetadata = async (integrationKey: IntegrationKey) => {
    setLoading(prev => ({ ...prev, [`${integrationKey}_sync`]: true }));
    setMessage({ type: "", text: "" });

    try {
      if (integrationKey === "dropbox") {
        // Start sync task
        const response = await fetch(`${API_BASE_URL}/integrations/dropbox/sync`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${localStorage.getItem("token")}`
          },
          body: JSON.stringify({
            root_path: "/",
            force: false
          })
        });

        if (!response.ok) {
          throw new Error(await response.text());
        }

        const data = await response.json() as { task_id: string };
        
        setMessage({
          type: "success",
          text: `Sync started for ${integrationConfigs[integrationKey].name}. Task ID: ${data.task_id}`
        });

        // Update sync status
        setIntegrations(prev => ({
          ...prev,
          [integrationKey]: {
            ...prev[integrationKey],
            last_sync: new Date().toISOString(),
            sync_status: "success"
          }
        }));

        // You could poll for task status here if needed
        // pollTaskStatus(data.task_id);
      } else {
        setMessage({
          type: "error",
          text: `Sync not available for ${integrationConfigs[integrationKey].name} yet`
        });
      }
    } catch (error) {
      console.error("Sync error:", error);
      
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

  // Check for OAuth callback on component mount
  useEffect(() => {
    // Check if we're returning from OAuth callback
    const urlParams = new URLSearchParams(window.location.search);
    const code = urlParams.get('code');
    const state = urlParams.get('state');
    
    if (code && state) {
      // Handle OAuth callback
      handleOAuthCallback(code, state);
    }
  }, []);

  // Handle OAuth callback
  const handleOAuthCallback = async (code: string, state: string) => {
    setMessage({ type: "", text: "Completing authorization..." });

    try {
      // Get the redirect URI that was used to start OAuth
      const redirectUri = sessionStorage.getItem('oauth_redirect_uri') || window.location.origin + window.location.pathname;
      
      // Determine which provider based on state or URL
      // For now, assume Dropbox (you might encode provider in state)
      const response = await api.post('/oauth/dropbox/callback', {
        code,
        state,
        redirect_uri: redirectUri
      });

      if (response.data.connected) {
        setMessage({
          type: "success",
          text: "Successfully connected to Dropbox!"
        });

        // Clear URL parameters
        window.history.replaceState({}, document.title, window.location.pathname);
        
        // Clear session storage
        sessionStorage.removeItem('oauth_redirect_uri');
        sessionStorage.removeItem('oauth_return_url');

        // Reload integration status
        await loadIntegrationStatus();
      }
    } catch (error) {
      console.error("OAuth callback error:", error);
      let errorMessage = "Failed to complete authorization";
      
      if (axios.isAxiosError(error) && error.response?.data?.detail) {
        errorMessage = error.response.data.detail;
      }
      
      setMessage({
        type: "error",
        text: errorMessage
      });

      // Clear URL parameters even on error
      window.history.replaceState({}, document.title, window.location.pathname);
    }
  };

  // Load integration status on mount and when user changes
  useEffect(() => {
    if (user) {
      setCurrentUserId(user.id);
      loadIntegrationStatus();
    }
  }, [user]);

  // Auto-clear messages after 5 seconds
  useEffect(() => {
    if (message.text) {
      const timer = setTimeout(() => {
        setMessage({ type: "", text: "" });
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [message]);

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
      <div style={styles.container} data-testid="cloud-integrations-container">
        {message.text && (
          <div 
            {...getTestAttributes({ 
              prefix: "msg", 
              feature: "integrations",
              status: message.type as StatusType
            })}
            style={{
              ...styles.messageBox,
              backgroundColor: message.type === "success" ? "#dcfce7" : "#fee2e2",
              border: `1px solid ${message.type === "success" ? "#bbf7d0" : "#fecaca"}`,
              color: message.type === "success" ? "#16a34a" : "#dc2626"
            }}
          >
            {message.type === "success" ? (
              <CheckCircle style={{ width: "16px", height: "16px" }} />
            ) : (
              <AlertCircle style={{ width: "16px", height: "16px" }} />
            )}
            <span data-testid={`msg-text-${message.type}`}>{message.text}</span>
          </div>
        )}

        <div style={styles.gridContainer} data-testid="integrations-grid">
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
    <div 
      style={styles.card}
      {...getTestAttributes({ 
        prefix: "card", 
        feature: "integrations",
        integration: integrationKey
      })}
    >
      <div style={styles.header}>
        <img 
          src={config.icon} 
          alt={`${config.name} logo`} 
          style={styles.icon}
          data-testid={generateTestId({
            prefix: "icon",
            integration: integrationKey
          })}
        />
        <div>
          <h3 
            style={styles.title}
            data-testid={generateTestId({
              prefix: "header",
              integration: integrationKey,
              suffix: "title"
            })}
          >
            {config.name}
          </h3>
          <div 
            style={styles.statusContainer}
            {...getTestAttributes({ 
              prefix: "status", 
              integration: integrationKey,
              status: integration.status as StatusType
            })}
          >
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
        <div 
          style={styles.warningBox}
          data-testid={generateTestId({
            prefix: "msg",
            integration: integrationKey,
            suffix: "reauth-warning"
          })}
        >
          <AlertTriangle style={{ width: "16px", height: "16px", color: "#f59e0b" }} />
          <span style={{ fontSize: "14px", color: "#92400e" }}>
            Your connection has expired. Please re-authorize to continue using this integration.
          </span>
        </div>
      )}

      {integration.connected && !integration.needs_reauth && (
        <div 
          style={styles.infoBox}
          data-testid={generateTestId({
            prefix: "section",
            integration: integrationKey,
            suffix: "connected-info"
          })}
        >
          <div style={{ fontSize: "14px", color: "#6b7280", marginBottom: "6px" }}>
            Connected Account
          </div>
          <div 
            style={{ fontSize: "16px", fontWeight: "500", color: "#111827" }}
            data-testid={generateTestId({
              prefix: "label",
              integration: integrationKey,
              suffix: "connected-email"
            })}
          >
            {integration.email || "Connected"}
          </div>
          {integration.connectedAt && (
            <div 
              style={{ fontSize: "14px", color: "#6b7280", marginTop: "6px" }}
              data-testid={generateTestId({
                prefix: "label",
                integration: integrationKey,
                suffix: "connected-date"
              })}
            >
              Connected on {new Date(integration.connectedAt).toLocaleDateString()}
            </div>
          )}
          {integration.last_sync && (
            <div 
              style={{ 
                fontSize: "12px", 
                color: "#6b7280", 
                marginTop: "4px",
                display: "flex",
                alignItems: "center",
                gap: "4px"
              }}
              data-testid={generateTestId({
                prefix: "label",
                integration: integrationKey,
                suffix: "last-sync"
              })}
            >
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
        <ul 
          style={styles.featuresList}
          data-testid={generateTestId({
            prefix: "list",
            integration: integrationKey,
            suffix: "features"
          })}
        >
          {config.features.map((feature, index) => (
            <li 
              key={index} 
              style={{ marginBottom: "4px" }}
              data-testid={generateTestId({
                prefix: "item",
                integration: integrationKey,
                identifier: index,
                suffix: "feature"
              })}
            >
              {feature}
            </li>
          ))}
        </ul>
      </div>

      <div style={styles.buttonContainer}>
        {integration.needs_reauth ? (
          <>
            <button
              {...getTestAttributes({ 
                prefix: "btn", 
                feature: "integrations",
                integration: integrationKey, 
                action: "reauth",
                ariaLabel: `Re-authorize ${config.name}`
              })}
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
              {...getTestAttributes({ 
                prefix: "btn", 
                feature: "integrations",
                integration: integrationKey, 
                action: "disconnect",
                ariaLabel: `Disconnect from ${config.name}`
              })}
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
              {...getTestAttributes({ 
                prefix: "btn", 
                feature: "integrations",
                integration: integrationKey, 
                action: "sync",
                ariaLabel: `Sync metadata from ${config.name}`
              })}
              onClick={() => onSync(integrationKey)}
              disabled={loading[`${integrationKey}_sync`] || integrationKey !== "dropbox"}
              style={{
                ...styles.primaryButton,
                backgroundColor: loading[`${integrationKey}_sync`] || integrationKey !== "dropbox" ? "#9ca3af" : config.color,
                opacity: loading[`${integrationKey}_sync`] || integrationKey !== "dropbox" ? 0.5 : 1,
                cursor: loading[`${integrationKey}_sync`] || integrationKey !== "dropbox" ? "not-allowed" : "pointer"
              }}
              title={integrationKey !== "dropbox" ? "Sync not available for this integration yet" : undefined}
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
              {...getTestAttributes({ 
                prefix: "btn", 
                feature: "integrations",
                integration: integrationKey, 
                action: "disconnect",
                ariaLabel: `Disconnect from ${config.name}`
              })}
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
              {...getTestAttributes({ 
                prefix: "btn", 
                feature: "integrations",
                integration: integrationKey, 
                action: "settings",
                ariaLabel: `Settings for ${config.name}`
              })}
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
            {...getTestAttributes({ 
              prefix: "btn", 
              feature: "integrations",
              integration: integrationKey, 
              action: "connect",
              ariaLabel: `Connect to ${config.name}`
            })}
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