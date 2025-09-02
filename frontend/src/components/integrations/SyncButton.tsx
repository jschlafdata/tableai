// src/components/integrations/SyncButton.tsx
import type { CSSProperties, ButtonHTMLAttributes } from "react";
import { RefreshCw } from "lucide-react";

interface SyncButtonProps extends Omit<ButtonHTMLAttributes<HTMLButtonElement>, "onClick"> {
  onSync: () => void;
  loading?: boolean;
  sx?: CSSProperties;
}

const SyncButton: React.FC<SyncButtonProps> = ({ 
  onSync, 
  loading = false, 
  disabled,
  sx,
  ...props 
}) => {
  const styles: Record<string, CSSProperties> = {
    button: {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      padding: "8px 16px",
      backgroundColor: "#2563eb",
      color: "white",
      border: "none",
      borderRadius: "6px",
      fontSize: "14px",
      fontWeight: 500,
      cursor: "pointer",
      transition: "background-color 0.2s, opacity 0.2s",
      ...sx,
    },
    buttonDisabled: {
      opacity: 0.5,
      cursor: "not-allowed",
    },
    buttonHover: {
      backgroundColor: "#1d4ed8",
    },
    spinner: {
      animation: "spin 1s linear infinite",
    },
  };

  const isDisabled = loading || disabled;

  return (
    <>
      <button
        {...props}
        onClick={onSync}
        disabled={isDisabled}
        style={{
          ...styles.button,
          ...(isDisabled ? styles.buttonDisabled : {}),
        }}
        onMouseEnter={(e) => {
          if (!isDisabled) {
            e.currentTarget.style.backgroundColor = "#1d4ed8";
          }
        }}
        onMouseLeave={(e) => {
          if (!isDisabled) {
            e.currentTarget.style.backgroundColor = "#2563eb";
          }
        }}
      >
        {loading ? (
          <span style={styles.spinner}>
            <RefreshCw size={20} />
          </span>
        ) : (
          <RefreshCw size={20} />
        )}
        {loading ? "Syncing..." : "Sync"}
      </button>
      
      <style>
        {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </>
  );
};

export default SyncButton;