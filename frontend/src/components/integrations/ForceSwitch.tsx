// src/components/integrations/ForceSwitch.tsx
import type { CSSProperties } from "react";

interface ForceSwitchProps {
  force: boolean;
  onForceChange: (force: boolean) => void;
}

const ForceSwitch: React.FC<ForceSwitchProps> = ({ force, onForceChange }) => {
  const styles: Record<string, CSSProperties> = {
    container: {
      display: "inline-flex",
      alignItems: "center",
      gap: "8px",
      marginRight: "16px",
    },
    switch: {
      position: "relative",
      width: "44px",
      height: "24px",
      backgroundColor: force ? "#2563eb" : "#d1d5db",
      borderRadius: "12px",
      cursor: "pointer",
      transition: "background-color 0.2s",
      border: "none",
      padding: 0,
    },
    slider: {
      position: "absolute",
      top: "2px",
      left: force ? "22px" : "2px",
      width: "20px",
      height: "20px",
      backgroundColor: "white",
      borderRadius: "50%",
      transition: "left 0.2s",
      boxShadow: "0 2px 4px rgba(0, 0, 0, 0.2)",
    },
    label: {
      fontSize: "14px",
      fontWeight: 500,
      color: "#374151",
      userSelect: "none",
      cursor: "pointer",
    },
  };

  return (
    <div style={styles.container}>
      <button
        type="button"
        style={styles.switch}
        onClick={() => onForceChange(!force)}
        aria-checked={force}
        role="switch"
      >
        <span style={styles.slider} />
      </button>
      <label 
        style={styles.label}
        onClick={() => onForceChange(!force)}
      >
        Force
      </label>
    </div>
  );
};

export default ForceSwitch;