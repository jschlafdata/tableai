// src/components/integrations/ProcessButton.tsx
import { useState } from "react";
import type { CSSProperties } from "react";
import { Play } from "lucide-react";
import type { ProcessResult } from "./EnhancedHeaderControls";

interface ProcessButtonProps {
  selectedFiles?: string[];
  force?: boolean;
  stage?: number;
  priority?: number;
  disabled?: boolean;
  onComplete?: (result: ProcessResult) => void;
}

const ProcessButton: React.FC<ProcessButtonProps> = ({
  selectedFiles = [],
  force = false,
  stage = 0,
  priority = 5,
  disabled = false,
  onComplete = () => {},
}) => {
  const [isProcessing, setIsProcessing] = useState(false);

  const handleClick = async () => {
    if (!selectedFiles || selectedFiles.length === 0) {
      alert("No files selected for processing");
      return;
    }

    setIsProcessing(true);

    try {
      console.log("Starting bulk file processing:", {
        fileIds: selectedFiles,
        stage,
        force,
        priority,
      });

      // Simulate processing for UI demonstration
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Mock successful response
      const mockResult: ProcessResult = {
        success: true,
        taskIds: ["task-001", "task-002", "task-003"],
        message: `Processing ${selectedFiles.length} files`,
        fileCount: selectedFiles.length,
      };

      console.log("Processing started successfully:", mockResult);
      
      alert(
        `✅ Processing started successfully!\n${mockResult.message}\nTask IDs: ${mockResult.taskIds?.slice(0, 3).join(", ")}${
          mockResult.taskIds && mockResult.taskIds.length > 3 ? "..." : ""
        }`
      );

      onComplete(mockResult);
    } catch (err: any) {
      console.error("Processing failed:", err);
      
      const errorResult: ProcessResult = {
        success: false,
        error: err.message || "Processing failed",
        fileCount: selectedFiles.length,
      };
      
      alert(`❌ Processing failed: ${errorResult.error}`);
      onComplete(errorResult);
    } finally {
      setIsProcessing(false);
    }
  };

  const buttonText = isProcessing
    ? "Processing..."
    : `Process ${selectedFiles.length} file${selectedFiles.length !== 1 ? "s" : ""}`;

  const styles: Record<string, CSSProperties> = {
    button: {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      padding: "8px 16px",
      backgroundColor: "#9333ea",
      color: "white",
      border: "none",
      borderRadius: "6px",
      fontSize: "14px",
      fontWeight: 500,
      cursor: "pointer",
      transition: "background-color 0.2s, opacity 0.2s",
    },
    buttonDisabled: {
      opacity: 0.5,
      cursor: "not-allowed",
    },
    buttonHover: {
      backgroundColor: "#7c2d12",
    },
    spinner: {
      width: "20px",
      height: "20px",
      border: "2px solid transparent",
      borderTop: "2px solid currentColor",
      borderRadius: "50%",
      animation: "spin 1s linear infinite",
    },
  };

  const isDisabled = !selectedFiles?.length || disabled || isProcessing;

  return (
    <>
      <button
        onClick={handleClick}
        disabled={isDisabled}
        style={{
          ...styles.button,
          ...(isDisabled ? styles.buttonDisabled : {}),
        }}
        onMouseEnter={(e) => {
          if (!isDisabled) {
            e.currentTarget.style.backgroundColor = "#7e22ce";
          }
        }}
        onMouseLeave={(e) => {
          if (!isDisabled) {
            e.currentTarget.style.backgroundColor = "#9333ea";
          }
        }}
        title={`Process ${selectedFiles.length} selected files${force ? " (forced)" : ""}`}
      >
        {isProcessing ? (
          <span style={styles.spinner} />
        ) : (
          <Play size={20} />
        )}
        {buttonText}
      </button>
      
      <style>
        {`
          @keyframes spin {
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </>
  );
};

export default ProcessButton;