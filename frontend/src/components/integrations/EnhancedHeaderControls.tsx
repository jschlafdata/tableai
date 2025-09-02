// src/components/integrations/EnhancedHeaderControls.tsx
import { useState } from "react";
import type { CSSProperties } from "react";
import { ArrowUp, ArrowDown } from "lucide-react";
import SyncButton from "./SyncButton";
import ProcessButton from "./ProcessButton";
import ForceSwitch from "./ForceSwitch";

// Types
export type SortDirection = "asc" | "desc";
export type FileType = "pdf" | "xlsx" | "docx" | "png";

interface EnhancedHeaderControlsProps {
  selectedFiles: string[];
  fileTypes: FileType[];
  sortDir: SortDirection;
  ignoreRegex: string;
  pathCategories: string;
  autoLabel: string;
  force: boolean;
  onForceChange: (force: boolean) => void;
  isSyncing: boolean;
  onSync: () => void;
  onSortChange: (dir: SortDirection) => void;
  onFileTypeChange: (types: FileType[]) => void;
  onRegexChange: (regex: string) => void;
  onCategoryChange: (category: string) => void;
  onLabelChange: (label: string) => void;
  onProcessComplete: (result: ProcessResult) => void;
  pathOrId?: string;
  isFolder?: boolean;
  filterCounts?: Record<string, number>;
  onRefresh?: () => void;
  globalLoading?: boolean;
}

export interface ProcessResult {
  success: boolean;
  taskIds?: string[];
  message?: string;
  fileCount: number;
  error?: string;
}

const EnhancedHeaderControls: React.FC<EnhancedHeaderControlsProps> = ({
  selectedFiles,
  fileTypes,
  sortDir,
  ignoreRegex,
  pathCategories,
  autoLabel,
  force,
  onForceChange,
  isSyncing,
  onSync,
  onSortChange,
  onFileTypeChange,
  onRegexChange,
  onCategoryChange,
  onLabelChange,
  onProcessComplete,
  globalLoading = false,
}) => {
  const styles: Record<string, CSSProperties> = {
    container: {
      display: "flex",
      flexDirection: "column",
      gap: "16px",
    },
    row: {
      display: "flex",
      alignItems: "center",
      gap: "16px",
      flexWrap: "wrap",
    },
    toggleGroup: {
      display: "inline-flex",
      borderRadius: "6px",
      overflow: "hidden",
      border: "1px solid #d1d5db",
    },
    toggleButton: {
      padding: "6px 12px",
      backgroundColor: "white",
      border: "none",
      borderRight: "1px solid #d1d5db",
      cursor: "pointer",
      fontSize: "14px",
      transition: "background-color 0.2s",
      display: "flex",
      alignItems: "center",
      gap: "4px",
    },
    toggleButtonActive: {
      backgroundColor: "#2563eb",
      color: "white",
    },
    textInput: {
      width: "200px",
      padding: "6px 12px",
      border: "1px solid #d1d5db",
      borderRadius: "6px",
      fontSize: "14px",
      outline: "none",
    },
  };

  const handleFileTypeToggle = (type: FileType) => {
    const newTypes = fileTypes.includes(type)
      ? fileTypes.filter(t => t !== type)
      : [...fileTypes, type];
    onFileTypeChange(newTypes);
  };

  return (
    <div style={styles.container}>
      {/* First row - Main action buttons */}
      <div style={styles.row}>
        <SyncButton
          onSync={onSync}
          loading={isSyncing}
          disabled={!selectedFiles.length}
        />

        <ForceSwitch force={force} onForceChange={onForceChange} />

        <ProcessButton
          selectedFiles={selectedFiles}
          force={force}
          stage={0}
          priority={5}
          disabled={!selectedFiles?.length || globalLoading}
          onComplete={onProcessComplete}
        />

        {/* Sort Direction Toggle */}
        <div style={styles.toggleGroup}>
          <button
            style={{
              ...styles.toggleButton,
              ...(sortDir === "asc" ? styles.toggleButtonActive : {}),
            }}
            onClick={() => onSortChange("asc")}
          >
            <ArrowUp size={16} />
          </button>
          <button
            style={{
              ...styles.toggleButton,
              borderRight: "none",
              ...(sortDir === "desc" ? styles.toggleButtonActive : {}),
            }}
            onClick={() => onSortChange("desc")}
          >
            <ArrowDown size={16} />
          </button>
        </div>

        {/* File Type Toggles */}
        <div style={styles.toggleGroup}>
          {(["pdf", "xlsx", "docx", "png"] as FileType[]).map((type, index, array) => (
            <button
              key={type}
              style={{
                ...styles.toggleButton,
                ...(fileTypes.includes(type) ? styles.toggleButtonActive : {}),
                ...(index === array.length - 1 ? { borderRight: "none" } : {}),
              }}
              onClick={() => handleFileTypeToggle(type)}
            >
              {type.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Second row - Text inputs */}
      <div style={styles.row}>
        <input
          type="text"
          placeholder="Ignore Folder Paths"
          value={ignoreRegex}
          onChange={(e) => onRegexChange(e.target.value)}
          style={styles.textInput}
        />
        <input
          type="text"
          placeholder="Path Categories"
          value={pathCategories}
          onChange={(e) => onCategoryChange(e.target.value)}
          style={styles.textInput}
        />
        <input
          type="text"
          placeholder="Auto Label"
          value={autoLabel}
          onChange={(e) => onLabelChange(e.target.value)}
          style={styles.textInput}
        />
      </div>
    </div>
  );
};

export default EnhancedHeaderControls;