// src/components/integrations/UnifiedFileViewer.tsx
import { useState, useMemo } from "react";
import type { CSSProperties } from "react";
import { 
  Folder, 
  File, 
  MoreVertical, 
  ChevronRight,
  Info
} from "lucide-react";
import PageLayout from "../layouts/PageLayout";
import UserDropdown from "../auth/UserDropdown";
import EnhancedHeaderControls from "./EnhancedHeaderControls";
import type { User } from "../../types/user";
import type { SortDirection, FileType, ProcessResult } from "./EnhancedHeaderControls";
import { setToken } from "../../api";

// Types
interface FileItem {
  id: string;
  name: string;
  path: string;
  file_type: "file" | "folder";
  file_ext?: string;
  size?: number;
  modified_time?: string;
  source: "dropbox" | "google_drive" | "local";
  source_metadata?: {
    shared_with?: Array<{ name: string; avatar_url?: string }>;
  };
}

interface UnifiedFileViewerProps {
  user: User | null;
}

interface SyncResult {
  type: "success" | "error";
  message: string;
  details?: {
    totalSynced?: number;
    totalMatched?: number;
    dropboxFiles?: number;
    googleDriveFiles?: number;
    backend?: string;
    totalErrors?: number;
  };
}

// Mock data for demonstration
const MOCK_FILES: FileItem[] = [
  {
    id: "1",
    name: "Documents",
    path: "/Documents",
    file_type: "folder",
    source: "dropbox",
    modified_time: "2024-12-20T10:00:00Z"
  },
  {
    id: "2",
    name: "report_2024.pdf",
    path: "/report_2024.pdf",
    file_type: "file",
    file_ext: "pdf",
    size: 2457600,
    source: "dropbox",
    modified_time: "2024-12-15T14:30:00Z"
  },
  {
    id: "3",
    name: "Projects",
    path: "/Projects",
    file_type: "folder",
    source: "google_drive",
    modified_time: "2024-12-18T09:00:00Z"
  },
  {
    id: "4",
    name: "presentation.pdf",
    path: "/Documents/presentation.pdf",
    file_type: "file",
    file_ext: "pdf",
    size: 5242880,
    source: "dropbox",
    modified_time: "2024-12-19T16:45:00Z"
  },
  {
    id: "5",
    name: "data_analysis.xlsx",
    path: "/Documents/data_analysis.xlsx",
    file_type: "file",
    file_ext: "xlsx",
    size: 1843200,
    source: "google_drive",
    modified_time: "2024-12-18T11:20:00Z",
    source_metadata: {
      shared_with: [
        { name: "John Doe", avatar_url: "" },
        { name: "Jane Smith", avatar_url: "" }
      ]
    }
  }
];

// Utility functions
const formatBytes = (bytes = 0) => {
  if (bytes < 1024) return `${bytes} B`;
  const k = 1024, dm = 2, sizes = ["KB", "MB", "GB", "TB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(dm)} ${sizes[i]}`;
};

const formatDate = (iso?: string) => {
  if (!iso) return "—";
  const d = new Date(iso);
  return (
    d.toLocaleDateString(undefined, { day: "2-digit", month: "short", year: "numeric" }) +
    " " +
    d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })
  );
};

const getSourceIcon = (source: string) => {
  switch (source) {
    case "dropbox":
      return "📦";
    case "google_drive":
      return "💾";
    default:
      return "📁";
  }
};

// Breadcrumbs Component
const FileBreadcrumbs: React.FC<{
  currentPath: string;
  onCrumbClick: (path: string) => void;
}> = ({ currentPath, onCrumbClick }) => {
  const crumbs = [
    { name: "Home", path: "" },
    ...String(currentPath)
      .split("/")
      .filter(Boolean)
      .map((seg, i, all) => ({
        name: decodeURIComponent(seg),
        path: all.slice(0, i + 1).join("/")
      }))
  ];

  const styles: Record<string, CSSProperties> = {
    container: {
      display: "flex",
      alignItems: "center",
      gap: "8px",
      marginBottom: "16px",
      fontSize: "14px"
    },
    crumb: {
      color: "#6b7280",
      cursor: "pointer",
      textDecoration: "none"
    },
    activeCrumb: {
      color: "#111827",
      fontWeight: 500
    },
    separator: {
      color: "#9ca3af"
    }
  };

  return (
    <div style={styles.container}>
      {crumbs.map((crumb, index) => (
        <React.Fragment key={crumb.path}>
          {index > 0 && <ChevronRight size={16} style={styles.separator} />}
          <span
            style={{
              ...styles.crumb,
              ...(crumb.path === currentPath ? styles.activeCrumb : {})
            }}
            onClick={() => onCrumbClick(crumb.path)}
          >
            {crumb.name}
          </span>
        </React.Fragment>
      ))}
    </div>
  );
};

// File Item Component
const FileItemRow: React.FC<{
  file: FileItem;
  isSelected: boolean;
  onSelect: (e: React.ChangeEvent<HTMLInputElement>, file: FileItem) => void;
  onItemClick: (file: FileItem) => void;
}> = ({ file, isSelected, onSelect, onItemClick }) => {
  const isFolder = file.file_type === "folder";
  const sharedWith = file.source_metadata?.shared_with || [];

  const styles: Record<string, CSSProperties> = {
    row: {
      display: "flex",
      alignItems: "center",
      backgroundColor: "white",
      padding: "12px 16px",
      borderRadius: "8px",
      boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
      marginBottom: "8px",
      cursor: "pointer",
      transition: "all 0.2s",
      ...(isSelected && {
        backgroundColor: "#eff6ff",
        boxShadow: "0 1px 3px rgba(37, 99, 235, 0.2)"
      })
    },
    checkbox: {
      marginRight: "12px",
      cursor: "pointer"
    },
    content: {
      display: "flex",
      alignItems: "center",
      flex: 1,
      gap: "8px",
      overflow: "hidden"
    },
    icon: {
      color: isFolder ? "#f59e0b" : "#6b7280"
    },
    name: {
      fontSize: "14px",
      fontWeight: 500,
      color: "#111827",
      whiteSpace: "nowrap" as const,
      overflow: "hidden",
      textOverflow: "ellipsis"
    },
    sourceIcon: {
      fontSize: "12px",
      marginLeft: "4px"
    },
    metadata: {
      display: "flex",
      alignItems: "center",
      gap: "24px"
    },
    metaItem: {
      fontSize: "13px",
      color: "#6b7280",
      minWidth: "80px"
    },
    avatarGroup: {
      display: "flex",
      marginLeft: "8px"
    },
    avatar: {
      width: "24px",
      height: "24px",
      borderRadius: "50%",
      backgroundColor: "#e5e7eb",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      fontSize: "10px",
      color: "#6b7280",
      marginLeft: "-8px",
      border: "2px solid white"
    },
    moreButton: {
      padding: "4px",
      backgroundColor: "transparent",
      border: "none",
      cursor: "pointer",
      borderRadius: "4px",
      display: "flex",
      alignItems: "center",
      justifyContent: "center"
    }
  };

  return (
    <div
      style={styles.row}
      onClick={() => onItemClick(file)}
      onMouseEnter={(e) => {
        if (!isSelected) {
          e.currentTarget.style.backgroundColor = "#f9fafb";
        }
      }}
      onMouseLeave={(e) => {
        if (!isSelected) {
          e.currentTarget.style.backgroundColor = "white";
        }
      }}
    >
      <input
        type="checkbox"
        checked={isSelected}
        onChange={(e) => onSelect(e, file)}
        onClick={(e) => e.stopPropagation()}
        style={styles.checkbox}
      />

      <div style={styles.content}>
        <div style={styles.icon}>
          {isFolder ? <Folder size={20} /> : <File size={20} />}
        </div>
        <span style={styles.name}>{file.name}</span>
        <span style={styles.sourceIcon} title={`Source: ${file.source}`}>
          {getSourceIcon(file.source)}
        </span>
      </div>

      <div style={styles.metadata}>
        <span style={styles.metaItem}>
          {file.size ? formatBytes(file.size) : "—"}
        </span>
        <span style={{ ...styles.metaItem, minWidth: "60px" }}>
          {file.file_ext || (isFolder ? "folder" : "—")}
        </span>
        <span style={{ ...styles.metaItem, minWidth: "140px" }}>
          {formatDate(file.modified_time)}
        </span>

        {sharedWith.length > 0 && (
          <div style={styles.avatarGroup}>
            {sharedWith.slice(0, 3).map((user, i) => (
              <div key={i} style={{ ...styles.avatar, zIndex: sharedWith.length - i }}>
                {user.name?.[0]?.toUpperCase() || "?"}
              </div>
            ))}
          </div>
        )}

        <button
          style={styles.moreButton}
          onClick={(e) => e.stopPropagation()}
        >
          <MoreVertical size={16} />
        </button>
      </div>
    </div>
  );
};

// Main Component
const UnifiedFileViewer: React.FC<UnifiedFileViewerProps> = ({ user }) => {
  // State
  const [files] = useState<FileItem[]>(MOCK_FILES);
  const [currentPath, setCurrentPath] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [sortDir, setSortDir] = useState<SortDirection>("asc");
  const [fileTypes, setFileTypes] = useState<FileType[]>(["pdf"]);
  const [sourceFilter, setSourceFilter] = useState("all");
  const [isLoading, setIsLoading] = useState(false);
  const [syncResult, setSyncResult] = useState<SyncResult | null>(null);

  // Enhanced controls state
  const [ignoreRegex, setIgnoreRegex] = useState("");
  const [pathCategories, setPathCategories] = useState("");
  const [autoLabel, setAutoLabel] = useState("");
  const [force, setForce] = useState(false);

  // Filter and sort files
  const filteredFiles = useMemo(() => {
    let filtered = [...files];

    // Filter by current path
    if (currentPath) {
      const normalizedCurrentPath = currentPath.startsWith("/") ? currentPath : `/${currentPath}`;
      filtered = filtered.filter(file => {
        const filePath = file.path;
        const parentPath = filePath.substring(0, filePath.lastIndexOf("/")) || "/";
        return parentPath === normalizedCurrentPath;
      });
    } else {
      filtered = filtered.filter(file => {
        const filePath = file.path;
        const parentPath = filePath.substring(0, filePath.lastIndexOf("/")) || "/";
        return parentPath === "/";
      });
    }

    // Filter by source
    if (sourceFilter !== "all") {
      filtered = filtered.filter(file => file.source === sourceFilter);
    }

    // Filter by file type
    if (fileTypes.length > 0 && !fileTypes.includes("png")) {
      filtered = filtered.filter(file =>
        file.file_type === "folder" ||
        (file.file_ext && fileTypes.includes(file.file_ext as FileType))
      );
    }

    // Apply ignore regex
    if (ignoreRegex) {
      try {
        const regex = new RegExp(ignoreRegex, "i");
        filtered = filtered.filter(file => !regex.test(file.path));
      } catch (e) {
        console.warn("Invalid regex pattern:", ignoreRegex);
      }
    }

    // Sort
    filtered.sort((a, b) => {
      if (a.file_type === "folder" && b.file_type !== "folder") return -1;
      if (a.file_type !== "folder" && b.file_type === "folder") return 1;
      
      const compareValue = a.name.localeCompare(b.name);
      return sortDir === "asc" ? compareValue : -compareValue;
    });

    return filtered;
  }, [files, currentPath, sourceFilter, fileTypes, sortDir, ignoreRegex]);

  // Handlers
  const handleItemClick = (file: FileItem) => {
    if (file.file_type === "folder") {
      const folderPath = file.path.startsWith("/") ? file.path.substring(1) : file.path;
      setCurrentPath(folderPath);
    } else {
      alert(`Opening file: ${file.name} (ID: ${file.id})`);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>, file: FileItem) => {
    e.stopPropagation();
    
    setSelectedFiles(prev => {
      return e.target.checked
        ? [...prev, file.id]
        : prev.filter(id => id !== file.id);
    });
  };

  const handleSync = async () => {
    setIsLoading(true);
    try {
      // Simulate sync operation
      await new Promise(resolve => setTimeout(resolve, 2000));

      setSyncResult({
        type: "success",
        message: "✅ Sync completed! Files synced successfully.",
        details: {
          totalSynced: selectedFiles.length,
          dropboxFiles: 3,
          googleDriveFiles: 2,
          backend: "local"
        }
      });

      setTimeout(() => setSyncResult(null), 5000);
      setSelectedFiles([]);
    } catch (error) {
      setSyncResult({
        type: "error",
        message: "❌ Sync failed: Connection error"
      });
      setTimeout(() => setSyncResult(null), 5000);
    } finally {
      setIsLoading(false);
    }
  };

  const handleProcessComplete = (result: ProcessResult) => {
    console.log("Process completed:", result);
  };

  const styles: Record<string, CSSProperties> = {
    container: {
      padding: "24px",
      maxWidth: "1400px",
      margin: "0 auto"
    },
    alert: {
      padding: "12px 16px",
      borderRadius: "8px",
      marginBottom: "16px",
      display: "flex",
      alignItems: "flex-start",
      gap: "8px"
    },
    alertSuccess: {
      backgroundColor: "#dcfce7",
      border: "1px solid #bbf7d0",
      color: "#16a34a"
    },
    alertError: {
      backgroundColor: "#fee2e2",
      border: "1px solid #fecaca",
      color: "#dc2626"
    },
    alertInfo: {
      backgroundColor: "#dbeafe",
      border: "1px solid #bfdbfe",
      color: "#1e40af"
    },
    header: {
      backgroundColor: "#f3f4f6",
      padding: "16px",
      borderRadius: "8px",
      marginBottom: "16px"
    },
    fileList: {
      marginTop: "16px"
    },
    selectedIndicator: {
      position: "fixed" as const,
      bottom: "24px",
      right: "24px",
      backgroundColor: "#2563eb",
      color: "white",
      padding: "8px 16px",
      borderRadius: "8px",
      boxShadow: "0 4px 6px rgba(0, 0, 0, 0.1)"
    }
  };

  return (
    <PageLayout 
      title="Documents"
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
        {/* Sync result banner */}
        {syncResult && (
          <div style={{
            ...styles.alert,
            ...(syncResult.type === "success" ? styles.alertSuccess : styles.alertError)
          }}>
            <div>
              <div style={{ fontWeight: 600 }}>{syncResult.message}</div>
              {syncResult.details && (
                <div style={{ fontSize: "12px", marginTop: "4px", opacity: 0.8 }}>
                  📦 Dropbox: {syncResult.details.dropboxFiles} files • 
                  💾 Google Drive: {syncResult.details.googleDriveFiles} files • 
                  💿 Backend: {syncResult.details.backend}
                </div>
              )}
            </div>
          </div>
        )}

        <FileBreadcrumbs
          currentPath={currentPath}
          onCrumbClick={setCurrentPath}
        />

        <div style={styles.header}>
          <EnhancedHeaderControls
            selectedFiles={selectedFiles}
            fileTypes={fileTypes}
            sortDir={sortDir}
            ignoreRegex={ignoreRegex}
            pathCategories={pathCategories}
            autoLabel={autoLabel}
            force={force}
            onForceChange={setForce}
            isSyncing={isLoading}
            onSync={handleSync}
            onSortChange={setSortDir}
            onFileTypeChange={setFileTypes}
            onRegexChange={setIgnoreRegex}
            onCategoryChange={setPathCategories}
            onLabelChange={setAutoLabel}
            onProcessComplete={handleProcessComplete}
            globalLoading={false}
          />
        </div>

        {/* File list */}
        <div style={styles.fileList}>
          {filteredFiles.length === 0 ? (
            <div style={{
              ...styles.alert,
              ...styles.alertInfo
            }}>
              <Info size={16} />
              {files.length > 0
                ? "No files found matching your criteria"
                : "No files found"}
            </div>
          ) : (
            filteredFiles.map(file => (
              <FileItemRow
                key={file.id}
                file={file}
                isSelected={selectedFiles.includes(file.id)}
                onSelect={handleFileSelect}
                onItemClick={handleItemClick}
              />
            ))
          )}
        </div>

        {/* Selected files indicator */}
        {selectedFiles.length > 0 && (
          <div style={styles.selectedIndicator}>
            {selectedFiles.length} file{selectedFiles.length > 1 ? "s" : ""} selected
          </div>
        )}
      </div>
    </PageLayout>
  );
};

export default UnifiedFileViewer;