// src/components/integrations/IntegrationsPage.tsx
import { useState } from "react";
import type { CSSProperties } from "react";
import PageLayout from "../layouts/PageLayout";
import UserDropdown from "../auth/UserDropdown";
import EnhancedHeaderControls from "./EnhancedHeaderControls";
import type { User } from "../../types/user";
import type { SortDirection, FileType, ProcessResult } from "./EnhancedHeaderControls";
import { setToken } from "../../api";

interface IntegrationsPageProps {
  user: User | null;
}

// Mock file data for demonstration
interface FileItem {
  id: string;
  name: string;
  type: FileType;
  size: number;
  modifiedAt: string;
  path: string;
  selected?: boolean;
}

const MOCK_FILES: FileItem[] = [
  { id: "1", name: "report_2024.pdf", type: "pdf", size: 2457600, modifiedAt: "2024-12-15", path: "/documents/reports" },
  { id: "2", name: "data_analysis.xlsx", type: "xlsx", size: 1843200, modifiedAt: "2024-12-18", path: "/documents/spreadsheets" },
  { id: "3", name: "presentation.pdf", type: "pdf", size: 5242880, modifiedAt: "2024-12-20", path: "/documents/presentations" },
  { id: "4", name: "contract.docx", type: "docx", size: 524288, modifiedAt: "2024-12-19", path: "/documents/legal" },
  { id: "5", name: "chart.png", type: "png", size: 307200, modifiedAt: "2024-12-17", path: "/documents/images" },
];

const IntegrationsPage: React.FC<IntegrationsPageProps> = ({ user }) => {
  // State management
  const [selectedFiles, setSelectedFiles] = useState<string[]>([]);
  const [fileTypes, setFileTypes] = useState<FileType[]>(["pdf"]);
  const [sortDir, setSortDir] = useState<SortDirection>("desc");
  const [ignoreRegex, setIgnoreRegex] = useState("");
  const [pathCategories, setPathCategories] = useState("");
  const [autoLabel, setAutoLabel] = useState("");
  const [force, setForce] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [files, setFiles] = useState<FileItem[]>(MOCK_FILES);

  // Handlers
  const handleSync = async () => {
    setIsSyncing(true);
    // Simulate sync operation
    setTimeout(() => {
      setIsSyncing(false);
      console.log("Sync completed");
    }, 2000);
  };

  const handleProcessComplete = (result: ProcessResult) => {
    console.log("Process completed:", result);
    if (result.success) {
      // Clear selection after successful processing
      setSelectedFiles([]);
    }
  };

  const handleFileSelect = (fileId: string) => {
    setSelectedFiles(prev => 
      prev.includes(fileId) 
        ? prev.filter(id => id !== fileId)
        : [...prev, fileId]
    );
  };

  const handleSelectAll = () => {
    const visibleFileIds = getFilteredFiles().map(f => f.id);
    if (selectedFiles.length === visibleFileIds.length) {
      setSelectedFiles([]);
    } else {
      setSelectedFiles(visibleFileIds);
    }
  };

  // Filter files based on selected types
  const getFilteredFiles = () => {
    let filtered = [...files];
    
    if (fileTypes.length > 0) {
      filtered = filtered.filter(file => fileTypes.includes(file.type));
    }
    
    if (ignoreRegex) {
      try {
        const regex = new RegExp(ignoreRegex, 'i');
        filtered = filtered.filter(file => !regex.test(file.path));
      } catch (e) {
        // Invalid regex, ignore filter
      }
    }
    
    // Sort files
    filtered.sort((a, b) => {
      const comparison = a.modifiedAt.localeCompare(b.modifiedAt);
      return sortDir === "asc" ? comparison : -comparison;
    });
    
    return filtered;
  };

  const filteredFiles = getFilteredFiles();

  const styles: Record<string, CSSProperties> = {
    container: {
      padding: "24px",
    },
    header: {
      marginBottom: "24px",
    },
    title: {
      fontSize: "1.5rem",
      fontWeight: 600,
      color: "#111827",
      marginBottom: "16px",
    },
    fileList: {
      backgroundColor: "white",
      borderRadius: "8px",
      boxShadow: "0 1px 3px rgba(0, 0, 0, 0.1)",
      overflow: "hidden",
    },
    fileHeader: {
      display: "grid",
      gridTemplateColumns: "40px 1fr 100px 150px 200px",
      padding: "12px 16px",
      backgroundColor: "#f9fafb",
      borderBottom: "1px solid #e5e7eb",
      fontSize: "12px",
      fontWeight: 600,
      color: "#6b7280",
      textTransform: "uppercase",
    },
    fileRow: {
      display: "grid",
      gridTemplateColumns: "40px 1fr 100px 150px 200px",
      padding: "12px 16px",
      borderBottom: "1px solid #e5e7eb",
      fontSize: "14px",
      color: "#374151",
      cursor: "pointer",
      transition: "background-color 0.2s",
    },
    fileRowSelected: {
      backgroundColor: "#eff6ff",
    },
    checkbox: {
      width: "18px",
      height: "18px",
      cursor: "pointer",
    },
    fileName: {
      fontWeight: 500,
      color: "#111827",
    },
    fileType: {
      display: "inline-block",
      padding: "2px 8px",
      backgroundColor: "#e5e7eb",
      borderRadius: "4px",
      fontSize: "12px",
      fontWeight: 500,
      textTransform: "uppercase",
    },
    stats: {
      padding: "16px",
      backgroundColor: "#f9fafb",
      borderRadius: "8px",
      marginBottom: "16px",
      display: "flex",
      gap: "32px",
    },
    statItem: {
      display: "flex",
      flexDirection: "column",
      gap: "4px",
    },
    statLabel: {
      fontSize: "12px",
      color: "#6b7280",
      textTransform: "uppercase",
    },
    statValue: {
      fontSize: "20px",
      fontWeight: 600,
      color: "#111827",
    },
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  return (
    <PageLayout 
      title="Integrations"
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
        <div style={styles.header}>
          <h2 style={styles.title}>Cloud Storage Integrations</h2>
          
          {/* Statistics */}
          <div style={styles.stats}>
            <div style={styles.statItem}>
              <span style={styles.statLabel}>Total Files</span>
              <span style={styles.statValue}>{files.length}</span>
            </div>
            <div style={styles.statItem}>
              <span style={styles.statLabel}>Selected</span>
              <span style={styles.statValue}>{selectedFiles.length}</span>
            </div>
            <div style={styles.statItem}>
              <span style={styles.statLabel}>Filtered</span>
              <span style={styles.statValue}>{filteredFiles.length}</span>
            </div>
          </div>

          {/* Header Controls */}
          <EnhancedHeaderControls
            selectedFiles={selectedFiles}
            fileTypes={fileTypes}
            sortDir={sortDir}
            ignoreRegex={ignoreRegex}
            pathCategories={pathCategories}
            autoLabel={autoLabel}
            force={force}
            onForceChange={setForce}
            isSyncing={isSyncing}
            onSync={handleSync}
            onSortChange={setSortDir}
            onFileTypeChange={setFileTypes}
            onRegexChange={setIgnoreRegex}
            onCategoryChange={setPathCategories}
            onLabelChange={setAutoLabel}
            onProcessComplete={handleProcessComplete}
          />
        </div>

        {/* File List */}
        <div style={styles.fileList}>
          <div style={styles.fileHeader}>
            <input
              type="checkbox"
              style={styles.checkbox}
              checked={selectedFiles.length === filteredFiles.length && filteredFiles.length > 0}
              onChange={handleSelectAll}
            />
            <span>Name</span>
            <span>Type</span>
            <span>Size</span>
            <span>Modified</span>
          </div>

          {filteredFiles.map(file => (
            <div
              key={file.id}
              style={{
                ...styles.fileRow,
                ...(selectedFiles.includes(file.id) ? styles.fileRowSelected : {}),
              }}
              onClick={() => handleFileSelect(file.id)}
            >
              <input
                type="checkbox"
                style={styles.checkbox}
                checked={selectedFiles.includes(file.id)}
                onChange={() => {}}
                onClick={(e) => e.stopPropagation()}
              />
              <span style={styles.fileName}>{file.name}</span>
              <span style={styles.fileType}>{file.type}</span>
              <span>{formatFileSize(file.size)}</span>
              <span>{file.modifiedAt}</span>
            </div>
          ))}

          {filteredFiles.length === 0 && (
            <div style={{ padding: "32px", textAlign: "center", color: "#6b7280" }}>
              No files match the selected filters
            </div>
          )}
        </div>
      </div>
    </PageLayout>
  );
};

export default IntegrationsPage;